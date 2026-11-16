"""Run a reproducible TCS retrieval comparison and generate an answer-review packet."""
from __future__ import annotations

import hashlib
import logging
import json
import os
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from backend.rag.chunker import load_documents, chunk_documents
from backend.rag.vector_store import get_collection
from backend.rag.graph_index import graph_retrieve
from backend.rag.rag_pipeline import answer_question
from evaluation.benchmark import evaluate_records
from evaluation.metrics import citation_coverage
from evaluation.traceability import page_for_excerpt, page_for_anchor

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "evaluation" / "tcs_benchmark.json"
OUTPUT_DIR = ROOT / "evaluation" / "results"
TOP_K = 5
CANDIDATE_K = 40
MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?", re.I)
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])[-+]?\d[\d,]*(?:\.\d+)?")
CITATION_RE = re.compile(r"(?:\[S(\d+)\]|【S(\d+)】)")
ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).casefold())


def stable_chunk_id(item: dict) -> str:
    payload = f'{item["source"]}|{item.get("page")}|{item["chunk_id"]}|{item["text"]}'
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def resolve_relevance(rows: list[dict], chunks: list[dict]) -> list[dict]:
    """Resolve each reviewed source anchor to the stable Chroma chunk IDs."""
    enriched = []
    for row in rows:
        source_chunks = [item for item in chunks if item["source"] == row["source"]]
        anchor = normalize(row["anchor"])
        hits = [item for item in source_chunks if anchor in normalize(item["text"])]
        if not hits:
            raise RuntimeError(
                f"No chunk contains reviewed anchor for {row['id']}: "
                f'{row["source"]} :: {row["anchor"]}'
            )
        enriched.append({**row, "relevant_chunk_ids": [stable_chunk_id(item) for item in hits]})
    return enriched


def dense_ranking(collection, embedding: list[float], candidate_k: int) -> list[str]:
    result = collection.query(
        query_embeddings=[embedding],
        n_results=min(candidate_k, collection.count()),
        include=["distances"],
    )
    return (result.get("ids") or [[]])[0]


def build_bm25(collection):
    """Build a full-corpus BM25 index for offline evaluation (not the 512 MiB API path)."""
    result = collection.get(include=["documents"])
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    if not ids or len(ids) != len(documents):
        raise RuntimeError("Unable to load aligned document IDs and text for BM25 evaluation.")
    corpus = [tokenize(document or "") for document in documents]
    if not any(corpus):
        raise RuntimeError("BM25 corpus contains no tokens.")
    return ids, BM25Okapi(corpus)


def bm25_ranking(query: str, ids: list[str], index: BM25Okapi, candidate_k: int) -> list[str]:
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = index.get_scores(tokens)
    ranked = sorted(
        ((float(score), chunk_id) for score, chunk_id in zip(scores, ids) if score > 0),
        key=lambda item: (-item[0], item[1]),
    )
    return [chunk_id for _, chunk_id in ranked[:candidate_k]]


def rrf(*rankings: list[str], limit: int = TOP_K) -> list[str]:
    scores = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)
    return [key for key, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def aggregate(method_rows: list[dict]) -> dict:
    records = [
        {"question": row["question"], "relevant_chunk_ids": row["relevant_chunk_ids"],
         "retrieved_chunk_ids": row["retrieved_chunk_ids"]}
        for row in method_rows
    ]
    return evaluate_records(records, k=TOP_K)


def cited_source_indices(answer: str, source_count: int) -> list[int]:
    values = []
    clean_answer = ZERO_WIDTH_RE.sub("", answer)
    for match in CITATION_RE.finditer(clean_answer):
        value = int(match.group(1) or match.group(2))
        if 1 <= value <= source_count and value not in values:
            values.append(value)
    return values


def cited_gold_support(answer: str, sources: list[dict], relevant_ids: set[str]) -> bool:
    return any(sources[index - 1].get("chunk_id") in relevant_ids
               for index in cited_source_indices(answer, len(sources)))


def numeric_tokens(text: str) -> list[Decimal]:
    values = []
    for token in NUMBER_RE.findall(str(text)):
        try:
            values.append(Decimal(token.replace(",", "")))
        except InvalidOperation:
            continue
    return values


def numeric_value_present(answer: str, expected: str, unit: str = "") -> bool:
    expected_values = numeric_tokens(expected)
    answer_values = numeric_tokens(answer)
    if not expected_values:
        return False
    target = expected_values[0]
    if target in answer_values:
        return True
    # Financial statements may express the same INR amount in million rather than crore.
    if "crore" in unit.casefold() and re.search(r"\bmillions?\b", answer, re.I):
        return any(value / Decimal(10) == target for value in answer_values)
    # Common abbreviated headcount notation, e.g. 270k+.
    if "employees" in unit.casefold() and re.search(r"\b\d+(?:\.\d+)?\s*k\+?\b", answer, re.I):
        return any(value * Decimal(1000) == target for value in answer_values)
    return False


def source_contains_expected(source_text: str, expected: str) -> bool:
    values = numeric_tokens(expected)
    if not values:
        return False
    normalized = re.sub(r"(?<=\d),(?=\d)", "", source_text)
    return any(value in numeric_tokens(normalized) for value in values)


def main() -> None:
    rows = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    chunks = chunk_documents(load_documents())
    rows = resolve_relevance(rows, chunks)

    collection = get_collection()
    count = collection.count()
    if count <= 0:
        raise RuntimeError("The downloaded Chroma collection is empty.")
    indexed_ids, bm25_index = build_bm25(collection)

    model = SentenceTransformer(MODEL_NAME)
    methods = {"dense_only": [], "bm25_only": [], "dense_plus_bm25": [], "dense_bm25_graph": []}
    for number, row in enumerate(rows, start=1):
        embedding = model.encode(row["question"], normalize_embeddings=True).tolist()
        dense = dense_ranking(collection, embedding, CANDIDATE_K)
        bm25 = bm25_ranking(row["question"], indexed_ids, bm25_index, CANDIDATE_K)
        graph = [item["id"] for item in graph_retrieve(row["question"], top_k=CANDIDATE_K)]

        ranked = {
            "dense_only": dense[:TOP_K],
            "bm25_only": bm25[:TOP_K],
            "dense_plus_bm25": rrf(dense, bm25),
            "dense_bm25_graph": rrf(dense, bm25, graph),
        }
        for method, retrieved in ranked.items():
            methods[method].append({**row, "retrieved_chunk_ids": retrieved})
        print(f"Evaluated {number}/{len(rows)}: {row['id']}")

    method_metrics = {}
    for method, records in methods.items():
        result = aggregate(records)
        method_metrics[method] = {
            "mean_recall_at_5": result["mean_recall_at_k"],
            "mean_mrr_at_5": result["mean_mrr_at_k"],
            "per_question": result["per_question"],
        }

    summary = {
        "benchmark": "TCS financial disclosures, source-anchored 50-question set",
        "question_count": len(rows),
        "k": TOP_K,
        "model": MODEL_NAME,
        "index_chunk_count": count,
        "bm25_corpus_count": len(indexed_ids),
        "methods": method_metrics,
        "labeling": "Relevant chunks resolved from manually curated source-file and evidence-text anchors; every anchor must resolve or the run fails.",
        "limitations": [
            "This benchmark measures retrieval against source-anchored relevant chunks, not answer correctness.",
            "Overlapping chunks containing the same anchor are all labelled relevant; this can affect Recall@K.",
            "Alternative documents supporting the same fact are not labelled unless explicitly included as relevance sources.",
            "The BM25 comparison is a full-corpus offline baseline; it is not the memory-bounded lexical candidate path used by the live API.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "retrieval_comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    answer_audits = []
    if os.getenv("GROQ_API_KEY"):
        # Generate an answer for every benchmark question, not only a hand-picked subset.
        audit_rows = rows
        for audit_number, row in enumerate(audit_rows, start=1):
            # The configured Groq free/on-demand TPM budget is shared across requests.
            # Two answer generations per minute leave room for both prompt and completion tokens.
            if audit_number > 1 and (audit_number - 1) % 2 == 0:
                time.sleep(65)
            embedding = model.encode(row["question"], normalize_embeddings=True).tolist()
            try:
                result = None
                last_error = None
                for attempt in range(2):
                    try:
                        result = answer_question(row["question"], query_embedding=embedding, top_k=5)
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt == 0:
                            logger.warning("Retrying empty/failed Groq answer for %s: %s", row["id"], exc)
                            time.sleep(15)
                if result is None:
                    raise last_error or RuntimeError("Answer generation returned no result")
            except Exception as exc:
                # Preserve one failed generation as an auditable record and continue.
                # A transient provider failure must not discard the completed retrieval report.
                logger.exception("Answer generation failed for %s", row["id"])
                answer_audits.append({
                    "id": row["id"],
                    "question": row["question"],
                    "expected_answer": row["expected_value"],
                    "expected_unit": row["unit"],
                    "expected_source": row["source"],
                    "expected_anchor": row["anchor"],
                    "expected_page": page_for_anchor(
                        (ROOT / row["source"]).read_text(encoding="utf-8"), row["anchor"]
                    ),
                    "answer": None,
                    "generation_error": f"{type(exc).__name__}: {exc}",
                    "sources": [],
                    "cited_evidence": [],
                    "citation_coverage": 0.0,
                    "cited_gold_evidence_present": False,
                    "expected_numeric_value_present": False,
                    "expected_value_present_in_cited_evidence": False,
                    "human_answer_relevance_review": "not_reviewable_generation_failed",
                    "human_faithfulness_review": "not_reviewable_generation_failed",
                    "human_numerical_review": "not_reviewable_generation_failed",
                    "human_citation_correctness_review": "not_reviewable_generation_failed",
                    "human_page_traceability_review": "not_reviewable_generation_failed",
                })
                continue
            answer = result["answer"]
            sources = result["sources"]
            cited_indices = cited_source_indices(answer, len(sources))
            cited_sources = [sources[index - 1] for index in cited_indices]
            evidence = []
            for source in cited_sources:
                found = collection.get(ids=[source["chunk_id"]], include=["documents"])
                docs = found.get("documents") or []
                excerpt = (docs[0] or "")[:1600] if docs else ""
                source_path = ROOT / source.get("source", "")
                source_text = source_path.read_text(encoding="utf-8") if source_path.is_file() else ""
                resolved_page = source.get("page") or page_for_excerpt(source_text, excerpt)
                evidence.append({
                    **source,
                    "page": resolved_page,
                    "excerpt": excerpt,
                    "page_traceable": resolved_page is not None,
                })
            answer_audits.append({
                "id": row["id"],
                "question": row["question"],
                "expected_answer": row["expected_value"],
                "expected_unit": row["unit"],
                "expected_source": row["source"],
                "expected_anchor": row["anchor"],
                "expected_page": page_for_anchor(
                    (ROOT / row["source"]).read_text(encoding="utf-8"), row["anchor"]
                ),
                "answer": answer,
                "sources": sources,
                "cited_evidence": evidence,
                "citation_coverage": citation_coverage(answer, len(sources)),
                "cited_gold_evidence_present": cited_gold_support(
                    answer, sources, set(row["relevant_chunk_ids"])
                ),
                "expected_numeric_value_present": numeric_value_present(
                    answer, row["expected_value"], row["unit"]
                ),
                "expected_value_present_in_cited_evidence": any(
                    source_contains_expected(item["excerpt"], row["expected_value"])
                    for item in evidence
                ),
                "human_answer_relevance_review": "pending",
                "human_faithfulness_review": "pending",
                "human_numerical_review": "pending",
                "human_citation_correctness_review": "pending",
                "human_page_traceability_review": "pending",
            })
    else:
        print("GROQ_API_KEY not configured; answer-generation audit skipped.")

    audit_packet = {
        "status": ("machine_checks_complete_human_review_pending" if all("generation_error" not in item for item in answer_audits) else "generation_completed_with_errors_human_review_pending"),
        "answer_count": len(answer_audits),
        "successful_answer_count": sum("generation_error" not in item for item in answer_audits),
        "generation_error_count": sum("generation_error" in item for item in answer_audits),
        "benchmark_question_count": len(rows),
        "answer_context_top_k": 5,
        "review_rubric": {
            "answer_relevance": "pass / partial / fail: directly answers the asked metric, entity, period and comparison.",
            "faithfulness": "pass / partial / fail: every material claim is entailed by the cited excerpt; no unsupported inference.",
            "numerical_accuracy": "pass / partial / fail: value, currency, unit, sign, period and accounting basis all match.",
            "citation_correctness": "pass / partial / fail: each citation points to evidence that supports its adjacent claim.",
            "page_traceability": "pass / partial / fail: source file and page are resolvable to the original disclosure.",
            "reviewer_notes": "Record the exact source page/line and explain every partial or failed check."
        },
        "checks": "Machine checks include citation marker coverage, expected-value token presence and source excerpt traceability. They are screening checks, not a substitute for reviewer sign-off. Review all five dimensions for every answer before reporting answer-quality rates.",
        "answers": answer_audits,
    }
    (OUTPUT_DIR / "answer_audit_packet.json").write_text(
        json.dumps(audit_packet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "retrieval_comparison": str(OUTPUT_DIR / "retrieval_comparison.json"),
        "answer_audit_packet": str(OUTPUT_DIR / "answer_audit_packet.json"),
        "metrics": {name: {
            "recall_at_5": value["mean_recall_at_5"],
            "mrr_at_5": value["mean_mrr_at_5"],
        } for name, value in method_metrics.items()},
        "answer_audit_count": len(answer_audits),
    }, indent=2))


if __name__ == "__main__":
    main()
