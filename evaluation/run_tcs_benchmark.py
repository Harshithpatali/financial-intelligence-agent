"""Run the frozen TCS retrieval benchmark and produce an answer-audit packet."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer

from backend.rag.chunker import load_documents, chunk_documents
from backend.rag.vector_store import get_collection
from backend.rag.retriever import _lexical_candidates
from backend.rag.graph_index import graph_retrieve
from backend.rag.rag_pipeline import answer_question
from evaluation.benchmark import evaluate_records

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "evaluation" / "tcs_benchmark.json"
OUTPUT_DIR = ROOT / "evaluation" / "results"
TOP_K = 5
CANDIDATE_K = 40
MODEL_NAME = "BAAI/bge-small-en-v1.5"
AUDIT_IDS = {"q01", "q05", "q07", "q09", "q12", "q16", "q20",
             "q23", "q28", "q35", "q40", "q47", "q49"}


def normalize(text: str) -> str:
    return re.sub(r"\\s+", " ", str(text)).strip().casefold()


def stable_chunk_id(item: dict) -> str:
    payload = f'{item["source"]}|{item.get("page")}|{item["chunk_id"]}|{item["text"]}'
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def resolve_relevance(rows: list[dict], chunks: list[dict]) -> list[dict]:
    """Resolve reviewed source anchors to the exact stable IDs used by Chroma."""
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


def cited_gold_support(answer: str, sources: list[dict], relevant_ids: set[str]) -> bool:
    cited = {int(value) for value in re.findall(r"\\[S(\\d+)\\]", answer)}
    return any(1 <= index <= len(sources)
               and sources[index - 1].get("chunk_id") in relevant_ids
               for index in cited)


def numeric_value_present(answer: str, expected: str) -> bool:
    digits = re.sub(r"[^0-9.\\-]", "", expected)
    answer_digits = re.sub(r"[^0-9.\\-]", "", answer.replace(",", ""))
    if not digits:
        return False
    return digits in answer_digits


def main() -> None:
    rows = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    chunks = chunk_documents(load_documents())
    rows = resolve_relevance(rows, chunks)
    collection = get_collection()
    if collection.count() <= 0:
        raise RuntimeError("The downloaded Chroma collection is empty.")

    model = SentenceTransformer(MODEL_NAME)
    methods = {"dense_only": [], "dense_plus_bm25": [], "dense_bm25_graph": []}
    for number, row in enumerate(rows, start=1):
        embedding = model.encode(row["question"], normalize_embeddings=True).tolist()
        dense = dense_ranking(collection, embedding, CANDIDATE_K)
        _, lexical_scores = _lexical_candidates(collection, row["question"], limit=CANDIDATE_K)
        lexical = sorted(lexical_scores, key=lambda key: (-lexical_scores[key], key))
        graph = [item["id"] for item in graph_retrieve(row["question"], top_k=CANDIDATE_K)]

        ranked = {
            "dense_only": dense[:TOP_K],
            "dense_plus_bm25": rrf(dense, lexical),
            "dense_bm25_graph": rrf(dense, lexical, graph),
        }
        for method, retrieved in ranked.items():
            methods[method].append({**row, "retrieved_chunk_ids": retrieved})
        print(f"Evaluated {number}/{len(rows)}: {row['id']}")

    summary = {
        "benchmark": "TCS financial disclosures, source-anchored 50-question set",
        "question_count": len(rows),
        "k": TOP_K,
        "model": MODEL_NAME,
        "index_chunk_count": collection.count(),
        "methods": {
            method: {
                "mean_recall_at_5": aggregate(records)["mean_recall_at_k"],
                "mean_mrr_at_5": aggregate(records)["mean_mrr_at_k"],
                "per_question": aggregate(records)["per_question"],
            }
            for method, records in methods.items()
        },
        "labeling": "Relevant chunks resolved from manually curated source-file and evidence-text anchors; anchor resolution fails closed if no chunk matches.",
        "limitations": [
            "This benchmark measures retrieval against source-anchored relevant chunks, not answer correctness.",
            "One anchor may resolve to overlapping chunks; all matching chunks are treated as relevant.",
            "The source documents are extracted text and may contain OCR or reporting-period issues; source provenance must be checked for material claims.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "retrieval_comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8"
    )

    answer_audits = []
    if __import__("os").getenv("GROQ_API_KEY"):
        for row in rows:
            if row["id"] not in AUDIT_IDS:
                continue
            embedding = model.encode(row["question"], normalize_embeddings=True).tolist()
            result = answer_question(row["question"], query_embedding=embedding)
            answer = result["answer"]
            sources = result["sources"]
            answer_audits.append({
                "id": row["id"],
                "question": row["question"],
                "expected_answer": row["expected_value"],
                "expected_unit": row["unit"],
                "answer": answer,
                "sources": sources,
                "citation_coverage": __import__("evaluation.metrics", fromlist=["citation_coverage"]).citation_coverage(
                    answer, len(sources)
                ),
                "cited_gold_evidence_present": cited_gold_support(
                    answer, sources, set(row["relevant_chunk_ids"])
                ),
                "expected_numeric_value_present": numeric_value_present(
                    answer, row["expected_value"]
                ),
                "human_faithfulness_review": "pending",
                "human_numerical_review": "pending",
            })
    else:
        print("GROQ_API_KEY not configured; answer-generation audit skipped.")

    audit_packet = {
        "status": "machine_checks_complete_human_review_pending",
        "answer_count": len(answer_audits),
        "checks": "Citation coverage, citation-to-gold-chunk overlap and expected numeric token presence are automated screening checks only.",
        "answers": answer_audits,
    }
    (OUTPUT_DIR / "answer_audit_packet.json").write_text(
        json.dumps(audit_packet, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8"
    )
    print(json.dumps({
        "retrieval_comparison": str(OUTPUT_DIR / "retrieval_comparison.json"),
        "answer_audit_packet": str(OUTPUT_DIR / "answer_audit_packet.json"),
        "metrics": {name: {
            "recall_at_5": value["mean_recall_at_5"],
            "mrr_at_5": value["mean_mrr_at_5"],
        } for name, value in summary["methods"].items()},
        "answer_audit_count": len(answer_audits),
    }, indent=2))


if __name__ == "__main__":
    main()
