"""Aggregate retrieval metrics over a manually reviewed JSONL benchmark.

Each JSONL row must contain question, relevant_chunk_ids and retrieved_chunk_ids.
The latter should be the ordered IDs produced by the retrieval system being tested.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from evaluation.metrics import recall_at_k, reciprocal_rank_at_k


def evaluate_records(records: list[dict], k: int = 5) -> dict:
    if not records:
        raise ValueError("Benchmark must contain at least one question.")
    if k < 1:
        raise ValueError("k must be at least 1.")

    recalls, reciprocal_ranks = [], []
    for row_number, row in enumerate(records, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Row {row_number} must be a JSON object.")
        question = row.get("question")
        relevant = row.get("relevant_chunk_ids")
        retrieved = row.get("retrieved_chunk_ids")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Row {row_number} has no valid question.")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(f"Row {row_number} needs reviewed relevant_chunk_ids.")
        if not isinstance(retrieved, list):
            raise ValueError(f"Row {row_number} needs ordered retrieved_chunk_ids.")
        if any(not isinstance(value, str) or not value for value in relevant + retrieved):
            raise ValueError(f"Row {row_number} contains an invalid chunk ID.")
        if len(set(relevant)) != len(relevant):
            raise ValueError(f"Row {row_number} contains duplicate relevant chunk IDs.")
        recalls.append(recall_at_k(retrieved, relevant, k))
        reciprocal_ranks.append(reciprocal_rank_at_k(retrieved, relevant, k))

    return {
        "question_count": len(records),
        "k": k,
        "mean_recall_at_k": round(mean(recalls), 4),
        "mean_mrr_at_k": round(mean(reciprocal_ranks), 4),
        "per_question": [
            {"question": row["question"], "recall_at_k": round(recalls[i], 4),
             "reciprocal_rank_at_k": round(reciprocal_ranks[i], 4)}
            for i, row in enumerate(records)
        ],
        "interpretation": "Offline retrieval metrics against manually reviewed relevance labels; not answer faithfulness or citation correctness.",
    }


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path, help="Reviewed JSONL with retrieved chunk IDs")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    result = evaluate_records(load_jsonl(args.benchmark), args.k)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
