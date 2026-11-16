import pytest

from evaluation.benchmark import evaluate_records
from evaluation.metrics import recall_at_k, reciprocal_rank_at_k, citation_coverage


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "c", "d"}, 2) == 1 / 3


def test_mrr_at_k():
    assert reciprocal_rank_at_k(["x", "b", "a"], {"a", "b"}, 2) == 0.5
    assert reciprocal_rank_at_k(["x"], {"a"}, 5) == 0


def test_citation_coverage_supports_square_unicode_and_zero_width_markers():
    assert citation_coverage("Revenue rose [S1]. Margin fell 【S3】.", 1) == 0.5
    assert citation_coverage("Revenue rose 【S1】. Margin grew [S2].", 2) == 1.0
    assert citation_coverage("Revenue rose [\u200bS1\u200b].", 1) == 1.0


def test_benchmark_aggregates_recall_and_mrr():
    result = evaluate_records([
        {"question": "Q1", "relevant_chunk_ids": ["a", "c"], "retrieved_chunk_ids": ["a", "b"]},
        {"question": "Q2", "relevant_chunk_ids": ["z"], "retrieved_chunk_ids": ["x", "z"]},
    ], k=2)
    assert result["question_count"] == 2
    assert result["mean_recall_at_k"] == 0.75
    assert result["mean_mrr_at_k"] == 0.75


def test_benchmark_rejects_empty_or_duplicate_relevance_labels():
    with pytest.raises(ValueError, match="relevant_chunk_ids"):
        evaluate_records([{"question": "Q", "relevant_chunk_ids": [], "retrieved_chunk_ids": []}])
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_records([{"question": "Q", "relevant_chunk_ids": ["a", "a"], "retrieved_chunk_ids": []}])


def test_benchmark_rejects_empty_benchmark():
    with pytest.raises(ValueError, match="at least one"):
        evaluate_records([])
