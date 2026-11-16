"""Offline retrieval and citation metrics for reviewed benchmarks."""
import re


def recall_at_k(retrieved_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)


def reciprocal_rank_at_k(retrieved_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def citation_coverage(answer, source_count):
    """Fraction of answer sentences carrying at least one valid [S#] or 【S#】 citation."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s]
    if not sentences:
        return 0.0
    valid = set(range(1, source_count + 1))
    cited = 0
    for sentence in sentences:
        refs = {
            int(match.group(1) or match.group(2))
            for match in re.finditer(r"(?:\[S(\d+)\]|【S(\d+)】)", sentence)
        }
        if refs & valid:
            cited += 1
    return cited / len(sentences)
