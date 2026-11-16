"""Memory-bounded hybrid retrieval.

Uses bounded Chroma document filters for lexical candidates rather than
materializing every document and a full in-memory BM25 index on the 512 MiB
Render instance. Dense and graph retrieval remain bounded to small candidate
sets.
"""
from threading import Lock
import math
import re

_CACHE_LOCK = Lock()
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}", re.I)


def _lexical_candidates(collection, query, limit=40):
    """Fetch a small candidate set for each informative query term."""
    terms = []
    for term in _TOKEN_RE.findall(query.lower()):
        if term not in terms and term not in {"the", "and", "for", "with", "from", "what", "how", "was", "were", "are"}:
            terms.append(term)
        if len(terms) >= 5:
            break

    scores = {}
    records = {}
    for term in terms:
        try:
            result = collection.get(
                where_document={"$contains": term},
                include=["documents", "metadatas"],
                limit=limit,
            )
        except Exception:
            continue
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        for index, chunk_id in enumerate(ids):
            if index >= len(docs):
                continue
            document = docs[index] or ""
            metadata = metas[index] if index < len(metas) else None
            records[chunk_id] = {
                "id": chunk_id,
                "text": document,
                "metadata": metadata or {},
            }
            scores[chunk_id] = scores.get(chunk_id, 0) + 1
    ranked = sorted(scores, key=lambda key: (-scores[key], key))[:limit]
    return {key: records[key] for key in ranked}, {key: float(scores[key]) for key in ranked}


def retrieve(query, top_k=3, query_embedding=None):
    if not query or not query.strip():
        raise ValueError("Query must not be empty")
    if top_k < 1 or top_k > 10:
        raise ValueError("top_k must be between 1 and 10")

    from backend.rag.vector_store import get_collection
    from backend.rag.graph_index import graph_retrieve

    collection = get_collection()
    count = collection.count()
    if not count:
        return {"documents": [], "sources": [], "distances": [], "retrieval": []}

    records, lexical_scores = _lexical_candidates(collection, query, limit=top_k * 8)
    graph_hits = graph_retrieve(query, top_k=top_k * 8)
    graph_scores = {item["id"]: float(item["score"]) for item in graph_hits}

    dense_scores = {}
    if query_embedding is not None:
        from backend.rag.embeddings import EMBEDDING_DIMENSION
        if not isinstance(query_embedding, list) or len(query_embedding) != EMBEDDING_DIMENSION:
            raise ValueError(f"query_embedding must contain exactly {EMBEDDING_DIMENSION} values for the configured Groq embedding model")
        if not all(math.isfinite(float(value)) for value in query_embedding):
            raise ValueError("query_embedding values must all be finite")
        norm = math.sqrt(sum(float(value) ** 2 for value in query_embedding))
        if not 0.90 <= norm <= 1.10:
            raise ValueError("query_embedding must be L2-normalized")
        dense_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 8, count),
            include=["documents", "metadatas", "distances"],
        )
        dense_ids = (dense_result.get("ids") or [[]])[0]
        dense_distances = (dense_result.get("distances") or [[]])[0]
        dense_docs = (dense_result.get("documents") or [[]])[0]
        dense_metas = (dense_result.get("metadatas") or [[]])[0]
        for index, (chunk_id, distance) in enumerate(zip(dense_ids, dense_distances)):
            dense_scores[chunk_id] = 1.0 / (1.0 + max(float(distance), 0.0))
            if chunk_id not in records and index < len(dense_docs):
                records[chunk_id] = {
                    "id": chunk_id,
                    "text": dense_docs[index] or "",
                    "metadata": dense_metas[index] if index < len(dense_metas) and dense_metas[index] else {},
                }

    rankings = [
        [key for key, _ in sorted(dense_scores.items(), key=lambda item: item[1], reverse=True)],
        [key for key, _ in sorted(lexical_scores.items(), key=lambda item: item[1], reverse=True)],
        [key for key, _ in sorted(graph_scores.items(), key=lambda item: item[1], reverse=True)],
    ]
    fused = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, 1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (60 + rank)

    ordered = sorted(fused, key=fused.get, reverse=True)[:top_k]
    selected = [records[key] for key in ordered if key in records]
    return {
        "documents": [item["text"][:5000] for item in selected],
        "sources": [
            {
                "source": item["metadata"].get("source", "Unknown"),
                "page": item["metadata"].get("page") or None,
                "chunk_id": item["id"],
                "score": round(fused[item["id"]], 6),
            }
            for item in selected
        ],
        "distances": [fused[item["id"]] for item in selected],
        "retrieval": (["dense"] if dense_scores else [])
        + (["bm25"] if lexical_scores else [])
        + (["knowledge_graph"] if graph_scores else []),
    }
