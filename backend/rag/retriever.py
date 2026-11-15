"""Memory-conscious hybrid retrieval using frontend query vectors, BM25 and graph evidence.

The hosted API does not load PyTorch/SentenceTransformers. Streamlit supplies
the normalized BGE query vector; Render searches the precomputed Chroma index.
"""
from threading import Lock

_records_cache = []
_bm25_cache = None
_cache_count = -1
_cache_lock = Lock()


def _all_records(collection):
    """Read Chroma in bounded pages rather than materializing one large response."""
    records = []
    offset = 0
    page_size = 250
    while True:
        result = collection.get(
            include=["documents", "metadatas"],
            limit=page_size,
            offset=offset,
        )
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        for index, chunk_id in enumerate(ids):
            if index >= len(docs):
                continue
            records.append({
                "id": chunk_id,
                "text": docs[index] or "",
                "metadata": (metas[index] if index < len(metas) else None) or {},
            })
        if len(ids) < page_size:
            break
        offset += len(ids)
    return records


def _get_lexical_index(collection):
    """Build BM25 once per collection size, guarded against concurrent builds."""
    global _records_cache, _bm25_cache, _cache_count
    count = collection.count()
    if count != _cache_count:
        with _cache_lock:
            count = collection.count()
            if count != _cache_count:
                records = _all_records(collection)
                if records:
                    from rank_bm25 import BM25Okapi
                    tokenized = [item["text"].lower().split() for item in records]
                    bm25 = BM25Okapi(tokenized)
                else:
                    bm25 = None
                _records_cache, _bm25_cache, _cache_count = records, bm25, count
    return _records_cache, _bm25_cache


def retrieve(query, top_k=3, query_embedding=None):
    if not query or not query.strip():
        raise ValueError("Query must not be empty")
    if top_k < 1 or top_k > 10:
        raise ValueError("top_k must be between 1 and 10")

    from backend.rag.vector_store import get_collection
    from backend.rag.graph_index import graph_retrieve

    collection = get_collection()
    if not collection.count():
        return {"documents": [], "sources": [], "distances": [], "retrieval": []}

    records, bm25 = _get_lexical_index(collection)
    lexical_scores = {}
    if bm25 is not None:
        scores = bm25.get_scores(query.lower().split())
        # Keep only a small candidate pool to avoid sorting/retaining every hit.
        candidates = sorted(
            ((records[i]["id"], float(score)) for i, score in enumerate(scores) if score > 0),
            key=lambda item: item[1],
            reverse=True,
        )[: top_k * 8]
        lexical_scores = dict(candidates)

    graph_hits = graph_retrieve(query, top_k=top_k * 8)
    graph_scores = {item["id"]: float(item["score"]) for item in graph_hits}

    dense_scores = {}
    if query_embedding is not None:
        if not isinstance(query_embedding, list) or len(query_embedding) != 384:
            raise ValueError("query_embedding must contain exactly 384 values for BAAI/bge-small-en-v1.5")
        dense_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 8, collection.count()),
            include=["distances"],
        )
        dense_ids = (dense_result.get("ids") or [[]])[0]
        dense_distances = (dense_result.get("distances") or [[]])[0]
        dense_scores = {
            chunk_id: 1.0 / (1.0 + max(float(distance), 0.0))
            for chunk_id, distance in zip(dense_ids, dense_distances)
        }

    # Reciprocal-rank fusion across dense, lexical and graph candidates.
    rankings = [
        [key for key, _ in sorted(dense_scores.items(), key=lambda item: item[1], reverse=True)],
        [key for key, _ in sorted(lexical_scores.items(), key=lambda item: item[1], reverse=True)],
        [key for key, _ in sorted(graph_scores.items(), key=lambda item: item[1], reverse=True)],
    ]
    fused = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, 1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (60 + rank)

    by_id = {item["id"]: item for item in records}
    ordered = sorted(fused, key=fused.get, reverse=True)[:top_k]
    selected = [by_id[key] for key in ordered if key in by_id]
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
        "retrieval": (["dense"] if dense_scores else []) + (["bm25"] if bm25 is not None else []) + (["knowledge_graph"] if graph_scores else []),
    }
