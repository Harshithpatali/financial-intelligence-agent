"""Create the graph index from an already-built Chroma collection (no re-embedding)."""
from backend.rag.vector_store import get_collection
from backend.rag.graph_index import build_graph, INDEX_PATH

def main():
    if INDEX_PATH.exists():
        print(f"Graph index already exists: {INDEX_PATH}")
        return
    collection = get_collection()
    result = collection.get(include=["documents", "metadatas"])
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    chunks = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] or {}
        chunks.append({
            "id": chunk_id,
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page") or None,
            "text": documents[index],
        })
    if not chunks:
        raise RuntimeError("Chroma collection is empty. Build the vector index first.")
    graph = build_graph(chunks)
    print(f"Built graph from {len(chunks)} indexed chunks: {graph.number_of_nodes()} entities, {graph.number_of_edges()} co-mention edges.")

if __name__ == "__main__":
    main()
