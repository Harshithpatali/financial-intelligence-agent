"""Build the Chroma vector index and evidence graph using Groq embeddings only."""
import hashlib

from tqdm import tqdm

from backend.rag.chunker import load_documents, chunk_documents
from backend.rag.embeddings import EMBEDDING_MODEL, EMBEDDING_DIMENSION, embed_texts
from backend.rag.vector_store import reset_collection
from backend.rag.graph_index import build_graph


def main():
    chunks = chunk_documents(load_documents())
    if not chunks:
        raise RuntimeError("No Markdown chunks found under ./knowledge")

    collection = reset_collection()
    graph_chunks = []
    batch_size = 32
    for start in tqdm(range(0, len(chunks), batch_size), desc="Groq indexing"):
        batch = chunks[start:start + batch_size]
        embeddings = embed_texts(
            [item["text"] for item in batch],
            input_type="search_document",
        )
        ids, metadata = [], []
        for item in batch:
            stable_id = hashlib.sha1(
                f'{item["source"]}|{item.get("page")}|{item["chunk_id"]}|{item["text"]}'.encode("utf-8")
            ).hexdigest()
            ids.append(stable_id)
            metadata.append({
                "source": item["source"],
                "chunk_id": str(item["chunk_id"]),
                "page": str(item["page"]) if item.get("page") is not None else "",
                "doc_type": item.get("doc_type", "document"),
                "embedding_provider": "groq",
                "embedding_model": EMBEDDING_MODEL,
            })
            graph_chunks.append({"id": stable_id, **item})

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=[item["text"] for item in batch],
            metadatas=metadata,
        )

    graph = build_graph(graph_chunks)
    if collection.count() != len(chunks):
        raise RuntimeError(
            f"Index count mismatch: expected {len(chunks)}, found {collection.count()}."
        )
    print(
        f"Indexed {len(chunks)} chunks with {EMBEDDING_MODEL} "
        f"({EMBEDDING_DIMENSION} dimensions); graph contains "
        f"{graph.number_of_nodes()} entities and {graph.number_of_edges()} co-mention relationships."
    )


if __name__ == "__main__":
    main()
