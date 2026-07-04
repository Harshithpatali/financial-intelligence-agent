from sentence_transformers import SentenceTransformer
from backend.rag.vector_store import collection

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

def retrieve(query, top_k=5):

    query_embedding = model.encode(
        query
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )


    results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)

    return {
        "documents": results["documents"][0],
        "sources": results["metadatas"][0]
    }