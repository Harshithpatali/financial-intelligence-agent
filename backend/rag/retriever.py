from backend.rag.vector_store import get_collection

_model = None

def get_model():
    global _model

    if _model is None:
        from sentence_transformers import SentenceTransformer

        print("Loading embedding model...")

        _model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5"
        )

        print("Embedding model loaded.")

    return _model


def retrieve(query, top_k=5):

    print("Loading model...")
    model = get_model()
    print("Model loaded")

    print("Loading collection...")
    collection = get_collection()
    print("Collection loaded")

    print("Creating embedding...")
    query_embedding = model.encode(
        query
    ).tolist()
    print("Embedding created")

    print("Querying Chroma...")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    print("Chroma query complete")

    return {
        "documents": results["documents"][0],
        "sources": results["metadatas"][0]
    }

print("retriever.py imported")