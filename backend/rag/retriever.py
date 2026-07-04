from backend.rag.vector_store import get_collection

print("retriever.py imported")

_model = None

def get_model():
    global _model

    if _model is None:
        print("STEP A: importing SentenceTransformer")

        from sentence_transformers import SentenceTransformer

        print("STEP B: creating SentenceTransformer")

        _model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)
        print("STEP C: SentenceTransformer loaded")

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

