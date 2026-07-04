from backend.rag.vector_store import get_collection

print("RETRIEVER VERSION 2026-07-04")
print("retriever.py imported")

_model = None


def get_model():
    global _model

    if _model is None:

        print("STEP A: importing SentenceTransformer")

        from sentence_transformers import SentenceTransformer

        print("STEP B: creating SentenceTransformer")

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        

        print("STEP C: SentenceTransformer loaded")

    return _model


def retrieve(query, top_k=5):

    print("STEP 1: retrieve() called")

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

    print(
    f"Embedding length: {len(query_embedding)}"
)
    print("Embedding created")

    print("Querying ChromaDB...")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    print("Chroma query complete")

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    return {
        "documents": documents,
        "sources": metadatas
    }