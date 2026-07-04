from backend.rag.vector_store import (
    get_collection
)

print("retriever.py imported")

_model = None


def get_model():

    global _model

    if _model is None:

        print("STEP A: importing SentenceTransformer")

        from sentence_transformers import (
            SentenceTransformer
        )

        print("STEP B: creating model")

        _model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5"
        )

        print("STEP C: model loaded")

    return _model

def retrieve(
    query: str,
    top_k: int = 5
):

    print("Retrieve called")

    model = get_model()

    collection = get_collection()

    query_embedding = (
        model.encode(query)
        .tolist()
    )

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = (
        results.get(
            "documents",
            [[]]
        )[0]
    )

    metadatas = (
        results.get(
            "metadatas",
            [[]]
        )[0]
    )

    distances = (
        results.get(
            "distances",
            [[]]
        )[0]
    )

    return {
        "documents": documents,
        "sources": metadatas,
        "distances": distances
    }