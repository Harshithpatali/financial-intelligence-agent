import os

from dotenv import load_dotenv
load_dotenv()

from backend.rag.vector_store import get_collection

print("retriever.py imported")

_model = None


def get_model():

    global _model

    if _model is None:

        print("STEP A: importing SentenceTransformer")

        try:
            import torch
            print("TORCH VERSION:", torch.__version__)

            from sentence_transformers import SentenceTransformer
            print("SentenceTransformer imported")

        except Exception as e:
            print("IMPORT ERROR:", str(e))
            raise

        print("STEP B: creating model")

        hf_token = os.getenv("HF_TOKEN")

        print(f"HF Token Found: {bool(hf_token)}")

        _model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5",
            token=hf_token
        )

        print("STEP C: model loaded")

    return _model

def retrieve(
    query: str,
    top_k: int = 5
):

    print("Retrieve called")

    model = get_model()

    print("Model obtained")

    collection = get_collection()

    print("Collection obtained")

    print("Creating embedding")

    query_embedding = (
        model.encode(query)
        .tolist()
    )

    print("Embedding created")

    print("Querying ChromaDB")

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

    print("Chroma query completed")

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

    print(
        f"Retrieved {len(documents)} documents"
    )

    return {
        "documents": documents,
        "sources": metadatas,
        "distances": distances
    }