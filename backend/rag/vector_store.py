from pathlib import Path
import chromadb

print("vector_store.py imported")

_collection = None


def get_collection():

    global _collection

    if _collection is None:

        PROJECT_ROOT = (
            Path(__file__).resolve().parents[2]
        )

        CHROMA_PATH = (
            PROJECT_ROOT
            / "backend"
            / "chroma_db"
        )

        print(
            f"Loading ChromaDB from: {CHROMA_PATH}"
        )

        if not CHROMA_PATH.exists():

            raise FileNotFoundError(
                f"ChromaDB folder not found: {CHROMA_PATH}"
            )

        client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        _collection = client.get_collection(
            name="tcs_knowledge_base"
        )

        print(
            f"Collection Count: {_collection.count()}"
        )

    return _collection