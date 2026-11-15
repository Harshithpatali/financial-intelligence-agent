from pathlib import Path
import os
import chromadb

_COLLECTION = None
_COLLECTION_NAME = "tcs_knowledge_base"

def get_db_path():
    return Path(os.getenv("CHROMA_DB_PATH", str(Path(__file__).resolve().parents[2] / "backend" / "chroma_db"))).resolve()

def get_client():
    path = get_db_path()
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))

def get_collection():
    global _COLLECTION
    if _COLLECTION is None:
        _COLLECTION = get_client().get_or_create_collection(
            name=_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    return _COLLECTION

def reset_collection():
    global _COLLECTION
    client = get_client()
    try: client.delete_collection(_COLLECTION_NAME)
    except Exception: pass
    _COLLECTION = client.get_or_create_collection(
        name=_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    return _COLLECTION
