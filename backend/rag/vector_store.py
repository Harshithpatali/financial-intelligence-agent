import chromadb

client = chromadb.PersistentClient(
    path="backend/chroma_db"
)

collection = client.get_or_create_collection(
    name="tcs_knowledge_base"
)