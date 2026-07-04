from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from backend.rag.chunker import (
    load_documents,
    chunk_documents
)

from backend.rag.vector_store import collection


model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

print("Loading documents...")

docs = load_documents()

print(
    f"Documents loaded: {len(docs)}"
)

chunks = chunk_documents(docs)

print(
    f"Chunks created: {len(chunks)}"
)

for idx, chunk in enumerate(
        tqdm(chunks)
):

    embedding = model.encode(
        chunk["text"]
    ).tolist()

    collection.add(
        ids=[f"chunk_{idx}"],
        embeddings=[embedding],
        documents=[chunk["text"]],
        metadatas=[{
            "source": chunk["source"]
        }]
    )

print(
    "Vector database built successfully!"
)