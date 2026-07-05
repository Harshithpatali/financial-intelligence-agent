from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents():

    docs = []

    for md_file in Path("knowledge").rglob("*.md"):

        try:
            text = md_file.read_text(
                encoding="utf-8"
            )

            docs.append({
                "source": str(md_file),
                "content": text
            })

        except Exception as e:
            print(
                f"Error reading {md_file}: {e}"
            )

    return docs


def chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []

    for doc in documents:

        texts = splitter.split_text(
            doc["content"]
        )

        for idx, chunk in enumerate(texts):

            chunks.append({
                "text": chunk,
                "source": doc["source"],
                "chunk_id": idx
            })

    return chunks