from fastapi import FastAPI
from backend.rag.rag_pipeline import answer_question

app = FastAPI(
    title="TCS Financial Intelligence Agent"
)

@app.get("/")
def root():
    return {
        "status": "healthy"
    }

@app.get("/ask")
def ask(question: str):

    print(f"Received question: {question}")

    result = answer_question(question)

    return result

from pathlib import Path

@app.get("/debug")
def debug():

    chroma_path = Path("backend/chroma_db")

    return {
        "chroma_exists": chroma_path.exists(),
        "num_files": len(list(chroma_path.glob("**/*"))),
        "files": [str(f) for f in chroma_path.glob("**/*")][:10]
    }