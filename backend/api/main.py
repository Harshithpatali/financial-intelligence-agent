from fastapi import FastAPI
from backend.rag.rag_pipeline import answer_question
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "healthy"}


app = FastAPI(
    title="TCS Financial Intelligence Agent"
)

@app.get("/ask")
def ask(question: str):

    result = answer_question(
        question
    )

    return result