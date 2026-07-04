from fastapi import FastAPI

app = FastAPI(
    title="TCS Financial Intelligence Agent"
)

@app.get("/")
def root():
    return {"status": "healthy"}

@app.get("/ask")
def ask(question: str):

    print("Importing rag_pipeline")

    from backend.rag.rag_pipeline import answer_question

    print("rag_pipeline imported")

    return answer_question(question)