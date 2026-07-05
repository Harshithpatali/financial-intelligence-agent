from fastapi import FastAPI
import traceback
import httpx
print("HTTPX VERSION:", httpx.__version__)
app = FastAPI(
    title="TCS Financial Intelligence Agent",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "healthy"}

@app.get("/ask")
def ask(question: str):
    try:
        print("=" * 80)
        print(f"QUESTION: {question}")

        from backend.rag.rag_pipeline import answer_question

        result = answer_question(question)

        return result

    except Exception as e:
        print("ERROR IN /ask")
        traceback.print_exc()

        return {
            "error": str(e),
            "exception_type": type(e).__name__
        }