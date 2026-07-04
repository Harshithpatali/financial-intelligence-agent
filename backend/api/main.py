from fastapi import FastAPI
import traceback

app = FastAPI(
    title="TCS Financial Intelligence Agent",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "status": "healthy"
    }


@app.get("/ask")
def ask(question: str):

    try:

        print("=" * 80)
        print(f"QUESTION: {question}")

        from backend.rag.rag_pipeline import (
            answer_question
        )

        result = answer_question(question)

        return result

    except Exception as e:

        print("ERROR IN /ask")
        print(str(e))
        traceback.print_exc()

        return {
            "error": str(e)
        }