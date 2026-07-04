from fastapi import FastAPI
import traceback

app = FastAPI(title="TCS Financial Intelligence Agent")

@app.get("/")
def root():
    return {"status": "healthy"}

@app.get("/ask")
def ask(question: str):
    try:
        print("Importing rag_pipeline")

        from backend.rag.rag_pipeline import answer_question

        print("rag_pipeline imported")

        return answer_question(question)

    except Exception as e:
        print("ERROR IN /ask")
        print(str(e))
        traceback.print_exc()

        return {
            "error": str(e)
        }