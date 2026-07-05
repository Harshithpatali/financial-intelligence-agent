from fastapi import FastAPI
import traceback

app = FastAPI(
    title="TCS Financial Intelligence Agent",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "healthy"}


@app.get("/ask")
def ask(question: str):

    print("ASK ENDPOINT HIT")

    try:
        print("QUESTION:", question)

        from backend.rag.rag_pipeline import answer_question

        print("RAG PIPELINE IMPORTED")

        result = answer_question(question)

        print("ANSWER GENERATED")

        return result

    except Exception as e:

        print("ERROR:")
        traceback.print_exc()

        return {
            "error": str(e)
        }