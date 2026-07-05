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
        import traceback

        print("ERROR:")
        traceback.print_exc()

        return {"error": str(e)}