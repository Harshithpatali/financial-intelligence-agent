from backend.rag.retriever import (
    retrieve
)

from backend.rag.llm import (
    generate_answer
)

print("rag_pipeline.py imported")


def answer_question(question):

    print(
        "Starting retrieval..."
    )

    results = retrieve(
        question,
        top_k=5
    )

    documents = results.get(
        "documents",
        []
    )

    sources = []

    for item in results.get(
        "sources",
        []
    ):

        if isinstance(item, dict):

            sources.append(
                item.get(
                    "source",
                    "Unknown"
                )
            )

    context = "\n\n".join(
        documents
    )

    prompt = f"""
You are a financial analyst.

Answer ONLY using the supplied context.

If information is unavailable,
state that clearly.

CONTEXT:
{context}

QUESTION:
{question}
"""

    answer = generate_answer(
        prompt
    )

    return {
        "question": question,
        "answer": answer,
        "sources": list(
            set(sources)
        ),
        "num_sources": len(
            set(sources)
        )
    }