from backend.rag.retriever import retrieve
from backend.rag.llm import generate_answer


def answer_question(question):

    results = retrieve(
        question,
        top_k=5
    )

    context = "\n\n".join(
        results["documents"]
    )

    sources = []

    for item in results["sources"]:
        sources.append(
            item["source"]
        )

    prompt = f"""
You are a senior equity research analyst.

Use ONLY the provided context.

Requirements:
- Give a concise executive summary.
- Mention financial figures exactly.
- Mention years and periods.
- Calculate growth rates if available.
- Compare with previous periods.
- If information is missing, say so.
- Never invent numbers.

Context:
{context}

Question:
{question}
"""

    answer = generate_answer(
        prompt
    )

    return {
    "question": question,
    "answer": answer,
    "sources": list(set(sources)),
    "num_sources": len(set(sources))
}