"""Source-grounded Graph RAG question answering with bounded context."""
from backend.rag.retriever import retrieve
from backend.rag.llm import generate_answer


def answer_question(question, query_embedding=None):
    if not question or not question.strip():
        raise ValueError("Question must not be empty")

    results = retrieve(question, top_k=3, query_embedding=query_embedding)
    documents = results["documents"]
    sources = results["sources"]
    if not documents:
        return {
            "question": question,
            "answer": "I could not find supporting information in the indexed TCS documents.",
            "sources": [],
            "num_sources": 0,
            "retrieval": results.get("retrieval", []),
        }

    blocks = []
    for i, doc in enumerate(documents, 1):
        src = sources[i - 1]
        page = f', page {src["page"]}' if src.get("page") else ""
        # A hard cap prevents unexpectedly large prompts and slow generations.
        excerpt = doc[:5000]
        blocks.append(f'[S{i}] Source: {src["source"]}{page}\\n{excerpt}')

    prompt = f"""You are a careful financial research assistant.
Answer only from the evidence below. Never invent figures, dates, causes or management statements.
If evidence is insufficient or conflicting, say so. Cite every material factual claim as [S1], [S2], etc.
Distinguish reported facts from interpretation. Do not provide investment advice.
Keep the answer concise and directly address the question.
EVIDENCE:
{chr(10).join(blocks)}
QUESTION:
{question}
"""
    answer = generate_answer(prompt)
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "num_sources": len(sources),
        "retrieval": results.get("retrieval", []),
    }
