"""Source-grounded Graph RAG question answering with bounded context."""
from backend.rag.retriever import retrieve
from backend.rag.llm import generate_answer
from backend.rag.embeddings import embed_query


def answer_question(question, query_embedding=None, top_k=3):
    if not question or not question.strip():
        raise ValueError("Question must not be empty")
    if not isinstance(top_k, int) or not 1 <= top_k <= 10:
        raise ValueError("top_k must be an integer between 1 and 10")

    if query_embedding is None:
        query_embedding = embed_query(question)
    results = retrieve(question, top_k=top_k, query_embedding=query_embedding)
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
        excerpt = doc[:5000]
        blocks.append(f'[S{i}] Source: {src["source"]}{page}\\n{excerpt}')

    prompt = f"""You are a careful financial research assistant.
Answer only from the evidence below. Never invent figures, dates, causes or management statements.
If evidence is insufficient or conflicting, say so. Cite every material factual claim using the exact format [S1], [S2], etc.
Distinguish reported facts from interpretation. Do not provide investment advice.
Keep the answer concise and directly address the question.
For financial figures, preserve the reporting period, currency, unit, sign and accounting basis shown in the cited evidence. Convert units only when the conversion is exact and state the converted unit.
Do not claim that a figure is absent merely because one excerpt omits it; inspect all supplied evidence before abstaining.
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
