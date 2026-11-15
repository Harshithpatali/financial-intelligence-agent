from unittest.mock import patch
from backend.rag.rag_pipeline import answer_question
@patch("backend.rag.rag_pipeline.generate_answer",return_value="Revenue increased [S1].")
@patch("backend.rag.rag_pipeline.retrieve")
def test_answer_question_sources(mock_retrieve,_mock_llm):
 mock_retrieve.return_value={"documents":["Revenue increased in FY2024."],"sources":[{"source":"annual.md","page":12,"chunk_id":"abc","score":0.02}],"retrieval":["dense","bm25","knowledge_graph"]}
 result=answer_question("What happened to revenue?")
 assert result["answer"].endswith("[S1].") and result["sources"][0]["page"]==12
@patch("backend.rag.rag_pipeline.retrieve")
def test_no_evidence_fallback(mock_retrieve):
 mock_retrieve.return_value={"documents":[],"sources":[],"retrieval":[]}
 result=answer_question("Unsupported question")
 assert result["sources"]==[] and "could not find supporting" in result["answer"].lower()
