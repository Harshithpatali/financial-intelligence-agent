from fastapi.testclient import TestClient
from backend.api.main import app
client=TestClient(app)
def test_health(): assert client.get("/health").json()["status"]=="ok"
def test_short_question_rejected(): assert client.get("/ask",params={"question":"x"}).status_code==422
def test_post_ask_accepts_question_only(monkeypatch):
    from backend.rag import rag_pipeline
    def fake_answer(question,query_embedding=None):
        return {"question":question,"answer":"mocked","sources":[],"received_embedding":query_embedding}
    monkeypatch.setattr(rag_pipeline,"answer_question",fake_answer)
    response=client.post("/ask",json={"question":"What was TCS revenue?"})
    assert response.status_code==200 and response.json()["received_embedding"] is None
