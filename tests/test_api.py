from fastapi.testclient import TestClient
from backend.api.main import app
client=TestClient(app)
def test_health():assert client.get("/health").json()["status"]=="ok"
def test_short_question_rejected():assert client.get("/ask",params={"question":"x"}).status_code==422
