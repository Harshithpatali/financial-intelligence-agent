from fastapi.testclient import TestClient
import pytest

from backend.api.main import app
from backend.api.rate_limit import InMemoryRateLimiter

client = TestClient(app)


def test_embedding_rejects_wrong_dimensions():
    response = client.post("/ask", json={"question": "TCS revenue?", "query_embedding": [0.0] * 12})
    assert response.status_code == 422


def test_embedding_rejects_nonfinite_values():
    response = client.post("/ask", json={"question": "TCS revenue?", "query_embedding": [float("nan")] * 384})
    assert response.status_code == 422


def test_rate_limiter_returns_retry_after():
    limiter = InMemoryRateLimiter(limit=1, window_seconds=60)

    class Client:
        host = "127.0.0.1"

    class Request:
        client = Client()

    limiter.check(Request())
    with pytest.raises(Exception) as error:
        limiter.check(Request())
    assert getattr(error.value, "status_code", None) == 429
