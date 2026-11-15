from fastapi.testclient import TestClient

from backend.api.main import app

client = TestClient(app)


def test_market_quote_endpoint(monkeypatch):
    from backend import market_data
    monkeypatch.setattr(market_data, "get_quote", lambda ticker: {"ticker": ticker, "price": 123.4})
    response = client.get("/market/quote", params={"ticker": "TCS.NS"})
    assert response.status_code == 200
    assert response.json()["price"] == 123.4


def test_market_history_rejects_unsupported_period(monkeypatch):
    from backend import market_data
    monkeypatch.setattr(market_data, "get_history", lambda ticker, period: (_ for _ in ()).throw(ValueError("Unsupported history period.")))
    response = client.get("/market/history", params={"ticker": "TCS.NS", "period": "10d"})
    assert response.status_code == 422


def test_news_endpoint(monkeypatch):
    from backend import market_data
    monkeypatch.setattr(market_data, "get_news", lambda query, limit: {"query": query, "items": []})
    response = client.get("/news", params={"q": "TCS earnings"})
    assert response.status_code == 200
    assert response.json()["query"] == "TCS earnings"


def test_ticker_validation():
    from backend.market_data import normalize_ticker
    import pytest
    with pytest.raises(ValueError):
        normalize_ticker("TCS.NS; DROP TABLE")
