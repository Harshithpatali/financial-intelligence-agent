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



def test_market_cache_returns_last_successful_value_when_provider_fails(monkeypatch):
    from backend import market_data

    key = ("quote", "STALE_TEST")
    monkeypatch.setitem(market_data._CACHE, key, (0.0, {"ticker": "STALE_TEST", "price": 100.0}))
    result = market_data._cached(
        "quote",
        ("STALE_TEST",),
        lambda: (_ for _ in ()).throw(RuntimeError("provider rate limited")),
    )
    assert result["price"] == 100.0
    assert result["data_status"] == "stale"
    assert "last successfully cached" in result["price_note"]



def test_daily_sentiment_aggregates_finbert_scores(monkeypatch):
    from backend import market_data

    monkeypatch.setattr(
        market_data,
        "get_news",
        lambda query, limit: {
            "items": [
                {"title": "Strong quarterly growth", "summary": "Revenue increased", "published": "Sun, 15 Nov 2026 08:00:00 GMT", "url": "https://example.com/1", "publisher": "Example"},
                {"title": "Profit warning", "summary": "Margins declined", "published": "Sun, 15 Nov 2026 09:00:00 GMT", "url": "https://example.com/2", "publisher": "Example"},
            ]
        },
    )
    labels = iter([
        {"label": "positive", "confidence": 0.91, "positive_score": 0.91, "negative_score": 0.03, "neutral_score": 0.06, "sentiment_score": 0.88},
        {"label": "negative", "confidence": 0.86, "positive_score": 0.04, "negative_score": 0.86, "neutral_score": 0.10, "sentiment_score": -0.82},
    ])
    monkeypatch.setattr(market_data, "_classify_financial_text", lambda text: next(labels))
    result = market_data.get_daily_sentiment("TEST COMPANY", 2)
    assert result["article_count"] == 2
    assert result["counts"] == {"positive": 1, "negative": 1, "neutral": 0}
    assert result["sentiment_label"] == "neutral"
    assert len(result["trend"]) == 1
    assert result["model"] == "ProsusAI/finbert"


def test_daily_sentiment_endpoint(monkeypatch):
    from backend import market_data
    monkeypatch.setattr(
        market_data,
        "get_daily_sentiment",
        lambda query, limit: {"query": query, "sentiment_label": "positive", "article_count": 1},
    )
    response = client.get("/sentiment/daily", params={"q": "TCS"})
    assert response.status_code == 200
    assert response.json()["sentiment_label"] == "positive"
