"""Unauthenticated market/news adapters with bounded responses and short caching.

Yahoo Finance via yfinance and Google News RSS are public, unofficial sources.
Quotes may be delayed; availability and terms can change without notice.
"""
from __future__ import annotations

import re
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

_CACHE: dict[tuple, tuple[float, Any]] = {}
_LOCK = threading.Lock()
_TTL = {"quote": 60, "history": 300, "news": 300, "sentiment": 86400}
_ALLOWED = re.compile(r"^[A-Z0-9^=.-]{1,20}$")


def _cached(kind: str, key: tuple, loader):
    cache_key = (kind,) + key
    now = time.monotonic()
    with _LOCK:
        item = _CACHE.get(cache_key)
        if item and now - item[0] < _TTL[kind]:
            return item[1]
    try:
        value = loader()
    except Exception:
        # Keep the last successful response available during provider outages.
        with _LOCK:
            stale_item = _CACHE.get(cache_key)
        if stale_item and isinstance(stale_item[1], dict):
            stale_value = dict(stale_item[1])
            stale_value["data_status"] = "stale"
            stale_value["price_note"] = (
                str(stale_value.get("price_note", "")).strip()
                + " Provider request failed; showing the last successfully cached observation."
            ).strip()
            return stale_value
        raise
    if isinstance(value, dict):
        value = dict(value)
        value["data_status"] = "fresh"
    with _LOCK:
        _CACHE[cache_key] = (now, value)
    return value


def normalize_ticker(ticker: str) -> str:
    value = (ticker or "").strip().upper()
    if not _ALLOWED.fullmatch(value):
        raise ValueError("Ticker contains unsupported characters.")
    return value


def get_quote(ticker: str) -> dict:
    symbol = normalize_ticker(ticker)

    def load():
        import yfinance as yf

        frame = yf.Ticker(symbol).history(period="5d", interval="1d", auto_adjust=False)
        if frame.empty:
            raise LookupError(f"No recent market data returned for {symbol}.")
        row = frame.iloc[-1]
        previous = float(frame.iloc[-2]["Close"]) if len(frame) > 1 else None
        price = float(row["Close"])
        change = price - previous if previous is not None else None
        return {
            "ticker": symbol,
            "price": price,
            "currency": "INR" if symbol.endswith((".NS", ".BO")) else None,
            "previous_close": previous,
            "change": change,
            "change_percent": (change / previous * 100) if previous else None,
            "day_high": float(row["High"]),
            "day_low": float(row["Low"]),
            "volume": int(row["Volume"]),
            "market_data_timestamp": row.name.to_pydatetime().isoformat() if hasattr(row.name, "to_pydatetime") else str(row.name),
            "source": "Yahoo Finance via yfinance",
            "price_note": "May be delayed; not an exchange-certified real-time quote.",
        }

    return _cached("quote", (symbol,), load)


def get_history(ticker: str, period: str = "1y") -> dict:
    symbol = normalize_ticker(ticker)
    if period not in {"1mo", "3mo", "6mo", "1y", "2y", "5y"}:
        raise ValueError("Unsupported history period.")

    def load():
        import yfinance as yf

        frame = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=False)
        if frame.empty:
            raise LookupError(f"No historical market data returned for {symbol}.")
        points = [
            {"date": index.strftime("%Y-%m-%d"), "close": round(float(row["Close"]), 4), "volume": int(row["Volume"])}
            for index, row in frame.iterrows()
        ]
        return {
            "ticker": symbol,
            "period": period,
            "points": points,
            "source": "Yahoo Finance via yfinance",
            "price_note": "Historical prices may be adjusted or delayed; verify before financial decisions.",
        }

    return _cached("history", (symbol, period), load)


def get_news(query: str, limit: int = 10) -> dict:
    cleaned = " ".join((query or "").split())
    if not cleaned or len(cleaned) > 160:
        raise ValueError("News query must contain 1–160 characters.")
    limit = max(1, min(int(limit), 20))

    def load():
        params = urllib.parse.urlencode({"q": cleaned, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
        url = "https://news.google.com/rss/search?" + params
        request = urllib.request.Request(url, headers={"User-Agent": "FinancialIntelligenceResearch/1.0"})
        with urllib.request.urlopen(request, timeout=12) as response:
            xml_bytes = response.read(1_500_000)
        root = ET.fromstring(xml_bytes)
        items = []
        for item in root.findall("./channel/item")[:limit]:
            source = item.findtext("source") or "Publisher not identified"
            items.append({
                "title": item.findtext("title") or "Untitled",
                "url": item.findtext("link") or "",
                "published": item.findtext("pubDate") or "",
                "publisher": source,
                "summary": item.findtext("description") or "",
            })
        return {
            "query": cleaned,
            "items": items,
            "source": "Google News RSS",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }

    return _cached("news", (cleaned, limit), load)


def _classify_financial_text(text: str) -> dict:
    """Classify one headline with ProsusAI/FinBERT through Hugging Face Inference."""
    import json
    import os

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("FinBERT is not configured. Add an HF_TOKEN with Inference Providers permission to Render.")
    url = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"
    payload = {"inputs": text[:1800], "parameters": {"function_to_apply": "softmax", "top_k": 3}}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=18) as response:
        result = json.loads(response.read(500_000).decode("utf-8"))
    # HF providers may wrap token-classification output in one extra list.
    while isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        result = result[0]
    if not isinstance(result, list) or not result:
        raise RuntimeError("FinBERT returned an unexpected classification response.")
    scores = {str(item["label"]).lower(): float(item["score"]) for item in result if isinstance(item, dict) and "label" in item and "score" in item}
    if not scores:
        raise RuntimeError("FinBERT did not return sentiment scores.")
    label = max(scores, key=scores.get)
    positive = scores.get("positive", 0.0)
    negative = scores.get("negative", 0.0)
    neutral = scores.get("neutral", 0.0)
    return {
        "label": label,
        "confidence": round(scores[label], 4),
        "positive_score": round(positive, 4),
        "negative_score": round(negative, 4),
        "neutral_score": round(neutral, 4),
        "sentiment_score": round(positive - negative, 4),
    }


def get_daily_sentiment(query: str, limit: int = 8) -> dict:
    """Fetch current headlines and score them with financial-domain FinBERT."""
    cleaned = " ".join((query or "").split())
    if not cleaned or len(cleaned) > 160:
        raise ValueError("Sentiment query must contain 1–160 characters.")
    limit = max(1, min(int(limit), 8))

    def load():
        from email.utils import parsedate_to_datetime

        news = get_news(cleaned, limit)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        candidates = []
        for article in news.get("items", [])[:limit]:
            title = str(article.get("title", "")).strip()
            if not title:
                continue
            summary = " ".join(str(article.get("summary", "")).split())
            candidates.append((article, title + (". " + summary[:900] if summary else "")))

        articles = []
        failed = 0

        def classify(candidate):
            article, text = candidate
            return article, _classify_financial_text(text)

        # Bounded concurrency lowers response time without loading FinBERT on Render.
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(classify, candidate) for candidate in candidates]
            for future in as_completed(futures):
                try:
                    article, scores = future.result()
                except Exception:
                    failed += 1
                    continue
                published = str(article.get("published", ""))
                try:
                    published_day = parsedate_to_datetime(published).date().isoformat()
                except (TypeError, ValueError, OverflowError):
                    published_day = "Date unavailable"
                articles.append({
                    "title": str(article.get("title", "Untitled")).strip(),
                    "url": str(article.get("url", "")),
                    "publisher": str(article.get("publisher", "Publisher not identified")),
                    "published": published,
                    "published_date": published_day,
                    **scores,
                })

        if not articles:
            raise RuntimeError("FinBERT could not classify any headlines. Check the Hugging Face token and Inference Providers access.")
        counts = {label: sum(item["label"] == label for item in articles) for label in ("positive", "negative", "neutral")}
        average = sum(item["sentiment_score"] for item in articles) / len(articles)
        daily = {}
        for item in articles:
            day = item["published_date"]
            if day == "Date unavailable":
                continue
            daily.setdefault(day, []).append(item["sentiment_score"])
        trend = [
            {"date": day, "sentiment_score": round(sum(values) / len(values), 4), "article_count": len(values)}
            for day, values in sorted(daily.items())
        ]
        return {
            "query": cleaned,
            "model": "ProsusAI/finbert",
            "source": "Google News RSS",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "article_count": len(articles),
            "failed_article_count": failed,
            "sentiment_score": round(average, 4),
            "sentiment_label": "positive" if average > 0.08 else "negative" if average < -0.08 else "neutral",
            "counts": counts,
            "trend": trend,
            "articles": articles,
            "methodology_note": "FinBERT classifies headline text; sentiment is not a prediction of stock returns or investment advice.",
        }

    return _cached("sentiment", (cleaned, limit), load)
