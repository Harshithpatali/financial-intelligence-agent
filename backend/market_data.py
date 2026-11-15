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
_TTL = {"quote": 60, "history": 300, "news": 300}
_ALLOWED = re.compile(r"^[A-Z0-9^=.-]{1,20}$")


def _cached(kind: str, key: tuple, loader):
    cache_key = (kind,) + key
    now = time.monotonic()
    with _LOCK:
        item = _CACHE.get(cache_key)
        if item and now - item[0] < _TTL[kind]:
            return item[1]
    value = loader()
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
