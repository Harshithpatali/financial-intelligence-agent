"""Build daily precomputed financial-news sentiment reports using Groq.

The report is written atomically: failed or incomplete runs never replace the
last successful published report.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from groq import Groq

MODEL = "openai/gpt-oss-20b"
MAX_HEADLINES = 8
COMPANIES = {
    "tcs": ("TCS · Tata Consultancy Services", "Tata Consultancy Services TCS"),
    "infosys": ("Infosys", "Infosys India"),
    "reliance": ("Reliance Industries", "Reliance Industries"),
    "hdfc_bank": ("HDFC Bank", "HDFC Bank"),
    "icici_bank": ("ICICI Bank", "ICICI Bank"),
    "nifty50": ("NIFTY 50", "NIFTY 50 Indian stock market"),
    "sensex": ("SENSEX", "BSE Sensex Indian stock market"),
}

SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "label": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                    "confidence": {"type": "number"},
                    "sentiment_score": {"type": "number"},
                },
                "required": ["index", "label", "confidence", "sentiment_score"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


def fetch_headlines(query: str, limit: int = MAX_HEADLINES) -> list[dict[str, str]]:
    params = urllib.parse.urlencode({"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
    request = urllib.request.Request(
        "https://news.google.com/rss/search?" + params,
        headers={"User-Agent": "FinancialIntelligenceResearch/1.0"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = response.read(1_500_000)
    root = ET.fromstring(payload)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in root.findall("./channel/item"):
        title = html.unescape((item.findtext("title") or "").strip())
        key = re.sub(r"\\W+", "", title.lower())
        if not title or key in seen:
            continue
        seen.add(key)
        rows.append({
            "title": title,
            "url": (item.findtext("link") or "").strip(),
            "published": (item.findtext("pubDate") or "").strip(),
            "publisher": html.unescape((item.findtext("source") or "Publisher not identified").strip()),
            "summary": html.unescape(re.sub(r"<[^>]+>", " ", item.findtext("description") or ""))[:900],
        })
        if len(rows) >= limit:
            break
    return rows


def classify_headlines(headlines: list[dict[str, str]], client: Groq) -> list[dict[str, Any]]:
    """Classify a complete company batch and validate model output against input indices."""
    indexed = [
        {"index": i, "headline": row["title"], "summary": row["summary"]}
        for i, row in enumerate(headlines)
    ]
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a financial-news sentiment classifier. Classify the tone expressed "
                            "by each headline and its short summary, not the likely future return of the asset. "
                            "Treat each item independently. Positive means clearly favourable business or "
                            "market tone; negative means clearly adverse tone; neutral means factual, mixed, "
                            "or unclear tone. Confidence must be between 0 and 1. Sentiment score must be "
                            "between -1 and 1, where -1 is strongly negative, 0 is neutral/mixed, and +1 "
                            "is strongly positive. Do not infer facts absent from the text."
                        ),
                    },
                    {"role": "user", "content": json.dumps(indexed, ensure_ascii=False)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "headline_sentiment_batch",
                        "strict": True,
                        "schema": SENTIMENT_SCHEMA,
                    },
                },
                temperature=0,
                max_completion_tokens=1800,
            )
            content = response.choices[0].message.content
            parsed = json.loads(content or "{}")
            items = parsed.get("items")
            if not isinstance(items, list) or len(items) != len(headlines):
                raise ValueError("Groq returned an incomplete headline batch.")
            by_index: dict[int, dict[str, Any]] = {}
            for item in items:
                idx = item.get("index")
                if not isinstance(idx, int) or idx in by_index or idx < 0 or idx >= len(headlines):
                    raise ValueError("Groq returned invalid or duplicate headline indices.")
                label = item.get("label")
                confidence = float(item.get("confidence"))
                score = float(item.get("sentiment_score"))
                if label not in {"positive", "negative", "neutral"}:
                    raise ValueError("Groq returned an unsupported sentiment label.")
                if not 0 <= confidence <= 1 or not -1 <= score <= 1:
                    raise ValueError("Groq returned a score outside the permitted range.")
                by_index[idx] = {
                    "label": label,
                    "confidence": round(confidence, 4),
                    "positive_score": round(max(score, 0.0), 4),
                    "negative_score": round(abs(min(score, 0.0)), 4),
                    "neutral_score": round(1.0 - confidence if label == "neutral" else 0.0, 4),
                    "sentiment_score": round(score, 4),
                }
            if set(by_index) != set(range(len(headlines))):
                raise ValueError("Groq omitted one or more headline indices.")
            return [by_index[i] for i in range(len(headlines))]
        except Exception as exc:
            last_error = exc
            status = getattr(exc, "status_code", None)
            if status not in (None, 429, 500, 502, 503, 504) and attempt < 2:
                raise RuntimeError(f"Groq classification failed (HTTP {status}): {exc}") from exc
            if attempt < 2:
                retry_after = getattr(exc, "response", None)
                delay = min(2 ** attempt, 8)
                if retry_after is not None:
                    try:
                        delay = min(float(retry_after.headers.get("retry-after", delay)), 30)
                    except (TypeError, ValueError):
                        pass
                time.sleep(delay)
    raise RuntimeError(f"Groq classification failed after retries: {last_error}") from last_error


def classify_company(
    slug: str, label: str, query: str, client: Groq
) -> dict[str, Any]:
    headlines = fetch_headlines(query)
    if not headlines:
        raise RuntimeError(f"No headlines found for {label}.")
    scores = classify_headlines(headlines, client)
    scored: list[dict[str, Any]] = []
    for row, sentiment in zip(headlines, scores, strict=True):
        try:
            published_day = parsedate_to_datetime(row["published"]).date().isoformat()
        except (TypeError, ValueError, OverflowError):
            published_day = "Date unavailable"
        scored.append({
            "title": row["title"],
            "url": row["url"],
            "publisher": row["publisher"],
            "published": row["published"],
            "published_date": published_day,
            **sentiment,
        })
    scored.sort(key=lambda row: row["published"], reverse=True)
    counts = {name: sum(row["label"] == name for row in scored) for name in ("positive", "negative", "neutral")}
    average = sum(row["sentiment_score"] for row in scored) / len(scored)
    by_day: dict[str, list[float]] = {}
    for row in scored:
        if row["published_date"] != "Date unavailable":
            by_day.setdefault(row["published_date"], []).append(row["sentiment_score"])
    trend = [
        {"date": day, "sentiment_score": round(sum(values) / len(values), 4), "article_count": len(values)}
        for day, values in sorted(by_day.items())
    ]
    return {
        "slug": slug,
        "company": label,
        "query": query,
        "model": MODEL,
        "source": "Google News RSS",
        "article_count": len(scored),
        "failed_article_count": 0,
        "sentiment_score": round(average, 4),
        "sentiment_label": "positive" if average > 0.08 else "negative" if average < -0.08 else "neutral",
        "counts": counts,
        "trend": trend,
        "articles": scored,
        "methodology_note": (
            "Groq openai/gpt-oss-20b classifies the tone of public news headlines and summaries. "
            "Scores are model-generated text assessments, not calibrated probabilities, stock-return "
            "forecasts, recommendations, or investment advice."
        ),
    }


def build_report(api_key: str) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it as a GitHub Actions repository secret.")
    client = Groq(api_key=api_key, timeout=45, max_retries=0)
    generated_at = datetime.now(timezone.utc).isoformat()
    reports: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for slug, (label, query) in COMPANIES.items():
        try:
            reports[slug] = classify_company(slug, label, query, client)
            print(f"OK {slug}: {reports[slug]['article_count']} headlines")
        except Exception as exc:
            failures[slug] = str(exc)[:500]
            print(f"FAILED {slug}: {failures[slug]}", file=sys.stderr)
    if failures or len(reports) != len(COMPANIES):
        raise RuntimeError(
            f"Report incomplete ({len(reports)}/{len(COMPANIES)} companies). "
            "Previous report will be retained."
        )
    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "report_date_utc": generated_at[:10],
        "timezone_note": "Generated daily at approximately 06:00 Asia/Kolkata (00:30 UTC).",
        "model": MODEL,
        "source": "Google News RSS",
        "companies": reports,
        "methodology_note": (
            "Groq openai/gpt-oss-20b assesses public headline tone only; scores are not calibrated "
            "probabilities, stock-return forecasts, recommendations, or investment advice."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="sentiment-report.json")
    args = parser.parse_args()
    report = build_report(os.getenv("GROQ_API_KEY", ""))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(f"Built report locally: {destination} ({len(report['companies'])} companies)")


if __name__ == "__main__":
    main()
