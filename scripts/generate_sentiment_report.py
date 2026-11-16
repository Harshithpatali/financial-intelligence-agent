"""Build daily, precomputed FinBERT sentiment reports for the Streamlit dashboard.

Run with HF_TOKEN set. The script writes atomically so a failed run never
replaces the last successful report.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

MODEL = "ProsusAI/finbert"
HF_URL = "https://router.huggingface.co/hf-inference/models/" + MODEL
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
        key = re.sub(r"\W+", "", title.lower())
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


def classify_text(text: str, token: str) -> dict[str, Any]:
    body = json.dumps({
        "inputs": text[:1800],
        "parameters": {"function_to_apply": "softmax", "top_k": 3},
    }).encode("utf-8")
    request = urllib.request.Request(
        HF_URL,
        data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read(500_000).decode("utf-8"))
            while isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
                result = result[0]
            if not isinstance(result, list) or not result:
                raise RuntimeError("FinBERT returned an unexpected response.")
            scores = {
                str(row["label"]).lower(): float(row["score"])
                for row in result
                if isinstance(row, dict) and "label" in row and "score" in row
            }
            if not scores:
                raise RuntimeError("FinBERT returned no sentiment scores.")
            label = max(scores, key=scores.get)
            positive, negative, neutral = (
                scores.get("positive", 0.0), scores.get("negative", 0.0), scores.get("neutral", 0.0)
            )
            return {
                "label": label,
                "confidence": round(scores[label], 4),
                "positive_score": round(positive, 4),
                "negative_score": round(negative, 4),
                "neutral_score": round(neutral, 4),
                "sentiment_score": round(positive - negative, 4),
            }
        except urllib.error.HTTPError as exc:
            detail = exc.read(1200).decode("utf-8", "replace")
            last_error = RuntimeError(f"Hugging Face inference returned HTTP {exc.code}: {detail}")
            if exc.code not in (429, 500, 502, 503, 504):
                break
        except Exception as exc:
            last_error = exc
        if attempt < 2:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"FinBERT classification failed: {last_error}") from last_error


def classify_company(slug: str, label: str, query: str, token: str) -> dict[str, Any]:
    headlines = fetch_headlines(query)
    if not headlines:
        raise RuntimeError(f"No headlines found for {label}.")
    scored: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                classify_text,
                row["title"] + (". " + row["summary"] if row["summary"] else ""),
                token,
            ): row
            for row in headlines
        }
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            scores = future.result()
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
                **scores,
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
        "methodology_note": "FinBERT classifies headline text; sentiment is not a prediction of stock returns or investment advice.",
    }


def build_report(token: str) -> dict[str, Any]:
    if not token:
        raise RuntimeError("HF_TOKEN is missing. Add it as a GitHub Actions repository secret.")
    generated_at = datetime.now(timezone.utc).isoformat()
    reports: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for slug, (label, query) in COMPANIES.items():
        try:
            reports[slug] = classify_company(slug, label, query, token)
            print(f"OK {slug}: {reports[slug]['article_count']} headlines")
        except Exception as exc:
            failures[slug] = str(exc)[:500]
            print(f"FAILED {slug}: {failures[slug]}", file=sys.stderr)
    # Fail closed: don't publish a report with missing companies or replace good data.
    if failures or len(reports) != len(COMPANIES):
        raise RuntimeError(f"Report incomplete ({len(reports)}/{len(COMPANIES)} companies). Previous report will be retained.")
    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "report_date_utc": generated_at[:10],
        "timezone_note": "Generated daily at approximately 06:00 Asia/Kolkata (00:30 UTC).",
        "model": MODEL,
        "source": "Google News RSS",
        "companies": reports,
        "methodology_note": "Headline tone only; not a forecast, recommendation, or investment advice.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="sentiment-report.json")
    args = parser.parse_args()
    report = build_report(os.getenv("HF_TOKEN", ""))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(f"Published report locally: {destination} ({len(report['companies'])} companies)")


if __name__ == "__main__":
    main()
