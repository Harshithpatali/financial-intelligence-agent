"""FastAPI endpoints with readiness, request validation and basic abuse controls."""
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from backend.api.rate_limit import limiter

class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "exception": self.formatException(record.exc_info) if record.exc_info else None,
        }, ensure_ascii=False)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
root_logger = logging.getLogger()
if not any(getattr(item, "_financial_json_handler", False) for item in root_logger.handlers):
    handler._financial_json_handler = True
    root_logger.addHandler(handler)
root_logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TCS Financial Intelligence Agent",
    version="2.1.0",
    description="Source-grounded financial research using hybrid Graph RAG.",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return safe validation details without echoing untrusted non-finite inputs."""
    errors = [
        {"location": [str(part) for part in error.get("loc", ())],
         "message": error.get("msg", "Invalid request"),
         "type": error.get("type", "value_error")}
        for error in exc.errors()
    ]
    return JSONResponse({"detail": errors}, status_code=422)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3, max_length=2000)


@app.get("/")
def root():
    return {"status": "healthy", "service": "tcs-financial-intelligence-agent", "version": "2.1.0"}


@app.get("/health")
def health():
    """Liveness only: confirms that the web process is responding."""
    return {"status": "ok", "service": "tcs-financial-intelligence-agent"}


@app.get("/ready")
def readiness():
    """Readiness: checks the local index and required generation configuration."""
    checks = {}
    try:
        from backend.rag.vector_store import get_collection
        count = get_collection().count()
        checks["chroma"] = {"ready": count > 0, "document_chunks": count}
    except Exception as exc:
        logger.exception("Readiness check failed for Chroma")
        checks["chroma"] = {"ready": False, "error": type(exc).__name__}
    try:
        from backend.rag.graph_index import INDEX_PATH, load_graph
        graph = load_graph()
        checks["knowledge_graph"] = {
            "ready": INDEX_PATH.exists() and graph.number_of_nodes() > 0,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
        }
    except Exception as exc:
        logger.exception("Readiness check failed for graph")
        checks["knowledge_graph"] = {"ready": False, "error": type(exc).__name__}
    checks["groq_configured"] = bool(os.getenv("GROQ_API_KEY"))
    ready = all([
        checks["chroma"]["ready"],
        checks["knowledge_graph"]["ready"],
        checks["groq_configured"],
    ])
    payload = {"status": "ready" if ready else "not_ready", "checks": checks}
    return JSONResponse(payload, status_code=200 if ready else 503)


@app.get("/ask")
def ask(request: Request, question: str = Query(min_length=3, max_length=2000)):
    limiter.check(request)
    return _answer(question, None)


@app.post("/ask")
def ask_post(request: Request, payload: AskRequest):
    limiter.check(request)
    return _answer(payload.question, None)


def _answer(question, query_embedding=None):
    request_id = "req-" + uuid.uuid4().hex[:12]
    started = time.monotonic()
    try:
        from backend.rag.rag_pipeline import answer_question
        result = answer_question(question, query_embedding=query_embedding)
        logger.info("question_answered request_id=%s elapsed_ms=%d sources=%d",
                    request_id, int((time.monotonic() - started) * 1000),
                    result.get("num_sources", 0))
        result["request_id"] = request_id
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        from backend.rag.llm import LLMRateLimitError
        if isinstance(exc, LLMRateLimitError):
            logger.warning("groq_rate_limited request_id=%s retry_after=%s", request_id, exc.retry_after)
            return JSONResponse(
                {"detail": "The AI provider is temporarily rate-limiting requests. Please retry shortly.",
                 "request_id": request_id},
                status_code=429,
                headers={"Retry-After": str(exc.retry_after)},
            )
        logger.exception("question_failed request_id=%s error_type=%s", request_id, type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail={"message": "The research service is temporarily unavailable.", "request_id": request_id},
        ) from exc



@app.get("/market/quote")
def market_quote(ticker: str = Query(min_length=1, max_length=20)):
    """Public market quote; data can be delayed and provider-limited."""
    from backend import market_data
    try:
        return market_data.get_quote(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("market_quote_failed ticker=%s error_type=%s", ticker, type(exc).__name__)
        raise HTTPException(status_code=502, detail="Market data provider is temporarily unavailable.") from exc


@app.get("/market/history")
def market_history(
    ticker: str = Query(min_length=1, max_length=20),
    period: str = Query(default="1y"),
):
    from backend import market_data
    try:
        return market_data.get_history(ticker, period)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("market_history_failed ticker=%s error_type=%s", ticker, type(exc).__name__)
        raise HTTPException(status_code=502, detail="Historical market data provider is temporarily unavailable.") from exc


@app.get("/sentiment/daily")
def daily_market_sentiment(
    q: str = Query(min_length=1, max_length=160),
    limit: int = Query(default=12, ge=1, le=12),
):
    """Daily headline sentiment scored with ProsusAI FinBERT."""
    from backend import market_data
    try:
        return market_data.get_daily_sentiment(q, limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.warning("sentiment_not_configured_or_unavailable error=%s", str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("daily_sentiment_failed query=%s error_type=%s", q, type(exc).__name__)
        raise HTTPException(status_code=502, detail="Daily sentiment analysis is temporarily unavailable.") from exc


@app.get("/news")
def financial_news(q: str = Query(min_length=1, max_length=160), limit: int = Query(default=10, ge=1, le=20)):
    from backend import market_data
    try:
        return market_data.get_news(q, limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("news_feed_failed query=%s error_type=%s", q, type(exc).__name__)
        raise HTTPException(status_code=502, detail="News feed is temporarily unavailable.") from exc
