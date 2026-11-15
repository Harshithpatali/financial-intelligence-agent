"""Groq generation with bounded retries and explicit provider rate-limit handling."""
import logging
import os

logger = logging.getLogger(__name__)


class LLMRateLimitError(RuntimeError):
    def __init__(self, retry_after=30):
        self.retry_after = max(1, int(retry_after))
        super().__init__("Groq rate limit reached")


def generate_answer(prompt):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")

    from groq import Groq, RateLimitError

    client = Groq(
        api_key=api_key,
        timeout=float(os.getenv("GROQ_TIMEOUT_SECONDS", "45")),
        max_retries=1,
    )
    try:
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=700,
        )
    except RateLimitError as exc:
        headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
        retry_after = headers.get("retry-after", os.getenv("GROQ_RETRY_AFTER_SECONDS", "30"))
        try:
            retry_after = int(float(retry_after))
        except (TypeError, ValueError):
            retry_after = 30
        logger.warning("Groq rate limit reached; retry_after_seconds=%s", retry_after)
        raise LLMRateLimitError(retry_after) from exc

    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("Groq returned an empty answer")
    return answer
