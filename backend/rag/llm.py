"""Groq generation with bounded request time and output size."""
import os
import logging

logger = logging.getLogger(__name__)


def generate_answer(prompt):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")

    from groq import Groq

    # Keep the request bounded so a free web instance is not held indefinitely.
    client = Groq(
        api_key=api_key,
        timeout=float(os.getenv("GROQ_TIMEOUT_SECONDS", "45")),
        max_retries=0,
    )
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=700,
    )
    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("Groq returned an empty answer")
    return answer
