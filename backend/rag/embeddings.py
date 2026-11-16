"""Groq-hosted Nomic embeddings shared by indexing, API queries and evaluation."""
from __future__ import annotations

import math
import os
from functools import lru_cache

from groq import Groq

EMBEDDING_MODEL = os.getenv("GROQ_EMBEDDING_MODEL", "nomic-embed-text-v1_5")
EMBEDDING_DIMENSION = int(os.getenv("GROQ_EMBEDDING_DIMENSION", "768"))
BATCH_SIZE = max(1, min(int(os.getenv("GROQ_EMBEDDING_BATCH_SIZE", "32")), 128))


@lru_cache(maxsize=1)
def _client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for Groq embeddings.")
    return Groq(api_key=api_key, timeout=60.0, max_retries=3)


def _normalize(vector: list[float]) -> list[float]:
    values = [float(value) for value in vector]
    if len(values) != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Groq embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, "
            f"received {len(values)}. Check GROQ_EMBEDDING_MODEL."
        )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("Groq returned a non-finite embedding value.")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        raise RuntimeError("Groq returned a zero-length embedding.")
    return [value / norm for value in values]


def embed_texts(texts: list[str], *, input_type: str = "search_document") -> list[list[float]]:
    """Embed texts in deterministic batches; Nomic distinguishes documents and queries."""
    if not texts:
        return []
    if input_type not in {"search_document", "search_query"}:
        raise ValueError("input_type must be search_document or search_query")
    clean = [str(text).strip() for text in texts]
    if any(not value for value in clean):
        raise ValueError("Embedding inputs must not contain empty text.")
    vectors: list[list[float] | None] = [None] * len(clean)
    for start in range(0, len(clean), BATCH_SIZE):
        batch = clean[start:start + BATCH_SIZE]
        prefixed = [f"{input_type}: {value}" for value in batch]
        response = _client().embeddings.create(
            model=EMBEDDING_MODEL,
            input=prefixed,
            encoding_format="float",
        )
        for item in response.data:
            index = start + int(item.index)
            if index < start or index >= start + len(batch):
                raise RuntimeError("Groq returned an out-of-range embedding index.")
            vectors[index] = _normalize(item.embedding)
    if any(vector is None for vector in vectors):
        raise RuntimeError("Groq returned an incomplete embedding batch.")
    return [vector for vector in vectors if vector is not None]


def embed_query(text: str) -> list[float]:
    return embed_texts([text], input_type="search_query")[0]
