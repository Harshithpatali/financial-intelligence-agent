from types import SimpleNamespace

from backend.rag import embeddings


def test_embed_texts_batches_and_restores_response_order(monkeypatch):
    calls = []

    class FakeEmbeddings:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(data=[
                SimpleNamespace(index=1, embedding=[0.0, 3.0, 4.0]),
                SimpleNamespace(index=0, embedding=[0.0, 0.0, 2.0]),
            ])

    monkeypatch.setattr(embeddings, "EMBEDDING_DIMENSION", 3)
    monkeypatch.setattr(embeddings, "BATCH_SIZE", 2)
    monkeypatch.setattr(embeddings, "_client", lambda: SimpleNamespace(embeddings=FakeEmbeddings()))

    vectors = embeddings.embed_texts(["first", "second"], input_type="search_query")

    assert calls[0]["model"] == embeddings.EMBEDDING_MODEL
    assert calls[0]["input"] == ["search_query: first", "search_query: second"]
    assert vectors == [[0.0, 0.0, 1.0], [0.0, 0.6, 0.8]]


def test_embed_texts_rejects_empty_input():
    try:
        embeddings.embed_texts([""])
    except ValueError as exc:
        assert "empty text" in str(exc)
    else:
        raise AssertionError("Expected empty input to be rejected")
