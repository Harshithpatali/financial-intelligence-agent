print("retriever.py imported")

_model = None

def get_model():
    global _model

    if _model is None:
        print("STEP A: before import")

        from sentence_transformers import SentenceTransformer

        print("STEP B: after import")

        _model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("STEP C: model loaded")

    return _model