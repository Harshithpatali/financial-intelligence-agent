from scripts.generate_sentiment_report import COMPANIES, MAX_HEADLINES, MODEL, classify_headlines


def test_expected_universe_is_configured():
    assert set(COMPANIES) == {"tcs", "infosys", "reliance", "hdfc_bank", "icici_bank", "nifty50", "sensex"}


def test_headline_limit_is_twenty_per_company():
    assert MAX_HEADLINES == 20


def test_groq_model_is_configured():
    assert MODEL == "openai/gpt-oss-20b"


def test_classification_preserves_headline_order_and_validates_scores():
    class FakeCompletions:
        def create(self, **kwargs):
            class Message:
                content = '{"items":[{"index":0,"label":"positive","confidence":0.9,"sentiment_score":0.7},{"index":1,"label":"negative","confidence":0.8,"sentiment_score":-0.6}]}'
            class Choice:
                message = Message()
            class Response:
                choices = [Choice()]
            return Response()

    class FakeClient:
        class Chat:
            class Completions:
                create = staticmethod(FakeCompletions().create)
            completions = Completions()
        chat = Chat()

    headlines = [
        {"title": "Results beat estimates", "summary": ""},
        {"title": "Profit falls sharply", "summary": ""},
    ]
    results = classify_headlines(headlines, FakeClient())
    assert [row["label"] for row in results] == ["positive", "negative"]
    assert [row["sentiment_score"] for row in results] == [0.7, -0.6]
    assert all(0 <= row["confidence"] <= 1 for row in results)
