from scripts.generate_sentiment_report import COMPANIES, MODEL


def test_expected_universe_is_configured():
    assert set(COMPANIES) == {"tcs", "infosys", "reliance", "hdfc_bank", "icici_bank", "nifty50", "sensex"}


def test_finbert_model_is_financial_domain_model():
    assert MODEL == "ProsusAI/finbert"
