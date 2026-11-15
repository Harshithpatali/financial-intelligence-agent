import ast
from pathlib import Path


def test_streamlit_frontend_is_valid_python():
    app_path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    source = app_path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(app_path))


def test_frontend_contains_core_research_views():
    app_path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    source = app_path.read_text(encoding="utf-8")
    for expected in ("AI research assistant", "Balance sheet strength", "Cash flow quality", "How the research engine works"):
        assert expected in source
