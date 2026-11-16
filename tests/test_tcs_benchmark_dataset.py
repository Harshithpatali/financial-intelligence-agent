import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_tcs_benchmark_has_50_unique_source_anchored_questions():
    rows = json.loads((ROOT / "evaluation/tcs_benchmark.json").read_text(encoding="utf-8"))
    assert len(rows) == 50
    assert len({row["id"] for row in rows}) == 50
    required = {"id", "question", "expected_value", "unit", "source", "anchor", "category"}
    for row in rows:
        assert required.issubset(row)
        assert row["question"].strip()
        assert row["expected_value"].strip()
        assert row["anchor"].strip()
        assert (ROOT / row["source"]).is_file()


def test_tcs_benchmark_covers_financial_and_operating_disclosures():
    rows = json.loads((ROOT / "evaluation/tcs_benchmark.json").read_text(encoding="utf-8"))
    sources = {row["source"] for row in rows}
    assert "knowledge/quarterly_reports/FY25_Q4_consolidated.md" in sources
    assert "knowledge/quarterly_reports/FY25_Q4_press_release.md" in sources
    assert "knowledge/quarterly_reports/FY25_Q3_consolidated.md" in sources
