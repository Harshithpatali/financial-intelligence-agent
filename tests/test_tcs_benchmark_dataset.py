import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _number_tokens(text):
    values = set()
    for raw in re.findall(r"\(?\s*[-+]?\d[\d,]*(?:\.\d+)?\s*\)?", text):
        token = raw.strip().replace(",", "")
        negative = token.startswith("-") or (token.startswith("(") and token.endswith(")"))
        token = token.strip("()").lstrip("+-").strip()
        values.add(("-" if negative else "") + token)
    return values


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


def test_each_benchmark_anchor_and_expected_value_are_present_nearby():
    rows = json.loads((ROOT / "evaluation/tcs_benchmark.json").read_text(encoding="utf-8"))
    for row in rows:
        source = (ROOT / row["source"]).read_text(encoding="utf-8")
        anchor_positions = [m.start() for m in re.finditer(re.escape(row["anchor"]), source, re.I)]
        assert anchor_positions, f'{row["id"]}: anchor missing from {row["source"]}'
        expected_numbers = _number_tokens(row["expected_value"])
        windows = [
            source[max(0, pos - 700):pos + 1100]
            for pos in anchor_positions
        ]
        # The source reports headcount as 270k+, while the normalized expected value is 270000.
        if row["id"] == "q34":
            assert any("270k" in window.casefold() for window in windows)
        elif row["id"] == "q14":
            # "Cash paid" is a positive magnitude although the statement presents the outflow in parentheses.
            assert any({value.lstrip("-") for value in expected_numbers} & {
                value.lstrip("-") for value in _number_tokens(window)
            } for window in windows)
        else:
            assert any(expected_numbers & _number_tokens(window) for window in windows), (
                f'{row["id"]}: expected value {row["expected_value"]!r} not found near anchor '
                f'{row["anchor"]!r}'
            )
