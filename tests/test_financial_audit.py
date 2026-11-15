import pytest

from backend.rag.financial_audit import (
    find_conflicting_disclosures,
    normalize_period,
    verify_snapshot_value,
)


def test_normalizes_fiscal_year_and_date():
    assert normalize_period("FY2025") == "FY2025"
    assert normalize_period("2025-03-31") == "FY2025"
    assert normalize_period("FY24-25") == "FY2025"


def test_snapshot_value_matches_income_statement():
    # Value is read from the repository's income-statement snapshot.
    result = verify_snapshot_value("Total Revenue", "FY2025", 2_553_240_000_000)
    assert result["status"] == "match"
    assert result["period"] == "FY2025"
    assert result["source_type"] == "structured_financial_snapshot_not_audited_source"


def test_wrong_period_or_value_is_not_accepted():
    assert verify_snapshot_value("Total Revenue", "FY2024", 2_553_240_000_000)["status"] == "mismatch"
    assert verify_snapshot_value("Total Revenue", "FY2030", 1)["status"] == "period_not_found"


def test_conflicting_disclosures_are_flagged_not_averaged():
    conflicts = find_conflicting_disclosures([
        {"metric": "Revenue", "period": "FY2025", "value": 100, "source": "annual_report.pdf", "unit": "INR", "basis": "consolidated"},
        {"metric": "Revenue", "period": "2025-03-31", "value": 110, "source": "earnings_call.md", "unit": "INR", "basis": "consolidated"},
    ])
    assert len(conflicts) == 1
    assert conflicts[0]["status"] == "conflict_requires_review"
    assert conflicts[0]["resolution"] is None


def test_same_value_disclosures_are_not_flagged():
    assert find_conflicting_disclosures([
        {"metric": "Revenue", "period": "FY2025", "value": 100, "source": "a", "unit": "INR", "basis": "consolidated"},
        {"metric": "Revenue", "period": "FY2025", "value": 100.1, "source": "b", "unit": "INR", "basis": "consolidated"},
    ]) == []


def test_nonfinite_claim_rejected():
    with pytest.raises(ValueError):
        verify_snapshot_value("Total Revenue", "FY2025", float("nan"))
