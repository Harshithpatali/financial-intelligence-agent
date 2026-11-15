"""Deterministic checks for structured financial snapshots and conflicting evidence.

The CSV snapshots are secondary structured inputs. A successful check does not
establish that a value agrees with an audited annual report.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STATEMENTS = {
    "income": ROOT / "data/financial_metrics/income_statement.csv",
    "balance": ROOT / "data/financial_metrics/balance_sheet.csv",
    "cash_flow": ROOT / "data/financial_metrics/cash_flow.csv",
}


def normalize_period(period: str) -> str:
    text = str(period).strip().upper().replace(" ", "")
    fiscal_span = re.fullmatch(r"FY?(\d{2})[-/](\d{2})", text)
    if fiscal_span:
        _, suffix = fiscal_span.groups()
        return "FY20" + suffix
    match = re.fullmatch(r"(?:FY)?(20\d{2})(?:[-/]?(\d{2,4}))?", text)
    if match:
        year, suffix = match.groups()
        if suffix:
            return "FY" + (suffix if len(suffix) == 4 else year[:2] + suffix)
        return "FY" + year
    date_match = re.fullmatch(r"(20\d{2})-(\d{2})-(\d{2})", text)
    if date_match:
        return "FY" + date_match.group(1)
    raise ValueError(f"Unsupported reporting period: {period}")


def _load_statement(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Financial snapshot is missing: {path}")
    return pd.read_csv(path, index_col=0)


def verify_snapshot_value(
    metric: str,
    period: str,
    reported_value: float,
    statement: str = "income",
    absolute_tolerance: float = 1.0,
) -> dict:
    """Compare a numeric claim to the selected local CSV snapshot."""
    if statement not in STATEMENTS:
        raise ValueError(f"Unknown statement '{statement}'")
    if not math.isfinite(float(reported_value)):
        raise ValueError("reported_value must be finite")
    if absolute_tolerance < 0:
        raise ValueError("absolute_tolerance cannot be negative")

    frame = _load_statement(STATEMENTS[statement])
    if metric not in frame.index:
        return {"status": "metric_not_found", "metric": metric, "period": normalize_period(period)}
    target_period = normalize_period(period)
    columns = {normalize_period(col): col for col in frame.columns}
    if target_period not in columns:
        return {"status": "period_not_found", "metric": metric, "period": target_period}

    source_value = frame.loc[metric, columns[target_period]]
    if pd.isna(source_value):
        return {"status": "source_value_missing", "metric": metric, "period": target_period}
    source_value = float(source_value)
    difference = float(reported_value) - source_value
    matches = abs(difference) <= absolute_tolerance
    return {
        "status": "match" if matches else "mismatch",
        "metric": metric,
        "period": target_period,
        "reported_value": float(reported_value),
        "snapshot_value": source_value,
        "difference": difference,
        "tolerance": float(absolute_tolerance),
        "source_file": STATEMENTS[statement].relative_to(ROOT).as_posix(),
        "source_type": "structured_financial_snapshot_not_audited_source",
    }


def find_conflicting_disclosures(records: list[dict], relative_tolerance: float = 0.005) -> list[dict]:
    """Return disagreements; never silently average or choose between sources.

    Each record requires metric, period, value and source. Values must already
    use the same unit and consolidation basis before they can be compared.
    """
    if not 0 <= relative_tolerance < 1:
        raise ValueError("relative_tolerance must be in [0, 1)")
    grouped = defaultdict(list)
    for item in records:
        required = {"metric", "period", "value", "source", "unit", "basis"}
        if not required.issubset(item):
            raise ValueError(f"Each disclosure must include {sorted(required)}")
        value = float(item["value"])
        if not math.isfinite(value):
            raise ValueError("Disclosure values must be finite")
        key = (str(item["metric"]).strip().lower(), normalize_period(item["period"]), str(item["unit"]).strip().lower(), str(item["basis"]).strip().lower())
        grouped[key].append({**item, "value": value, "period": key[1]})

    conflicts = []
    for (metric, period, unit, basis), items in grouped.items():
        if len(items) < 2:
            continue
        values = [item["value"] for item in items]
        scale = max(abs(value) for value in values) or 1.0
        spread = max(values) - min(values)
        if spread / scale > relative_tolerance:
            conflicts.append({
                "metric": metric,
                "period": period,
                "status": "conflict_requires_review",
                "spread": spread,
                "disclosures": items,
                "resolution": None,
            })
    return conflicts
