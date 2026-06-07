"""Data quality checks for EBS ingestion and D1 canonical snapshot.

Implements the checks specified in EBS_SCHEMA_SPEC.md Section 9 and
CANONICAL_SNAPSHOT_SPEC.md Section 8.

Each check returns a ``QualityCheckResult`` and the overall report is a list
of results that can be serialised for audit.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from src.ingestion.schema import EXPECTED_FIELD_COUNT

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckResult:
    check_id: str
    description: str
    passed: bool
    detail: dict[str, Any]


def check_field_count(df: pd.DataFrame) -> QualityCheckResult:
    """Q-raw-1: Most lines should have the expected number of fields (52)."""
    total = len(df)
    if total == 0:
        return QualityCheckResult("Q-raw-1", "Field count check", False, {"error": "empty dataframe"})

    matching = (df["raw_field_count"] == EXPECTED_FIELD_COUNT).sum()
    pct = matching / total * 100

    return QualityCheckResult(
        check_id="Q-raw-1",
        description="Field count check (expected 52)",
        passed=pct >= 95.0,
        detail={
            "total_rows": int(total),
            "matching_rows": int(matching),
            "pct_matching": round(pct, 2),
            "expected": EXPECTED_FIELD_COUNT,
        },
    )


def check_required_columns_presence(df: pd.DataFrame) -> QualityCheckResult:
    """Q-raw-2: event_id, imsi, event_time must be non-null for most modeling candidates."""
    required = ["event_id", "imsi", "event_time"]
    total = len(df)
    if total == 0:
        return QualityCheckResult("Q-raw-2", "Required columns presence", False, {"error": "empty dataframe"})

    detail: dict[str, Any] = {"total_rows": int(total)}
    all_ok = True
    for col in required:
        if col not in df.columns:
            detail[col] = "MISSING_COLUMN"
            all_ok = False
            continue
        non_null = df[col].notna().sum()
        non_empty = (df[col].astype(str).str.strip() != "").sum() if non_null > 0 else 0
        pct = non_empty / total * 100
        detail[col] = {"non_null": int(non_null), "non_empty": int(non_empty), "pct": round(pct, 2)}
        if pct < 90.0:
            all_ok = False

    return QualityCheckResult(
        check_id="Q-raw-2",
        description="Required columns presence (event_id, imsi, event_time)",
        passed=all_ok,
        detail=detail,
    )


def check_timestamp_validity(df: pd.DataFrame) -> QualityCheckResult:
    """Q-D1-2: event_ts must parse successfully for most records."""
    total = len(df)
    if total == 0:
        return QualityCheckResult("Q-D1-2", "Timestamp validity", False, {"error": "empty dataframe"})

    if "event_ts" not in df.columns:
        return QualityCheckResult("Q-D1-2", "Timestamp validity", False, {"error": "event_ts column missing"})

    valid = df["event_ts"].notna().sum()
    pct = valid / total * 100

    return QualityCheckResult(
        check_id="Q-D1-2",
        description="Timestamp validity (event_ts parsed from event_time)",
        passed=pct >= 95.0,
        detail={"total_rows": int(total), "valid_ts": int(valid), "pct_valid": round(pct, 2)},
    )


def check_event_id_distribution(df: pd.DataFrame) -> QualityCheckResult:
    """Q-raw-4: Event ID distribution -- verify parser isn't column-shifted."""
    if "event_id" not in df.columns or len(df) == 0:
        return QualityCheckResult("Q-raw-4", "Event ID distribution", False, {"error": "no data"})

    dist = df["event_id"].value_counts().head(10).to_dict()
    known_events = {"l_handover", "l_service_request", "l_tau", "l_attach",
                    "l_bearer_modify", "l_pdn_connect", "l_detach", "l_pdn_disconnect"}
    top_events = set(dist.keys())
    overlap = top_events & known_events

    return QualityCheckResult(
        check_id="Q-raw-4",
        description="Event ID distribution (top 10 should contain known event types)",
        passed=len(overlap) >= 2,
        detail={"top_events": {str(k): int(v) for k, v in dist.items()}, "known_overlap": list(overlap)},
    )


def check_imsi_availability(df: pd.DataFrame) -> QualityCheckResult:
    """Q-D1-3: Subscriber identity (imsi) must be present for most modeling records."""
    total = len(df)
    if total == 0:
        return QualityCheckResult("Q-D1-3", "IMSI availability", False, {"error": "empty"})

    non_null = df["imsi"].notna().sum() if "imsi" in df.columns else 0
    pct = non_null / total * 100

    return QualityCheckResult(
        check_id="Q-D1-3",
        description="IMSI availability for modeling records",
        passed=pct >= 90.0,
        detail={"total": int(total), "non_null_imsi": int(non_null), "pct": round(pct, 2)},
    )


def check_cell_identity(df: pd.DataFrame) -> QualityCheckResult:
    """Q-D1-4: Cell identity (eci or ci) must be present for most handover records."""
    total = len(df)
    if total == 0:
        return QualityCheckResult("Q-D1-4", "Cell identity availability", False, {"error": "empty"})

    eci_ok = df["eci"].notna() if "eci" in df.columns else pd.Series([False] * total)
    ci_ok = df["ci"].notna() if "ci" in df.columns else pd.Series([False] * total)
    either = (eci_ok | ci_ok).sum()
    pct = either / total * 100

    return QualityCheckResult(
        check_id="Q-D1-4",
        description="Cell identity availability (eci or ci)",
        passed=pct >= 80.0,
        detail={"total": int(total), "with_cell_id": int(either), "pct": round(pct, 2)},
    )


def check_provenance_completeness(df: pd.DataFrame) -> QualityCheckResult:
    """Q-D1-5: Provenance columns must be present and fully populated."""
    provenance_cols = ["dataset_snapshot_id", "source_file", "raw_field_count", "schema_version"]
    missing_cols = [c for c in provenance_cols if c not in df.columns]
    if missing_cols:
        return QualityCheckResult(
            "Q-D1-5", "Provenance completeness", False,
            {"missing_columns": missing_cols},
        )

    total = len(df)
    detail: dict[str, Any] = {"total_rows": int(total)}
    all_ok = True
    for col in provenance_cols:
        non_null = df[col].notna().sum()
        detail[col] = int(non_null)
        if non_null < total:
            all_ok = False

    return QualityCheckResult(
        check_id="Q-D1-5",
        description="Provenance completeness",
        passed=all_ok,
        detail=detail,
    )


def check_event_filter(df: pd.DataFrame, expected_event: str = "l_handover") -> QualityCheckResult:
    """Q-D1-1: All records in D1 should match the event filter."""
    total = len(df)
    if total == 0:
        return QualityCheckResult("Q-D1-1", "Event filter", False, {"error": "empty"})

    matching = (df["event_id"] == expected_event).sum()
    pct = matching / total * 100

    return QualityCheckResult(
        check_id="Q-D1-1",
        description=f"Event filter ({expected_event})",
        passed=pct >= 99.9,
        detail={"total": int(total), "matching": int(matching), "pct": round(pct, 2)},
    )


def run_raw_quality_checks(df: pd.DataFrame) -> list[QualityCheckResult]:
    """Run all quality checks applicable to raw parsed data (before filtering)."""
    return [
        check_field_count(df),
        check_required_columns_presence(df),
        check_event_id_distribution(df),
    ]


def run_d1_quality_checks(df: pd.DataFrame) -> list[QualityCheckResult]:
    """Run all quality checks applicable to the D1 canonical snapshot."""
    return [
        check_event_filter(df),
        check_timestamp_validity(df),
        check_imsi_availability(df),
        check_cell_identity(df),
        check_provenance_completeness(df),
    ]


def format_quality_report(results: list[QualityCheckResult]) -> str:
    """Format quality check results as a human-readable report."""
    lines = ["Data Quality Report", "=" * 60]
    passed_count = sum(1 for r in results if r.passed)
    lines.append(f"Checks passed: {passed_count}/{len(results)}")
    lines.append("")

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"[{status}] {r.check_id}: {r.description}")
        for k, v in r.detail.items():
            lines.append(f"       {k}: {v}")
        lines.append("")

    return "\n".join(lines)


def quality_results_to_dicts(results: list[QualityCheckResult]) -> list[dict]:
    """Convert quality check results to serialisable dicts."""
    return [asdict(r) for r in results]
