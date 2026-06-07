"""Data quality checks for D2 feature datasets and D5 weak-label datasets.

Follows the same pattern as ``src.ingestion.quality`` -- each check returns
a ``QualityCheckResult`` that can be serialised into metadata sidecars.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckResult:
    check_id: str
    description: str
    passed: bool
    detail: dict[str, Any]


# ---------------------------------------------------------------------------
# D2 quality checks
# ---------------------------------------------------------------------------

def check_d2_row_count(df: pd.DataFrame) -> QualityCheckResult:
    """Q-D2-1: D2 must have at least one row."""
    n = len(df)
    return QualityCheckResult(
        "Q-D2-1", "D2 non-empty check",
        passed=n > 0,
        detail={"row_count": n},
    )


def check_d2_key_columns(df: pd.DataFrame) -> QualityCheckResult:
    """Q-D2-2: Key columns (imsi, window_start) must be present and non-null."""
    required = ["imsi", "window_start"]
    total = len(df)
    if total == 0:
        return QualityCheckResult("Q-D2-2", "Key columns present", False, {"error": "empty"})
    detail: dict[str, Any] = {"total_rows": total}
    all_ok = True
    for col in required:
        if col not in df.columns:
            detail[col] = "MISSING"
            all_ok = False
        else:
            nn = int(df[col].notna().sum())
            detail[col] = {"non_null": nn, "pct": round(nn / total * 100, 2)}
            if nn < total * 0.95:
                all_ok = False
    return QualityCheckResult("Q-D2-2", "Key columns present and non-null", all_ok, detail)


def check_d2_feature_columns(df: pd.DataFrame) -> QualityCheckResult:
    """Q-D2-3: All expected feature columns must exist."""
    expected = [
        "n_handover", "n_unique_cells", "pingpong_count", "pingpong_rate",
        "mean_inter_ho_s", "std_inter_ho_s", "entropy_cell_seq",
    ]
    missing = [c for c in expected if c not in df.columns]
    return QualityCheckResult(
        "Q-D2-3", "Feature columns present",
        passed=len(missing) == 0,
        detail={"expected": expected, "missing": missing},
    )


def check_d2_provenance(df: pd.DataFrame) -> QualityCheckResult:
    """Q-D2-4: Provenance columns (feature_version, source_snapshot_id) must be present."""
    prov = ["feature_version", "source_snapshot_id"]
    missing = [c for c in prov if c not in df.columns]
    if missing:
        return QualityCheckResult("Q-D2-4", "Provenance columns", False,
                                  {"missing": missing})
    total = len(df)
    detail: dict[str, Any] = {"total_rows": total}
    all_ok = True
    for col in prov:
        nn = int(df[col].notna().sum())
        detail[col] = nn
        if nn < total:
            all_ok = False
    return QualityCheckResult("Q-D2-4", "Provenance columns fully populated", all_ok, detail)


def check_d2_no_negative_counts(df: pd.DataFrame) -> QualityCheckResult:
    """Q-D2-5: Count features should not have negative values."""
    count_cols = ["n_handover", "n_unique_cells", "pingpong_count"]
    detail: dict[str, Any] = {}
    all_ok = True
    for col in count_cols:
        if col in df.columns:
            neg = int((pd.to_numeric(df[col], errors="coerce").fillna(0) < 0).sum())
            detail[col] = {"negative_rows": neg}
            if neg > 0:
                all_ok = False
    return QualityCheckResult("Q-D2-5", "No negative count features", all_ok, detail)


def run_d2_quality_checks(df: pd.DataFrame) -> list[QualityCheckResult]:
    """Run all D2 quality checks."""
    return [
        check_d2_row_count(df),
        check_d2_key_columns(df),
        check_d2_feature_columns(df),
        check_d2_provenance(df),
        check_d2_no_negative_counts(df),
    ]


# ---------------------------------------------------------------------------
# D5 quality checks
# ---------------------------------------------------------------------------

def check_d5_label_column(df: pd.DataFrame) -> QualityCheckResult:
    """Q-D5-1: weak_label column must exist and contain only 0/1."""
    if "weak_label" not in df.columns:
        return QualityCheckResult("Q-D5-1", "weak_label column present", False,
                                  {"error": "column missing"})
    unique_vals = set(df["weak_label"].dropna().unique())
    valid = unique_vals.issubset({0, 1})
    return QualityCheckResult(
        "Q-D5-1", "weak_label column valid (binary 0/1)",
        passed=valid,
        detail={"unique_values": sorted(unique_vals), "is_binary": valid},
    )


def check_d5_label_distribution(df: pd.DataFrame) -> QualityCheckResult:
    """Q-D5-2: Label distribution should not be 100% one class (degenerate)."""
    if "weak_label" not in df.columns or len(df) == 0:
        return QualityCheckResult("Q-D5-2", "Label distribution", False,
                                  {"error": "no data"})
    dist = df["weak_label"].value_counts().to_dict()
    n_classes = len(dist)
    return QualityCheckResult(
        "Q-D5-2", "Label distribution non-degenerate (both classes present)",
        passed=n_classes >= 2,
        detail={"distribution": {str(k): int(v) for k, v in dist.items()},
                "n_classes": n_classes},
    )


def check_d5_provenance(df: pd.DataFrame) -> QualityCheckResult:
    """Q-D5-3: D5 provenance columns must be present."""
    prov = ["weak_label_version", "source_feature_version", "source_snapshot_id"]
    missing = [c for c in prov if c not in df.columns]
    if missing:
        return QualityCheckResult("Q-D5-3", "D5 provenance columns", False,
                                  {"missing": missing})
    total = len(df)
    detail: dict[str, Any] = {"total_rows": total}
    all_ok = True
    for col in prov:
        nn = int(df[col].notna().sum())
        detail[col] = nn
        if nn < total:
            all_ok = False
    return QualityCheckResult("Q-D5-3", "D5 provenance columns fully populated", all_ok, detail)


def check_d5_row_match(df: pd.DataFrame) -> QualityCheckResult:
    """Q-D5-4: D5 should have the same rows as its D2 source (no data loss)."""
    total = len(df)
    has_label = int(df["weak_label"].notna().sum()) if "weak_label" in df.columns else 0
    return QualityCheckResult(
        "Q-D5-4", "All rows have weak_label assigned",
        passed=has_label == total and total > 0,
        detail={"total_rows": total, "labeled_rows": has_label},
    )


def run_d5_quality_checks(df: pd.DataFrame) -> list[QualityCheckResult]:
    """Run all D5 quality checks."""
    return [
        check_d5_label_column(df),
        check_d5_label_distribution(df),
        check_d5_provenance(df),
        check_d5_row_match(df),
    ]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_feature_quality_report(results: list[QualityCheckResult]) -> str:
    """Human-readable quality report for D2/D5 checks."""
    lines = ["Feature Quality Report", "=" * 60]
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
