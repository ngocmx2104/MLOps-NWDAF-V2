# src/experiments/metrics/business.py
from __future__ import annotations

from typing import Any

import numpy as np


def business_metrics(y_true, y_pred, *, c_fp: float, c_fn: float,
                     window_hours: float | None = None) -> dict[str, Any]:
    """Business-impact group. expected_cost = FP*C(FP) + FN*C(FN) (Elkan 2001 cost-sensitive).
    C(FP)/C(FN) are parameters; their VALUES + justification are set by the caller (P8b/thesis)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    out: dict[str, Any] = {
        "fp": fp, "fn": fn, "tp": tp,
        "expected_cost": fp * c_fp + fn * c_fn,
        "cost_params": {"c_fp": c_fp, "c_fn": c_fn},
    }
    if window_hours:
        out["detections_per_hour"] = int(np.sum(y_pred == 1)) / window_hours
    return out


def cost_sensitivity_curve(*, fp: int, fn: int, c_fp: float, ratios: list[float]) -> list[dict[str, Any]]:
    """Report expected_cost across a range of cost ratios r = C(FN)/C(FP) (Elkan 2001), instead of
    committing to a single subjective cost. C(FP) fixed; C(FN) = r * C(FP). One row per ratio.
    Takes confusion counts directly (fp, fn) — the experiment harness reads these from the emitted
    validation_summary.confusion rather than holding per-sample predictions."""
    out: list[dict[str, Any]] = []
    for r in ratios:
        c_fn = r * c_fp
        out.append({"ratio": r, "c_fp": c_fp, "c_fn": c_fn, "expected_cost": fp * c_fp + fn * c_fn})
    return out
