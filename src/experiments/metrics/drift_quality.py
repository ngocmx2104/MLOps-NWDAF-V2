# src/experiments/metrics/drift_quality.py
from __future__ import annotations

from typing import Any

import pandas as pd

from src.monitoring.psi import PSIDriftMonitor
from src.monitoring.schema import MonitoringConfig


def drift_metrics(reference: pd.DataFrame, observed: pd.DataFrame, *,
                  min_features_alert: int = 3) -> dict[str, Any]:
    """Drift group (PSI 3-tier + KS) via the verified monitoring numerics."""
    cfg = MonitoringConfig(min_features_alert=min_features_alert, min_observed_rows=1)
    monitor = PSIDriftMonitor(reference_frame=reference, config=cfg)
    return monitor.evaluate(observed)


def data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Data-quality group: null/duplicate rates + row count."""
    n = len(df)
    null_rate = float(df.isna().mean().mean()) if n else 0.0
    dup_rate = float(df.duplicated().mean()) if n else 0.0
    return {"n_rows": n, "n_cols": df.shape[1], "null_rate": null_rate, "duplicate_rate": dup_rate}
