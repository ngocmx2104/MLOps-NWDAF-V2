"""Evidently drift cross-check (industry-standard validation of the custom PSI detector).

Returns the share of drifted columns from Evidently's DataDriftPreset. Used to
cross-validate PSI (answers the 'manual/cảm tính' critique); PSI remains the primary
measured detector. Evidently 0.7.x API: Report([DataDriftPreset]).run(current, reference).
"""
from __future__ import annotations

import pandas as pd


def evidently_drift_share(reference: pd.DataFrame, current: pd.DataFrame) -> float:
    """Fraction of columns Evidently flags as drifted (0.0–1.0)."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    snapshot = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
    for metric in snapshot.dict().get("metrics", []):
        value = metric.get("value")
        if isinstance(value, dict) and "share" in value:
            return float(value["share"])
    return 0.0
