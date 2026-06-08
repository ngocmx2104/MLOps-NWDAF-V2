"""DriftDetector — covariate drift over serving predictions vs a training reference.

Combines PSI (primary measured detector) with Evidently (standard cross-check).
`drift_detected` follows PSI (the detector measured in Exp-4); the Evidently share is
reported alongside for validation. Observed features come from the serving
predictions.jsonl (each record carries feature_values).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.monitoring.evidently_drift import evidently_drift_share
from src.monitoring.psi import PSIDriftMonitor
from src.monitoring.schema import MonitoringConfig
from src.training.schema import FEATURE_COLUMNS

_FEATS = list(FEATURE_COLUMNS)


class DriftDetector:
    def __init__(self, config: MonitoringConfig | None = None) -> None:
        self.config = config or MonitoringConfig()

    def load_observed(self, predictions_path: str | Path) -> pd.DataFrame:
        rows: list[dict[str, float]] = []
        with Path(predictions_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                feats = rec.get("feature_values")
                if feats:
                    rows.append({c: float(feats[c]) for c in _FEATS if c in feats})
        return pd.DataFrame(rows, columns=_FEATS)

    def load_reference(self, reference_path: str | Path) -> pd.DataFrame:
        df = pd.read_parquet(reference_path)
        return df[[c for c in _FEATS if c in df.columns]].copy()

    def detect(self, reference_path: str | Path, predictions_path: str | Path) -> dict[str, Any]:
        reference = self.load_reference(reference_path)
        observed = self.load_observed(predictions_path)
        psi_result = PSIDriftMonitor(reference_frame=reference, config=self.config).evaluate(observed)
        share = (evidently_drift_share(reference, observed)
                 if not observed.empty and len(observed) >= self.config.min_observed_rows else 0.0)
        return {"drift_detected": bool(psi_result["drift_detected"]),
                "psi": psi_result, "evidently_drift_share": share,
                "observed_rows": int(len(observed))}
