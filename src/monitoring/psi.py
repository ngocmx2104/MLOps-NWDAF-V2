"""PSI-based covariate drift detection (3-tier ok/warn/alert) + KS cross-statistic.

Ported from the verified MLOps_Project monitoring numerics. PSI is the primary
*measured* detector for Exp-4; the KS p-value is reported alongside as a check.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.monitoring.schema import MonitoringConfig


def compute_psi(reference: np.ndarray, observed: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1-D arrays."""
    eps = 1e-6
    breakpoints = np.linspace(float(np.min(reference)), float(np.max(reference)), bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    ref_counts = np.histogram(reference, bins=breakpoints)[0].astype(float)
    obs_counts = np.histogram(observed, bins=breakpoints)[0].astype(float)
    ref_pct = np.clip(ref_counts / (ref_counts.sum() + eps), eps, None)
    obs_pct = np.clip(obs_counts / (obs_counts.sum() + eps), eps, None)
    return float(np.sum((obs_pct - ref_pct) * np.log(obs_pct / ref_pct)))


@dataclass(frozen=True)
class PSIDriftMonitor:
    """3-tier PSI covariate drift monitor (C8). evaluate(observed) -> status dict with
    per-feature psi/level (ok/warn/alert) + KS, and drift_detected when >= min_features_alert
    features hit 'alert'. (PSI is undefined for zero-variance reference columns -> they yield
    psi=0; the EBS handover features all have variance.)"""

    reference_frame: pd.DataFrame | None = None
    config: MonitoringConfig = field(default_factory=MonitoringConfig)

    def evaluate(self, observed_frame: pd.DataFrame | None) -> dict[str, Any]:
        cfg = self.config
        if self.reference_frame is None:
            return {"status": "reference-unavailable", "drift_detected": False, "per_feature": {}}
        if observed_frame is None or observed_frame.empty:
            return {"status": "insufficient-live-window", "drift_detected": False, "per_feature": {}}
        if observed_frame.shape[0] < cfg.min_observed_rows:
            return {"status": "waiting-min-window", "drift_detected": False,
                    "observed_rows": int(observed_frame.shape[0]),
                    "required_rows": cfg.min_observed_rows, "per_feature": {}}

        per_feature: dict[str, Any] = {}
        alerted = 0
        for col in observed_frame.columns:
            if col not in self.reference_frame.columns:
                continue
            ref = self.reference_frame[col].dropna().to_numpy()
            obs = observed_frame[col].dropna().to_numpy()
            if len(ref) == 0 or len(obs) == 0:
                continue
            psi_val = compute_psi(ref, obs, cfg.psi_bins)
            ks_stat, ks_p = ks_2samp(ref, obs)
            level = "alert" if psi_val >= cfg.psi_alert else ("warn" if psi_val >= cfg.psi_warn else "ok")
            per_feature[col] = {"psi": round(psi_val, 4), "psi_level": level,
                                "ks_statistic": round(float(ks_stat), 4),
                                "ks_pvalue": round(float(ks_p), 6)}
            if psi_val >= cfg.psi_alert:
                alerted += 1

        return {"status": "evaluated", "drift_detected": alerted >= cfg.min_features_alert,
                "alerted_features": alerted, "total_features": len(per_feature),
                "psi_alert_threshold": cfg.psi_alert, "psi_warn_threshold": cfg.psi_warn,
                "min_features_for_alert": cfg.min_features_alert, "per_feature": per_feature}
