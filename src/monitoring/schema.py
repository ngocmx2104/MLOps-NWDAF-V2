"""Monitoring subsystem (component C8) — config + record schemas.

Monitoring is C1's capability (no backend abstraction); the C0/C1 contrast is the
drift->retrain loop ON vs OFF, measured in Exp-4 (P8).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

PSI_WARN = 0.20
PSI_ALERT = 0.25
PSI_BINS = 10


@dataclass(frozen=True)
class MonitoringConfig:
    psi_warn: float = PSI_WARN
    psi_alert: float = PSI_ALERT
    psi_bins: int = PSI_BINS
    min_features_alert: int = 3       # drift_detected when >= this many features hit 'alert'
    min_observed_rows: int = 50       # below this, report 'waiting-min-window'
    evidently_drift_threshold: float = 0.5  # evidently drift share >= this -> flags (cross-check)
    cooldown_seconds: float = 3600.0  # min gap between retrains
    retrain_min_auc: float = 0.5      # eval gate: deploy only if new val ROC-AUC >= this
    window_days: int = 5              # sliding window for retraining data

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
