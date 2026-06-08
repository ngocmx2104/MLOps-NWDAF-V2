"""Auto-retrain closed loop: detect drift -> (cooldown) -> retrain on a sliding window
-> eval gate -> ServingRuntime.reload(). Returns whether a retrain was deployed so the
P8 Exp-4 harness can compare loop ON (C1) vs OFF (C0).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.monitoring.detector import DriftDetector
from src.monitoring.schema import MonitoringConfig
from src.serving.records import append_jsonl, utc_now_iso
from src.training.data import apply_sliding_window
from src.training.pipeline import run_training
from src.training.schema import TrainingConfig


@dataclass
class AutoRetrainTrigger:
    cooldown_seconds: float = 3600.0
    _last_retrain_time: float = field(default=0.0, init=False, repr=False)
    _total_retrain_count: int = field(default=0, init=False, repr=False)

    def should_retrain(self, drift_result: dict[str, Any]) -> bool:
        if not drift_result.get("drift_detected", False):
            return False
        return (time.time() - self._last_retrain_time) >= self.cooldown_seconds

    def record_retrain(self) -> None:
        self._last_retrain_time = time.time()
        self._total_retrain_count += 1


def run_retrain_cycle(*, predictions_path: Path, reference_path: Path, dataset_path: Path,
                      output_dir: Path, config: MonitoringConfig | None = None,
                      model_type: str = "iforest", backend: str = "noop",
                      runtime: Any = None,
                      trigger: AutoRetrainTrigger | None = None) -> dict[str, Any]:
    # NOTE: when backend="mlflow", the caller sets MLFLOW_TRACKING_URI in the env (the
    # standard MLflow pattern) so the retrained model registers where the serving
    # runtime reads from. run_training picks up that global URI.
    config = config or MonitoringConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trigger = trigger or AutoRetrainTrigger(cooldown_seconds=config.cooldown_seconds)

    # 1. detect drift over the serving predictions
    drift = DriftDetector(config).detect(reference_path, predictions_path)
    append_jsonl(output_dir / "drift_history.jsonl",
                 {"recorded_at": utc_now_iso(), "event": "drift_check", **_drift_summary(drift)})
    if not trigger.should_retrain(drift):
        return {"retrained": False, "drift": drift}

    # 2. retrain on a sliding window of the dataset (MTLF strategy)
    df = pd.read_parquet(dataset_path)
    windowed = apply_sliding_window(df, window_days=config.window_days)
    window_path = output_dir / "retrain_window.parquet"
    windowed.to_parquet(window_path, index=False)
    result = run_training(window_path, model_type=model_type, backend=backend,
                          output_dir=output_dir / "model",
                          cfg=TrainingConfig(use_labels_for_evaluation=True, label_column="label"))
    new_auc = float(result["metrics"].get("roc_auc", 0.0))

    # 3. eval gate — do not deploy a model below the bar. We skip reload() so the SERVING
    # runtime keeps the prior model. KNOWN LIMIT (registry/C1 path): run_training already
    # moved the MLflow 'staging' alias to the rejected model, so a later reload/restart
    # would pick it up. True registry governance (gate decides alias promotion, i.e. split
    # register-from-promote in run_training + revert alias on rejection) is deferred to P7.
    if new_auc < config.retrain_min_auc:
        append_jsonl(output_dir / "retrain_history.jsonl",
                     {"recorded_at": utc_now_iso(), "event": "retrain_rejected",
                      "new_auc": new_auc, "gate": config.retrain_min_auc})
        return {"retrained": False, "gate_failed": True, "new_auc": new_auc, "drift": drift}

    # 4. deploy — reload serving (registry-alias loaders pick up the new staging version).
    # reload() FIRST: only stamp the cooldown + write the deployed record once the new
    # model is actually serving, so a failed reload doesn't waste the cooldown or leave a
    # drift trigger with no matching deployment record.
    if runtime is not None:
        runtime.reload()
    trigger.record_retrain()
    append_jsonl(output_dir / "retrain_history.jsonl",
                 {"recorded_at": utc_now_iso(), "event": "retrain_deployed",
                  "new_auc": new_auc, "model_version": result.get("model_version"),
                  "model_path": result.get("model_path")})
    # retrain_count: cumulative when the caller threads a persistent trigger across cycles
    # (the P8 Exp-4 harness does this -> # of auto-retrains is an RQ4 operational metric).
    return {"retrained": True, "retrain_count": trigger._total_retrain_count,
            "new_auc": new_auc, "model_version": result.get("model_version"),
            "model_path": result.get("model_path"), "drift": drift}


def _drift_summary(drift: dict[str, Any]) -> dict[str, Any]:
    return {"drift_detected": drift["drift_detected"],
            "evidently_drift_share": drift["evidently_drift_share"],
            "alerted_features": drift["psi"].get("alerted_features"),
            "observed_rows": drift["observed_rows"]}
