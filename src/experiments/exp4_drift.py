"""Exp-4 drift scenario-clock + closed-loop retrain ON/OFF (RQ4a).

Fixes §9.14: a SEPARATE predictions file is written per step (no accumulation across
steps). The AutoRetrainTrigger is persistent across all steps so the cumulative
retrain_count is tracked correctly.

P6 format contracts honoured here:
- DriftDetector.load_reference  reads a parquet of FEATURE_COLUMNS.
  -> baseline_path must be a parquet with those columns (guaranteed by callers).
- DriftDetector.load_observed   reads a JSONL where each record has {"feature_values": {...}}.
  -> _write_step_predictions writes exactly that format.
- run_retrain_cycle              calls apply_sliding_window(df, window_days=…).
  apply_sliding_window returns df unchanged when "window_start" column is absent,
  so dataset parquets without that column are safe.
  run_retrain_cycle uses TrainingConfig(label_column="label") hardcoded ->
  dataset parquets MUST have a "label" column.
- MonitoringConfig.min_observed_rows must be <= window size; we set it to 1 for
  tiny experiment runs (tiny test) but callers may increase it for real runs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.records import write_json
from src.monitoring.detector import DriftDetector
from src.monitoring.retrain import AutoRetrainTrigger, run_retrain_cycle
from src.monitoring.schema import MonitoringConfig
from src.training.schema import FEATURE_COLUMNS

_FEAT_COLS = list(FEATURE_COLUMNS)


def _write_step_predictions(window: pd.DataFrame, path: Path) -> Path:
    """Write a per-step predictions.jsonl in the format DriftDetector.load_observed expects.

    Each record: {"event": "predict", "feature_values": {col: float, ...}}.
    NO accumulation across steps — a fresh file is written for each step (fixes §9.14).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _, row in window.iterrows():
            fv = {c: float(row[c]) for c in _FEAT_COLS if c in window.columns}
            f.write(json.dumps({"event": "predict", "feature_values": fv}) + "\n")
    return path


def run_exp4_scenario(
    *,
    scenario: str,
    baseline_path: Path,
    drift_path: Path,
    output_root: Path,
    n_steps: int = 4,
    loop: str = "on",
    min_observed_rows: int = 1,
    cooldown_seconds: float = 0.0,
) -> dict[str, Any]:
    """One drift scenario: step a fixed window across baseline->drift, writing a SEPARATE
    predictions file per step (fixed window, no accumulation). Per step: detect drift;
    if loop 'on' and drift, run the closed-loop retrain (persistent trigger, cooldown=0).

    Returns:
        scenario, loop, n_steps, step_drift_flags (list[bool]), drift_detected_any,
        detection_latency_steps (steps after onset, -1 if never detected post-onset),
        retrain_count.
    """
    output_root = Path(output_root) / f"exp4_{scenario}_{loop}"
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = MonitoringConfig(
        cooldown_seconds=cooldown_seconds,
        min_observed_rows=min_observed_rows,
    )
    # Persistent trigger across steps so cumulative retrain_count is tracked
    trigger = AutoRetrainTrigger(cooldown_seconds=cooldown_seconds)

    base_df = pd.read_parquet(baseline_path)
    drift_df = pd.read_parquet(drift_path)
    # Step the drift scenario in TEMPORAL order (sorted by window_start) so gradual/recurring
    # drift accumulates across steps. Random sampling would scramble the time progression and
    # make slow drift look undetectable -- a harness artifact, not a real detector property.
    if "window_start" in drift_df.columns:
        drift_df = drift_df.sort_values("window_start").reset_index(drop=True)
    if "window_start" in base_df.columns:
        base_df = base_df.sort_values("window_start").reset_index(drop=True)

    # Step layout: first half uses baseline (pre-onset); second half steps THROUGH the drift file
    onset = n_steps // 2
    n_drift_steps = max(1, n_steps - onset)
    drift_win = max(1, len(drift_df) // n_drift_steps)
    base_win = max(1, len(base_df) // max(1, onset))

    step_flags: list[bool] = []
    detection_latency: int | None = None
    retrain_count = 0

    for k in range(n_steps):
        if k < onset:
            src = base_df
            window = base_df.iloc[k * base_win:(k + 1) * base_win]
        else:
            src = drift_df
            j = k - onset
            window = drift_df.iloc[j * drift_win:(j + 1) * drift_win]
        if window.empty:
            window = src.head(max(1, len(src) // n_steps))

        pred_path = _write_step_predictions(
            window, output_root / f"step_{k}_pred.jsonl"
        )

        drift_result = DriftDetector(cfg).detect(baseline_path, pred_path)
        flagged = bool(drift_result["drift_detected"])
        step_flags.append(flagged)

        # Record detection latency: first step >= onset where drift is flagged
        if flagged and detection_latency is None and k >= onset:
            detection_latency = k - onset

        if loop == "on" and flagged:
            # retrain dataset: the current source window, with "label" column
            # (run_retrain_cycle uses TrainingConfig(label_column="label"))
            ds_path = output_root / f"retrain_ds_{k}.parquet"
            src.to_parquet(ds_path, index=False)
            r = run_retrain_cycle(
                predictions_path=pred_path,
                reference_path=baseline_path,
                dataset_path=ds_path,
                output_dir=output_root / f"rt_{k}",
                config=cfg,
                model_type="iforest",
                backend="noop",
                trigger=trigger,
            )
            if r.get("retrained"):
                # trigger._total_retrain_count is the authoritative cumulative count
                retrain_count = r.get("retrain_count", retrain_count + 1)

    return {
        "scenario": scenario,
        "loop": loop,
        "n_steps": n_steps,
        "step_drift_flags": step_flags,
        "drift_detected_any": any(step_flags),
        "detection_latency_steps": detection_latency if detection_latency is not None else -1,
        "retrain_count": retrain_count,
    }


def run_exp4(
    *,
    baseline_path: Path,
    scenarios: dict[str, Path],
    output_root: Path,
    n_steps: int = 6,
    min_observed_rows: int = 50,
    cooldown_seconds: float = 0.0,
) -> dict[str, Any]:
    """Exp-4 (RQ4a): run each drift scenario ON and OFF, summarise.

    ON vs OFF retrain_count difference is the closed-loop value signal.
    Writes exp4_summary.json to output_root/exp4_drift/.
    """
    output_root = Path(output_root)
    out: dict[str, Any] = {"experiment_id": "exp4_drift", "scenarios": {}}
    for name, path in scenarios.items():
        on = run_exp4_scenario(
            scenario=name,
            baseline_path=baseline_path,
            drift_path=path,
            output_root=output_root,
            n_steps=n_steps,
            loop="on",
            min_observed_rows=min_observed_rows,
            cooldown_seconds=cooldown_seconds,
        )
        off = run_exp4_scenario(
            scenario=name,
            baseline_path=baseline_path,
            drift_path=path,
            output_root=output_root,
            n_steps=n_steps,
            loop="off",
            min_observed_rows=min_observed_rows,
            cooldown_seconds=cooldown_seconds,
        )
        out["scenarios"][name] = {"on": on, "off": off}
    write_json(output_root / "exp4_drift" / "exp4_summary.json", out)
    return out
