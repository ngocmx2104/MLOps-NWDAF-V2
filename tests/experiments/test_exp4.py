"""Tests for Exp-4: drift scenario-clock + closed-loop retrain ON/OFF (RQ4a).

Key integration checks:
- Big covariate shift (shift=5.0) -> drift_detected_any=True
- loop="on" + drift -> retrain_count >= 1
- loop="off" -> retrain_count == 0

The test frames carry:
- All 7 FEATURE_COLUMNS (for DriftDetector.load_reference and _write_step_predictions)
- A "label" column (for run_retrain_cycle -> TrainingConfig(label_column="label"))
- No "window_start" column (apply_sliding_window returns df unchanged when column is absent)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.exp4_drift import run_exp4_scenario
from src.training.schema import FEATURE_COLUMNS


def _frame(n: int, shift: float, seed: int) -> pd.DataFrame:
    """Build a synthetic feature frame with a 'label' column for retrain_cycle."""
    rng = np.random.RandomState(seed)
    base = rng.rand(n, len(FEATURE_COLUMNS)) + shift
    df = pd.DataFrame(base, columns=list(FEATURE_COLUMNS))
    # Ensure all feature values are positive (PSI bins need valid ranges)
    df = df.abs()
    # "label" column required by run_retrain_cycle (TrainingConfig(label_column="label"))
    df["label"] = (rng.rand(n) < 0.3).astype(int)
    return df


def test_run_exp4_scenario_detects_and_counts(tmp_path):
    """Big covariate shift -> drift detected; loop=on -> at least one retrain."""
    baseline = _frame(200, shift=0.0, seed=1)
    drift = _frame(200, shift=5.0, seed=2)  # large covariate shift -> PSI alerts
    baseline.to_parquet(tmp_path / "baseline.parquet")
    drift.to_parquet(tmp_path / "drift.parquet")

    res = run_exp4_scenario(
        scenario="sudden",
        baseline_path=tmp_path / "baseline.parquet",
        drift_path=tmp_path / "drift.parquet",
        output_root=tmp_path / "out",
        n_steps=4,
        loop="on",
    )
    assert res["scenario"] == "sudden"
    assert res["drift_detected_any"] is True, (
        f"Expected drift to be detected with shift=5.0; got step_flags={res['step_drift_flags']}"
    )
    assert isinstance(res["detection_latency_steps"], int)
    assert res["retrain_count"] >= 1, (
        f"Expected at least one retrain (loop=on, drift detected); got {res['retrain_count']}"
    )

    # OFF run: drift may still be detected, but retrain is never triggered
    res_off = run_exp4_scenario(
        scenario="sudden",
        baseline_path=tmp_path / "baseline.parquet",
        drift_path=tmp_path / "drift.parquet",
        output_root=tmp_path / "out_off",
        n_steps=4,
        loop="off",
    )
    assert res_off["retrain_count"] == 0, (
        f"Expected no retrains with loop=off; got {res_off['retrain_count']}"
    )
