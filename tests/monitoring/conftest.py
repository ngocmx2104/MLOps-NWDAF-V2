"""Monitoring-test fixtures: a reference feature parquet + serving predictions.jsonl
(drift / no-drift) + a trained model. All under tmp_path."""
import json

import numpy as np
import pandas as pd
import pytest

from src.training.pipeline import run_training
from src.training.schema import FEATURE_COLUMNS, TrainingConfig

FEATURES = list(FEATURE_COLUMNS)


def _frame(rng, n, scale=1.0):
    df = pd.DataFrame({f: rng.gamma(2.0, 1.0, n) * scale for f in FEATURES})
    df["imsi"] = [str(i) for i in range(n)]
    df["window_start"] = pd.date_range("2024-06-26", periods=n, freq="5min", tz="UTC")
    return df


@pytest.fixture
def reference_parquet(tmp_path):
    rng = np.random.RandomState(0)
    df = _frame(rng, 400)
    df["label"] = 0
    anom = rng.choice(400, size=20, replace=False)
    df.loc[anom, FEATURES] *= 5.0
    df.loc[anom, "label"] = 1
    path = tmp_path / "reference.parquet"
    df.to_parquet(path, index=False)
    return path


def _write_predictions(path, frame):
    """Write a serving-style predictions.jsonl (each line has feature_values)."""
    with path.open("w", encoding="utf-8") as f:
        for _, row in frame.iterrows():
            rec = {"recorded_at": "2024-06-27T00:00:00Z", "event": "predict",
                   "is_anomaly": False, "anomaly_score": 0.1, "model_type": "iforest",
                   "model_version": "v1", "latency_ms": 1.0,
                   "feature_values": {f: float(row[f]) for f in FEATURES}}
            f.write(json.dumps(rec) + "\n")


@pytest.fixture
def predictions_no_drift(reference_parquet, tmp_path):
    # Resample rows from the reference itself -> SAME distribution -> guaranteed no drift
    # (avoids a false PSI signal from the reference's injected anomaly tail).
    ref = pd.read_parquet(reference_parquet)
    sample = ref.sample(n=200, replace=True, random_state=1)
    path = tmp_path / "pred_nodrift.jsonl"
    _write_predictions(path, sample)
    return path


@pytest.fixture
def predictions_drift(tmp_path):
    rng = np.random.RandomState(2)
    path = tmp_path / "pred_drift.jsonl"
    _write_predictions(path, _frame(rng, 200, scale=4.0))  # shifted -> drift
    return path


@pytest.fixture
def iforest_on_reference(reference_parquet, tmp_path):
    out = run_training(reference_parquet, model_type="iforest", backend="noop",
                       output_dir=tmp_path / "m",
                       cfg=TrainingConfig(use_labels_for_evaluation=True, label_column="label"))
    return out["model_path"]
