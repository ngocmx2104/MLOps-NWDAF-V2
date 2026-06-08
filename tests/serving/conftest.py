"""Shared serving-test fixtures: a small labeled parquet + trained models + a
sqlite-MLflow registry + a materialized Feast repo. All artifacts live under tmp_path."""
import numpy as np
import pandas as pd
import pytest

from src.training.pipeline import run_training
from src.training.schema import FEATURE_COLUMNS, TrainingConfig

FEATURES = list(FEATURE_COLUMNS)


@pytest.fixture
def labeled_parquet(tmp_path):
    """A 300-row feature parquet with imsi/window_start/label (5% extreme anomalies)."""
    rng = np.random.RandomState(0)
    n = 300
    df = pd.DataFrame({f: rng.gamma(2.0, 1.0, n) for f in FEATURES})
    df["imsi"] = [str(i) for i in range(n)]
    df["window_start"] = pd.date_range("2024-06-26", periods=n, freq="5min", tz="UTC")
    df["label"] = 0
    anom = rng.choice(n, size=15, replace=False)
    df.loc[anom, FEATURES] = df.loc[anom, FEATURES] * 5.0
    df.loc[anom, "label"] = 1
    path = tmp_path / "features.parquet"
    df.to_parquet(path, index=False)
    return path, df


@pytest.fixture
def _cfg():
    return TrainingConfig(use_labels_for_evaluation=True, label_column="label")


@pytest.fixture
def iforest_model_path(labeled_parquet, tmp_path, _cfg):
    path, _ = labeled_parquet
    out = run_training(path, model_type="iforest", backend="noop",
                       output_dir=tmp_path / "m_if", cfg=_cfg)
    return out["model_path"]  # a self-contained joblib (bare IsolationForest)


@pytest.fixture
def lstm_model_path(labeled_parquet, tmp_path, _cfg):
    path, _ = labeled_parquet
    out = run_training(path, model_type="lstm_ae", backend="noop",
                       output_dir=tmp_path / "m_lstm", cfg=_cfg)
    return out["model_path"]  # a joblib meta bundle (references the .pt by abs path)


@pytest.fixture
def mlflow_registry(labeled_parquet, tmp_path, monkeypatch, _cfg):
    """A sqlite MLflow with TWO registered iforest versions; alias 'staging' on the latest.
    Returns (tracking_uri, first_version_number, latest_version_number)."""
    import mlflow
    from mlflow.tracking import MlflowClient
    path, _ = labeled_parquet
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    run_training(path, model_type="iforest", backend="mlflow",
                 output_dir=tmp_path / "r1", cfg=_cfg, run_name="v_first")
    run_training(path, model_type="iforest", backend="mlflow",
                 output_dir=tmp_path / "r2", cfg=_cfg, run_name="v_latest")
    versions = MlflowClient().search_model_versions("name='iforest_pingpong'")
    nums = sorted(int(mv.version) for mv in versions)
    return uri, str(nums[0]), str(nums[-1])


@pytest.fixture
def sample_rows():
    """One normal-ish feature row as the API would receive it."""
    return [{f: 1.0 for f in FEATURES}]
