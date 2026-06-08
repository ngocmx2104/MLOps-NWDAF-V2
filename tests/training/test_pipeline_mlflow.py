"""MLflow C1 end-to-end smoke tests.

Exercises run_training → MLflowTracker → sqlite registry (no server needed).
Mirrors the sqlite fixture pattern from tests/tracking/test_mlflow_tracker.py.
"""
import pytest

from src.training.pipeline import run_training
from src.training.schema import TrainingConfig


@pytest.fixture
def mlflow_sqlite(tmp_path, monkeypatch):
    import mlflow

    db = tmp_path / "mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{db}")
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    return tmp_path


def test_pipeline_iforest_mlflow_registers(mlflow_sqlite, labeled_features, tmp_path):
    """C1 end-to-end: pipeline trains iforest and registers a REAL model version."""
    path, _ = labeled_features
    out = run_training(path, model_type="iforest", backend="mlflow",
                       output_dir=tmp_path / "m",
                       cfg=TrainingConfig(use_labels_for_evaluation=True, label_column="label"))
    assert out["backend"] == "mlflow"
    assert out["model_version"] is not None
    assert out["model_version"].startswith("models:/iforest_pingpong/")


def test_pipeline_lstm_mlflow_distinct_name(mlflow_sqlite, labeled_features, tmp_path):
    """Model-swap (RQ4) registers under a DISTINCT name so the staging alias is not clobbered."""
    path, _ = labeled_features
    out = run_training(path, model_type="lstm_ae", backend="mlflow",
                       output_dir=tmp_path / "m",
                       cfg=TrainingConfig(use_labels_for_evaluation=True, label_column="label"))
    assert out["backend"] == "mlflow"
    assert out["model_version"] is not None
    assert out["model_version"].startswith("models:/lstm_ae_pingpong/")
