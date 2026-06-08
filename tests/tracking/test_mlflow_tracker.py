"""Tests for MLflowTracker — uses sqlite backend so the model registry works without a server."""
import pytest

from src.tracking import create_tracker
from src.tracking.mlflow_tracker import MLflowTracker
from src.tracking.schema import ExperimentConfig


@pytest.fixture
def mlflow_local(tmp_path, monkeypatch):
    import mlflow

    db = tmp_path / "mlflow.db"
    tracking_uri = f"sqlite:///{db}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    return tmp_path


def test_factory_returns_mlflow(mlflow_local):
    assert isinstance(create_tracker("mlflow"), MLflowTracker)


def test_full_lifecycle_with_registry(mlflow_local, tmp_path):
    t = MLflowTracker()
    h = t.init_experiment(
        ExperimentConfig(
            experiment_name="t_exp",
            run_name="r1",
            backend="mlflow",
            tags={"model": "iforest"},
        )
    )
    assert h.run_id and h.backend == "mlflow"

    t.log_params({"contamination": 0.01, "n_estimators": 200})
    t.log_metrics({"roc_auc": 0.98, "pr_auc": 0.55})

    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"dummy-model-bytes")

    version_uri = t.register_model(
        str(model_file),
        "iforest_pingpong",
        metrics={"roc_auc": 0.98},
        alias="staging",
    )
    assert version_uri and version_uri.startswith("models:/iforest_pingpong/")

    t.end_experiment()

    # Verify the alias resolves to a real registered model version.
    # MLflow 3.x adaptation: register_model uses log_artifact + get_artifact_uri URI
    # (not runs:/<id>/model) because MLflow 3.x requires a proper MLflow-logged model for
    # the runs:/ scheme. The resulting registered model version + alias are real and queryable.
    from mlflow.tracking import MlflowClient

    mv = MlflowClient().get_model_version_by_alias("iforest_pingpong", "staging")
    assert mv.version is not None
