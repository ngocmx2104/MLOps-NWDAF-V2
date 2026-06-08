"""promote_model: separate registration from alias promotion (eval-gate governance)."""
from src.tracking.noop import NoopTracker


def test_noop_promote_returns_none():
    assert NoopTracker().promote_model("m", "staging", "1") is None


def test_mlflow_promote_moves_alias(tmp_path, monkeypatch):
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)

    from src.tracking.mlflow_tracker import MLflowTracker
    from src.tracking.schema import ExperimentConfig

    tr = MLflowTracker()
    tr.init_experiment(ExperimentConfig(experiment_name="t", run_name="r", backend="mlflow"))
    model = tmp_path / "m.joblib"
    model.write_bytes(b"x")
    uri_v = tr.register_model(str(model), "demo_model", alias="candidate")
    tr.end_experiment()
    version = uri_v.rsplit("/", 1)[-1]

    out = tr.promote_model("demo_model", "staging", version)
    assert out == f"models:/demo_model/{version}"
    mv = MlflowClient().get_model_version_by_alias("demo_model", "staging")
    assert str(mv.version) == version
