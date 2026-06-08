import pytest

from src.serving.model_loader import (
    PathModelLoader, RegistryModelLoader, create_loader,
)
from src.serving.schema import REGISTRY_NAMES, ServingConfig


def test_path_loader_iforest(iforest_model_path):
    lm = PathModelLoader(iforest_model_path, "iforest").load()
    assert lm.model_type == "iforest" and lm.model_obj is not None


def test_path_loader_lstm(lstm_model_path):
    lm = PathModelLoader(lstm_model_path, "lstm_ae").load()
    assert lm.model_type == "lstm_ae" and lm.meta_path is not None


def test_path_loader_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PathModelLoader(tmp_path / "nope.joblib", "iforest").load()


def test_registry_loader_iforest(mlflow_registry):
    uri, _first, latest = mlflow_registry
    lm = RegistryModelLoader("iforest_pingpong", "iforest", alias="staging",
                             tracking_uri=uri).load()
    assert lm.model_type == "iforest" and lm.model_obj is not None
    assert lm.version.endswith(f"/{latest}")  # alias 'staging' -> latest version


def test_registry_loader_by_version(mlflow_registry):
    uri, first, _latest = mlflow_registry
    lm = RegistryModelLoader("iforest_pingpong", "iforest", version=first,
                             tracking_uri=uri).load()
    assert lm.version.endswith(f"/{first}")


def test_factory_path(iforest_model_path):
    cfg = ServingConfig(loader="path", model_type="iforest", model_path=str(iforest_model_path))
    assert create_loader(cfg).load().model_type == "iforest"


def test_factory_registry(mlflow_registry):
    uri, _f, _l = mlflow_registry
    cfg = ServingConfig(loader="registry", model_type="iforest", tracking_uri=uri)
    assert create_loader(cfg).load().model_obj is not None


def test_registry_names_match_training():
    """Guard: serving's REGISTRY_NAMES must stay in sync with training's private map."""
    from src.training.pipeline import _REGISTRY_NAMES
    assert REGISTRY_NAMES == _REGISTRY_NAMES


def test_registry_loader_lstm_warns(labeled_parquet, tmp_path, monkeypatch):
    """lstm_ae via registry must WARN (only the meta is registered; .pt is by abs path).
    On this single node the .pt still exists at its training path, so load() succeeds."""
    import mlflow

    from src.cicd.eval_gate import run_eval_gate
    from src.cicd.schema import GateConfig
    from src.tracking import create_tracker
    from src.training.pipeline import run_training
    from src.training.schema import TrainingConfig
    path, _ = labeled_parquet
    uri = f"sqlite:///{tmp_path / 'lstm_reg.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    out = run_training(path, model_type="lstm_ae", backend="mlflow", output_dir=tmp_path / "l",
                       cfg=TrainingConfig(use_labels_for_evaluation=True, label_column="label"))
    # promote candidate -> staging so the registry loader can find the alias
    # (min_roc_auc=0.0 -> always promote; this is test setup, not a real gate)
    run_eval_gate(metrics=out["metrics"], model_name=out["model_name"],
                  model_version=out["model_version"], tracker=create_tracker("mlflow"),
                  cfg=GateConfig(min_roc_auc=0.0))
    with pytest.warns(UserWarning, match="lstm_ae"):
        lm = RegistryModelLoader("lstm_ae_pingpong", "lstm_ae", alias="staging",
                                 tracking_uri=uri).load()
    assert lm.model_type == "lstm_ae" and lm.meta_path is not None
