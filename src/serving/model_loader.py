"""ModelLoader abstraction — the C0/C1 deployment control variable.

PathModelLoader (C0)     -> load a model artifact by local path (ad-hoc deploy).
RegistryModelLoader (C1) -> load from the MLflow registry by alias or version
                            (governed deploy + alias-based rollback).
create_loader(config) selects one, mirroring src.tracking.create_tracker.

Security note: joblib.load below deserializes model artifacts that are produced by
THIS project's own training pipeline (src.training.run_training -> joblib.dump) and
served either from a deployer-controlled local path or our own MLflow registry. They
are trusted internal artifacts (single-node thesis scope), not untrusted input -> the
pickle-execution risk does not apply here. Do NOT point a loader at an untrusted file.

MLflow 3.13 adaptation (documented):
  The `models:/name@alias` and `models:/name/version` URI forms passed to
  `mlflow.artifacts.download_artifacts()` raise MlflowException("No such artifact: ''")
  in MLflow 3.13.0 with a sqlite backend. The workaround: resolve the model version
  via `MlflowClient.get_model_version_by_alias()` or `get_model_version()`, then call
  `download_artifacts(artifact_uri=mv.source)` where `mv.source` is the actual
  file-level artifact URI stored at registration time. This is functionally equivalent
  and tested in test_model_loader.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib

from src.serving.predictor import LoadedModel
from src.serving.schema import ServingConfig
from src.training.schema import FEATURE_COLUMNS

_FEATS = list(FEATURE_COLUMNS)


class ModelLoader(ABC):
    @abstractmethod
    def load(self) -> LoadedModel: ...


def _load_artifact(model_file: Path, model_type: str, version: str) -> LoadedModel:
    if model_type == "iforest":
        return LoadedModel(model_type="iforest", version=version,
                           model_obj=joblib.load(model_file), feature_columns=_FEATS)
    if model_type == "lstm_ae":
        # NOTE single-node: the meta bundle references its .pt by an absolute training
        # path. Self-contained packaging for remote/Docker is deferred to P7/P8.
        return LoadedModel(model_type="lstm_ae", version=version,
                           meta_path=str(model_file), feature_columns=_FEATS)
    raise ValueError(f"Unknown model_type={model_type!r}")


class PathModelLoader(ModelLoader):
    def __init__(self, model_path: str | Path, model_type: str) -> None:
        self.model_path = Path(model_path)
        self.model_type = model_type

    def load(self) -> LoadedModel:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
        return _load_artifact(self.model_path, self.model_type, version=self.model_path.stem)


class RegistryModelLoader(ModelLoader):
    def __init__(self, name: str, model_type: str, *, alias: str | None = None,
                 version: str | None = None, tracking_uri: str | None = None) -> None:
        if not alias and not version:
            raise ValueError("RegistryModelLoader needs an alias or a version")
        self.name = name
        self.model_type = model_type
        self.alias = alias
        self.version_num = version
        self.tracking_uri = tracking_uri

    def load(self) -> LoadedModel:
        import mlflow
        from mlflow.tracking import MlflowClient
        # Pass tracking_uri explicitly (no global mlflow.set_tracking_uri) so repeated
        # loads (Task 5 reload/rollback) never mutate process-global MLflow state.
        client = MlflowClient(tracking_uri=self.tracking_uri)
        # Resolve alias or version -> ModelVersion to obtain mv.source (the actual
        # artifact file URI). Direct models:/name@alias and models:/name/version URIs
        # fail in MLflow 3.13.0 with sqlite backend ("No such artifact: ''").
        if self.version_num:
            version_num = str(self.version_num)
            mv = client.get_model_version(self.name, version_num)
        else:
            mv = client.get_model_version_by_alias(self.name, self.alias)
            version_num = str(mv.version)
        if self.model_type == "lstm_ae":
            # Registry stores only the meta bundle; the .pt is referenced by an absolute
            # training-time path inside it -> predict() FileNotFoundErrors if that path is
            # unavailable (remote/Docker). Single-node works; full packaging is P7/P8.
            import warnings
            warnings.warn(
                "RegistryModelLoader for lstm_ae downloads only the meta bundle; the .pt "
                "weights are referenced by an absolute training-time path and are not in "
                "the registry. predict() fails if that path is unavailable (remote/Docker). "
                "Use PathModelLoader on the training node, or wait for P7/P8 packaging.",
                stacklevel=2,
            )
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=mv.source, tracking_uri=self.tracking_uri)
        model_file = Path(local_path)
        if model_file.is_dir():  # fallback: dir-registered artifact (assumes one .joblib)
            files = sorted(model_file.rglob("*.joblib"))
            if not files:
                raise FileNotFoundError(
                    f"No *.joblib under downloaded artifact dir: {model_file}"
                )
            model_file = files[0]
        return _load_artifact(model_file, self.model_type,
                              version=f"models:/{self.name}/{version_num}")


def create_loader(config: ServingConfig) -> ModelLoader:
    if config.loader == "path":
        if not config.model_path:
            raise ValueError("path loader requires config.model_path")
        return PathModelLoader(config.model_path, config.model_type)
    if config.loader == "registry":
        # Factory always loads via the alias (the operational rollback primitive). Loading
        # a specific version number is done directly via RegistryModelLoader(version=...)
        # — used by ServingRuntime.rollback (Task 5).
        return RegistryModelLoader(config.resolved_registry_name(), config.model_type,
                                   alias=config.registry_alias, tracking_uri=config.tracking_uri)
    raise ValueError(f"Unknown loader={config.loader!r}")
