"""MLflow tracker (Config C1) — real experiment tracking + model registry.

MLflow 3.x API adaptation note:
- ``mlflow.register_model`` in MLflow 3.x requires a URI pointing to a proper MLflow-logged
  model (one with an MLmodel manifest). A generic ``log_artifact`` path followed by a
  ``runs:/<id>/model`` URI is rejected ("Unable to find a logged_model with artifact_path").
- Workaround (minimal): call ``log_artifact`` without an artifact subdirectory to land the
  file at the run's artifact root, then register via ``mlflow.get_artifact_uri() + '/' + name``.
  This gives a real registered model version + alias while keeping the tracker backend-agnostic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.tracking.base import BaseTracker
from src.tracking.schema import ExperimentConfig, RunHandle


class MLflowTracker(BaseTracker):
    def __init__(self) -> None:
        self._run = None
        self._client: MlflowClient | None = None
        self._artifact_uri: str | None = None

    def init_experiment(self, config: ExperimentConfig) -> RunHandle:
        mlflow.set_experiment(config.experiment_name)
        self._run = mlflow.start_run(run_name=config.run_name, tags=config.tags or {})
        self._client = MlflowClient()
        self._artifact_uri = mlflow.get_artifact_uri()
        info = self._run.info
        uri = mlflow.get_tracking_uri()
        url = f"{uri}/#/experiments/{info.experiment_id}/runs/{info.run_id}"
        return RunHandle(run_id=info.run_id, backend="mlflow", url=url)

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_dataset(self, path: str, name: str | None = None) -> None:
        mlflow.log_artifact(str(path), artifact_path="datasets")

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def register_model(
        self,
        model_path: str,
        name: str,
        metrics: dict[str, float] | None = None,
        alias: str | None = None,
    ) -> str | None:
        # Log artifact at the run's artifact root (no sub-path) so the URI resolves correctly.
        # MLflow 3.x: register_model requires the URI to point to an existing artifact file or
        # a proper MLflow-logged model directory — using get_artifact_uri() + filename is the
        # minimal viable approach for generic model files (joblib, .pt, etc.).
        filename = Path(model_path).name
        mlflow.log_artifact(str(model_path))
        model_uri = f"{self._artifact_uri}/{filename}"
        mv = mlflow.register_model(model_uri=model_uri, name=name)
        if alias and self._client is not None:
            self._client.set_registered_model_alias(name, alias, mv.version)
        return f"models:/{name}/{mv.version}"

    def end_experiment(self, status: str = "FINISHED") -> None:
        mlflow.end_run(status=status)
