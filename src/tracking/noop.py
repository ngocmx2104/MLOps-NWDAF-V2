from typing import Any

from src.tracking.base import BaseTracker
from src.tracking.schema import ExperimentConfig, RunHandle


class NoopTracker(BaseTracker):
    """C0 baseline: ablated MLOps layer — accepts all calls, persists nothing."""

    def init_experiment(self, config: ExperimentConfig) -> RunHandle:
        return RunHandle(run_id=None, backend="noop")

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def log_dataset(self, path: str, name: str | None = None) -> None:
        pass

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        pass

    def register_model(self, model_path: str, name: str,
                       metrics: dict[str, float] | None = None,
                       alias: str | None = None) -> str | None:
        return None

    def end_experiment(self, status: str = "FINISHED") -> None:
        pass
