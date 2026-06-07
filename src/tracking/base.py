from abc import ABC, abstractmethod
from typing import Any

from src.tracking.schema import ExperimentConfig, RunHandle


class BaseTracker(ABC):
    """Backend-agnostic experiment/registry interface. Core pipeline depends only on this."""

    @abstractmethod
    def init_experiment(self, config: ExperimentConfig) -> RunHandle: ...

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...

    @abstractmethod
    def log_dataset(self, path: str, name: str | None = None) -> None: ...

    @abstractmethod
    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        """Upload an arbitrary file (plot, drift report, serialized snapshot) to the artifact store."""
        ...

    @abstractmethod
    def register_model(self, model_path: str, name: str,
                       metrics: dict[str, float] | None = None,
                       alias: str | None = None) -> str | None:
        """Register a model version. Real backends return a version URI/ID string;
        NoopTracker returns None (nothing registered). `alias` (e.g. 'champion')
        promotes the version if the backend supports it."""
        ...

    @abstractmethod
    def end_experiment(self, status: str = "FINISHED") -> None:
        """End the run. Callers needing the run URL should read it from the
        RunHandle returned by init_experiment."""
        ...
