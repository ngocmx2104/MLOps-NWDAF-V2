from abc import ABC, abstractmethod

from src.tracking.schema import ExperimentConfig, RunHandle


class BaseTracker(ABC):
    """Backend-agnostic experiment/registry interface. Core pipeline depends only on this."""

    @abstractmethod
    def init_experiment(self, config: ExperimentConfig) -> RunHandle: ...

    @abstractmethod
    def log_params(self, params: dict) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: dict, step: int | None = None) -> None: ...

    @abstractmethod
    def log_dataset(self, path: str, name: str | None = None) -> None: ...

    @abstractmethod
    def register_model(self, model_path: str, name: str,
                       metrics: dict | None = None) -> str | None: ...

    @abstractmethod
    def end_experiment(self, status: str = "FINISHED") -> None: ...
