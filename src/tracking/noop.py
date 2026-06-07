from src.tracking.base import BaseTracker
from src.tracking.schema import ExperimentConfig, RunHandle


class NoopTracker(BaseTracker):
    """C0 baseline: ablated MLOps layer — accepts all calls, persists nothing."""

    def init_experiment(self, config: ExperimentConfig) -> RunHandle:
        self._config = config
        return RunHandle(run_id=None, backend="noop")

    def log_params(self, params: dict) -> None:
        pass

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        pass

    def log_dataset(self, path: str, name: str | None = None) -> None:
        pass

    def register_model(self, model_path: str, name: str,
                       metrics: dict | None = None) -> str | None:
        return None

    def end_experiment(self, status: str = "FINISHED") -> None:
        pass
