import os

from src.tracking.base import BaseTracker
from src.tracking.noop import NoopTracker


def create_tracker(backend: str | None = None) -> BaseTracker:
    """Return a tracker for the given backend (or env MLOPS_BACKEND, default 'noop')."""
    backend = (backend or os.environ.get("MLOPS_BACKEND", "noop")).lower()
    if backend == "noop":
        return NoopTracker()
    if backend == "mlflow":
        from src.tracking.mlflow_tracker import MLflowTracker  # implemented in P4
        return MLflowTracker()
    if backend == "clearml":
        from src.tracking.clearml_tracker import ClearMLTracker  # implemented in P4
        return ClearMLTracker()
    raise ValueError(f"Unknown MLOPS_BACKEND={backend!r} (expected noop|mlflow|clearml)")
