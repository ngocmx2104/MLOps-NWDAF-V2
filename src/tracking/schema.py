from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentConfig:
    experiment_name: str
    run_name: str
    # Metadata label for which config this run belongs to (e.g. "C0"/"noop", "C1"/"mlflow").
    # Does NOT control dispatch — create_tracker() chooses the backend.
    backend: str = "noop"
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunHandle:
    run_id: str | None
    backend: str
    url: str | None = None
