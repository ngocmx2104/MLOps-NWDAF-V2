from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    experiment_name: str
    run_name: str
    backend: str = "noop"
    tags: dict = field(default_factory=dict)


@dataclass
class RunHandle:
    run_id: str | None
    backend: str
    url: str | None = None
