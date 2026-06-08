from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

EXPERIMENT_RECORD_VERSION = "experiment_record_v1"
RESULT_SUMMARY_VERSION = "result_summary_v1"
DEFAULT_EXPERIMENT_ROOT = "artifacts/experiments"


@dataclass(frozen=True)
class RunConfig:
    """One workload run repeated n_runs times with fixed seeds; warmup_drop leading runs
    are excluded from statistics (THESIS_SPEC §4.4 protocol)."""
    workload: list[str]                 # subprocess argv; "{seed}" tokens get substituted
    n_runs: int = 10
    warmup_drop: int = 1
    seeds: list[int] = field(default_factory=lambda: list(range(42, 52)))
    env: dict[str, str] = field(default_factory=dict)

    def scored_seeds(self) -> list[int]:
        return self.seeds[self.warmup_drop:self.n_runs]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    experiment_root: str = DEFAULT_EXPERIMENT_ROOT
    configs: dict[str, RunConfig] = field(default_factory=dict)  # e.g. {"C0": rc0, "C1": rc1}

    def to_dict(self) -> dict[str, Any]:
        return {"experiment_id": self.experiment_id, "experiment_root": self.experiment_root,
                "configs": {k: v.to_dict() for k, v in self.configs.items()}}
