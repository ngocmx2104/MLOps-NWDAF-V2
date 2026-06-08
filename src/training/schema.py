from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.features.schema import D2_FEATURE_COLUMNS

TRAINING_CONFIG_VERSION = "training_config_v1"
TRAINING_PROCESS_VERSION = "phase3_training_v1"
REGISTRY_SCHEMA_VERSION = "registry_record_v1"
MODEL_FAMILY = "iforest_pingpong"
MODEL_CLASS = "sklearn.ensemble.IsolationForest"
CANDIDATE_STATE = "candidate"

FEATURE_COLUMNS: tuple[str, ...] = tuple(col.name for col in D2_FEATURE_COLUMNS)


@dataclass(frozen=True)
class TrainingConfig:
    contamination: float = 0.01
    random_state: int = 42
    n_estimators: int = 200
    test_size: float = 0.2
    label_column: str = "weak_label"
    use_labels_for_evaluation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingContext:
    dataset_path: str
    dataset_id: str
    dataset_layer: str
    feature_version: str
    source_snapshot_id: str
    row_count: int
    label_column: str | None = None
    label_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
