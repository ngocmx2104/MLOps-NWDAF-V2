"""Serving subsystem (component C7) — request/response + runtime config schemas."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel, model_validator

from src.training.schema import FEATURE_COLUMNS, MODEL_FAMILY

# Registry name per model family. iforest reuses the canonical MODEL_FAMILY; the
# lstm name mirrors src.training.pipeline._REGISTRY_NAMES (a drift-guard test in
# test_model_loader.py asserts they stay in sync).
REGISTRY_NAMES: dict[str, str] = {"iforest": MODEL_FAMILY, "lstm_ae": "lstm_ae_pingpong"}

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_ALIAS = "staging"
DEFAULT_OUTPUT_ROOT = "artifacts/serving"


class PredictRequest(BaseModel):
    """A prediction request: supply EXACTLY one of imsi (Feast online lookup) or features."""
    imsi: str | None = None
    features: dict[str, float] | None = None
    request_id: str | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "PredictRequest":
        if (self.imsi is None) == (self.features is None):
            raise ValueError("provide exactly one of: imsi, features")
        return self


class PredictResponse(BaseModel):
    request_id: str
    is_anomaly: bool
    anomaly_score: float
    model_type: str
    model_version: str
    latency_ms: float
    feature_values: dict[str, float]


class ModelInfo(BaseModel):
    model_type: str
    model_version: str
    loader: str
    feature_columns: list[str]


@dataclass(frozen=True)
class ServingConfig:
    model_type: str = "iforest"          # iforest | lstm_ae
    loader: str = "path"                 # path (C0) | registry (C1)
    model_path: str | None = None        # for path loader
    registry_name: str | None = None     # for registry loader (default: REGISTRY_NAMES[model_type])
    registry_alias: str = DEFAULT_ALIAS
    tracking_uri: str | None = None      # mlflow sqlite/server uri
    feast_repo_path: str | None = None   # required for imsi (online) requests
    output_root: str = DEFAULT_OUTPUT_ROOT
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT

    def resolved_registry_name(self) -> str:
        if self.registry_name:
            return self.registry_name
        if self.model_type not in REGISTRY_NAMES:
            raise ValueError(f"unknown model_type {self.model_type!r}; valid: {list(REGISTRY_NAMES)}")
        return REGISTRY_NAMES[self.model_type]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_columns"] = list(FEATURE_COLUMNS)
        return payload
