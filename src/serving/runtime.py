"""ServingRuntime — holds the active model + feature provider; predict/reload/rollback.

Backend (path=C0 / registry=C1) is the controlled deployment variable for RQ3;
model_type (iforest/lstm_ae) swaps the served model for RQ4. Latency is measured
with perf_counter (operational metric).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from src.serving.feature_provider import FeastOnlineProvider
from src.serving.model_loader import PathModelLoader, RegistryModelLoader, create_loader
from src.serving.predictor import LoadedModel
from src.serving.records import (
    append_jsonl, build_deployment_record, build_prediction_record, build_rollback_record,
)
from src.serving.schema import ModelInfo, PredictRequest, PredictResponse, ServingConfig


@dataclass
class ServingRuntime:
    config: ServingConfig
    model: LoadedModel
    provider: FeastOnlineProvider | None
    output_root: Path

    @classmethod
    def build(cls, config: ServingConfig) -> "ServingRuntime":
        model = create_loader(config).load()
        provider = FeastOnlineProvider(config.feast_repo_path) if config.feast_repo_path else None
        rt = cls(config=config, model=model, provider=provider,
                 output_root=Path(config.output_root))
        append_jsonl(rt.output_root / "deployment_history.jsonl",
                     build_deployment_record(model_type=model.model_type,
                                             model_version=model.version, loader=config.loader))
        return rt

    def _rows_for(self, request: PredictRequest) -> list[dict[str, float]]:
        if request.imsi is not None:
            if self.provider is None:
                raise ValueError("imsi requests require config.feast_repo_path")
            return self.provider.get([request.imsi])
        return [request.features]

    def predict(self, request: PredictRequest) -> PredictResponse:
        t0 = perf_counter()
        rows = self._rows_for(request)
        out = self.model.predict(rows)[0]
        latency_ms = (perf_counter() - t0) * 1000.0
        resp = PredictResponse(
            request_id=request.request_id or f"pred-{uuid.uuid4().hex[:12]}",
            is_anomaly=out["is_anomaly"], anomaly_score=out["anomaly_score"],
            model_type=self.model.model_type, model_version=self.model.version,
            latency_ms=latency_ms, feature_values=rows[0])
        append_jsonl(self.output_root / "predictions.jsonl",
                     build_prediction_record(resp.model_dump()))
        return resp

    def reload(self) -> dict[str, Any]:
        self.model = create_loader(self.config).load()
        return self.status()

    def rollback(self, target_version: str, reason: str = "manual rollback") -> dict[str, Any]:
        """Roll back the active model to a prior artifact.

        ``target_version`` is interpreted by the configured loader:
        - loader='registry': an MLflow version NUMBER string (e.g. "3").
        - loader='path':     a file-system PATH to the model artifact (e.g. "m_v2.joblib").

        If the target fails to load, the current model is kept and no record is written
        (``self.model`` is reassigned only after ``loader.load()`` returns).
        """
        previous = self.model.version
        if self.config.loader == "registry":
            loader: RegistryModelLoader | PathModelLoader = RegistryModelLoader(
                self.config.resolved_registry_name(), self.config.model_type,
                version=target_version, tracking_uri=self.config.tracking_uri)
        else:
            loader = PathModelLoader(target_version, self.config.model_type)
        self.model = loader.load()
        record = build_rollback_record(from_version=previous, to_version=self.model.version,
                                       reason=reason)
        append_jsonl(self.output_root / "rollback_history.jsonl", record)
        return record

    def status(self) -> dict[str, Any]:
        return ModelInfo(model_type=self.model.model_type, model_version=self.model.version,
                         loader=self.config.loader,
                         feature_columns=self.model.feature_columns).model_dump()
