"""LoadedModel: a model + its inference dispatch (tracker/loader-agnostic).

iforest -> sklearn IsolationForest (score_samples/predict).
lstm_ae -> reuse src.training.lstm_detector.predict_lstm_ae (train/serve consistency).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.training.schema import FEATURE_COLUMNS


@dataclass
class LoadedModel:
    model_type: str
    version: str
    # sklearn model object (iforest path). Trusted internal artifact — loaded by
    # the caller (path/registry loader) via joblib from pipeline-produced .joblib files.
    model_obj: Any = None
    meta_path: str | None = None     # joblib meta bundle path (lstm_ae)
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    def _matrix(self, rows: list[dict[str, float]]) -> np.ndarray:
        return np.asarray(
            [[float(r[c]) for c in self.feature_columns] for r in rows], dtype=float
        )

    def predict(self, rows: list[dict[str, float]]) -> list[dict[str, Any]]:
        x = self._matrix(rows)  # raises KeyError if a feature column is missing
        if self.model_type == "iforest":
            scores = -self.model_obj.score_samples(x)
            preds = (self.model_obj.predict(x) == -1).astype(int)
        elif self.model_type == "lstm_ae":
            from src.training.lstm_detector import predict_lstm_ae
            preds, scores = predict_lstm_ae(x, Path(self.meta_path))
        else:
            raise ValueError(f"Unknown model_type={self.model_type!r}")
        return [{"is_anomaly": bool(p), "anomaly_score": float(s)}
                for p, s in zip(preds, scores)]
