from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from src.training.schema import FEATURE_COLUMNS, MODEL_CLASS, MODEL_FAMILY, TrainingConfig


@dataclass
class TrainingResult:
    model: Any
    metrics: dict[str, float]
    fit_summary: dict[str, Any]
    validation_summary: dict[str, Any]


def _describe_scores(scores: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


def train_isolation_forest(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray | None,
    cfg: TrainingConfig,
) -> TrainingResult:
    model = IsolationForest(
        n_estimators=cfg.n_estimators,
        contamination=cfg.contamination,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    model.fit(x_train)

    train_scores = -model.score_samples(x_train)
    val_scores = -model.score_samples(x_val)
    val_pred = (model.predict(x_val) == -1).astype(int)

    metrics: dict[str, float] = {}
    if y_val is not None and len(set(y_val.tolist())) > 1:
        metrics = {
            "precision": float(precision_score(y_val, val_pred, zero_division=0)),
            "recall": float(recall_score(y_val, val_pred, zero_division=0)),
            "f1": float(f1_score(y_val, val_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_val, val_scores)),
            "pr_auc": float(average_precision_score(y_val, val_scores)),
        }

    fit_summary = {
        "model_family": MODEL_FAMILY,
        "model_class": MODEL_CLASS,
        "feature_columns": list(FEATURE_COLUMNS),
        "n_train_rows": int(len(x_train)),
        "n_validation_rows": int(len(x_val)),
    }
    validation_summary = {
        "train_score_summary": _describe_scores(train_scores),
        "validation_score_summary": _describe_scores(val_scores),
        "validation_has_labels": y_val is not None,
        "predicted_anomaly_rate": float(np.mean(val_pred)),
    }
    return TrainingResult(model=model, metrics=metrics, fit_summary=fit_summary, validation_summary=validation_summary)
