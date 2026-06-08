from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)


def compute_model_metrics(y_true, y_score, *, threshold: float = 0.5) -> dict[str, Any]:
    """Model-performance group. roc_auc/pr_auc are None when y_true has a single class
    (undefined). y_pred is derived by thresholding y_score."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    out: dict[str, Any] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true.tolist())) > 1 else None,
    }
    if len(set(y_true.tolist())) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out
