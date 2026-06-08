# tests/experiments/test_model_perf.py
import numpy as np

from src.experiments.metrics.model_perf import compute_model_metrics


def test_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_model_metrics(y_true, y_score, threshold=0.5)
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_single_class_guard():
    m = compute_model_metrics(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]), threshold=0.5)
    assert m["roc_auc"] is None and m["pr_auc"] is None  # undefined with one class
