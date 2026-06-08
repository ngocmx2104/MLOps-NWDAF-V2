# tests/experiments/test_business.py
import numpy as np

from src.experiments.metrics.business import business_metrics


def test_expected_cost():
    # y_true vs y_pred: 1 FP (pred1,true0), 1 FN (pred0,true1)
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    m = business_metrics(y_true, y_pred, c_fp=2.0, c_fn=10.0, window_hours=4.0)
    assert m["fp"] == 1 and m["fn"] == 1
    assert m["expected_cost"] == 1 * 2.0 + 1 * 10.0    # 12.0
    assert m["detections_per_hour"] == 2 / 4.0         # 2 positive predictions / 4h
