import numpy as np

from src.training.core import train_isolation_forest
from src.training.schema import TrainingConfig


def test_iforest_trains_and_scores():
    rng = np.random.RandomState(0)
    x_train = rng.normal(size=(200, 7))
    x_val = rng.normal(size=(50, 7))
    y_val = np.zeros(50, dtype=int)
    y_val[:5] = 1
    res = train_isolation_forest(x_train, x_val, y_val, TrainingConfig())
    assert res.model is not None
    assert "roc_auc" in res.metrics and 0.0 <= res.metrics["roc_auc"] <= 1.0
    assert res.fit_summary["n_train_rows"] == 200
