import numpy as np
import pandas as pd

from src.experiments.metrics.drift_quality import data_quality, drift_metrics


def test_drift_metrics_detects_shift():
    rng = np.random.RandomState(0)
    ref = pd.DataFrame({"f1": rng.normal(0, 1, 500), "f2": rng.normal(0, 1, 500)})
    obs = pd.DataFrame({"f1": rng.normal(5, 1, 500), "f2": rng.normal(5, 1, 500)})  # big shift
    m = drift_metrics(ref, obs, min_features_alert=1)
    assert m["drift_detected"] is True
    assert m["per_feature"]["f1"]["psi"] > 0.25


def test_data_quality():
    df = pd.DataFrame({"a": [1, 1, None], "b": [1, 1, 1]})
    q = data_quality(df)
    assert q["null_rate"] > 0 and q["duplicate_rate"] >= 0 and q["n_rows"] == 3
