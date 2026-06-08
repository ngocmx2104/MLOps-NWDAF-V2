import numpy as np
import pandas as pd
import pytest

from src.features.schema import D2_FEATURE_COLUMNS

FEATURES = [c.name for c in D2_FEATURE_COLUMNS]


@pytest.fixture
def labeled_features(tmp_path):
    """A small calibrated-ish feature parquet with a `label` column (5% anomalies)."""
    rng = np.random.RandomState(0)
    n = 300
    data = {f: rng.gamma(2.0, 1.0, n) for f in FEATURES}
    df = pd.DataFrame(data)
    df["imsi"] = [str(i) for i in range(n)]
    df["window_start"] = pd.date_range("2024-06-26", periods=n, freq="5min", tz="UTC")
    df["label"] = 0
    anom = rng.choice(n, size=15, replace=False)
    df.loc[anom, FEATURES] = df.loc[anom, FEATURES] * 5.0  # make anomalies extreme
    df.loc[anom, "label"] = 1
    df["weak_label"] = df["label"]
    path = tmp_path / "features.parquet"
    df.to_parquet(path, index=False)
    return path, df
