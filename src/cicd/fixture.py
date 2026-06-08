"""Self-contained synthetic labeled dataset for CI smoke (no real data / profile.json needed).
NOT the calibrated synthetic generator (src/data, P2) used for P8 experiments — this only has
to exercise feature->train->eval-gate->deploy and let the gate PASS deterministically."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.schema import D2_FEATURE_COLUMNS

_FEATURES = [c.name for c in D2_FEATURE_COLUMNS]


def make_fixture_dataset(path: Path, *, n: int = 300, seed: int = 0) -> Path:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({f: rng.gamma(2.0, 1.0, n) for f in _FEATURES})
    df["imsi"] = [str(i) for i in range(n)]
    df["window_start"] = pd.date_range("2024-06-26", periods=n, freq="5min", tz="UTC")
    df["label"] = 0
    anom = rng.choice(n, size=max(1, n // 20), replace=False)  # ~5% anomalies
    df.loc[anom, _FEATURES] = df.loc[anom, _FEATURES] * 5.0     # make them extreme/separable
    df.loc[anom, "label"] = 1
    df["weak_label"] = df["label"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
