import numpy as np
import pandas as pd

from src.monitoring.evidently_drift import evidently_drift_share

COLS = ["n_handover", "pingpong_count", "entropy_cell_seq"]


def test_no_drift_low_share():
    rng = np.random.RandomState(0)
    ref = pd.DataFrame({c: rng.gamma(2.0, 1.0, 400) for c in COLS})
    cur = pd.DataFrame({c: rng.gamma(2.0, 1.0, 400) for c in COLS})
    assert evidently_drift_share(ref, cur) <= 0.5


def test_drift_high_share():
    rng = np.random.RandomState(0)
    ref = pd.DataFrame({c: rng.gamma(2.0, 1.0, 400) for c in COLS})
    cur = pd.DataFrame({c: rng.gamma(2.0, 1.0, 400) * 4.0 for c in COLS})
    assert evidently_drift_share(ref, cur) >= 0.5
