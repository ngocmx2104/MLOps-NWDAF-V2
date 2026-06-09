import numpy as np
import pandas as pd

from src.training.lstm_detector import train_lstm_ae
from src.training.schema import FEATURE_COLUMNS


def _tiny(path, n=60, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(rng.rand(n, len(FEATURE_COLUMNS)), columns=list(FEATURE_COLUMNS))
    df["label"] = (rng.rand(n) < 0.35).astype(int)
    df.to_parquet(path)
    return path


def test_lstm_respects_test_size(tmp_path):
    ds = _tiny(tmp_path / "d.parquet", n=100)
    out = train_lstm_ae(ds, tmp_path / "m", random_state=42, test_size=0.25, epochs=2)
    # 25% of 100 -> 25 val rows (sklearn rounding); train = 75
    assert out["val_rows"] == 25 and out["train_rows"] == 75
