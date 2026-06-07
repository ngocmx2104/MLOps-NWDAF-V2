import pandas as pd

from src.features.schema import WeakLabelConfig
from src.features.weak_labels import apply_weak_labels


def test_weak_label_rule():
    df = pd.DataFrame({
        "pingpong_count": [2, 0, 1, 5],
        "n_handover": [5, 1, 2, 10],
    })
    out = apply_weak_labels(df, WeakLabelConfig())  # min_pp=1, min_ho=3
    # row0: pp2>=1 & ho5>=3 -> 1; row1: 0 -> 0; row2: ho2<3 -> 0; row3: 1
    assert out["weak_label"].tolist() == [1, 0, 0, 1]


def test_weak_label_is_binary():
    df = pd.DataFrame({"pingpong_count": [0, 3], "n_handover": [0, 4]})
    out = apply_weak_labels(df, WeakLabelConfig())
    assert set(out["weak_label"].unique()) <= {0, 1}
