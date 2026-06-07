import pandas as pd

from src.features.builder import compute_ue_window_features
from src.features.schema import WindowConfig

FEATURES = ["n_handover", "n_unique_cells", "pingpong_count", "pingpong_rate",
            "mean_inter_ho_s", "std_inter_ho_s", "entropy_cell_seq"]


def test_computes_7_features(d1_like_df):
    out = compute_ue_window_features(d1_like_df, WindowConfig())
    assert all(f in out.columns for f in FEATURES)
    assert {"imsi", "window_start"} <= set(out.columns)


def test_pingpong_detected(d1_like_df):
    out = compute_ue_window_features(d1_like_df, WindowConfig())
    imsi_111 = out[out["imsi"] == "111"].iloc[0]
    assert imsi_111["n_handover"] == 3
    assert imsi_111["n_unique_cells"] == 2
    assert imsi_111["pingpong_count"] == 1


def test_single_handover_has_nan_timing(d1_like_df):
    out = compute_ue_window_features(d1_like_df, WindowConfig())
    imsi_222 = out[out["imsi"] == "222"].iloc[0]
    assert imsi_222["n_handover"] == 1
    assert imsi_222["pingpong_count"] == 0
    assert pd.isna(imsi_222["mean_inter_ho_s"])
