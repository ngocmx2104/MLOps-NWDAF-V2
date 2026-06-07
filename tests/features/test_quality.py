import pandas as pd

from src.features.quality import run_d2_quality_checks, run_d5_quality_checks


def _d2_df():
    return pd.DataFrame({
        "imsi": ["1", "2"], "window_start": pd.to_datetime(["2024-06-26", "2024-06-26"], utc=True),
        "n_handover": [3, 1], "n_unique_cells": [2, 1], "pingpong_count": [1, 0],
        "pingpong_rate": [0.33, 0.0], "mean_inter_ho_s": [10.0, float("nan")],
        "std_inter_ho_s": [2.0, float("nan")], "entropy_cell_seq": [0.9, 0.0],
        "feature_version": ["ho_features_v1"] * 2, "source_snapshot_id": ["D1_X"] * 2,
    })


def test_d2_checks_pass_on_valid():
    results = run_d2_quality_checks(_d2_df())
    assert len(results) == 5
    assert all(r.passed for r in results)


def test_d5_checks_detect_binary_labels():
    df = _d2_df()
    df["weak_label"] = [1, 0]
    df["weak_label_version"] = "weak_label_v1"
    df["source_feature_version"] = "ho_features_v1"
    results = run_d5_quality_checks(df)
    assert len(results) == 4
    assert all(r.passed for r in results)
