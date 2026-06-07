import pandas as pd

from src.data.synthetic_generator import generate_and_save, generate_synthetic_data

FEATURES = ["n_handover", "n_unique_cells", "pingpong_count", "pingpong_rate",
            "mean_inter_ho_s", "std_inter_ho_s", "entropy_cell_seq"]


def test_has_7_features_and_labels():
    df = generate_synthetic_data(n_samples=200, random_state=42, anomaly_rate=0.05)
    assert all(f in df.columns for f in FEATURES)
    assert len(df) == 200
    assert df["label"].sum() == int(200 * 0.05)
    assert {"imsi", "window_start", "label"} <= set(df.columns)


def test_deterministic_by_seed():
    a = generate_synthetic_data(n_samples=100, random_state=7, anomaly_rate=0.05)
    b = generate_synthetic_data(n_samples=100, random_state=7, anomaly_rate=0.05)
    pd.testing.assert_frame_equal(a, b)


def test_generate_and_save_with_drift(tmp_path):
    out = tmp_path / "f.parquet"
    meta = generate_and_save(out, n_samples=300, random_state=42,
                             anomaly_rate=0.05, drift_type="sudden", drift_start=150)
    assert out.exists()
    assert meta["n_samples"] == 300
    assert meta["drift_type"] == "sudden"
    assert meta["n_drifted_samples"] > 0
    df = pd.read_parquet(out)
    assert df["has_drift"].sum() > 0


def test_no_drift_default(tmp_path):
    meta = generate_and_save(tmp_path / "g.parquet", n_samples=100, random_state=1)
    assert meta["drift_type"] == "none"
    assert meta["n_drifted_samples"] == 0
