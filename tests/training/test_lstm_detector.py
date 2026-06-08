import numpy as np

from src.training.lstm_detector import FEATURE_COLS, predict_lstm_ae, train_lstm_ae
from src.training.schema import FEATURE_COLUMNS


def test_feature_cols_match_schema():
    """LSTM feature order MUST equal the canonical schema order.

    Both detectors must consume the exact same features in the same order, or the
    IForest vs LSTM-AE comparison (RQ3/RQ4) is invalid. This guards against silent
    drift between lstm_detector.FEATURE_COLS and src.training.schema.FEATURE_COLUMNS.
    """
    assert list(FEATURE_COLS) == list(FEATURE_COLUMNS)


def test_lstm_ae_trains_and_predicts(labeled_features, tmp_path):
    path, df = labeled_features
    out = train_lstm_ae(path, tmp_path / "lstm", epochs=5, label_column="label")
    assert "metrics" in out and "threshold" in out["metrics"]
    assert (tmp_path / "lstm" / "model_lstm_ae.pt").exists()
    # inference round-trips
    x = df[FEATURE_COLS].to_numpy(float)
    preds, scores = predict_lstm_ae(x, out["model_path"])
    assert len(preds) == len(df) and len(scores) == len(df)
    assert set(np.unique(preds)).issubset({0, 1})


def test_lstm_ae_deterministic(labeled_features, tmp_path):
    path, _ = labeled_features
    a = train_lstm_ae(path, tmp_path / "a", epochs=5, random_state=1, label_column="label")
    b = train_lstm_ae(path, tmp_path / "b", epochs=5, random_state=1, label_column="label")
    assert abs(a["metrics"]["threshold"] - b["metrics"]["threshold"]) < 1e-6
    # also compare mean_val_mse: catches weight drift, not just the percentile scalar
    assert abs(a["metrics"]["mean_val_mse"] - b["metrics"]["mean_val_mse"]) < 1e-5
