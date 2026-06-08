from src.training.pipeline import run_training
from src.training.schema import TrainingConfig


def test_pipeline_iforest_noop(labeled_features, tmp_path):
    path, _ = labeled_features
    out = run_training(path, model_type="iforest", backend="noop",
                       output_dir=tmp_path / "m",
                       cfg=TrainingConfig(use_labels_for_evaluation=True, label_column="label"))
    assert out["backend"] == "noop"
    assert "roc_auc" in out["metrics"], "expected labeled eval (fixture must have 2-class labels)"
    assert out["metrics"]["roc_auc"] >= 0.0
    assert out["model_version"] is None  # noop registers nothing


def test_model_swap_iforest_to_lstm(labeled_features, tmp_path):
    """Same pipeline, swap model_type — proves RQ4 flexibility (no pipeline change)."""
    path, _ = labeled_features
    cfg = TrainingConfig(use_labels_for_evaluation=True, label_column="label")
    a = run_training(path, model_type="iforest", backend="noop", output_dir=tmp_path / "a", cfg=cfg)
    b = run_training(path, model_type="lstm_ae", backend="noop", output_dir=tmp_path / "b", cfg=cfg)
    assert a["model_type"] == "iforest" and b["model_type"] == "lstm_ae"
    assert "metrics" in a and "metrics" in b  # both ran through the SAME pipeline
    assert a["metrics"] and b["metrics"]  # both models produced at least one metric
