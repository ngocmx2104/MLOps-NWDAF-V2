from src.training.pipeline import run_training
from src.training.schema import TrainingConfig


def test_result_carries_fit_validation_and_confusion(labeled_features, tmp_path):
    path, _ = labeled_features
    out = run_training(path, model_type="iforest", backend="noop", output_dir=tmp_path / "m",
                       cfg=TrainingConfig(use_labels_for_evaluation=True, label_column="label"))
    assert "fit_summary" in out and "validation_summary" in out
    assert out["fit_summary"]["feature_columns"]            # non-empty list
    assert "predicted_anomaly_rate" in out["validation_summary"]
    conf = out["validation_summary"]["confusion"]           # tp/fp/fn/tn from the validation split
    assert set(conf) == {"tp", "fp", "fn", "tn"} and all(isinstance(v, int) for v in conf.values())
