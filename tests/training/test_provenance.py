from src.training.pipeline import run_training
from src.training.schema import TrainingConfig


def test_training_logs_lineage_tags(labeled_features, tmp_path, monkeypatch):
    """run_training should log dataset lineage (dataset_id/feature_version/source_snapshot_id)
    as params via the tracker (provenance, HANDOFF §9.6)."""
    captured = {}
    import src.training.pipeline as pipe

    real_create = pipe.create_tracker

    def spy_create(backend=None):
        tr = real_create(backend)
        orig = tr.log_params
        tr.log_params = lambda params: (captured.update(params), orig(params))[1]
        return tr

    monkeypatch.setattr(pipe, "create_tracker", spy_create)
    path, _ = labeled_features
    run_training(path, model_type="iforest", backend="noop", output_dir=tmp_path / "m",
                 cfg=TrainingConfig(use_labels_for_evaluation=True, label_column="label"))
    assert {"dataset_id", "feature_version", "source_snapshot_id", "dataset_layer", "row_count"}.issubset(captured)
