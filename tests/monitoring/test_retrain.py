from src.monitoring.retrain import AutoRetrainTrigger, run_retrain_cycle
from src.monitoring.schema import MonitoringConfig
from src.serving.runtime import ServingRuntime
from src.serving.schema import ServingConfig


def test_trigger_respects_cooldown():
    trig = AutoRetrainTrigger(cooldown_seconds=10_000.0)
    assert trig.should_retrain({"drift_detected": True}) is True
    trig.record_retrain()
    assert trig.should_retrain({"drift_detected": True}) is False  # within cooldown
    assert trig.should_retrain({"drift_detected": False}) is False


def test_no_drift_skips_retrain(reference_parquet, predictions_no_drift, tmp_path):
    out = run_retrain_cycle(
        predictions_path=predictions_no_drift, reference_path=reference_parquet,
        dataset_path=reference_parquet, output_dir=tmp_path / "rc",
        config=MonitoringConfig(min_features_alert=3, min_observed_rows=50))
    assert out["retrained"] is False
    assert out["drift"]["drift_detected"] is False


def test_drift_triggers_retrain_and_reload(reference_parquet, predictions_drift, tmp_path, monkeypatch):
    """C1 closed loop: drift -> retrain (mlflow registry) -> ServingRuntime.reload() serves new version."""
    import mlflow
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    from src.training.pipeline import run_training
    from src.training.schema import TrainingConfig
    from src.cicd.eval_gate import run_eval_gate
    from src.cicd.schema import GateConfig
    from src.tracking import create_tracker
    boot = run_training(reference_parquet, model_type="iforest", backend="mlflow",
                        output_dir=tmp_path / "m0",
                        cfg=TrainingConfig(use_labels_for_evaluation=True, label_column="label"))
    # bootstrap: put an initial model at 'staging' so the registry runtime can load it
    # (min_roc_auc=0.0 -> always promote; this is test setup, not a real gate)
    run_eval_gate(metrics=boot["metrics"], model_name=boot["model_name"],
                  model_version=boot["model_version"], tracker=create_tracker("mlflow"),
                  cfg=GateConfig(min_roc_auc=0.0))
    runtime = ServingRuntime.build(ServingConfig(loader="registry", model_type="iforest",
                                                 tracking_uri=uri, output_root=str(tmp_path / "s")))
    before = runtime.model.version
    out = run_retrain_cycle(
        predictions_path=predictions_drift, reference_path=reference_parquet,
        dataset_path=reference_parquet, output_dir=tmp_path / "rc",
        config=MonitoringConfig(min_features_alert=3, min_observed_rows=50, cooldown_seconds=0.0),
        model_type="iforest", backend="mlflow", runtime=runtime)
    assert out["retrained"] is True
    assert out["new_auc"] >= 0.0
    assert out["retrain_count"] == 1  # surfaced for the P8 Exp-4 retrain-count metric
    assert runtime.model.version != before  # reload() deployed the freshly retrained version


def test_eval_gate_blocks_bad_model(reference_parquet, predictions_drift, tmp_path):
    """A retrain whose ROC-AUC is below the gate is NOT deployed."""
    out = run_retrain_cycle(
        predictions_path=predictions_drift, reference_path=reference_parquet,
        dataset_path=reference_parquet, output_dir=tmp_path / "rc",
        config=MonitoringConfig(min_features_alert=3, min_observed_rows=50,
                                cooldown_seconds=0.0, retrain_min_auc=1.1),  # impossible gate
        model_type="iforest", backend="noop")
    assert out["retrained"] is False and out["gate_failed"] is True
