from src.tracking.base import BaseTracker
from src.tracking.noop import NoopTracker
from src.tracking.schema import ExperimentConfig


def test_noop_is_basetracker():
    assert issubclass(NoopTracker, BaseTracker)


def test_noop_lifecycle_safe_and_silent():
    t = NoopTracker()
    h = t.init_experiment(ExperimentConfig(experiment_name="e", run_name="r"))
    assert h.backend == "noop" and h.run_id is None
    t.log_params({"contamination": 0.01})
    t.log_metrics({"roc_auc": 0.9}, step=1)
    t.log_dataset("/tmp/x.parquet", name="d1")
    t.log_artifact("/tmp/roc.png", artifact_path="plots")
    assert t.register_model("/tmp/m.joblib", "iforest", {"roc_auc": 0.9}) is None
    t.end_experiment("FINISHED")
