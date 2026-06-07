from src.tracking.schema import ExperimentConfig, RunHandle

def test_experiment_config_defaults():
    cfg = ExperimentConfig(experiment_name="exp", run_name="run1")
    assert cfg.backend == "noop"
    assert cfg.tags == {}

def test_run_handle():
    h = RunHandle(run_id=None, backend="noop")
    assert h.run_id is None and h.backend == "noop" and h.url is None
