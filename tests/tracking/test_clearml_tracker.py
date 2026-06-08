"""Tests for ClearMLTracker — runs ClearML in offline mode (no server required)."""
import pytest

from src.tracking.clearml_tracker import ClearMLTracker
from src.tracking.schema import ExperimentConfig


@pytest.fixture(autouse=True)
def clearml_offline(tmp_path, monkeypatch):
    from clearml import Task

    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
    Task.set_offline(offline_mode=True)
    yield
    Task.set_offline(offline_mode=False)


def test_init_and_log_offline(tmp_path):
    t = ClearMLTracker()
    h = t.init_experiment(
        ExperimentConfig(
            experiment_name="NWDAF-MLOps",
            run_name="clearml_r1",
            backend="clearml",
            tags={"model": "iforest"},
        )
    )
    assert h.backend == "clearml" and h.run_id

    t.log_params({"contamination": 0.01})
    t.log_metrics({"roc_auc": 0.97})

    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"dummy")

    # register_model: in offline mode OutputModel raises AttributeError (DummyModel limitation);
    # the tracker catches it and returns a deterministic offline placeholder id (non-None).
    mid = t.register_model(str(model_file), "iforest_pingpong", {"roc_auc": 0.97}, alias="staging")
    assert mid is not None

    t.end_experiment()
