"""ClearMLTracker tests.

ClearML spawns background threads (resource monitor, dev worker) that keep the
host process alive and would block pytest's own exit. So the offline functional
flow runs in an ISOLATED SUBPROCESS that force-exits via os._exit(0) — the
threads die with it and the pytest process stays clean. Full server-backed
validation (real registry/overhead) happens in Phase 8 (Exp-3).

Note: this test module imports only the ClearMLTracker *class* (clearml itself is
lazy-imported inside init_experiment), so no clearml threads start in pytest.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

from src.tracking import create_tracker
from src.tracking.base import BaseTracker
from src.tracking.clearml_tracker import ClearMLTracker


def test_clearml_is_basetracker_and_factory():
    assert issubclass(ClearMLTracker, BaseTracker)
    assert isinstance(create_tracker("clearml"), ClearMLTracker)


def test_clearml_offline_flow_in_subprocess():
    """init → log_params → log_metrics → register_model → end, in ClearML offline mode."""
    script = textwrap.dedent(
        """
        import os, sys, tempfile, pathlib
        os.environ["CLEARML_OFFLINE_MODE"] = "1"
        from clearml import Task
        Task.set_offline(True)
        from src.tracking.clearml_tracker import ClearMLTracker
        from src.tracking.schema import ExperimentConfig
        t = ClearMLTracker()
        h = t.init_experiment(ExperimentConfig(experiment_name="NWDAF-MLOps",
                                               run_name="clearml_smoke",
                                               backend="clearml", tags={"model": "iforest"}))
        assert h.backend == "clearml" and h.run_id, "init failed"
        t.log_params({"contamination": 0.01})
        t.log_metrics({"roc_auc": 0.97})
        mf = pathlib.Path(tempfile.mkdtemp()) / "model.joblib"
        mf.write_bytes(b"dummy")
        mid = t.register_model(str(mf), "iforest_pingpong", {"roc_auc": 0.97}, alias="staging")
        assert mid is not None, "register_model returned None"
        t.end_experiment()
        print("CLEARML_OK", h.run_id)
        sys.stdout.flush()
        os._exit(0)  # force-exit so lingering clearml threads cannot block process exit
        """
    )
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr[-2000:]}"
    assert "CLEARML_OK" in r.stdout
