# tests/experiments/test_exp2.py
"""Exp-2 C0-vs-C1 ablation test (tiny-N fixture, N=2).

FIX 1 — storage_delta_bytes must capture the REAL MLflow tracking store overhead
(sqlite DB + artifacts inside _c1_store/), not just the training output files that both
C0 and C1 share identically.  We point C1's store to a known tmp dir via
MLFLOW_TRACKING_URI + pre-created experiment, then assert the store dir grew.

FIX 2 — traceability must genuinely distinguish C0 from C1.
NoopTracker returns run_id=None and model_version=None (confirmed by reading
src/tracking/noop.py).  MLflowTracker returns a real UUID run_id and
"models:/.../<version>" model_version.  traceability_ok() derives the flag
from the emitted backend/run_id/model_version — no hand-assignment.
"""
from __future__ import annotations

import sys

from src.experiments.exp2_ablation import run_exp2


def test_run_exp2_ablation(tmp_path):
    from tests.experiments._fixtures import write_tiny_labeled_parquet

    ds = write_tiny_labeled_parquet(tmp_path / "d5.parquet")

    # FIX 1: point C1's MLflow store to a known directory inside tmp_path.
    # _c1_store will contain mlflow.db + mlartifacts — this is the real overhead.
    c1_store = tmp_path / "_c1_store"
    c1_db = c1_store / "mlflow.db"
    c1_env = {"MLFLOW_TRACKING_URI": f"sqlite:///{c1_db}"}

    def workload(backend):
        return [
            sys.executable, "-m", "src.training", "train",
            "--dataset", str(ds),
            "--backend", backend,
            "--output-dir", "{output_dir}",
            "--seed", "{seed}",
        ]

    summary = run_exp2(
        dataset=str(ds),
        seeds=[1, 2],
        output_root=tmp_path / "out",
        c0_workload=workload("noop"),
        c1_workload=workload("mlflow"),
        c1_env=c1_env,
        c1_store_dir=c1_store,
    )

    # ---- 6 groups present ----
    for group in ("model_perf", "operational", "resource", "maturity", "business", "data_quality"):
        assert group in summary, f"missing group: {group}"

    # ---- FIX 3: verify both configs produced actual successful runs ----
    # Without this, traceability/control assertions could be vacuously satisfied
    # (e.g. traceable_C0=False because c0_ok is empty, not because noop was detected).
    assert summary["n_c0_ok"] >= 1, (
        f"C0 must have at least 1 successful run; got n_c0_ok={summary['n_c0_ok']}"
    )
    assert summary["n_c1_ok"] >= 1, (
        f"C1 must have at least 1 successful run; got n_c1_ok={summary['n_c1_ok']}"
    )

    # ---- operational: wilcoxon present ----
    assert "wilcoxon_wall_s" in summary["operational"]

    # ---- FIX 1: storage_delta_bytes > 0 (MLflow genuinely writes sqlite db) ----
    # The C1 store includes at minimum a ~700KB sqlite DB — always > 0 when C1 ran.
    # C0/noop writes NO tracking store.
    assert summary["resource"]["storage_delta_bytes"] > 0, (
        "storage_delta_bytes should be >0: MLflow writes a sqlite DB + artifacts "
        "but got 0 — the store dir may not have been measured correctly."
    )

    # ---- model_perf is a CONTROL ----
    # control_equal is None only when no perf keys could be compared (no data).
    # With n_c0_ok>=1 and n_c1_ok>=1 confirmed above, it must be True or False.
    assert summary["model_perf"]["control_equal"] in (True, False), (
        "control_equal must be True or False when both configs have successful runs; "
        f"got {summary['model_perf']['control_equal']!r} — check that perf metrics are emitted"
    )

    # ---- FIX 2: traceability genuinely distinguishes C0 from C1 ----
    # NoopTracker emits run_id=None + model_version=None -> traceable_C0 must be False.
    # MLflowTracker emits a real run_id UUID + models:/.../1 -> traceable_C1 must be True.
    # With n_c0_ok>=1/n_c1_ok>=1 (asserted above) this is a real comparison, not vacuous.
    assert summary["maturity"]["traceable_C0"] is False, (
        "C0 (noop backend, run_id=None, model_version=None) must NOT be traceable"
    )
    assert summary["maturity"]["traceable_C1"] is True, (
        "C1 (mlflow backend, real run_id + model_version) must be traceable"
    )

    # maturity delta > 0: C1 has more MLops-tagged tests credited than C0
    assert summary["maturity"]["delta"] > 0, (
        "ML Test Score must improve from C0 to C1 (requires_mlops tests credited in C1)"
    )

    # ---- exp2_summary.json was written ----
    assert (tmp_path / "out" / "exp2_ablation" / "exp2_summary.json").exists()
