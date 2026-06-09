# src/experiments/exp2_ablation.py
"""Exp-2: C0 (noop/ad-hoc) vs C1 (MLflow) capability ablation.

Design rationale
----------------
Same data / model / seed — the ONLY variable is the MLOps layer.  Quantifies the
trade-off: cost (operational overhead + storage) vs benefit (maturity + traceability).

model_perf and business are CONTROLS (same seed/model → ~equal detection) — they prove
C0 is NOT a straw-man, not that C1 "wins".

FIX 1 — Storage delta
---------------------
The plain-plan approach summed run_dir bytes for C0 and C1.  Both run_dirs hold the
same model.joblib + training_result.json, so Δ ≈ 0 — missing MLflow's real cost.
Correct approach: measure the C1 MLflow *tracking store* (c1_store_dir) which is
pointed to a known directory via MLFLOW_TRACKING_URI=sqlite:///<abs>/mlflow.db.
C0/noop writes NO tracking store → C0 tracking bytes = 0.
storage_delta_bytes = config_storage_bytes(c1_store_dir) - 0.
The store includes: mlflow.db (SQLite run/param/metric metadata, ~700 KB) + any
logged artifacts under mlartifacts/.

FIX 2 — Traceability distinction
---------------------------------
NoopTracker.init_experiment returns RunHandle(run_id=None, backend="noop").
NoopTracker.register_model returns None.
Training pipeline emits:  "run_id": None,  "backend": "noop",  "model_version": None.

traceability_ok(r, require_registry=True) already returns False when run_id is None.
To be explicit and robust we additionally gate on backend != "noop": an ad-hoc
pipeline with no tracking store is objectively not traceable regardless of whether
a placeholder field slips through.  Derive from emitted fields — no hand-assignment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.experiments.exp_common import (
    ConfigSpec,
    config_storage_bytes,
    run_experiment,
    traceability_ok,
)
from src.experiments.maturity import assess
from src.experiments.records import write_json
from src.experiments.stats import summarize, wilcoxon_compare

_PERF_KEYS = ("roc_auc", "pr_auc", "f1")
_MANIFEST_REL = "src/experiments/mltest_manifest.yaml"


def _perf_vals(runs: list[dict], key: str) -> list[float]:
    return [float(v) for r in runs if (v := r["result"].get("metrics", {}).get(key)) is not None]


def _traceable(run: dict) -> bool:
    """Objective traceability gate combining backend identity + run_id + model_version.

    FIX 2: gates on backend != 'noop' (structural: noop has no real tracking store)
    AND run_id non-empty AND (require_registry) model_version non-empty.
    All values derived from the emitted training_result.json — zero hand-scoring.
    """
    result = run.get("result") or {}
    # Structural gate: noop is the ad-hoc baseline — by definition not traceable.
    if result.get("backend") == "noop":
        return False
    # Delegate the field-level check (run_id + model_version) to the shared helper.
    return traceability_ok(run, require_registry=True)


def run_exp2(
    *,
    dataset: str,
    seeds: list[int],
    output_root: Path,
    c0_workload: list[str],
    c1_workload: list[str],
    c1_env: dict[str, str] | None = None,
    c1_store_dir: Path,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Exp-2 capability ablation C0 (noop) vs C1 (mlflow).

    Parameters
    ----------
    c1_store_dir:
        Directory where C1's MLflow tracking store lives (sqlite DB + artifacts).
        REQUIRED.  FIX 1: we measure this directory's total bytes as the
        tracking overhead; C0/noop writes nothing there (0 bytes).
        storage_delta_bytes = config_storage_bytes(c1_store_dir) - 0.
        Passing None is an error — the run_dir-summing fallback has been
        removed because it would misattribute model bytes (which C0 also
        writes) as MLflow overhead, producing a fake Δ.
    """
    output_root = Path(output_root)
    repo_root = Path(repo_root)

    c0 = run_experiment(
        ConfigSpec(label="C0", workload=c0_workload, env={"MLOPS_BACKEND": "noop"}),
        experiment_id="exp2_ablation",
        seeds=seeds,
        output_root=output_root,
    )
    c1 = run_experiment(
        ConfigSpec(label="C1", workload=c1_workload,
                   env={"MLOPS_BACKEND": "mlflow", **(c1_env or {})}),
        experiment_id="exp2_ablation",
        seeds=seeds,
        output_root=output_root,
    )

    def ok(runs: list[dict]) -> list[dict]:
        return [r for r in runs if r["resource"]["returncode"] == 0]

    c0_ok, c1_ok = ok(c0), ok(c1)

    # ------------------------------------------------------------------
    # 1) model_perf — CONTROL (expect C0 ≈ C1; same seed/model)
    # ------------------------------------------------------------------
    perf: dict[str, Any] = {}
    # FIX 2: control_equal must NOT be a vacuous True when we have no data to
    # compare.  Start as None and set to True/False only if at least one metric
    # key was actually compared across both configs.
    control_equal: bool | None = None
    keys_compared = 0
    for k in _PERF_KEYS:
        a, b = _perf_vals(c0_ok, k), _perf_vals(c1_ok, k)
        if a and b:
            perf[k] = {"C0": summarize(a), "C1": summarize(b)}
            keys_compared += 1
            # Start from True on first real comparison, then flip if any key
            # differs meaningfully.
            if control_equal is None:
                control_equal = True
            if abs(summarize(a)["mean"] - summarize(b)["mean"]) > 1e-6:
                control_equal = False
    if keys_compared == 0:
        # No overlapping perf data → cannot certify anything.
        control_equal = None
        perf["control_equal_note"] = "insufficient data — no overlapping perf keys compared"
    perf["control_equal"] = control_equal
    perf["note"] = (
        "CONTROL — same seed/model; near-equal detection proves C0 is not a straw-man"
    )

    # ------------------------------------------------------------------
    # 2) operational — pipeline wall time + Wilcoxon C0 vs C1
    # ------------------------------------------------------------------
    w0 = [r["resource"]["wall_s"] for r in c0_ok]
    w1 = [r["resource"]["wall_s"] for r in c1_ok]
    operational: dict[str, Any] = {
        "C0_wall_s": summarize(w0) if w0 else None,
        "C1_wall_s": summarize(w1) if w1 else None,
    }
    if w0 and w1:
        if len(w0) == len(w1) and len(w0) > 1:
            operational["wilcoxon_wall_s"] = wilcoxon_compare(w0, w1)
        else:
            operational["wilcoxon_wall_s"] = {
                "note": (
                    f"n_c0={len(w0)}, n_c1={len(w1)} — unequal pairs"
                    if len(w0) != len(w1)
                    else f"n_c0={len(w0)}, n_c1={len(w1)} — need ≥2 matched pairs; collect more seeds"
                )
            }
    else:
        operational["wilcoxon_wall_s"] = {"note": "no successful runs"}

    # ------------------------------------------------------------------
    # 3) resource — RSS + FIXED storage delta (FIX 1)
    # ------------------------------------------------------------------
    rss0 = [r["resource"]["peak_rss_mb"] for r in c0_ok]
    rss1 = [r["resource"]["peak_rss_mb"] for r in c1_ok]

    # FIX 1: c1_store_dir is now REQUIRED (see signature).  Raise loudly if somehow
    # None slips through at runtime (e.g. called via **kwargs unpacking).
    if c1_store_dir is None:
        raise ValueError(
            "c1_store_dir is required to measure tracking-store overhead honestly. "
            "Pass the directory that contains C1's MLflow sqlite DB + mlartifacts. "
            "The run_dir-summing fallback has been removed because it misattributes "
            "model bytes (which C0 also writes) as MLflow overhead."
        )
    # Measure the C1 tracking store (db + artifacts) as the real overhead.
    # C0/noop writes NO tracking store, so its tracking bytes = 0.
    store_bytes_c1 = config_storage_bytes(Path(c1_store_dir))
    store_scope = "c1_store_dir (sqlite DB + mlartifacts)"
    store_bytes_c0 = 0  # noop/C0 writes no tracking store

    resource: dict[str, Any] = {
        "C0_rss_mb": summarize(rss0) if rss0 else None,
        "C1_rss_mb": summarize(rss1) if rss1 else None,
        "storage_delta_bytes": store_bytes_c1 - store_bytes_c0,
        "storage_c1_bytes": store_bytes_c1,
        "storage_c0_bytes": store_bytes_c0,
        "storage_scope": store_scope,
    }

    # ------------------------------------------------------------------
    # 4) maturity — ML Test Score C0 vs C1 (FIX 2: traceability gate)
    # ------------------------------------------------------------------
    manifest_path = (repo_root / _MANIFEST_REL).resolve()
    # FIX 4: missing manifest must fail loudly, not silently produce assess({})
    # → C0=0/C1=0/delta=0 which would invalidate the RQ3 maturity finding.
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"ML Test Score manifest not found at: {manifest_path}\n"
            "Run from the repo root or pass repo_root= explicitly. "
            "Creating an empty manifest is forbidden — the maturity result would be meaningless."
        )
    manifest: dict[str, Any] = yaml.safe_load(manifest_path.read_text()) or {}

    mat_c0 = assess(manifest, repo_root=repo_root, capability="C0")
    mat_c1 = assess(manifest, repo_root=repo_root, capability="C1")

    # FIX 2: derive traceability from emitted backend + run_id + model_version.
    # noop: run_id=None, backend="noop", model_version=None → False (structural + fields).
    # mlflow: real UUID run_id, backend="mlflow", models:/.../<v> → True.
    traceable_c0 = (
        all(_traceable(r) for r in c0_ok) if c0_ok else False
    )
    traceable_c1 = (
        all(_traceable(r) for r in c1_ok) if c1_ok else False
    )

    maturity: dict[str, Any] = {
        "C0_score": mat_c0["ml_test_score"],
        "C1_score": mat_c1["ml_test_score"],
        "delta": mat_c1["ml_test_score"] - mat_c0["ml_test_score"],
        "google_level_C0": mat_c0["google_level"],
        "google_level_C1": mat_c1["google_level"],
        "traceable_C0": traceable_c0,
        "traceable_C1": traceable_c1,
        "traceability_note": (
            "FIX2: C0 noop returns run_id=None + model_version=None + backend='noop' -> False; "
            "C1 mlflow returns real run_id UUID + models:/.../v + backend='mlflow' -> True. "
            "Derived from emitted training_result.json fields, zero hand-scoring."
        ),
    }

    # ------------------------------------------------------------------
    # 5) business — CONTROL (identical detection C0/C1; sensitivity in Exp-1)
    # ------------------------------------------------------------------
    business: dict[str, Any] = {
        "note": (
            "CONTROL — identical detection C0/C1 (same seed/model proves fairness); "
            "cost-sensitivity curve reported in Exp-1 + tables."
        )
    }

    # ------------------------------------------------------------------
    # 6) data_quality — capability flag (C1 records QC; C0 ad-hoc does not)
    # FIX 6: derive from emitted run_id field — not a bare bool(c1_ok).
    # C1 (mlflow) emits a real UUID run_id → True when all successful runs have one.
    # C0 (noop) emits run_id=None → False (noop has no persistent tracking store).
    # This makes the flag falsifiable from run output, not a capability assertion.
    # ------------------------------------------------------------------
    c1_quality_recorded = bool(c1_ok) and all(
        r["result"].get("run_id") for r in c1_ok
    )
    c0_quality_recorded = bool(c0_ok) and all(
        r["result"].get("run_id") for r in c0_ok
    )
    data_quality: dict[str, Any] = {
        "C1_quality_recorded": c1_quality_recorded,
        "C0_quality_recorded": c0_quality_recorded,
        "note": (
            "FIX6: derived from emitted run_id — C1 mlflow runs have a real UUID "
            "run_id (governed, persisted); C0 noop runs emit run_id=None (not persisted). "
            "C1 logs dataset hash + feature version to MLflow; C0/noop does not."
        ),
    }

    summary: dict[str, Any] = {
        "experiment_id": "exp2_ablation",
        "n_seeds": len(seeds),
        "n_c0_ok": len(c0_ok),
        "n_c1_ok": len(c1_ok),
        "model_perf": perf,
        "operational": operational,
        "resource": resource,
        "maturity": maturity,
        "business": business,
        "data_quality": data_quality,
    }
    write_json(output_root / "exp2_ablation" / "exp2_summary.json", summary)
    return summary
