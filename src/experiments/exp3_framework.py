# src/experiments/exp3_framework.py
"""Exp-3: framework overhead + governance comparison (Noop / MLflow / ClearML).

Design rationale
----------------
Same training+tracking workload under each backend, varying only the
MLOPS_BACKEND env variable (and tracking URI for mlflow).  Measures:

1. **Overhead** — wall-clock time + peak RSS vs the noop (no-tracking) baseline.
   delta_wall_s = mean(backend) - mean(noop).  Positive → tracking adds latency.
   Justifies choosing MLflow: its overhead must be acceptable.

2. **Governance** — did runs produce a registered model_version AND a run_id?
   Derived from emitted fields in training_result.json:
   - noop : run_id=None, model_version=None  → registry=False, run_id=False
   - mlflow: real UUID run_id + models:/…/<v> → registry=True,  run_id=True
   No hand-assignment — governance flags are falsifiable from subprocess output.

ML metrics are NOT compared here (same seed/model → should be equal; if they
differ that is a bug, not a finding).  Governance + overhead IS the finding.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.experiments.exp_common import ConfigSpec, run_experiment
from src.experiments.records import write_json
from src.experiments.stats import summarize


def run_exp3(
    *,
    dataset: str,
    seeds: list[int],
    output_root: Path,
    workload: list[str],
    backends: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Exp-3: run the same training workload under each backend; measure overhead delta
    vs noop and governance presence.

    Parameters
    ----------
    dataset:
        Path to the labeled feature parquet (passed as ``--dataset`` to the CLI via
        workload tokens, but stored for provenance in the summary).
    seeds:
        List of integer seeds; one subprocess per seed per backend.
    output_root:
        Base directory for per-backend, per-seed run directories + summary JSON.
    workload:
        Command template with ``{output_dir}`` and ``{seed}`` tokens.  The backend
        name is injected via ``MLOPS_BACKEND`` env variable; the workload itself
        must accept ``--backend`` via env (the real training CLI reads
        ``MLOPS_BACKEND`` if ``--backend`` is not passed).
    backends:
        Mapping from backend name to extra env vars.  Must include ``"noop"`` as the
        baseline.  Example::

            {"noop": {},
             "mlflow": {"MLFLOW_TRACKING_URI": "sqlite:///path/to/mlflow.db"},
             "clearml": {"CLEARML_OFFLINE_MODE": "1"}}

    Returns
    -------
    dict with keys:
        ``experiment_id``, ``overhead`` (per-backend wall/rss + delta vs noop),
        ``governance`` (per-backend registry + run_id bool flags).
    Also writes ``exp3_summary.json`` under ``output_root/exp3_framework/``.
    """
    output_root = Path(output_root)
    overhead: dict[str, Any] = {}
    governance: dict[str, Any] = {}

    for name, env in backends.items():
        runs = run_experiment(
            ConfigSpec(label=name, workload=workload, env={"MLOPS_BACKEND": name, **env}),
            experiment_id="exp3_framework",
            seeds=seeds,
            output_root=output_root,
        )
        ok = [r for r in runs if r["resource"]["returncode"] == 0]

        overhead[name] = {
            "wall_s": summarize([r["resource"]["wall_s"] for r in ok]) if ok else None,
            "rss_mb": summarize([r["resource"]["peak_rss_mb"] for r in ok]) if ok else None,
        }

        # Governance: objective derivation from emitted training_result.json fields.
        # noop: model_version=None, run_id=None → both False.
        # mlflow: model_version="models:/…/<v>" (non-None), run_id=UUID → both True.
        # all() on an empty list is True (vacuously) — guard with bool(ok) first.
        governance[name] = {
            "registry": bool(ok) and all(
                r["result"].get("model_version") not in (None, "")
                for r in ok
            ),
            "run_id": bool(ok) and all(
                r["result"].get("run_id") not in (None, "")
                for r in ok
            ),
        }

    # Compute deltas vs the noop baseline.
    base_wall = overhead.get("noop", {}).get("wall_s")
    base_rss = overhead.get("noop", {}).get("rss_mb")
    for name, ov in overhead.items():
        if ov.get("wall_s") and base_wall:
            ov["delta_wall_s"] = ov["wall_s"]["mean"] - base_wall["mean"]
        if ov.get("rss_mb") and base_rss:
            ov["delta_rss_mb"] = ov["rss_mb"]["mean"] - base_rss["mean"]

    summary: dict[str, Any] = {
        "experiment_id": "exp3_framework",
        "dataset": dataset,
        "n_seeds": len(seeds),
        "backends": list(backends.keys()),
        "overhead": overhead,
        "governance": governance,
    }
    write_json(output_root / "exp3_framework" / "exp3_summary.json", summary)
    return summary
