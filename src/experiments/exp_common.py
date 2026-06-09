# src/experiments/exp_common.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.experiments.metrics.resource import measure_subprocess
from src.experiments.records import append_jsonl, build_run_record


@dataclass(frozen=True)
class ConfigSpec:
    """One experiment configuration: a labelled workload run under a fixed env.
    workload tokens '{seed}' and '{output_dir}' are substituted per run."""
    label: str
    workload: list[str]
    env: dict[str, str] = field(default_factory=dict)


def read_training_result(output_dir: Path) -> dict[str, Any] | None:
    """Read training_result.json emitted by the training CLI; None if absent/bad."""
    p = Path(output_dir) / "training_result.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _bind(workload: list[str], seed: int, output_dir: Path) -> list[str]:
    return [t.replace("{seed}", str(seed)).replace("{output_dir}", str(output_dir)) for t in workload]


def run_experiment(spec: ConfigSpec, *, experiment_id: str, seeds: list[int],
                   output_root: Path) -> list[dict[str, Any]]:
    """Run spec.workload once per seed as an isolated subprocess (honest resource), parse the
    emitted training_result.json, and record every run to runs.jsonl. Returns per-run dicts."""
    cfg_dir = Path(output_root) / experiment_id / spec.label
    cfg_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        run_dir = cfg_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        res = measure_subprocess(_bind(spec.workload, seed, run_dir), env=spec.env)
        result = read_training_result(run_dir)
        rec = build_run_record(experiment_id=experiment_id, run_index=i, seed=seed,
                               resource=res, config_label=spec.label)
        rec["result"] = result or {}
        rec["run_dir"] = str(run_dir)
        append_jsonl(cfg_dir / "runs.jsonl", rec)
        runs.append(rec)
    return runs


def inference_latency_jsonl(model_path, x, out_jsonl) -> Path:
    """Score each row of x with the joblib model, timing per-sample inference (perf_counter),
    writing a predictions.jsonl (one record/sample, has latency_ms) for operational.latency_percentiles.
    Note: loads a trusted first-party model produced by this same pipeline (not untrusted deserialization)."""
    model = joblib.load(model_path)
    out_jsonl = Path(out_jsonl)
    x = np.asarray(x, dtype=float)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in x:
            t0 = time.perf_counter()
            score = -model.score_samples(row.reshape(1, -1))[0]
            dt_ms = (time.perf_counter() - t0) * 1000.0
            f.write(json.dumps({"event": "predict", "anomaly_score": float(score),
                                "latency_ms": dt_ms}) + "\n")
    return out_jsonl


def config_storage_bytes(path) -> int:
    """Total bytes under a config's tracking dir (mlruns/registry). 0 if absent — the C0/noop case."""
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def traceability_ok(run_record: dict, *, require_registry: bool) -> bool:
    """Objective lineage check: a run is traceable iff it produced a run_id AND (when require_registry)
    a registered model_version. C0/noop returns a placeholder run but no real model_version -> not
    traceable to a governed artifact. No hand-scoring: derived from emitted fields."""
    result = run_record.get("result") or {}
    if run_record.get("resource", {}).get("returncode") != 0:
        return False
    if not result.get("run_id"):
        return False
    if require_registry and not result.get("model_version"):
        return False
    return True
