# src/experiments/exp_common.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
