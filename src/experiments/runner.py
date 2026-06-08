from __future__ import annotations

from pathlib import Path
from typing import Any

from src.experiments.metrics.resource import measure_subprocess
from src.experiments.records import append_jsonl, build_run_record


def _bind(workload: list[str], seed: int) -> list[str]:
    return [tok.replace("{seed}", str(seed)) for tok in workload]


def run_n(rc, *, experiment_id: str, config_label: str, output_dir: Path) -> dict[str, Any]:
    """Run the workload n_runs times (one subprocess each, fixed seeds), measuring resource;
    write every run to runs.jsonl. Warmup runs are recorded but flagged scored=False."""
    output_dir = Path(output_dir)
    runs: list[dict[str, Any]] = []
    for i, seed in enumerate(rc.seeds[:rc.n_runs]):
        res = measure_subprocess(_bind(rc.workload, seed), env=rc.env)
        scored = i >= rc.warmup_drop
        rec = build_run_record(experiment_id=experiment_id, run_index=i, seed=seed,
                               resource=res, config_label=config_label)
        rec["scored"] = scored
        append_jsonl(output_dir / "runs.jsonl", rec)
        runs.append(rec)
    return {"experiment_id": experiment_id, "config_label": config_label,
            "n_total": len(runs), "n_scored": sum(1 for r in runs if r["scored"]),
            "runs": runs}
