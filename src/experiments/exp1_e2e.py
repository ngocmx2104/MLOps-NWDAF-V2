# src/experiments/exp1_e2e.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.experiments.exp_common import ConfigSpec, run_experiment
from src.experiments.metrics.business import cost_sensitivity_curve
from src.experiments.records import build_result_summary, write_json
from src.experiments.stats import summarize

_DEFAULT_WORKLOAD = ["python", "-m", "src.training", "train", "--dataset", "{dataset}",
                     "--backend", "mlflow", "--output-dir", "{output_dir}", "--seed", "{seed}"]
_PERF_KEYS = ("roc_auc", "pr_auc", "f1", "precision", "recall")
_RATIOS = [1, 2, 5, 10, 20]


def run_exp1(*, dataset: str, seeds: list[int], output_root: Path,
             workload: list[str] | None = None) -> dict[str, Any]:
    """Exp-1: run the C1 pipeline E2E on real D5 over N seeds; report detection quality
    (ROC-AUC/PR-AUC/F1) + pipeline wall-time + cost-sensitivity curve. Establishes the pipeline
    solves ping-pong (RQ1/RQ2)."""
    wl = [t.replace("{dataset}", dataset) for t in (workload or _DEFAULT_WORKLOAD)]
    spec = ConfigSpec(label="C1", workload=wl, env={})
    runs = run_experiment(spec, experiment_id="exp1_e2e", seeds=seeds, output_root=output_root)
    ok = [r for r in runs if r["resource"]["returncode"] == 0]
    perf: dict[str, Any] = {}
    for key in _PERF_KEYS:
        vals = [r["result"].get("metrics", {}).get(key) for r in ok]
        vals = [float(v) for v in vals if v is not None]
        if vals:
            perf[key] = summarize(vals)
    wall = [r["resource"]["wall_s"] for r in ok]
    # cost sensitivity over aggregated confusion (sum FP/FN across seeds; weak-label confusion)
    fp = sum(int(r["result"].get("validation_summary", {}).get("confusion", {}).get("fp", 0)) for r in ok)
    fn = sum(int(r["result"].get("validation_summary", {}).get("confusion", {}).get("fn", 0)) for r in ok)
    cost = cost_sensitivity_curve(fp=fp, fn=fn, c_fp=1.0, ratios=_RATIOS)
    summary = build_result_summary(experiment_id="exp1_e2e",
                                   configs={"C1": {"model_perf": perf,
                                                   "pipeline_wall_s": summarize(wall) if wall else None}})
    summary["model_perf"] = perf
    summary["cost_sensitivity"] = cost
    summary["confusion_total"] = {"fp": fp, "fn": fn}
    summary["n_runs"] = len(runs)
    write_json(Path(output_root) / "exp1_e2e" / "exp1_summary.json", summary)
    return summary
