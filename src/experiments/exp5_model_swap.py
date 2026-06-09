# src/experiments/exp5_model_swap.py
"""Exp-5 model-swap: IsolationForest vs LSTM-AE (RQ4b).

HEADLINE = quantified comparison of the two model families on the SAME data/split;
'swap_core_changes=0' (the model is a config flag, not a code change) is the
SUPPORTING flexibility fact, not the main claim.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.experiments.exp_common import ConfigSpec, run_experiment
from src.experiments.records import write_json
from src.experiments.stats import summarize, wilcoxon_compare

_PERF_KEYS = ("roc_auc", "pr_auc", "f1")


def _collect(runs, label):
    ok = [r for r in runs if r["resource"]["returncode"] == 0]
    perf: dict[str, Any] = {}
    for k in _PERF_KEYS:
        vals = [float(r["result"]["metrics"][k]) for r in ok
                if r["result"].get("metrics", {}).get(k) is not None]
        if vals:
            perf[k] = summarize(vals)
    train_s = [r["resource"]["wall_s"] for r in ok]
    return {"label": label, "n_ok": len(ok), "model_perf": perf,
            "train_wall_s": summarize(train_s) if train_s else None}


def run_exp5(*, dataset: str, seeds: list[int], output_root: Path,
             iforest_workload: list[str], lstm_workload: list[str]) -> dict[str, Any]:
    """Exp-5 model-swap (RQ4b). HEADLINE = quantified comparison of the two model families on the
    SAME data/split; 'swap_core_changes=0' (the model is a config flag, not a code change) is the
    SUPPORTING flexibility fact, not the main claim."""
    output_root = Path(output_root)
    rf = run_experiment(ConfigSpec(label="iforest", workload=iforest_workload, env={}),
                        experiment_id="exp5_model_swap", seeds=seeds, output_root=output_root)
    lstm = run_experiment(ConfigSpec(label="lstm_ae", workload=lstm_workload, env={}),
                          experiment_id="exp5_model_swap", seeds=seeds, output_root=output_root)
    iforest_c = _collect(rf, "iforest")
    lstm_c = _collect(lstm, "lstm_ae")
    summary: dict[str, Any] = {"experiment_id": "exp5_model_swap",
                               "iforest": iforest_c, "lstm_ae": lstm_c,
                               "swap_core_changes": 0}  # only --model-type differs; pipeline unchanged
    # Wilcoxon on roc_auc when both have paired per-seed values
    a = [float(r["result"]["metrics"].get("roc_auc")) for r in rf
         if r["result"].get("metrics", {}).get("roc_auc") is not None]
    b = [float(r["result"]["metrics"].get("roc_auc")) for r in lstm
         if r["result"].get("metrics", {}).get("roc_auc") is not None]
    if len(a) == len(b) and len(a) > 1:
        summary["wilcoxon_roc_auc"] = wilcoxon_compare(a, b)
    write_json(output_root / "exp5_model_swap" / "exp5_summary.json", summary)
    return summary
