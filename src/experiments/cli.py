from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from src.experiments.maturity import assess
from src.experiments.records import write_json


def cmd_assess(args: argparse.Namespace) -> int:
    try:
        manifest = yaml.safe_load(Path(args.manifest).read_text())
    except FileNotFoundError as exc:
        print(f"[error] cannot load manifest: {exc}", file=sys.stderr)
        return 1
    except yaml.YAMLError as exc:
        print(f"[error] cannot load manifest: {exc}", file=sys.stderr)
        return 1
    if not isinstance(manifest, dict):
        print(
            f"[error] cannot load manifest: expected a mapping, got {type(manifest).__name__}",
            file=sys.stderr,
        )
        return 1
    report = assess(manifest, repo_root=Path(args.repo_root))
    if args.output:
        write_json(Path(args.output), report)
    print(
        f"[ml-test-score] {report['ml_test_score']} "
        f"(sections={report['section_scores']}, google L{report['google_level']})"
    )
    return 0


def _parse_seeds(seeds_str: str) -> list[int]:
    """Parse comma-separated seeds string to list[int], e.g. '1,2,3' → [1,2,3]."""
    return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]


def cmd_run_exp1(args: argparse.Namespace) -> int:
    """Run Exp-1 E2E quality experiment and print the summary path."""
    try:
        from src.experiments.exp1_e2e import run_exp1

        dataset = args.dataset
        if not Path(dataset).exists():
            print(f"[error] dataset not found: {dataset}", file=sys.stderr)
            return 1

        seeds = _parse_seeds(args.seeds)
        output_root = Path(args.output_root)
        backend = args.backend  # "noop" or "mlflow"

        # Build the training workload: real argv with {output_dir} and {seed} tokens.
        workload = [
            sys.executable, "-m", "src.training", "train",
            "--dataset", dataset,
            "--backend", backend,
            "--output-dir", "{output_dir}",
            "--seed", "{seed}",
        ]

        run_exp1(
            dataset=dataset,
            seeds=seeds,
            output_root=output_root,
            workload=workload,
        )
        summary_path = output_root / "exp1_e2e" / "exp1_summary.json"
        print(f"[exp1] summary written to {summary_path}")
        return 0
    except Exception as exc:
        print(f"[error] run-exp1 failed: {exc}", file=sys.stderr)
        return 1


def cmd_run_exp2(args: argparse.Namespace) -> int:
    """Run Exp-2 C0 vs C1 ablation and print the summary path."""
    try:
        from src.experiments.exp2_ablation import run_exp2

        dataset = args.dataset
        if not Path(dataset).exists():
            print(f"[error] dataset not found: {dataset}", file=sys.stderr)
            return 1

        seeds = _parse_seeds(args.seeds)
        output_root = Path(args.output_root)

        # c1_store_dir: dedicated directory for C1's MLflow tracking store.
        # REQUIRED for honest storage measurement (FIX 1 in exp2_ablation).
        c1_store_dir = output_root / "exp2_ablation" / "_c1_store"
        c1_store_dir.mkdir(parents=True, exist_ok=True)

        c1_tracking_uri = getattr(args, "c1_tracking_uri", None)
        if c1_tracking_uri is None:
            # Default to a sqlite DB inside c1_store_dir
            c1_tracking_uri = f"sqlite:///{c1_store_dir / 'mlflow.db'}"

        c0_workload = [
            sys.executable, "-m", "src.training", "train",
            "--dataset", dataset,
            "--backend", "noop",
            "--output-dir", "{output_dir}",
            "--seed", "{seed}",
        ]
        c1_workload = [
            sys.executable, "-m", "src.training", "train",
            "--dataset", dataset,
            "--backend", "mlflow",
            "--output-dir", "{output_dir}",
            "--seed", "{seed}",
        ]

        c1_env = {"MLFLOW_TRACKING_URI": c1_tracking_uri}

        run_exp2(
            dataset=dataset,
            seeds=seeds,
            output_root=output_root,
            c0_workload=c0_workload,
            c1_workload=c1_workload,
            c1_env=c1_env,
            c1_store_dir=c1_store_dir,
            repo_root=Path("."),
        )
        summary_path = output_root / "exp2_ablation" / "exp2_summary.json"
        print(f"[exp2] summary written to {summary_path}")
        return 0
    except Exception as exc:
        print(f"[error] run-exp2 failed: {exc}", file=sys.stderr)
        return 1


def cmd_run_exp3(args: argparse.Namespace) -> int:
    """Run Exp-3 framework overhead/governance comparison and print the summary path."""
    try:
        from src.experiments.exp3_framework import run_exp3

        dataset = args.dataset
        if not Path(dataset).exists():
            print(f"[error] dataset not found: {dataset}", file=sys.stderr)
            return 1

        seeds = _parse_seeds(args.seeds)
        output_root = Path(args.output_root)

        # Build the base workload (backend injected via MLOPS_BACKEND env by exp3 runner)
        workload = [
            sys.executable, "-m", "src.training", "train",
            "--dataset", dataset,
            "--output-dir", "{output_dir}",
            "--seed", "{seed}",
        ]

        # Parse backends: default noop+mlflow. Optional --backends "noop,mlflow,clearml"
        backends_arg = getattr(args, "backends", None) or "noop,mlflow"
        backend_names = [b.strip() for b in backends_arg.split(",") if b.strip()]

        # Build env per backend (mlflow gets a sqlite URI in output_root)
        mlflow_db = output_root / "exp3_framework" / "_mlflow.db"
        mlflow_db.parent.mkdir(parents=True, exist_ok=True)
        backends: dict[str, dict[str, str]] = {}
        for name in backend_names:
            if name == "noop":
                backends["noop"] = {}
            elif name == "mlflow":
                backends["mlflow"] = {"MLFLOW_TRACKING_URI": f"sqlite:///{mlflow_db}"}
            elif name == "clearml":
                backends["clearml"] = {"CLEARML_OFFLINE_MODE": "1"}
            else:
                backends[name] = {}

        run_exp3(
            dataset=dataset,
            seeds=seeds,
            output_root=output_root,
            workload=workload,
            backends=backends,
        )
        summary_path = output_root / "exp3_framework" / "exp3_summary.json"
        print(f"[exp3] summary written to {summary_path}")
        return 0
    except Exception as exc:
        print(f"[error] run-exp3 failed: {exc}", file=sys.stderr)
        return 1


def cmd_run_exp4(args: argparse.Namespace) -> int:
    """Run Exp-4 drift scenario-clock + closed-loop retrain ON/OFF and print the summary path."""
    try:
        from src.experiments.exp4_drift import run_exp4

        baseline = args.baseline
        if not Path(baseline).exists():
            print(f"[error] baseline not found: {baseline}", file=sys.stderr)
            return 1

        scenarios: dict[str, Path] = {}
        if args.sudden_drift and Path(args.sudden_drift).exists():
            scenarios["sudden"] = Path(args.sudden_drift)
        if args.gradual_drift and Path(args.gradual_drift).exists():
            scenarios["gradual"] = Path(args.gradual_drift)
        if args.recurring_drift and Path(args.recurring_drift).exists():
            scenarios["recurring"] = Path(args.recurring_drift)

        if not scenarios:
            print("[error] no valid drift scenario paths provided", file=sys.stderr)
            return 1

        output_root = Path(args.output_root)
        n_steps = int(args.n_steps)

        run_exp4(
            baseline_path=Path(baseline),
            scenarios=scenarios,
            output_root=output_root,
            n_steps=n_steps,
        )
        summary_path = output_root / "exp4_drift" / "exp4_summary.json"
        print(f"[exp4] summary written to {summary_path}")
        return 0
    except Exception as exc:
        print(f"[error] run-exp4 failed: {exc}", file=sys.stderr)
        return 1


def cmd_run_exp5(args: argparse.Namespace) -> int:
    """Run Exp-5 model-swap (IForest vs LSTM-AE) and print the summary path."""
    try:
        from src.experiments.exp5_model_swap import run_exp5

        dataset = args.dataset
        if not Path(dataset).exists():
            print(f"[error] dataset not found: {dataset}", file=sys.stderr)
            return 1

        seeds = _parse_seeds(args.seeds)
        output_root = Path(args.output_root)

        def _make_workload(model_type: str) -> list[str]:
            return [
                sys.executable, "-m", "src.training", "train",
                "--dataset", dataset,
                "--model-type", model_type,
                "--backend", "noop",
                "--output-dir", "{output_dir}",
                "--seed", "{seed}",
            ]

        run_exp5(
            dataset=dataset,
            seeds=seeds,
            output_root=output_root,
            iforest_workload=_make_workload("iforest"),
            lstm_workload=_make_workload("lstm_ae"),
        )
        summary_path = output_root / "exp5_model_swap" / "exp5_summary.json"
        print(f"[exp5] summary written to {summary_path}")
        return 0
    except Exception as exc:
        print(f"[error] run-exp5 failed: {exc}", file=sys.stderr)
        return 1


def cmd_run_exp6(args: argparse.Namespace) -> int:
    """Run Exp-6 modifiability (3 NWDAF mods, worktree + diff + regression) and print the summary."""
    try:
        from src.experiments.exp6_modifiability import run_exp6
        from src.experiments.modifications import get_default_mods

        output_root = Path(args.output_root)

        run_exp6(
            mods=get_default_mods(),
            output_root=output_root,
        )
        summary_path = output_root / "exp6_modifiability" / "exp6_summary.json"
        print(f"[exp6] summary written to {summary_path}")
        return 0
    except Exception as exc:
        print(f"[error] run-exp6 failed: {exc}", file=sys.stderr)
        return 1


def cmd_tables(args: argparse.Namespace) -> int:
    """Generate Markdown result tables from available experiment summaries."""
    try:
        from src.experiments.tables import (
            exp1_table,
            exp2_table,
            exp3_table,
            exp4_table,
            exp5_table,
            exp6_table,
            render_markdown,
        )

        output_root = Path(args.output_root)
        output_file = Path(args.output) if args.output else None

        sections: list[str] = []

        # Helper: load a summary JSON, return {} on missing/bad
        def load_summary(rel: str) -> dict:
            p = output_root / rel
            if p.exists():
                try:
                    return json.loads(p.read_text())
                except (json.JSONDecodeError, OSError):
                    return {}
            return {}

        # Exp-1 table
        exp1 = load_summary("exp1_e2e/exp1_summary.json")
        exp1_rows = exp1_table(exp1)
        exp1_cols = ["metric", "n", "mean", "std", "min", "max"]
        sections.append("## Exp-1: E2E Quality (ROC-AUC / PR-AUC / F1)\n")
        sections.append(render_markdown(exp1_rows, columns=exp1_cols))

        # Exp-2 table
        exp2 = load_summary("exp2_ablation/exp2_summary.json")
        exp2_rows = exp2_table(exp2)
        exp2_cols = ["metric", "C0", "C1", "delta", "p_value", "significant"]
        sections.append("\n## Exp-2: C0 vs C1 Ablation\n")
        sections.append(render_markdown(exp2_rows, columns=exp2_cols))

        # Exp-3 table
        exp3 = load_summary("exp3_framework/exp3_summary.json")
        exp3_rows = exp3_table(exp3)
        exp3_cols = ["backend", "wall_s_mean", "wall_s_std", "delta_wall_s",
                     "rss_mb_mean", "delta_rss_mb", "registry", "run_id"]
        sections.append("\n## Exp-3: Framework Overhead + Governance\n")
        sections.append(render_markdown(exp3_rows, columns=exp3_cols))

        # Exp-4 table (only rendered when summary exists)
        exp4 = load_summary("exp4_drift/exp4_summary.json")
        if exp4:
            exp4_rows = exp4_table(exp4)
            exp4_cols = ["scenario", "drift_detected_any", "detection_latency_steps",
                         "retrain_on", "retrain_off"]
            sections.append("\n## Exp-4: Drift Detection + Closed-Loop Retrain (ON vs OFF)\n")
            sections.append(render_markdown(exp4_rows, columns=exp4_cols))

        # Exp-5 table (only rendered when summary exists)
        exp5 = load_summary("exp5_model_swap/exp5_summary.json")
        if exp5:
            exp5_rows = exp5_table(exp5)
            exp5_cols = ["model", "roc_auc_mean", "pr_auc_mean", "f1_mean",
                         "train_wall_s_mean", "n_ok", "swap_core_changes"]
            sections.append("\n## Exp-5: Model-Swap IForest vs LSTM-AE\n")
            sections.append(render_markdown(exp5_rows, columns=exp5_cols))

        # Exp-6 table (only rendered when summary exists)
        exp6 = load_summary("exp6_modifiability/exp6_summary.json")
        if exp6:
            exp6_rows = exp6_table(exp6)
            exp6_cols = ["mod_id", "section", "files_changed", "lines_changed",
                         "regression_count", "pass"]
            sections.append("\n## Exp-6: Modifiability (NWDAF Mods + Regression)\n")
            sections.append(render_markdown(exp6_rows, columns=exp6_cols))

        content = "\n".join(sections)

        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(content, encoding="utf-8")
            print(f"[tables] written to {output_file}")
        else:
            print(content)
        return 0
    except Exception as exc:
        print(f"[error] tables failed: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- assess (existing P8a subcommand) ----
    a = sub.add_parser("assess", help="Run the ML Test Score assessor over an evidence manifest.")
    a.add_argument("--manifest", required=True)
    a.add_argument("--repo-root", dest="repo_root", default=".")
    a.add_argument("--output", default=None)
    a.set_defaults(func=cmd_assess)

    # ---- run-exp1 ----
    e1 = sub.add_parser("run-exp1", help="Run Exp-1: E2E quality experiment (C1 pipeline).")
    e1.add_argument("--dataset", required=True, help="Path to labeled feature Parquet.")
    e1.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51",
                    help="Comma-separated seeds (default: 10 seeds for real run).")
    e1.add_argument("--output-root", dest="output_root", default="artifacts/experiments",
                    help="Base directory for outputs.")
    e1.add_argument("--backend", default="mlflow",
                    help="Tracking backend for C1 workload (default: mlflow).")
    e1.set_defaults(func=cmd_run_exp1)

    # ---- run-exp2 ----
    e2 = sub.add_parser("run-exp2", help="Run Exp-2: C0 vs C1 ablation (6 groups).")
    e2.add_argument("--dataset", required=True, help="Path to labeled feature Parquet.")
    e2.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51",
                    help="Comma-separated seeds.")
    e2.add_argument("--output-root", dest="output_root", default="artifacts/experiments",
                    help="Base directory for outputs.")
    e2.add_argument("--c1-tracking-uri", dest="c1_tracking_uri", default=None,
                    help="MLflow tracking URI for C1 (default: sqlite under output-root).")
    e2.set_defaults(func=cmd_run_exp2)

    # ---- run-exp3 ----
    e3 = sub.add_parser("run-exp3", help="Run Exp-3: framework overhead + governance comparison.")
    e3.add_argument("--dataset", required=True, help="Path to labeled feature Parquet.")
    e3.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51",
                    help="Comma-separated seeds.")
    e3.add_argument("--output-root", dest="output_root", default="artifacts/experiments",
                    help="Base directory for outputs.")
    e3.add_argument("--backends", default="noop,mlflow",
                    help="Comma-separated backend names to compare (default: noop,mlflow).")
    e3.set_defaults(func=cmd_run_exp3)

    # ---- run-exp4 ----
    e4 = sub.add_parser("run-exp4", help="Run Exp-4: drift scenario-clock + closed-loop retrain ON/OFF.")
    e4.add_argument("--baseline", required=True, help="Path to baseline feature Parquet.")
    e4.add_argument("--sudden-drift", dest="sudden_drift", default=None,
                    help="Path to sudden-drift feature Parquet.")
    e4.add_argument("--gradual-drift", dest="gradual_drift", default=None,
                    help="Path to gradual-drift feature Parquet.")
    e4.add_argument("--recurring-drift", dest="recurring_drift", default=None,
                    help="Path to recurring-drift feature Parquet.")
    e4.add_argument("--n-steps", dest="n_steps", default=6, type=int,
                    help="Number of scenario steps (default: 6).")
    e4.add_argument("--output-root", dest="output_root", default="artifacts/experiments",
                    help="Base directory for outputs.")
    e4.set_defaults(func=cmd_run_exp4)

    # ---- run-exp5 ----
    e5 = sub.add_parser("run-exp5", help="Run Exp-5: model-swap IForest vs LSTM-AE.")
    e5.add_argument("--dataset", required=True, help="Path to labeled feature Parquet.")
    e5.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51",
                    help="Comma-separated seeds (default: 10 seeds for real run).")
    e5.add_argument("--output-root", dest="output_root", default="artifacts/experiments",
                    help="Base directory for outputs.")
    e5.set_defaults(func=cmd_run_exp5)

    # ---- run-exp6 ----
    e6 = sub.add_parser("run-exp6", help="Run Exp-6: NWDAF modifiability (3 mods, diff + regression).")
    e6.add_argument("--output-root", dest="output_root", default="artifacts/experiments",
                    help="Base directory for outputs.")
    e6.set_defaults(func=cmd_run_exp6)

    # ---- tables ----
    t = sub.add_parser("tables", help="Generate Markdown result tables from experiment summaries.")
    t.add_argument("--output-root", dest="output_root", default="artifacts/experiments",
                   help="Directory containing exp{1,2,3}_summary.json files.")
    t.add_argument("--output", default=None,
                   help="Output .md file path (default: print to stdout).")
    t.set_defaults(func=cmd_tables)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
