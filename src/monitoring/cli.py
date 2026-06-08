"""Monitoring CLI: `python -m src.monitoring check ...` runs a one-shot drift check
over a serving predictions.jsonl against a training reference parquet."""
from __future__ import annotations

import argparse
import json

from src.monitoring.detector import DriftDetector
from src.monitoring.schema import MonitoringConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.monitoring")
    sub = parser.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("check", help="One-shot drift check (predictions vs reference)")
    sp.add_argument("--reference", required=True, help="Training reference feature parquet")
    sp.add_argument("--predictions", required=True, help="Serving predictions.jsonl")
    sp.add_argument("--min-features-alert", type=int, default=MonitoringConfig.min_features_alert)
    sp.add_argument("--min-observed-rows", type=int, default=MonitoringConfig.min_observed_rows)
    sp.set_defaults(func=cmd_check)
    rp = sub.add_parser("retrain", help="Run the drift->retrain->eval-gate->reload cycle (CT)")
    rp.add_argument("--predictions", required=True)
    rp.add_argument("--reference", required=True)
    rp.add_argument("--dataset", required=True)
    rp.add_argument("--output-dir", dest="output_dir", required=True)
    rp.add_argument("--model-type", dest="model_type", default="iforest", choices=["iforest", "lstm_ae"])
    rp.add_argument("--backend", default="noop")
    rp.set_defaults(func=cmd_retrain)
    return parser


def cmd_check(args: argparse.Namespace) -> None:
    config = MonitoringConfig(min_features_alert=args.min_features_alert,
                              min_observed_rows=args.min_observed_rows)
    result = DriftDetector(config).detect(args.reference, args.predictions)
    summary = {"drift_detected": result["drift_detected"],
               "evidently_drift_share": result["evidently_drift_share"],
               "alerted_features": result["psi"].get("alerted_features"),
               "observed_rows": result["observed_rows"]}
    print(json.dumps(summary, indent=2))


def cmd_retrain(args: argparse.Namespace) -> int:
    from pathlib import Path

    from src.monitoring.retrain import run_retrain_cycle
    out = run_retrain_cycle(
        predictions_path=Path(args.predictions), reference_path=Path(args.reference),
        dataset_path=Path(args.dataset), output_dir=Path(args.output_dir),
        model_type=args.model_type, backend=args.backend)
    print(f"[retrain] retrained={out['retrained']} "
          f"drift={out['drift']['drift_detected']} count={out.get('retrain_count')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    main()
