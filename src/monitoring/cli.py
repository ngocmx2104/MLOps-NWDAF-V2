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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
