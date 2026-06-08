"""CI/CD CLI: `python -m src.cicd gate ...` (eval-gate, exit 1 blocks deploy) and
`python -m src.cicd fixture ...` (self-contained synthetic CI dataset)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.cicd.eval_gate import run_eval_gate
from src.cicd.schema import GateConfig
from src.tracking import create_tracker


def cmd_gate(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.metrics).read_text())
    cfg = GateConfig(min_roc_auc=args.min_roc_auc, min_pr_auc=args.min_pr_auc)
    res = run_eval_gate(metrics=data.get("metrics", data), model_name=data.get("model_name"),
                        model_version=data.get("model_version"),
                        tracker=create_tracker(args.backend), cfg=cfg)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(res.to_dict(), indent=2))
    status = "PASS" if res.passed else "FAIL"
    print(f"[eval-gate] {status} roc_auc={res.metrics['roc_auc']:.4f} "
          f"reasons={res.reasons} promoted_to={res.promoted_to}")
    return 0 if res.passed else 1


def cmd_fixture(args: argparse.Namespace) -> int:
    from src.cicd.fixture import make_fixture_dataset
    out = make_fixture_dataset(Path(args.output), n=args.rows, seed=args.seed)
    print(f"[fixture] wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cicd")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gate", help="Eval-gate: exit 0 if model passes, 1 if blocked.")
    g.add_argument("--metrics", required=True, help="training_result.json path")
    g.add_argument("--backend", default="noop")
    g.add_argument("--min-roc-auc", dest="min_roc_auc", type=float, default=0.70)
    g.add_argument("--min-pr-auc", dest="min_pr_auc", type=float, default=0.0)
    g.add_argument("--output", default=None, help="write gate_result.json here")
    g.set_defaults(func=cmd_gate)

    f = sub.add_parser("fixture", help="Generate a small synthetic labeled dataset for CI.")
    f.add_argument("--output", required=True)
    f.add_argument("--rows", type=int, default=300)
    f.add_argument("--seed", type=int, default=0)
    f.set_defaults(func=cmd_fixture)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
