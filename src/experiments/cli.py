from __future__ import annotations

import argparse
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
        print(f"[error] cannot load manifest: expected a mapping, got {type(manifest).__name__}", file=sys.stderr)
        return 1
    report = assess(manifest, repo_root=Path(args.repo_root))
    if args.output:
        write_json(Path(args.output), report)
    print(f"[ml-test-score] {report['ml_test_score']} "
          f"(sections={report['section_scores']}, google L{report['google_level']})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("assess", help="Run the ML Test Score assessor over an evidence manifest.")
    a.add_argument("--manifest", required=True)
    a.add_argument("--repo-root", dest="repo_root", default=".")
    a.add_argument("--output", default=None)
    a.set_defaults(func=cmd_assess)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
