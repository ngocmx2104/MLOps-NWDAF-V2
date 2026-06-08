"""Serving CLI: `python -m src.serving serve ...` runs the FastAPI app via uvicorn.

Config resolves from CLI args, then env vars (MLOPS_SERVING_*), then defaults — so a
Docker container can be configured entirely through the environment.
"""
from __future__ import annotations

import argparse
import os

from src.serving.schema import DEFAULT_HOST, DEFAULT_OUTPUT_ROOT, DEFAULT_PORT, ServingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.serving")
    sub = parser.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("serve", help="Run the FastAPI serving app")
    sp.add_argument("--model-type", default=os.environ.get("MLOPS_SERVING_MODEL_TYPE", "iforest"),
                    choices=["iforest", "lstm_ae"])
    sp.add_argument("--loader", default=os.environ.get("MLOPS_SERVING_LOADER", "path"),
                    choices=["path", "registry"])
    sp.add_argument("--model-path", default=os.environ.get("MLOPS_SERVING_MODEL_PATH"))
    sp.add_argument("--registry-name", default=os.environ.get("MLOPS_SERVING_REGISTRY_NAME"))
    sp.add_argument("--registry-alias", default=os.environ.get("MLOPS_SERVING_REGISTRY_ALIAS", "staging"))
    sp.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    sp.add_argument("--feast-repo-path", default=os.environ.get("MLOPS_SERVING_FEAST_REPO"))
    sp.add_argument("--output-root", default=os.environ.get("MLOPS_SERVING_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
    sp.add_argument("--host", default=os.environ.get("MLOPS_SERVING_HOST", DEFAULT_HOST))
    sp.add_argument("--port", type=int, default=int(os.environ.get("MLOPS_SERVING_PORT", DEFAULT_PORT)))
    sp.set_defaults(func=_cmd_serve)
    return parser


def build_config_from_args(args: argparse.Namespace) -> ServingConfig:
    return ServingConfig(
        model_type=args.model_type, loader=args.loader, model_path=args.model_path,
        registry_name=args.registry_name, registry_alias=args.registry_alias,
        tracking_uri=args.tracking_uri, feast_repo_path=args.feast_repo_path,
        output_root=args.output_root, host=args.host, port=args.port)


def _cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from src.monitoring.metrics import add_metrics
    from src.serving.app import create_app
    from src.serving.runtime import ServingRuntime
    config = build_config_from_args(args)
    app = add_metrics(create_app(ServingRuntime.build(config)))
    uvicorn.run(app, host=config.host, port=config.port)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
