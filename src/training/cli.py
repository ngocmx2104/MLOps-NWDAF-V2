"""CLI entry point for Phase 4 training flow.

Usage examples::

    # Train IsolationForest with noop tracker (C0 baseline)
    python -m src.training train --dataset artifacts/features/features.parquet

    # Train with MLflow tracker (C1)
    python -m src.training train --dataset artifacts/features/features.parquet --backend mlflow

    # Train LSTM-AE (model-swap, same pipeline -- RQ4)
    python -m src.training train --dataset artifacts/features/features.parquet --model-type lstm_ae

    # Specify output directory and seed
    python -m src.training train --dataset features.parquet --output-dir artifacts/models --seed 42
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from src.training.pipeline import run_training
from src.training.schema import TrainingConfig

logger = logging.getLogger("src.training")

# Project root (used for default paths only)
_project_root = Path(__file__).resolve().parents[2]

# Default random_state from the dataclass (read from class default, not hardcoded)
_DEFAULT_SEED = TrainingConfig.random_state  # type: ignore[attr-defined]  # frozen dataclass default


def cmd_train(args: argparse.Namespace) -> None:
    """Run training pipeline with selected backend and model type."""
    backend = args.backend or os.environ.get("MLOPS_BACKEND") or None
    cfg = TrainingConfig(
        random_state=args.seed,
        use_labels_for_evaluation=True,
    )

    result = run_training(
        dataset_path=Path(args.dataset),
        model_type=args.model_type,
        backend=backend,
        output_dir=Path(args.output_dir),
        cfg=cfg,
    )

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"  Backend:       {result['backend']}")
    print(f"  Model type:    {result['model_type']}")
    print(f"  Run ID:        {result['run_id']}")
    print(f"  Model path:    {result['model_path']}")
    print(f"  Model version: {result['model_version']}")
    print(f"  Train seconds: {result['train_seconds']:.3f}s")
    print("  Metrics:")
    for k, v in result["metrics"].items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.training",
        description="Phase 4 training CLI -- train, evaluate, and register models.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    sp_train = subparsers.add_parser(
        "train",
        help="Train a model and (optionally) register it via the configured tracker.",
    )
    sp_train.add_argument(
        "--dataset", required=True,
        help="Path to the feature Parquet file.",
    )
    sp_train.add_argument(
        "--model-type", dest="model_type", choices=["iforest", "lstm_ae"], default="iforest",
        help="Model type to train (default: iforest).",
    )
    sp_train.add_argument(
        "--backend", default=None,
        help=(
            "Tracking backend: noop (C0) / mlflow (C1) / clearml. "
            "Reads MLOPS_BACKEND env var if not set (default: noop)."
        ),
    )
    sp_train.add_argument(
        "--output-dir", dest="output_dir",
        default=str(_project_root / "artifacts" / "models"),
        help="Directory for saved model artifacts (default: artifacts/models).",
    )
    sp_train.add_argument(
        "--seed", type=int, default=_DEFAULT_SEED,
        help=f"Random seed (default: {_DEFAULT_SEED}, matches TrainingConfig.random_state).",
    )
    sp_train.set_defaults(func=cmd_train)

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args.func(args)


if __name__ == "__main__":
    main()
