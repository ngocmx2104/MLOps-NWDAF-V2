"""CLI entry point for Phase 3 feature pipeline.

Usage examples::

    # Build D2 feature dataset from default D1 snapshot
    python -m src.features.cli build-d2

    # Build D2 from a specific D1 parquet
    python -m src.features.cli build-d2 --d1 artifacts/snapshots/D1_*.parquet \\
        --output artifacts/features

    # Apply weak labels from default D2 feature dataset
    python -m src.features.cli weak-label

    # Apply weak labels from a specific D2 parquet
    python -m src.features.cli weak-label --d2 artifacts/features/D2_*.parquet \\
        --output artifacts/features_labeled
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.features.builder import build_d2_feature_dataset
from src.features.weak_labels import build_d5_weak_label_dataset

logger = logging.getLogger("src.features")

# Project root (for default paths only — not sys.path manipulation)
_project_root = Path(__file__).resolve().parents[2]

_DEFAULT_SNAPSHOTS_DIR = _project_root / "artifacts" / "snapshots"
_DEFAULT_FEATURES_DIR = _project_root / "artifacts" / "features"
_DEFAULT_LABELED_DIR = _project_root / "artifacts" / "features_labeled"


def cmd_build_d2(args: argparse.Namespace) -> None:
    """Build the D2 feature dataset from a D1 canonical snapshot."""
    if args.d1:
        d1_path = Path(args.d1)
    else:
        candidates = sorted(_DEFAULT_SNAPSHOTS_DIR.glob("D1_*.parquet"))
        if not candidates:
            raise FileNotFoundError(
                f"No D1_*.parquet found in {_DEFAULT_SNAPSHOTS_DIR}. "
                "Run `dvc repro snapshot` first, or pass --d1 <path>."
            )
        d1_path = candidates[0]
        logger.info("Using D1 snapshot: %s", d1_path)

    output_dir = Path(args.output)

    result = build_d2_feature_dataset(d1_path, output_dir)

    print("\n" + "=" * 60)
    print("D2 Feature Dataset Build Complete")
    print("=" * 60)
    print(f"  D2 ID:          {result['d2_id']}")
    print(f"  D1 rows:        {result['d1_row_count']}")
    print(f"  D2 rows:        {result['d2_row_count']}")
    print(f"  Feature ver:    {result['feature_version']}")
    print(f"  Source snap:    {result['source_snapshot_id']}")
    print(f"  Parquet:        {result['parquet_path']}")
    print(f"  Metadata:       {result['metadata_path']}")
    print(f"  Quality OK:     {result['all_quality_checks_passed']}")
    print("=" * 60)


def cmd_weak_label(args: argparse.Namespace) -> None:
    """Apply weak labels to a D2 feature dataset to produce D5."""
    if args.d2:
        d2_path = Path(args.d2)
    else:
        candidates = sorted(_DEFAULT_FEATURES_DIR.glob("D2_*.parquet"))
        if not candidates:
            raise FileNotFoundError(
                f"No D2_*.parquet found in {_DEFAULT_FEATURES_DIR}. "
                "Run `build-d2` first, or pass --d2 <path>."
            )
        d2_path = candidates[0]
        logger.info("Using D2 feature dataset: %s", d2_path)

    output_dir = Path(args.output)

    result = build_d5_weak_label_dataset(d2_path, output_dir)

    print("\n" + "=" * 60)
    print("D5 Weak-Label Dataset Build Complete")
    print("=" * 60)
    print(f"  D5 ID:          {result['d5_id']}")
    print(f"  D2 rows:        {result['d2_row_count']}")
    print(f"  D5 rows:        {result['d5_row_count']}")
    print(f"  Positive (1):   {result['n_positive']}")
    print(f"  Negative (0):   {result['n_negative']}")
    print(f"  Positive rate:  {result['positive_rate_pct']}%")
    print(f"  Parquet:        {result['parquet_path']}")
    print(f"  Metadata:       {result['metadata_path']}")
    print(f"  Quality OK:     {result['all_quality_checks_passed']}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.features.cli",
        description=(
            "Phase 3 feature pipeline CLI -- build D2 feature datasets and "
            "D5 weak-label support datasets from D1 canonical snapshots."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- build-d2 ---
    sp_d2 = subparsers.add_parser(
        "build-d2",
        help="Build D2 feature dataset from a D1 canonical snapshot.",
    )
    sp_d2.add_argument(
        "--d1", default=None,
        help=(
            "Path to D1 Parquet file. "
            "If omitted, uses the first D1_*.parquet in artifacts/snapshots/."
        ),
    )
    sp_d2.add_argument(
        "--output", default=str(_DEFAULT_FEATURES_DIR),
        help="Output directory for D2 Parquet + metadata (default: artifacts/features).",
    )
    sp_d2.set_defaults(func=cmd_build_d2)

    # --- weak-label ---
    sp_wl = subparsers.add_parser(
        "weak-label",
        help="Apply rule-based weak labels to a D2 dataset, producing D5.",
    )
    sp_wl.add_argument(
        "--d2", default=None,
        help=(
            "Path to D2 Parquet file. "
            "If omitted, uses the first D2_*.parquet in artifacts/features/."
        ),
    )
    sp_wl.add_argument(
        "--output", default=str(_DEFAULT_LABELED_DIR),
        help=(
            "Output directory for D5 Parquet + metadata "
            "(default: artifacts/features_labeled)."
        ),
    )
    sp_wl.set_defaults(func=cmd_weak_label)

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
