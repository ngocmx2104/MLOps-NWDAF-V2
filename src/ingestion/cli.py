"""CLI entry point for Phase 1 ingestion flow.

Usage examples::

    # Build D1 canonical snapshot from default EBS files
    python -m src.ingestion.cli build-snapshot

    # Build from custom files to a custom output directory
    python -m src.ingestion.cli build-snapshot \\
        --output-dir artifacts/snapshots \\
        path/to/file1_ebs path/to/file2_ebs

    # Parse raw EBS without filtering (all event types)
    python -m src.ingestion.cli parse-raw --no-filter --out raw_all.parquet

    # Run quality checks on an existing parquet file
    python -m src.ingestion.cli check-quality artifacts/snapshots/D1_*.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so ``src.`` imports work
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.ingestion.parser import normalize_timestamps, parse_ebs_files  # noqa: E402
from src.ingestion.quality import (  # noqa: E402
    format_quality_report,
    run_d1_quality_checks,
    run_raw_quality_checks,
)
from src.ingestion.snapshot import build_d1_snapshot  # noqa: E402

logger = logging.getLogger("src.ingestion")

# Default raw EBS file locations (relative to project root)
_DATA_DIR = _project_root / "data" / "raw_ebs"
DEFAULT_EBS_FILES = sorted(_DATA_DIR.glob("*_ebs"))


def cmd_build_snapshot(args: argparse.Namespace) -> None:
    """Build the D1 canonical snapshot."""
    ebs_files = [Path(p) for p in args.ebs_files] if args.ebs_files else DEFAULT_EBS_FILES
    output_dir = Path(args.output_dir)

    result = build_d1_snapshot(
        source_files=ebs_files,
        output_dir=output_dir,
        event_filter=args.event_filter,
        snapshot_id=args.snapshot_id,
    )

    print("\n" + "=" * 60)
    print("D1 Canonical Snapshot Build Complete")
    print("=" * 60)
    print(f"  Snapshot ID:    {result['snapshot_id']}")
    print(f"  Raw rows:       {result['raw_row_count']}")
    print(f"  D1 rows:        {result['d1_row_count']}")
    print(f"  Parquet:        {result['parquet_path']}")
    print(f"  Metadata:       {result['metadata_path']}")
    print(f"  Quality OK:     {result['all_quality_checks_passed']}")
    print("=" * 60)


def cmd_parse_raw(args: argparse.Namespace) -> None:
    """Parse raw EBS files to Parquet (without D1 filtering/metadata)."""
    ebs_files = [Path(p) for p in args.ebs_files] if args.ebs_files else DEFAULT_EBS_FILES
    out_path = Path(args.out)

    df = parse_ebs_files(ebs_files)
    df = normalize_timestamps(df)

    if not args.no_filter:
        df = df[df["event_id"] == "l_handover"].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Written {len(df)} rows to {out_path}")


def cmd_check_quality(args: argparse.Namespace) -> None:
    """Run quality checks on an existing Parquet file."""
    import pandas as pd

    path = Path(args.parquet_file)
    df = pd.read_parquet(path)

    print(f"\nRunning quality checks on: {path}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}\n")

    # Determine which checks to run based on available columns
    has_d1_cols = "dataset_snapshot_id" in df.columns and "event_ts" in df.columns

    if has_d1_cols:
        results = run_d1_quality_checks(df)
    else:
        results = run_raw_quality_checks(df)

    print(format_quality_report(results))

    if args.json_out:
        from src.ingestion.quality import quality_results_to_dicts
        json_path = Path(args.json_out)
        with json_path.open("w") as f:
            json.dump(quality_results_to_dicts(results), f, indent=2)
        print(f"Quality report written to: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.ingestion.cli",
        description="Phase 1 EBS ingestion CLI -- parse, validate, and build D1 canonical snapshots.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- build-snapshot ---
    sp_snap = subparsers.add_parser(
        "build-snapshot",
        help="Build the D1 canonical snapshot from raw EBS files.",
    )
    sp_snap.add_argument(
        "ebs_files", nargs="*",
        help="Paths to raw EBS files. If empty, uses default files in repo.",
    )
    sp_snap.add_argument(
        "--output-dir", default=str(_project_root / "artifacts" / "snapshots"),
        help="Directory for output Parquet + metadata JSON.",
    )
    sp_snap.add_argument(
        "--event-filter", default="l_handover",
        help="Event type to filter (default: l_handover).",
    )
    sp_snap.add_argument(
        "--snapshot-id", default=None,
        help="Override the auto-generated snapshot ID.",
    )
    sp_snap.set_defaults(func=cmd_build_snapshot)

    # --- parse-raw ---
    sp_raw = subparsers.add_parser(
        "parse-raw",
        help="Parse raw EBS files to Parquet (no D1 metadata).",
    )
    sp_raw.add_argument(
        "ebs_files", nargs="*",
        help="Paths to raw EBS files.",
    )
    sp_raw.add_argument(
        "--out", default=str(_project_root / "artifacts" / "raw_parsed.parquet"),
        help="Output Parquet path.",
    )
    sp_raw.add_argument(
        "--no-filter", action="store_true",
        help="Do not filter by event_id.",
    )
    sp_raw.set_defaults(func=cmd_parse_raw)

    # --- check-quality ---
    sp_qc = subparsers.add_parser(
        "check-quality",
        help="Run quality checks on an existing Parquet file.",
    )
    sp_qc.add_argument(
        "parquet_file",
        help="Path to a Parquet file to check.",
    )
    sp_qc.add_argument(
        "--json-out", default=None,
        help="Write quality report as JSON to this path.",
    )
    sp_qc.set_defaults(func=cmd_check_quality)

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
