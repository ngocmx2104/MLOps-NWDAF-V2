"""D1 Canonical Snapshot builder.

Builds the ``D1 canonical snapshot`` as specified in
docs/CANONICAL_SNAPSHOT_SPEC.md.  The snapshot is:

- parsed from raw EBS files via the positional parser
- filtered to ``event_id == 'l_handover'``
- enriched with provenance metadata
- timestamp-normalised
- saved as Parquet + JSON metadata sidecar

Reference: docs/CANONICAL_SNAPSHOT_SPEC.md
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.ingestion.parser import normalize_timestamps, parse_ebs_files
from src.ingestion.quality import (
    format_quality_report,
    quality_results_to_dicts,
    run_d1_quality_checks,
    run_raw_quality_checks,
)
from src.ingestion.schema import (
    D1_CORE_COLUMNS,
    D1_OPTIONAL_COLUMNS,
    D1_PROVENANCE_COLUMNS,
    SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)

PARSER_VERSION = "positional_parser_v1"
DEFAULT_EVENT_FILTER = "l_handover"


def _build_snapshot_id(source_files: list[Path]) -> str:
    """Generate a D1 snapshot ID from the source file metadata.

    Format: ``D1_EBS_HO_YYYYMMDD_HHMM_HHMM_V1``
    Falls back to a timestamp-based ID if file names are unusual.
    """
    try:
        # Extract date/time hints from file names like
        # A20240626.1400+0700-20240626.1403+0700_842_ebs
        names = sorted(f.name for f in source_files)
        first = names[0]
        last = names[-1]
        # Extract date (YYYYMMDD) and start time (HHMM) from first file
        date_part = first.split(".")[0][1:]  # "20240626"
        start_time = first.split(".")[1][:4]  # "1400"
        # Extract end time from last file
        end_segment = last.split("-")[1] if "-" in last else last
        end_time = end_segment.split(".")[1][:4] if "." in end_segment else "9999"
        return f"D1_EBS_HO_{date_part}_{start_time}_{end_time}_V1"
    except (IndexError, ValueError):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"D1_EBS_HO_{ts}_V1"


def _select_d1_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select columns for D1, keeping core + provenance + available optional."""
    columns_to_keep: list[str] = []
    for col in D1_CORE_COLUMNS:
        if col in df.columns:
            columns_to_keep.append(col)
    for col in D1_PROVENANCE_COLUMNS:
        if col in df.columns:
            columns_to_keep.append(col)
    for col in D1_OPTIONAL_COLUMNS:
        if col in df.columns:
            columns_to_keep.append(col)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_cols: list[str] = []
    for c in columns_to_keep:
        if c not in seen:
            unique_cols.append(c)
            seen.add(c)
    return df[unique_cols].copy()


def build_d1_snapshot(
    source_files: list[Path],
    output_dir: Path,
    *,
    event_filter: str = DEFAULT_EVENT_FILTER,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Build a D1 canonical snapshot from raw EBS files.

    Parameters
    ----------
    source_files:
        Paths to raw EBS ``*_ebs`` files.
    output_dir:
        Directory to write the Parquet data file and JSON metadata.
    event_filter:
        Event type to filter on (default ``l_handover``).
    snapshot_id:
        Override the auto-generated snapshot ID.

    Returns
    -------
    dict
        Summary including paths to output files, quality check results,
        row counts, and metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Parse raw EBS files
    # ------------------------------------------------------------------
    logger.info("Step 1/6: Parsing raw EBS files (%d files)", len(source_files))
    df_raw = parse_ebs_files(source_files)
    raw_row_count = len(df_raw)
    logger.info("  -> %d total raw rows parsed", raw_row_count)

    # ------------------------------------------------------------------
    # Step 2: Run raw quality checks
    # ------------------------------------------------------------------
    logger.info("Step 2/6: Running raw data quality checks")
    raw_qc = run_raw_quality_checks(df_raw)
    logger.info(format_quality_report(raw_qc))

    # ------------------------------------------------------------------
    # Step 3: Normalize timestamps
    # ------------------------------------------------------------------
    logger.info("Step 3/6: Normalizing timestamps (event_time -> event_ts)")
    df_raw = normalize_timestamps(df_raw)

    # ------------------------------------------------------------------
    # Step 4: Filter to modeling events
    # ------------------------------------------------------------------
    logger.info("Step 4/6: Filtering to event_id == '%s'", event_filter)
    if event_filter and "event_id" in df_raw.columns:
        df_filtered = df_raw[df_raw["event_id"] == event_filter].copy()
    else:
        df_filtered = df_raw.copy()
    filtered_row_count = len(df_filtered)
    logger.info("  -> %d rows after filter", filtered_row_count)

    # ------------------------------------------------------------------
    # Step 5: Attach snapshot metadata
    # ------------------------------------------------------------------
    logger.info("Step 5/6: Attaching snapshot metadata")
    sid = snapshot_id or _build_snapshot_id(source_files)
    df_filtered["dataset_snapshot_id"] = sid
    # record_index: position within the filtered snapshot
    df_filtered["record_index"] = range(len(df_filtered))

    # Select D1 columns
    df_d1 = _select_d1_columns(df_filtered)

    # ------------------------------------------------------------------
    # Step 6: Run D1 quality checks and write outputs
    # ------------------------------------------------------------------
    logger.info("Step 6/6: Running D1 quality checks and writing outputs")
    d1_qc = run_d1_quality_checks(df_d1)
    logger.info(format_quality_report(d1_qc))

    # Null profile for key columns
    null_profile: dict[str, int] = {}
    for col in df_d1.columns:
        null_count = int(df_d1[col].isna().sum())
        if null_count > 0:
            null_profile[col] = null_count

    # Write Parquet
    parquet_path = output_dir / f"{sid}.parquet"
    df_d1.to_parquet(parquet_path, index=False)
    logger.info("D1 Parquet written: %s (%d rows)", parquet_path, len(df_d1))

    # Build metadata
    created_at = datetime.now(timezone.utc).isoformat()
    metadata: dict[str, Any] = {
        "dataset_snapshot_id": sid,
        "created_at": created_at,
        "created_by_process": "src.ingestion.snapshot.build_d1_snapshot",
        "parser_version": PARSER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "source_files": [f.name for f in source_files],
        "event_filter": event_filter,
        "raw_row_count": raw_row_count,
        "filtered_row_count": filtered_row_count,
        "columns": list(df_d1.columns),
        "null_profile": null_profile,
        "raw_quality_checks": quality_results_to_dicts(raw_qc),
        "d1_quality_checks": quality_results_to_dicts(d1_qc),
    }

    # Write JSON metadata sidecar
    meta_path = output_dir / f"{sid}_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    logger.info("D1 metadata written: %s", meta_path)

    all_passed = all(r.passed for r in raw_qc + d1_qc)

    return {
        "snapshot_id": sid,
        "parquet_path": str(parquet_path),
        "metadata_path": str(meta_path),
        "raw_row_count": raw_row_count,
        "d1_row_count": len(df_d1),
        "all_quality_checks_passed": all_passed,
        "metadata": metadata,
    }
