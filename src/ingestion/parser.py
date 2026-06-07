"""Single source-of-truth EBS positional parser.

Parses raw EBS text files into a pandas DataFrame using the positional schema
defined in ``schema.py``.  Every row carries full provenance metadata
(source_file, line_number, raw_field_count).

This parser intentionally does NOT read ``EBS_fields_explained.xlsx`` for
column ordering.  The Excel file is a semantic dictionary only.

Reference: docs/EBS_SCHEMA_SPEC.md Section 6
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.ingestion.schema import (
    EXPECTED_FIELD_COUNT,
    FIELD_NAMES,
    SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)

SEPARATOR = ";"


def _parse_line(
    line: str,
    source_file: str,
    line_number: int,
) -> dict[str, object] | None:
    """Parse a single raw EBS line into a dict with provenance metadata.

    Returns ``None`` for blank lines.
    """
    stripped = line.rstrip("\n\r")
    if not stripped:
        return None

    parts = stripped.split(SEPARATOR)
    raw_field_count = len(parts)

    row: dict[str, object] = {}
    for i, name in enumerate(FIELD_NAMES):
        value = parts[i] if i < raw_field_count else ""
        row[name] = value if value != "" else None

    # Provenance columns
    row["source_file"] = source_file
    row["line_number"] = line_number
    row["raw_field_count"] = raw_field_count

    return row


def iter_parsed_rows(
    file_path: Path,
) -> Iterator[dict[str, object]]:
    """Yield parsed row dicts from a single raw EBS file.

    Each dict contains:
    - All 52 positional fields (empty values as ``None``)
    - ``source_file``: basename of the file
    - ``line_number``: 1-based line number within the file
    - ``raw_field_count``: actual number of semicolons + 1 in the raw line
    """
    source_file = file_path.name
    with file_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_number, line in enumerate(fh, start=1):
            row = _parse_line(line, source_file, line_number)
            if row is not None:
                yield row


def parse_ebs_files(
    file_paths: list[Path],
    *,
    warn_field_count: bool = True,
) -> pd.DataFrame:
    """Parse one or more raw EBS files into a single DataFrame.

    Parameters
    ----------
    file_paths:
        List of paths to raw EBS ``*_ebs`` text files.
    warn_field_count:
        If True, log a warning for lines with unexpected field counts.

    Returns
    -------
    pd.DataFrame
        DataFrame with 52 positional columns plus provenance columns.
    """
    all_rows: list[dict[str, object]] = []
    anomaly_count = 0

    for fp in file_paths:
        fp = Path(fp)
        if not fp.exists():
            logger.warning("EBS file not found, skipping: %s", fp)
            continue

        logger.info("Parsing EBS file: %s", fp.name)
        file_row_count = 0
        for row in iter_parsed_rows(fp):
            if warn_field_count and row["raw_field_count"] != EXPECTED_FIELD_COUNT:
                anomaly_count += 1
                if anomaly_count <= 5:
                    logger.warning(
                        "Unexpected field count %d (expected %d) at %s:%d",
                        row["raw_field_count"],
                        EXPECTED_FIELD_COUNT,
                        row["source_file"],
                        row["line_number"],
                    )
            all_rows.append(row)
            file_row_count += 1
        logger.info("  -> %d rows parsed from %s", file_row_count, fp.name)

    if anomaly_count > 5:
        logger.warning(
            "Total lines with unexpected field count: %d (only first 5 shown above)",
            anomaly_count,
        )

    if not all_rows:
        raise FileNotFoundError("No valid rows parsed from the given EBS files.")

    df = pd.DataFrame(all_rows)

    # Ensure schema_version column
    df["schema_version"] = SCHEMA_VERSION

    return df


def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ``event_time`` (epoch-ms string) to ``event_ts`` (UTC datetime).

    The original ``event_time`` column is preserved.  A new ``event_ts``
    column is added.

    Reference: EBS_SCHEMA_SPEC.md Section 8.2 (T3)
    """
    df = df.copy()
    df["event_time_ms"] = pd.to_numeric(df["event_time"], errors="coerce")
    df["event_ts"] = pd.to_datetime(
        df["event_time_ms"], unit="ms", errors="coerce", utc=True,
    )
    df.drop(columns=["event_time_ms"], inplace=True)
    return df
