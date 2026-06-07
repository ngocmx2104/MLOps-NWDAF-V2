"""D2 Feature Dataset builder.

Transforms a D1 canonical snapshot into a versioned D2 feature dataset
suitable for anomaly-detection training on subscriber handover behaviour.

The feature logic is adapted from the old prototype
(``nwdaf_mlops/features/pingpong_features.py``) but restructured for
versioning, provenance, and thesis-grade traceability.

Reference: docs/DATASET_STRATEGY.md (D2), IMPLEMENTATION_ROADMAP.md Phase 2
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.quality import run_d2_quality_checks
from src.features.schema import (
    D2_NUMERIC_FEATURE_NAMES,
    FEATURE_VERSION,
    WindowConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure feature computation helpers
# ---------------------------------------------------------------------------

def _entropy(values: list[str]) -> float:
    """Shannon entropy of a discrete sequence."""
    if not values:
        return 0.0
    s = pd.Series(values, dtype="string")
    p = s.value_counts(normalize=True)
    return float(-(p * np.log2(p)).sum())


def compute_ue_window_features(
    df_d1: pd.DataFrame,
    cfg: WindowConfig,
) -> pd.DataFrame:
    """Compute per-(subscriber, window) features from D1 handover records.

    Parameters
    ----------
    df_d1 : pd.DataFrame
        D1 canonical snapshot with at least ``imsi``, ``event_ts``, and
        ``eci`` (or ``ci``) columns.
    cfg : WindowConfig
        Aggregation window parameters.

    Returns
    -------
    pd.DataFrame
        One row per (imsi, window_start) with handover behaviour features.
    """
    df = df_d1.copy()

    # Validate required columns
    if "event_ts" not in df.columns:
        raise ValueError("Missing column event_ts -- run ingestion first.")
    if "imsi" not in df.columns:
        raise ValueError("Missing column imsi.")

    cell_col = "eci" if "eci" in df.columns else ("ci" if "ci" in df.columns else None)
    if cell_col is None:
        raise ValueError("Missing cell identity column (eci or ci).")

    df[cell_col] = df[cell_col].astype("string")
    df = df.dropna(subset=["event_ts", "imsi", cell_col]).sort_values(["imsi", "event_ts"])

    # Assign fixed windows
    window_td = pd.to_timedelta(cfg.window_seconds, unit="s")  # noqa: F841
    df["window_start"] = df["event_ts"].dt.floor(f"{cfg.window_seconds}s").astype("datetime64[ns, UTC]")

    rows: list[dict[str, Any]] = []
    for (imsi, w), g in df.groupby(["imsi", "window_start"], sort=False):
        g_sorted = g.sort_values("event_ts")
        seq = g_sorted[cell_col].tolist()
        ts = g_sorted["event_ts"].tolist()

        n_ho = len(seq)
        n_unique = len(set(seq))

        # Ping-pong: A->B->A within gap threshold
        pingpong = 0
        gaps: list[float] = []
        for i in range(1, len(ts)):
            gaps.append((ts[i] - ts[i - 1]).total_seconds())

        for i in range(2, len(seq)):
            if seq[i] == seq[i - 2] and seq[i] != seq[i - 1]:
                gap_prev = (ts[i - 1] - ts[i - 2]).total_seconds()
                gap_next = (ts[i] - ts[i - 1]).total_seconds()
                if (gap_prev <= cfg.pingpong_max_gap_seconds
                        and gap_next <= cfg.pingpong_max_gap_seconds):
                    pingpong += 1

        gaps_arr = np.array(gaps, dtype=float) if gaps else np.array([], dtype=float)
        rows.append({
            "imsi": imsi,
            "window_start": w,
            "n_handover": n_ho,
            "n_unique_cells": n_unique,
            "pingpong_count": pingpong,
            "pingpong_rate": pingpong / max(1, n_ho),
            "mean_inter_ho_s": float(gaps_arr.mean()) if gaps_arr.size else np.nan,
            "std_inter_ho_s": float(gaps_arr.std()) if gaps_arr.size else np.nan,
            "entropy_cell_seq": _entropy(seq),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# D2 snapshot builder (with provenance + metadata sidecar)
# ---------------------------------------------------------------------------

def _build_d2_id(source_snapshot_id: str) -> str:
    """Generate a D2 dataset ID from its D1 source."""
    # Replace D1 prefix with D2
    base = source_snapshot_id.replace("D1_", "D2_FE_", 1)
    return base


def build_d2_feature_dataset(
    d1_parquet_path: Path,
    output_dir: Path,
    *,
    window_config: WindowConfig | None = None,
    feature_version: str = FEATURE_VERSION,
) -> dict[str, Any]:
    """Build a versioned D2 feature dataset from a D1 canonical snapshot.

    Parameters
    ----------
    d1_parquet_path : Path
        Path to D1 Parquet file.
    output_dir : Path
        Where to write D2 Parquet + metadata JSON.
    window_config : WindowConfig, optional
        Override default aggregation window parameters.
    feature_version : str
        Feature schema version tag.

    Returns
    -------
    dict
        Summary with paths, row counts, quality checks, metadata.
    """
    d1_path = Path(d1_parquet_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = window_config or WindowConfig()

    # ------------------------------------------------------------------
    # Step 1: Load D1
    # ------------------------------------------------------------------
    logger.info("Step 1/5: Loading D1 snapshot from %s", d1_path)
    df_d1 = pd.read_parquet(d1_path)
    d1_row_count = len(df_d1)
    logger.info("  -> %d D1 rows loaded", d1_row_count)

    # Extract source snapshot ID from D1 data
    source_snapshot_id = "unknown"
    if "dataset_snapshot_id" in df_d1.columns:
        ids = df_d1["dataset_snapshot_id"].dropna().unique()
        if len(ids) > 0:
            source_snapshot_id = str(ids[0])

    # Also try reading from D1 metadata sidecar
    d1_meta_path = d1_path.with_name(d1_path.stem + "_metadata.json")
    if d1_meta_path.exists():
        with d1_meta_path.open() as f:
            d1_meta = json.load(f)
            source_snapshot_id = d1_meta.get("dataset_snapshot_id", source_snapshot_id)

    # ------------------------------------------------------------------
    # Step 2: Compute features
    # ------------------------------------------------------------------
    logger.info("Step 2/5: Computing UE-window features (window=%ds, gap=%ds)",
                cfg.window_seconds, cfg.pingpong_max_gap_seconds)
    df_feat = compute_ue_window_features(df_d1, cfg)
    logger.info("  -> %d feature rows produced", len(df_feat))

    # ------------------------------------------------------------------
    # Step 3: Attach provenance
    # ------------------------------------------------------------------
    logger.info("Step 3/5: Attaching provenance metadata")
    df_feat["feature_version"] = feature_version
    df_feat["source_snapshot_id"] = source_snapshot_id

    # ------------------------------------------------------------------
    # Step 4: Quality checks
    # ------------------------------------------------------------------
    logger.info("Step 4/5: Running D2 quality checks")
    from src.features.quality import format_feature_quality_report
    d2_qc = run_d2_quality_checks(df_feat)
    logger.info(format_feature_quality_report(d2_qc))

    # ------------------------------------------------------------------
    # Step 5: Write outputs
    # ------------------------------------------------------------------
    d2_id = _build_d2_id(source_snapshot_id)
    parquet_path = output_dir / f"{d2_id}.parquet"
    df_feat.to_parquet(parquet_path, index=False)
    logger.info("D2 Parquet written: %s (%d rows)", parquet_path, len(df_feat))

    # Feature distribution summary
    feat_stats: dict[str, Any] = {}
    for col in D2_NUMERIC_FEATURE_NAMES:
        if col in df_feat.columns:
            s = df_feat[col].dropna()
            feat_stats[col] = {
                "count": int(s.count()),
                "mean": round(float(s.mean()), 4) if len(s) > 0 else None,
                "std": round(float(s.std()), 4) if len(s) > 0 else None,
                "min": round(float(s.min()), 4) if len(s) > 0 else None,
                "max": round(float(s.max()), 4) if len(s) > 0 else None,
            }

    created_at = datetime.now(timezone.utc).isoformat()
    metadata: dict[str, Any] = {
        "dataset_id": d2_id,
        "dataset_layer": "D2",
        "description": "Versioned feature dataset for handover anomaly detection",
        "created_at": created_at,
        "created_by_process": "src.features.builder.build_d2_feature_dataset",
        "feature_version": feature_version,
        "source_snapshot_id": source_snapshot_id,
        "d1_parquet_path": str(d1_path),
        "window_config": cfg.to_dict(),
        "d1_row_count": d1_row_count,
        "d2_row_count": len(df_feat),
        "columns": list(df_feat.columns),
        "feature_statistics": feat_stats,
        "quality_checks": [
            {"check_id": r.check_id, "description": r.description,
             "passed": r.passed, "detail": r.detail}
            for r in d2_qc
        ],
    }

    meta_path = output_dir / f"{d2_id}_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    logger.info("D2 metadata written: %s", meta_path)

    all_passed = all(r.passed for r in d2_qc)

    return {
        "d2_id": d2_id,
        "parquet_path": str(parquet_path),
        "metadata_path": str(meta_path),
        "d1_row_count": d1_row_count,
        "d2_row_count": len(df_feat),
        "feature_version": feature_version,
        "source_snapshot_id": source_snapshot_id,
        "all_quality_checks_passed": all_passed,
        "metadata": metadata,
    }
