"""D5 Weak-Label Support Dataset builder.

Creates rule-derived weak labels from the D2 feature dataset. These labels
are **not ground truth** -- they are supporting evaluation artifacts used for
internal model-level signal assessment.

The labeling logic is adapted from the old prototype
(``nwdaf_mlops/labeling/rule_label.py``) and documented as weak supervision
per DATASET_STRATEGY.md (D5).

IMPORTANT: Weak labels must always be presented as supporting / heuristic
signals, never as authoritative ground truth.

Reference: docs/DATASET_STRATEGY.md (D5), USE_CASE_LOCK.md Section 4.1
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.quality import run_d5_quality_checks
from src.features.schema import (
    WEAK_LABEL_VERSION,
    WeakLabelConfig,
)

logger = logging.getLogger(__name__)


def apply_weak_labels(
    df_d2: pd.DataFrame,
    cfg: WeakLabelConfig,
) -> pd.DataFrame:
    """Apply rule-based weak labels to a D2 feature dataset.

    Parameters
    ----------
    df_d2 : pd.DataFrame
        D2 feature dataset with at least ``pingpong_count`` and ``n_handover``.
    cfg : WeakLabelConfig
        Thresholds for the weak labeling rule.

    Returns
    -------
    pd.DataFrame
        Copy of input with ``weak_label`` column appended.
        1 = anomalous (ping-pong behaviour detected), 0 = normal.
    """
    df = df_d2.copy()
    pp = pd.to_numeric(df["pingpong_count"], errors="coerce").fillna(0)
    nho = pd.to_numeric(df["n_handover"], errors="coerce").fillna(0)
    df["weak_label"] = ((pp >= cfg.min_pingpong_count) & (nho >= cfg.min_handover)).astype(int)
    return df


def _build_d5_id(source_d2_id: str) -> str:
    """Generate a D5 dataset ID from its D2 source."""
    return source_d2_id.replace("D2_FE_", "D5_WL_", 1)


def build_d5_weak_label_dataset(
    d2_parquet_path: Path,
    output_dir: Path,
    *,
    weak_label_config: WeakLabelConfig | None = None,
    weak_label_version: str = WEAK_LABEL_VERSION,
) -> dict[str, Any]:
    """Build a versioned D5 weak-label support dataset from D2.

    Parameters
    ----------
    d2_parquet_path : Path
        Path to D2 Parquet file.
    output_dir : Path
        Where to write D5 Parquet + metadata JSON.
    weak_label_config : WeakLabelConfig, optional
        Override default thresholds.
    weak_label_version : str
        Weak label version tag.

    Returns
    -------
    dict
        Summary with paths, row counts, label distribution, quality checks.
    """
    d2_path = Path(d2_parquet_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = weak_label_config or WeakLabelConfig()

    # ------------------------------------------------------------------
    # Step 1: Load D2
    # ------------------------------------------------------------------
    logger.info("Step 1/4: Loading D2 feature dataset from %s", d2_path)
    df_d2 = pd.read_parquet(d2_path)
    d2_row_count = len(df_d2)
    logger.info("  -> %d D2 rows loaded", d2_row_count)

    # Extract lineage from D2
    source_feature_version = "unknown"
    source_snapshot_id = "unknown"
    if "feature_version" in df_d2.columns:
        vals = df_d2["feature_version"].dropna().unique()
        if len(vals) > 0:
            source_feature_version = str(vals[0])
    if "source_snapshot_id" in df_d2.columns:
        vals = df_d2["source_snapshot_id"].dropna().unique()
        if len(vals) > 0:
            source_snapshot_id = str(vals[0])

    # Also try D2 metadata sidecar
    d2_meta_path = d2_path.with_name(d2_path.stem + "_metadata.json")
    if d2_meta_path.exists():
        with d2_meta_path.open() as f:
            d2_meta = json.load(f)
            source_feature_version = d2_meta.get("feature_version", source_feature_version)
            source_snapshot_id = d2_meta.get("source_snapshot_id", source_snapshot_id)
            d2_id_from_meta = d2_meta.get("dataset_id", "")
    else:
        d2_id_from_meta = d2_path.stem

    # ------------------------------------------------------------------
    # Step 2: Apply weak labels
    # ------------------------------------------------------------------
    logger.info("Step 2/4: Applying weak labels (min_pp=%d, min_ho=%d)",
                cfg.min_pingpong_count, cfg.min_handover)
    df_d5 = apply_weak_labels(df_d2, cfg)

    # Attach D5 provenance
    df_d5["weak_label_version"] = weak_label_version
    df_d5["source_feature_version"] = source_feature_version
    # Ensure source_snapshot_id is carried forward (already in D2 provenance)

    # ------------------------------------------------------------------
    # Step 3: Quality checks
    # ------------------------------------------------------------------
    logger.info("Step 3/4: Running D5 quality checks")
    from src.features.quality import format_feature_quality_report
    d5_qc = run_d5_quality_checks(df_d5)
    logger.info(format_feature_quality_report(d5_qc))

    # ------------------------------------------------------------------
    # Step 4: Write outputs
    # ------------------------------------------------------------------
    d5_id = _build_d5_id(d2_id_from_meta if d2_id_from_meta else d2_path.stem)
    parquet_path = output_dir / f"{d5_id}.parquet"
    df_d5.to_parquet(parquet_path, index=False)
    logger.info("D5 Parquet written: %s (%d rows)", parquet_path, len(df_d5))

    # Label distribution
    label_dist = df_d5["weak_label"].value_counts().to_dict()
    n_positive = int(label_dist.get(1, 0))
    n_negative = int(label_dist.get(0, 0))
    total = n_positive + n_negative
    positive_rate = round(n_positive / total * 100, 2) if total > 0 else 0.0

    created_at = datetime.now(timezone.utc).isoformat()
    metadata: dict[str, Any] = {
        "dataset_id": d5_id,
        "dataset_layer": "D5",
        "description": (
            "Weak-label support dataset for evaluation. "
            "Labels are rule-derived heuristics, NOT ground truth."
        ),
        "created_at": created_at,
        "created_by_process": "src.features.weak_labels.build_d5_weak_label_dataset",
        "weak_label_version": weak_label_version,
        "source_feature_version": source_feature_version,
        "source_snapshot_id": source_snapshot_id,
        "d2_parquet_path": str(d2_path),
        "weak_label_config": cfg.to_dict(),
        "d2_row_count": d2_row_count,
        "d5_row_count": len(df_d5),
        "label_distribution": {
            "positive (weak_label=1)": n_positive,
            "negative (weak_label=0)": n_negative,
            "positive_rate_pct": positive_rate,
        },
        "disclaimer": (
            "These weak labels are derived from rule-based heuristics on "
            "handover features. They should NOT be treated as ground truth. "
            "They serve as supporting evaluation artifacts for internal "
            "model-level signal assessment only."
        ),
        "columns": list(df_d5.columns),
        "quality_checks": [
            {"check_id": r.check_id, "description": r.description,
             "passed": r.passed, "detail": r.detail}
            for r in d5_qc
        ],
    }

    meta_path = output_dir / f"{d5_id}_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    logger.info("D5 metadata written: %s", meta_path)

    all_passed = all(r.passed for r in d5_qc)

    return {
        "d5_id": d5_id,
        "parquet_path": str(parquet_path),
        "metadata_path": str(meta_path),
        "d2_row_count": d2_row_count,
        "d5_row_count": len(df_d5),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_rate_pct": positive_rate,
        "weak_label_version": weak_label_version,
        "all_quality_checks_passed": all_passed,
        "metadata": metadata,
    }
