from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.training.schema import FEATURE_COLUMNS, TrainingConfig, TrainingContext


DEFAULT_WINDOW_DAYS = 5
DEFAULT_TIMESTAMP_COL = "window_start"


def apply_sliding_window(
    df: pd.DataFrame,
    window_days: int = DEFAULT_WINDOW_DAYS,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
) -> pd.DataFrame:
    """Keep only the most recent *window_days* of data (MTLF sliding window).

    This implements the sliding-window training strategy described in Chap3,
    where the model is always trained on the N most recent days of data to
    stay current with evolving subscriber mobility patterns.

    Args:
        df: Feature DataFrame with a datetime column.
        window_days: Number of days in the training window (default 5).
        timestamp_col: Name of the timestamp column to filter on.

    Returns:
        Filtered DataFrame containing only rows within the window.
        Returns ``df`` unchanged if the timestamp column is missing.
    """
    if timestamp_col not in df.columns:
        return df
    ts = pd.to_datetime(df[timestamp_col], utc=True)
    cutoff = ts.max() - pd.Timedelta(days=window_days)
    return df.loc[ts >= cutoff].reset_index(drop=True)


def _load_metadata_sidecar(dataset_path: Path) -> dict[str, Any]:
    meta_path = dataset_path.with_name(dataset_path.stem + "_metadata.json")
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_training_dataset(dataset_path: Path, cfg: TrainingConfig) -> tuple[pd.DataFrame, TrainingContext, dict[str, Any]]:
    dataset_path = Path(dataset_path)
    df = pd.read_parquet(dataset_path)
    meta = _load_metadata_sidecar(dataset_path)

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required feature columns: {missing}")

    fallback_series = df.get("source_snapshot_id", pd.Series(["unknown"]))
    dataset_id = str(meta.get("dataset_id") or meta.get("dataset_snapshot_id") or fallback_series.iloc[0])
    dataset_layer = str(meta.get("dataset_layer") or ("D5" if cfg.label_column in df.columns else "D2"))
    feature_version = str(meta.get("feature_version") or df.get("feature_version", pd.Series(["unknown"])).iloc[0])
    source_snapshot_id = str(meta.get("source_snapshot_id") or fallback_series.iloc[0])

    ctx = TrainingContext(
        dataset_path=str(dataset_path),
        dataset_id=dataset_id,
        dataset_layer=dataset_layer,
        feature_version=feature_version,
        source_snapshot_id=source_snapshot_id,
        row_count=int(len(df)),
        label_column=cfg.label_column if cfg.label_column in df.columns else None,
        label_available=cfg.label_column in df.columns,
    )
    return df, ctx, meta


def prepare_training_matrices(df: pd.DataFrame, cfg: TrainingConfig) -> tuple[np.ndarray, np.ndarray | None]:
    x = df.loc[:, FEATURE_COLUMNS].copy()
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = None
    if cfg.use_labels_for_evaluation and cfg.label_column in df.columns:
        y = df[cfg.label_column].astype(int).to_numpy()
    return x.to_numpy(dtype=float), y


def split_training_data(x: np.ndarray, y: np.ndarray | None, cfg: TrainingConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if y is not None and len(set(y.tolist())) > 1:
        return train_test_split(
            x,
            y,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=y,
        )
    x_train, x_val = train_test_split(
        x,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )
    return x_train, x_val, None, None
