"""Feast feature store helpers (component C3) for handover features.

Provides programmatic apply/materialize/retrieve over a `local` Feast repo
(File offline + SQLite online), proving train/serve feature consistency.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from feast import Entity, FeatureStore, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64

FEATURE_NAMES = [
    "n_handover", "n_unique_cells", "pingpong_count", "pingpong_rate",
    "mean_inter_ho_s", "std_inter_ho_s", "entropy_cell_seq",
]
_DTYPES = {
    "n_handover": Int64, "n_unique_cells": Int64, "pingpong_count": Int64,
    "pingpong_rate": Float64, "mean_inter_ho_s": Float64,
    "std_inter_ho_s": Float64, "entropy_cell_seq": Float64,
}
FEATURE_REFS = [f"handover_features:{n}" for n in FEATURE_NAMES]


def build_definitions(source_parquet: str) -> list:
    """Build the Feast Entity + FeatureView for a given source parquet."""
    imsi = Entity(name="imsi", join_keys=["imsi"], value_type=ValueType.STRING)
    source = FileSource(
        name="handover_feature_source",
        path=str(source_parquet),
        timestamp_field="window_start",
    )
    fv = FeatureView(
        name="handover_features",
        entities=[imsi],
        ttl=timedelta(days=3650),
        schema=[Field(name=n, dtype=_DTYPES[n]) for n in FEATURE_NAMES],
        online=True,
        source=source,
    )
    return [imsi, fv]


def apply_and_materialize(repo_path: Path, source_parquet: Path,
                          end_date: datetime | None = None) -> FeatureStore:
    """Apply definitions and materialize features into the online store."""
    store = FeatureStore(repo_path=str(repo_path))
    store.apply(build_definitions(str(source_parquet)))
    end = end_date or datetime.now(timezone.utc)
    store.materialize(start_date=datetime(2020, 1, 1, tzinfo=timezone.utc), end_date=end)
    return store


def get_online_features(store: FeatureStore, imsis: list[str]) -> pd.DataFrame:
    """Serving path: low-latency online features for given IMSIs."""
    rows = [{"imsi": i} for i in imsis]
    return store.get_online_features(features=FEATURE_REFS, entity_rows=rows).to_df()


def get_training_features(store: FeatureStore, entity_df: pd.DataFrame) -> pd.DataFrame:
    """Training path: point-in-time-correct historical features."""
    return store.get_historical_features(entity_df=entity_df, features=FEATURE_REFS).to_df()
