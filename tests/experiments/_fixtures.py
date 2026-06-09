"""Shared test fixtures for experiment harness tests.

Provides a tiny labeled parquet with the 7 FEATURE_COLUMNS + a `weak_label`
column (the default TrainingConfig.label_column). 60 rows, ~35% positives so
that a stratified 80/20 split keeps both classes in validation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.training.schema import FEATURE_COLUMNS


def write_tiny_labeled_parquet(path: Path | str) -> Path:
    """Write a tiny labeled feature parquet suitable for running the real training CLI.

    - 60 rows, ~21 positives (~35% label=1) — ensures stratified train/val split
      keeps both classes in validation (test_size=0.2 → 12 val rows, ≥1 positive).
    - Feature values are random but positives are shifted to be separable enough
      that iforest scores differ and metrics are non-degenerate.
    - label_column = ``weak_label`` (matches TrainingConfig default).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(7)
    n = 60
    n_pos = 21  # 35% positives

    # Normal rows: gamma(2,1) for all 7 features
    data = {col: rng.gamma(2.0, 1.0, n).tolist() for col in FEATURE_COLUMNS}
    df = pd.DataFrame(data)

    # Make positives separable: shift and scale a random subset
    pos_idx = rng.choice(n, size=n_pos, replace=False)
    for col in FEATURE_COLUMNS:
        df.loc[pos_idx, col] = df.loc[pos_idx, col] * 5.0 + 3.0

    df["weak_label"] = 0
    df.loc[pos_idx, "weak_label"] = 1

    df.to_parquet(path, index=False)
    return path
