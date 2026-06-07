"""Synthetic feature-level data generator with controlled concept drift.

Generates synthetic NWDAF handover *feature* data (7 features per IMSI/window)
calibrated to match real EBS distributions from profile.json.

Used for:
  - Exp-2 (ClearML vs MLflow): same data, different tracking backends
  - Exp-4 (IsolationForest vs LSTM-AE): same data, different models

Drift types (for Exp-3 validation at feature level):
  - gradual: Feature distributions shift slowly over time
  - sudden: Abrupt distribution change at a specific point
  - recurring: Drift pattern repeats periodically

Calibration source: 3 real EBS files (14:00-14:03 UTC+7, 2024-06-26)
  8,393 IMSI-window rows, 24,347 handover events.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class DriftConfig:
    """Configuration for drift injection."""
    drift_type: Literal["gradual", "sudden", "recurring"] = "gradual"
    drift_start_idx: int = 500
    drift_magnitude: float = 2.0
    recurring_period: int = 200
    affected_features: list[str] | None = None

    def __post_init__(self):
        if self.affected_features is None:
            self.affected_features = ["pingpong_rate", "mean_inter_ho_s", "entropy_cell_seq"]


# ---------------------------------------------------------------------------
# Real-data calibration (from profile.json + pipeline feature extraction)
# ---------------------------------------------------------------------------
# n_handover:      mean=2.90, std=3.33, median=2, p95=9, p99=14
# n_unique_cells:  mean=1.96, std=1.26, median=2, p95=5, p99=6
# pingpong_count:  mean=0.47, std=1.47, median=0, p90=2, p95=3  (82% zero)
# pingpong_rate:   mean=0.066, std=0.15, median=0, p95=0.43     (derived)
# mean_inter_ho_s: mean=22.0, std=25.0, median=14.0  (only 59% of rows)
# std_inter_ho_s:  mean=10.4, std=14.4, median=3.2   (only 59% of rows)
# entropy_cell_seq:mean=0.72, std=0.74, median=0.92
# ---------------------------------------------------------------------------

# Parameters for correlated generation (not simple normal draws)
_HO_COUNT_P = 0.35          # geometric p → mean ≈ 1/p ≈ 2.86 (min 1)
_CELL_P = 0.42              # geometric p for unique cells → mean ≈ 2.38, capped by n_ho
_PP_IMSI_RATIO = 0.18       # 18% of IMSIs have any ping-pong
_PP_MEAN_GIVEN_ACTIVE = 2.4 # mean PP count among active IMSIs
_INTER_HO_LOG_MU = 2.64     # log-normal μ → median ≈ exp(2.64) ≈ 14.0s
_INTER_HO_LOG_SIGMA = 1.05  # log-normal σ → spread
_STD_RATIO_MEAN = 0.65      # std_inter_ho ≈ 0.65 × mean_inter_ho


def _apply_drift(values: np.ndarray, drift_config: DriftConfig, feature_name: str) -> np.ndarray:
    """Apply drift pattern to a feature array."""
    if feature_name not in (drift_config.affected_features or []):
        return values

    n = len(values)
    result = values.copy()
    mag = drift_config.drift_magnitude

    if drift_config.drift_type == "sudden":
        start = drift_config.drift_start_idx
        if start < n:
            result[start:] = result[start:] * mag + mag

    elif drift_config.drift_type == "gradual":
        start = drift_config.drift_start_idx
        if start < n:
            drift_range = n - start
            ramp = np.linspace(0, mag, drift_range)
            result[start:] = result[start:] + result[start:] * ramp

    elif drift_config.drift_type == "recurring":
        period = drift_config.recurring_period
        for i in range(n):
            cycle_pos = (i - drift_config.drift_start_idx) % (period * 2)
            if drift_config.drift_start_idx <= i and cycle_pos < period:
                result[i] = result[i] * mag

    return result


def generate_synthetic_data(
    n_samples: int = 1000,
    random_state: int = 42,
    anomaly_rate: float = 0.05,
    drift_config: DriftConfig | None = None,
) -> pd.DataFrame:
    """Generate synthetic NWDAF handover feature data.

    Features are generated with inter-feature correlations that match
    real EBS data distributions (right-skewed, zero-inflated).

    Args:
        n_samples: Number of samples to generate.
        random_state: Random seed for reproducibility.
        anomaly_rate: Fraction of samples that are anomalies.
        drift_config: Optional drift injection config.

    Returns:
        DataFrame with 7 features, labels, and drift metadata.
    """
    rng = np.random.RandomState(random_state)

    # -- 1. n_handover: geometric(p) + 1  → min=1, right-skewed ----
    n_handover = rng.geometric(p=_HO_COUNT_P, size=n_samples).astype(float)
    # Cap extreme tail (real p99=14, max=103 but rare)
    n_handover = np.minimum(n_handover, 50)

    # -- 2. n_unique_cells: for n_ho>=2, Poisson(1)+1; else 1 ----
    n_unique_cells = np.ones(n_samples)
    multi = n_handover >= 2
    n_multi_ho = multi.sum()
    if n_multi_ho > 0:
        raw_cells = rng.poisson(lam=1.5, size=n_multi_ho) + 1
        n_unique_cells[multi] = np.minimum(raw_cells.astype(float), n_handover[multi])

    # -- 3. pingpong_count: zero-inflated Poisson ----
    has_pp = rng.random(n_samples) < _PP_IMSI_RATIO
    pp_counts = np.zeros(n_samples)
    n_active = has_pp.sum()
    if n_active > 0:
        pp_counts[has_pp] = rng.poisson(lam=_PP_MEAN_GIVEN_ACTIVE - 1, size=n_active) + 1
    # PP count can't exceed n_handover - 1 (need at least 2 HOs for 1 PP)
    pp_counts = np.minimum(pp_counts, np.maximum(n_handover - 1, 0))
    pingpong_count = pp_counts

    # -- 4. pingpong_rate: derived = pingpong_count / n_handover ----
    pingpong_rate = np.where(n_handover > 0, pingpong_count / n_handover, 0.0)
    pingpong_rate = np.clip(pingpong_rate, 0.0, 1.0)

    # -- 5. mean_inter_ho_s: log-normal, NaN for n_handover=1 ----
    mean_inter_ho = np.full(n_samples, np.nan)
    multi_ho = n_handover >= 2
    n_multi = multi_ho.sum()
    if n_multi > 0:
        mean_inter_ho[multi_ho] = rng.lognormal(
            mean=_INTER_HO_LOG_MU, sigma=_INTER_HO_LOG_SIGMA, size=n_multi
        )
        # Cap to realistic range
        mean_inter_ho[multi_ho] = np.clip(mean_inter_ho[multi_ho], 0.01, 200.0)

    # -- 6. std_inter_ho_s: proportional to mean, 0 for n_handover<=2 ----
    std_inter_ho = np.full(n_samples, np.nan)
    # Need >=3 HOs for non-zero std
    has_std = n_handover >= 3
    n_has_std = has_std.sum()
    if n_has_std > 0:
        ratio = np.abs(rng.normal(_STD_RATIO_MEAN, 0.35, size=n_has_std))
        std_inter_ho[has_std] = mean_inter_ho[has_std] * ratio
        std_inter_ho[has_std] = np.clip(std_inter_ho[has_std], 0.0, 100.0)
    # n_handover==2 → std exists but is determined by exactly 2 gaps
    exactly_2 = n_handover == 2
    if exactly_2.sum() > 0:
        std_inter_ho[exactly_2] = 0.0

    # -- 7. entropy_cell_seq: based on n_unique_cells ----
    entropy = np.zeros(n_samples)
    multi_cell = n_unique_cells >= 2
    n_mc = multi_cell.sum()
    if n_mc > 0:
        # Base entropy ≈ log2(n_unique_cells) with noise
        base_ent = np.log2(n_unique_cells[multi_cell])
        noise = rng.normal(0, 0.15, size=n_mc)
        entropy[multi_cell] = np.clip(base_ent + noise, 0.0, 4.0)

    # -- Build DataFrame ----
    data = {
        "n_handover": n_handover,
        "n_unique_cells": n_unique_cells,
        "pingpong_count": pingpong_count,
        "pingpong_rate": pingpong_rate,
        "mean_inter_ho_s": mean_inter_ho,
        "std_inter_ho_s": std_inter_ho,
        "entropy_cell_seq": entropy,
    }

    # Apply drift if configured
    if drift_config is not None:
        for feat_name in data:
            vals = data[feat_name]
            if not np.all(np.isnan(vals)):
                data[feat_name] = _apply_drift(vals, drift_config, feat_name)

    df = pd.DataFrame(data)

    # Generate synthetic IMSIs and timestamps
    df["imsi"] = [f"45204{rng.randint(1000000000, 9999999999)}" for _ in range(n_samples)]
    base_ts = pd.Timestamp("2024-06-26", tz="UTC")
    df["window_start"] = [base_ts + pd.Timedelta(minutes=5 * i) for i in range(n_samples)]

    # -- Anomaly labels ----
    n_anomaly = int(n_samples * anomaly_rate)
    labels = np.zeros(n_samples, dtype=int)
    anomaly_indices = rng.choice(n_samples, size=n_anomaly, replace=False)
    labels[anomaly_indices] = 1

    # Make anomaly features more extreme (high-mobility, high-PP pattern)
    for idx in anomaly_indices:
        df.loc[idx, "n_handover"] = float(rng.randint(8, 30))
        df.loc[idx, "n_unique_cells"] = float(min(rng.randint(3, 8), df.loc[idx, "n_handover"]))
        pp = float(rng.poisson(3) + 2)
        pp = min(pp, df.loc[idx, "n_handover"] - 1)
        df.loc[idx, "pingpong_count"] = pp
        df.loc[idx, "pingpong_rate"] = pp / df.loc[idx, "n_handover"]
        df.loc[idx, "mean_inter_ho_s"] = float(rng.uniform(1.0, 8.0))
        df.loc[idx, "std_inter_ho_s"] = float(rng.uniform(0.5, 4.0))
        df.loc[idx, "entropy_cell_seq"] = float(rng.uniform(1.5, 3.0))

    df["label"] = labels

    # -- Drift ground truth metadata ----
    if drift_config is not None:
        df["has_drift"] = False
        if drift_config.drift_type == "sudden":
            df.loc[drift_config.drift_start_idx:, "has_drift"] = True
        elif drift_config.drift_type == "gradual":
            df.loc[drift_config.drift_start_idx:, "has_drift"] = True
        elif drift_config.drift_type == "recurring":
            period = drift_config.recurring_period
            for i in range(n_samples):
                cycle_pos = (i - drift_config.drift_start_idx) % (period * 2)
                if drift_config.drift_start_idx <= i and cycle_pos < period:
                    df.loc[i, "has_drift"] = True
        df["drift_type"] = drift_config.drift_type
    else:
        df["has_drift"] = False
        df["drift_type"] = "none"

    return df


def generate_and_save(
    output_path: Path,
    n_samples: int = 1000,
    random_state: int = 42,
    anomaly_rate: float = 0.05,
    drift_type: str | None = None,
    drift_start: int = 500,
    drift_magnitude: float = 2.0,
) -> dict:
    """Generate synthetic data and save to parquet.

    Returns metadata dict for experiment logging.
    """
    drift_config = None
    if drift_type and drift_type != "none":
        drift_config = DriftConfig(
            drift_type=drift_type,  # type: ignore
            drift_start_idx=drift_start,
            drift_magnitude=drift_magnitude,
        )

    df = generate_synthetic_data(
        n_samples=n_samples,
        random_state=random_state,
        anomaly_rate=anomaly_rate,
        drift_config=drift_config,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    return {
        "path": str(output_path),
        "n_samples": n_samples,
        "anomaly_rate": anomaly_rate,
        "n_anomalies": int(df["label"].sum()),
        "drift_type": drift_type or "none",
        "drift_start": drift_start if drift_config else None,
        "n_drifted_samples": int(df["has_drift"].sum()),
        "random_state": random_state,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Generate synthetic NWDAF data with drift")
    parser.add_argument("--output", type=Path, required=True, help="Output parquet path")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anomaly-rate", type=float, default=0.05)
    parser.add_argument("--drift-type", choices=["none", "gradual", "sudden", "recurring"], default="none")
    parser.add_argument("--drift-start", type=int, default=500)
    parser.add_argument("--drift-magnitude", type=float, default=2.0)
    args = parser.parse_args()

    result = generate_and_save(
        output_path=args.output,
        n_samples=args.n_samples,
        random_state=args.seed,
        anomaly_rate=args.anomaly_rate,
        drift_type=args.drift_type,
        drift_start=args.drift_start,
        drift_magnitude=args.drift_magnitude,
    )
    print(json.dumps(result, indent=2))
