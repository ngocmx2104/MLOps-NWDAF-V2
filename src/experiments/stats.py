from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import wilcoxon


def summarize(samples: list[float]) -> dict[str, Any]:
    arr = np.asarray(samples, dtype=float)
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()), "max": float(arr.max())}


def bootstrap_ci(samples: list[float], *, ci: float = 0.95, n_boot: int = 10000,
                 seed: int = 12345) -> dict[str, Any]:
    rng = np.random.RandomState(seed)
    arr = np.asarray(samples, dtype=float)
    means = arr[rng.randint(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return {"ci": ci, "ci95": (lo, hi)}


def wilcoxon_compare(c0: list[float], c1: list[float]) -> dict[str, Any]:
    """Paired Wilcoxon signed-rank (C0 vs C1), alpha=0.05 (THESIS_SPEC §4.4)."""
    stat, p = wilcoxon(np.asarray(c0, dtype=float), np.asarray(c1, dtype=float))
    return {"n": len(c0), "statistic": float(stat), "p_value": float(p),
            "significant_0_05": bool(p < 0.05),
            "c0": summarize(c0), "c1": summarize(c1)}
