# src/experiments/metrics/resource.py
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil


def measure_subprocess(cmd: list[str], *, env: dict[str, str] | None = None,
                       sample_interval: float = 0.02, timeout: float | None = None) -> dict[str, Any]:
    """Cost/Resource group. Run cmd as a subprocess, sampling peak RSS + CPU via psutil
    while it lives, and wall time via perf_counter. Isolated process => honest measurement."""
    full_env = {**dict(os.environ), **(env or {})}
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, env=full_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p = psutil.Process(proc.pid)
    peak_rss = 0
    cpu_samples: list[float] = []
    try:
        while proc.poll() is None:
            try:
                peak_rss = max(peak_rss, p.memory_info().rss)
                cpu_samples.append(p.cpu_percent(interval=None))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(sample_interval)
    finally:
        proc.wait(timeout=timeout)
    wall_s = time.perf_counter() - t0
    return {"returncode": proc.returncode, "wall_s": wall_s,
            "peak_rss_mb": peak_rss / (1024 * 1024),
            "cpu_pct_mean": (sum(cpu_samples) / len(cpu_samples)) if cpu_samples else 0.0}


def storage_bytes(path: Path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def tracking_overhead(noop_wall_s: float, backend_wall_s: float,
                      noop_rss_mb: float, backend_rss_mb: float) -> dict[str, float]:
    """Δ(backend - noop): the tracking-layer overhead (lõi Exp-2/3, P8b feeds the numbers)."""
    return {"delta_wall_s": backend_wall_s - noop_wall_s,
            "delta_rss_mb": backend_rss_mb - noop_rss_mb}
