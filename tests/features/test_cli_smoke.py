"""Smoke test for the src.features CLI (build-d2 and weak-label subcommands).

Builds a minimal D1 parquet in a tmp directory and verifies that the CLI
produces a D2 Parquet (and subsequently a D5 Parquet) without errors.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def tiny_d1_parquet(tmp_path: Path) -> Path:
    """Write a minimal D1-like parquet that build_d2 can process."""
    base = pd.Timestamp("2024-06-26T14:00:00", tz="UTC")
    rows = [
        # IMSI 111: A->B->A ping-pong + 3 handovers
        {"imsi": "111", "eci": "10", "event_ts": base, "dataset_snapshot_id": "D1_SMOKE"},
        {"imsi": "111", "eci": "20", "event_ts": base + pd.Timedelta(seconds=10), "dataset_snapshot_id": "D1_SMOKE"},
        {"imsi": "111", "eci": "10", "event_ts": base + pd.Timedelta(seconds=20), "dataset_snapshot_id": "D1_SMOKE"},
        # IMSI 222: single handover, no ping-pong
        {"imsi": "222", "eci": "30", "event_ts": base + pd.Timedelta(seconds=5), "dataset_snapshot_id": "D1_SMOKE"},
        # IMSI 333: 4 handovers with ping-pong (should get positive weak label)
        {"imsi": "333", "eci": "10", "event_ts": base + pd.Timedelta(seconds=1), "dataset_snapshot_id": "D1_SMOKE"},
        {"imsi": "333", "eci": "20", "event_ts": base + pd.Timedelta(seconds=8), "dataset_snapshot_id": "D1_SMOKE"},
        {"imsi": "333", "eci": "10", "event_ts": base + pd.Timedelta(seconds=15), "dataset_snapshot_id": "D1_SMOKE"},
        {"imsi": "333", "eci": "20", "event_ts": base + pd.Timedelta(seconds=22), "dataset_snapshot_id": "D1_SMOKE"},
    ]
    df = pd.DataFrame(rows)
    out = tmp_path / "D1_SMOKE.parquet"
    df.to_parquet(out, index=False)
    return out


def test_build_d2_cli_smoke(tiny_d1_parquet: Path, tmp_path: Path) -> None:
    """build-d2 subcommand exits 0 and produces a D2 parquet."""
    out_dir = tmp_path / "features"
    result = subprocess.run(
        [
            sys.executable, "-m", "src.features.cli",
            "build-d2",
            "--d1", str(tiny_d1_parquet),
            "--output", str(out_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"CLI exited {result.returncode}.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    d2_files = list(out_dir.glob("D2_*.parquet"))
    assert d2_files, f"No D2_*.parquet found in {out_dir}; stdout:\n{result.stdout}"


def test_weak_label_cli_smoke(tiny_d1_parquet: Path, tmp_path: Path) -> None:
    """weak-label subcommand exits 0 and produces a D5 parquet."""
    features_dir = tmp_path / "features"
    labeled_dir = tmp_path / "labeled"

    # First build D2
    build_result = subprocess.run(
        [
            sys.executable, "-m", "src.features.cli",
            "build-d2",
            "--d1", str(tiny_d1_parquet),
            "--output", str(features_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert build_result.returncode == 0, (
        f"build-d2 failed.\nstdout:\n{build_result.stdout}\nstderr:\n{build_result.stderr}"
    )

    d2_files = list(features_dir.glob("D2_*.parquet"))
    assert d2_files

    # Then weak-label from the D2 output
    wl_result = subprocess.run(
        [
            sys.executable, "-m", "src.features.cli",
            "weak-label",
            "--d2", str(d2_files[0]),
            "--output", str(labeled_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert wl_result.returncode == 0, (
        f"CLI exited {wl_result.returncode}.\n"
        f"stdout:\n{wl_result.stdout}\n"
        f"stderr:\n{wl_result.stderr}"
    )
    d5_files = list(labeled_dir.glob("D5_*.parquet"))
    assert d5_files, f"No D5_*.parquet found in {labeled_dir}; stdout:\n{wl_result.stdout}"
