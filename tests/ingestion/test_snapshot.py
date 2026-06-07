import json
from pathlib import Path

import pandas as pd

from src.ingestion.snapshot import build_d1_snapshot


def test_build_snapshot_filters_handover(tiny_ebs_file, tmp_path):
    out = tmp_path / "snap"
    result = build_d1_snapshot([tiny_ebs_file], out, snapshot_id="D1_TEST")
    assert result["raw_row_count"] == 3
    assert result["d1_row_count"] == 2  # only the 2 l_handover rows
    df = pd.read_parquet(result["parquet_path"])
    assert (df["event_id"] == "l_handover").all()
    assert "dataset_snapshot_id" in df.columns and "event_ts" in df.columns


def test_snapshot_writes_metadata_sidecar(tiny_ebs_file, tmp_path):
    result = build_d1_snapshot([tiny_ebs_file], tmp_path / "snap", snapshot_id="D1_TEST")
    meta = json.loads(Path(result["metadata_path"]).read_text())
    assert meta["dataset_snapshot_id"] == "D1_TEST"
    assert meta["event_filter"] == "l_handover"
    assert meta["raw_row_count"] == 3 and meta["filtered_row_count"] == 2
    assert "raw_quality_checks" in meta and "d1_quality_checks" in meta


def test_snapshot_returns_quality_flag(tiny_ebs_file, tmp_path):
    result = build_d1_snapshot([tiny_ebs_file], tmp_path / "snap")
    assert isinstance(result["all_quality_checks_passed"], bool)
