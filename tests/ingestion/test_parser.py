import pandas as pd

from src.ingestion.parser import normalize_timestamps, parse_ebs_files
from src.ingestion.schema import EXPECTED_FIELD_COUNT


def test_parse_basic(tiny_ebs_file):
    df = parse_ebs_files([tiny_ebs_file])
    assert len(df) == 3
    assert df["event_id"].tolist() == ["l_handover", "l_handover", "l_service_request"]
    assert df["imsi"].tolist() == ["111", "111", "222"]
    assert (df["raw_field_count"] == EXPECTED_FIELD_COUNT).all()
    assert df["source_file"].iloc[0] == tiny_ebs_file.name
    assert (df["schema_version"] == "ebs_raw_positional_v1").all()


def test_empty_values_become_none(tiny_ebs_file):
    df = parse_ebs_files([tiny_ebs_file])
    # msisdn (pos 5) was left empty in the fixture
    assert df["msisdn"].isna().all()


def test_normalize_timestamps(tiny_ebs_file):
    df = normalize_timestamps(parse_ebs_files([tiny_ebs_file]))
    assert pd.api.types.is_datetime64_any_dtype(df["event_ts"])
    assert df["event_ts"].notna().all()
    assert str(df["event_ts"].dt.tz) == "UTC"


def test_missing_file_skipped_but_others_parsed(tiny_ebs_file, tmp_path):
    df = parse_ebs_files([tmp_path / "nope_ebs", tiny_ebs_file])
    assert len(df) == 3
