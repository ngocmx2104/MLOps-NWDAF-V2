from src.ingestion.parser import normalize_timestamps, parse_ebs_files
from src.ingestion.quality import (
    check_event_filter, check_field_count,
    run_raw_quality_checks, quality_results_to_dicts,
)


def test_field_count_check_passes(tiny_ebs_file):
    df = parse_ebs_files([tiny_ebs_file])
    res = check_field_count(df)
    assert res.passed
    assert res.detail["pct_matching"] == 100.0


def test_raw_checks_run(tiny_ebs_file):
    df = parse_ebs_files([tiny_ebs_file])
    results = run_raw_quality_checks(df)
    assert len(results) == 3
    assert all(hasattr(r, "passed") for r in results)


def test_event_filter_on_handover_only(tiny_ebs_file):
    df = normalize_timestamps(parse_ebs_files([tiny_ebs_file]))
    ho = df[df["event_id"] == "l_handover"].copy()
    assert check_event_filter(ho).passed  # 100% are l_handover


def test_results_serialisable(tiny_ebs_file):
    df = parse_ebs_files([tiny_ebs_file])
    dicts = quality_results_to_dicts(run_raw_quality_checks(df))
    assert isinstance(dicts, list) and "check_id" in dicts[0]
