from pathlib import Path

import pytest

from src.data.real_profile import extract_profile

REAL_EBS = sorted(Path("data/raw_ebs").glob("*_ebs"))


@pytest.mark.skipif(not REAL_EBS, reason="real EBS files not present (run dvc pull)")
def test_extract_profile_keys():
    p = extract_profile(REAL_EBS)
    assert p["source"]["total_handovers"] > 1000
    assert p["topology"]["cell_pool_size"] > 0
    assert "pingpong" in p["handover"]
    assert p["handover"]["unique_imsis"] > 0
    assert "event_type_mix" in p


@pytest.mark.skipif(not REAL_EBS, reason="real EBS files not present")
def test_extract_profile_handover_stats():
    p = extract_profile(REAL_EBS)
    assert "ho_count_per_imsi" in p["handover"]
    assert "buckets" in p["handover"]["ho_count_per_imsi"]
    assert isinstance(p["topology"]["cell_pool"], list)
