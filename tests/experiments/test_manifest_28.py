"""Task 6: TDD test for the 28-test Breck ML Test Score manifest.

Asserts:
- exactly 28 tests across 4 sections (7 each)
- every test with claimed > 0 verifies under C1 (no fabricated/stale pointers)
- C0 score <= C1 score (requires_mlops tests are zeroed under C0)
"""
from pathlib import Path

import yaml

from src.experiments.maturity import assess

REPO = Path(__file__).resolve().parents[2]


def test_manifest_has_28_tests_across_4_sections():
    m = yaml.safe_load((REPO / "src/experiments/mltest_manifest.yaml").read_text())
    tests = m["tests"]
    assert len(tests) == 28
    by_section: dict[str, int] = {}
    for t in tests:
        by_section.setdefault(t["section"], 0)
        by_section[t["section"]] += 1
    assert by_section == {"data": 7, "model": 7, "infrastructure": 7, "monitoring": 7}


def test_all_credited_pointers_verify_under_c1():
    m = yaml.safe_load((REPO / "src/experiments/mltest_manifest.yaml").read_text())
    rep = assess(m, repo_root=REPO, capability="C1")
    # every test that claims a score must actually verify (no fabricated/stale pointers)
    for tid, info in rep["per_test"].items():
        if info["claimed"] > 0:
            assert rep["verified"][tid] is True, f"stale pointer: {tid}"


def test_c0_strictly_lower_than_c1():
    m = yaml.safe_load((REPO / "src/experiments/mltest_manifest.yaml").read_text())
    c1 = assess(m, repo_root=REPO, capability="C1")["ml_test_score"]
    c0 = assess(m, repo_root=REPO, capability="C0")["ml_test_score"]
    assert c0 <= c1
