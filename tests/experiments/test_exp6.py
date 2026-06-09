"""Tests for Exp-6 modifiability harness (RQ4c).

Tests:
1. noop_comment mod: appends a comment -> files_changed>=1, lines_changed>=1,
   regression_count==0, pass=True.
2. breaking_mod: edits a src file so a targeted test FAILS ->
   regression_count>=1, pass=False.  This proves the harness actually tests
   the MODIFIED src (anti-false-pass guard).
3. clean_repo: main working tree is clean (no leftover edits) after
   measure_modification returns.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from src.experiments.exp6_modifiability import Modification, measure_modification

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Test 1: harmless noop comment
# ---------------------------------------------------------------------------

def test_measure_modification_counts_diff_and_regression():
    """A harmless comment-append must show diff footprint but zero regressions."""
    mod = Modification(
        mod_id="noop_comment",
        section="data",
        apply=lambda root: (root / "src/experiments/__init__.py").open("a").write(
            "# exp6 probe\n"
        ),
        test_targets=["tests/experiments/test_schema_records.py"],
    )
    result = measure_modification(mod)
    assert result["files_changed"] >= 1
    assert result["lines_changed"] >= 1
    assert result["regression_count"] == 0, (
        f"noop comment broke {result['regression_count']} tests — unexpected"
    )
    assert result["pass"] is True


# ---------------------------------------------------------------------------
# Test 2: breaking mod — proves harness detects a REAL regression
# ---------------------------------------------------------------------------

def test_breaking_mod_is_detected():
    """A mod that corrupts scored_seeds() logic must cause regression_count>=1.

    This is the anti-false-pass guard: if the harness were testing unmodified
    code (e.g. editable-install finder bypassing the mod), this test would fail
    with regression_count==0, revealing the bug in the harness.

    Target test: tests/experiments/test_schema_records.py::test_runconfig_seeds_and_warmup
    The existing test asserts scored_seeds() == [2, 3, 4, 5].
    Our mod changes EXPERIMENT_RECORD_VERSION so that test still passes but
    we need a mod that actually breaks the target test.

    Actual strategy: overwrite scored_seeds() to always return [] — the existing
    assert `rc.scored_seeds() == [2, 3, 4, 5]` will fail.
    """
    target_file = "src/experiments/schema.py"
    target_test = "tests/experiments/test_schema_records.py::test_runconfig_seeds_and_warmup"

    original_line = "        return self.seeds[self.warmup_drop:self.n_runs]"
    broken_line = "        return []  # exp6-break-probe"

    def apply_breaking_mod(root: Path) -> None:
        schema_path = root / target_file
        text = schema_path.read_text()
        assert original_line in text, (
            f"Could not find target line in {target_file}; test needs updating"
        )
        schema_path.write_text(text.replace(original_line, broken_line))

    mod = Modification(
        mod_id="break_scored_seeds",
        section="test_guard",
        apply=apply_breaking_mod,
        test_targets=[target_test],
    )
    result = measure_modification(mod)
    assert result["regression_count"] >= 1, (
        "HARNESS BUG: breaking mod yielded regression_count==0. "
        "The harness is testing unmodified code — fix the import path strategy."
    )
    assert result["pass"] is False


# ---------------------------------------------------------------------------
# Test 3: main repo working tree is CLEAN after both runs above
# ---------------------------------------------------------------------------

def test_repo_is_clean_after_measure_modification():
    """No leftover edits in the main working tree after measure_modification.

    Runs a noop mod and then checks git status --porcelain is empty for the
    file that was modified.
    """
    mod = Modification(
        mod_id="clean_check",
        section="data",
        apply=lambda root: (root / "src/experiments/__init__.py").open("a").write(
            "# clean-check probe\n"
        ),
        test_targets=["tests/experiments/test_schema_records.py"],
    )
    measure_modification(mod)

    result = subprocess.run(
        ["git", "status", "--porcelain", "src/experiments/__init__.py"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.stdout.strip() == "", (
        f"Main repo has leftover edits after measure_modification: {result.stdout!r}"
    )
