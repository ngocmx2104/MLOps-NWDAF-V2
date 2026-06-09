"""Exp-6 modifiability experiment (RQ4c).

Measures the cost and safety of applying NWDAF-specific modifications to the
MLOps pipeline.  For each modification:

1. The ``apply`` callable edits files in the MAIN repository working tree
   (which the editable install points to, so pytest immediately sees the
   modified code).
2. ``git diff --numstat HEAD`` measures the footprint (files_changed,
   lines_changed).
3. ``pytest <test_targets>`` counts regressions.
4. A ``finally`` block reverts every modified file via
   ``git checkout -- <file>``, leaving the main tree clean.

Why in-place rather than a git worktree + PYTHONPATH?
------------------------------------------------------
The editable install uses a custom MetaPathFinder (``_EditableFinder``) that
is appended to ``sys.meta_path`` by the ``.pth`` hook.  MetaPathFinders are
consulted *before* ``sys.path`` entries, so ``PYTHONPATH=<worktree>`` cannot
override it; ``import src.X`` always resolves to the main repo.  The only
reliable approach is to apply the mod in-place where the editable install
already points, run pytest, and revert in ``finally``.

PRIMARY metric: ``regression_count`` — number of targeted tests that break.
SECONDARY: ``files_changed``, ``lines_changed`` (git diff footprint).
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.experiments.records import write_json

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Modification:
    """A single named NWDAF pipeline modification.

    Parameters
    ----------
    mod_id : str
        Unique identifier, e.g. ``"add_feature"``.
    section : str
        Logical section: ``"data"``, ``"label"``, ``"monitoring"``, etc.
    apply : Callable[[Path], Any]
        A callable that receives the repository root path and edits one or
        more files in-place.  It MUST NOT commit — the harness manages
        revert.
    test_targets : list[str]
        Pytest node IDs (paths or ``path::test_name``) to check for
        regressions after the modification is applied.
    """

    mod_id: str
    section: str
    apply: Callable[[Path], Any]
    test_targets: list[str] = field(default_factory=list)


def _git_diff_numstat(repo_root: Path) -> tuple[int, int]:
    """Return (files_changed, lines_changed) from ``git diff --numstat HEAD``."""
    result = subprocess.run(
        ["git", "diff", "--numstat", "HEAD"],
        capture_output=True, text=True, cwd=str(repo_root), check=True,
    )
    files_changed = 0
    lines_changed = 0
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                added = int(parts[0])
                deleted = int(parts[1])
                files_changed += 1
                lines_changed += added + deleted
            except ValueError:
                # binary file: parts[0] or parts[1] may be '-'
                files_changed += 1
    return files_changed, lines_changed


def _git_modified_files(repo_root: Path) -> list[str]:
    """Return list of repo-relative paths of files modified in the working tree."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        capture_output=True, text=True, cwd=str(repo_root), check=True,
    )
    return [p.strip() for p in result.stdout.splitlines() if p.strip()]


def _run_pytest(test_targets: list[str], repo_root: Path) -> int:
    """Run pytest on test_targets; return number of FAILED tests."""
    if not test_targets:
        return 0
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *test_targets, "-q", "--tb=no"],
        capture_output=True, text=True, cwd=str(repo_root),
    )
    # Count FAILED lines in output; returncode==1 means failures exist
    failed = 0
    for line in result.stdout.splitlines():
        # pytest summary line: "X failed, Y passed" or "FAILED path::test"
        if line.strip().startswith("FAILED "):
            failed += 1
    # Fallback: if returncode!=0 and no "FAILED " lines found, count as 1
    if result.returncode not in (0, 5) and failed == 0:
        # returncode 5 = no tests collected; treat as 0 failures
        # Any other non-zero = at least 1 failure that we couldn't parse
        failed = 1
    return failed


def measure_modification(
    mod: Modification,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Apply ``mod``, measure diff footprint + regression count, then revert.

    The modified files are reverted via ``git checkout --`` in a ``finally``
    block, guaranteeing the main repo working tree is clean after this call
    regardless of any exception during pytest.

    Returns
    -------
    dict with keys:
        mod_id, section, files_changed, lines_changed,
        regression_count (PRIMARY), pass, test_targets
    """
    repo_root = Path(repo_root)
    modified_files: list[str] = []

    try:
        # 1. Apply modification in-place
        mod.apply(repo_root)

        # 2. Measure git diff footprint
        modified_files = _git_modified_files(repo_root)
        files_changed, lines_changed = _git_diff_numstat(repo_root)

        # 3. Count regressions (PRIMARY metric)
        regression_count = _run_pytest(mod.test_targets, repo_root)

    finally:
        # 4. Always revert — never leave the main tree dirty
        if modified_files:
            subprocess.run(
                ["git", "checkout", "--", *modified_files],
                cwd=str(repo_root), check=False,
            )
        else:
            # Fallback: revert any working-tree changes
            subprocess.run(
                ["git", "checkout", "--", "."],
                cwd=str(repo_root), check=False,
            )

    return {
        "mod_id": mod.mod_id,
        "section": mod.section,
        "files_changed": files_changed,
        "lines_changed": lines_changed,
        "regression_count": regression_count,  # PRIMARY
        "pass": regression_count == 0,
        "test_targets": mod.test_targets,
        "_note": "regression_count is PRIMARY; files/lines are secondary footprint metrics",
    }


def run_exp6(
    mods: list[Modification],
    *,
    repo_root: Path = REPO_ROOT,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Run Exp-6: apply each modification, measure, summarise.

    regression_count is the PRIMARY metric for RQ4c (modifiability cost
    in terms of test breakage).  files_changed / lines_changed are secondary
    footprint indicators.

    Parameters
    ----------
    mods : list[Modification]
        The NWDAF-specific modifications to evaluate.
    repo_root : Path
        Repository root (defaults to the main checkout).
    output_root : Path, optional
        If given, writes ``exp6_summary.json`` under
        ``<output_root>/exp6_modifiability/``.

    Returns
    -------
    dict  — exp6 summary payload.
    """
    results = [measure_modification(m, repo_root=repo_root) for m in mods]
    summary: dict[str, Any] = {
        "experiment_id": "exp6_modifiability",
        "primary_metric": "regression_count",
        "primary_metric_note": (
            "regression_count (# targeted tests broken) is the headline RQ4c metric. "
            "files_changed and lines_changed are secondary diff-footprint indicators."
        ),
        "mods": results,
    }
    if output_root is not None:
        out_dir = Path(output_root) / "exp6_modifiability"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "exp6_summary.json", summary)
    return summary
