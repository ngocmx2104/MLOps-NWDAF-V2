from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

_SECTIONS = ("data", "model", "infrastructure", "monitoring")


def _verify(pointer: dict[str, Any], repo_root: Path) -> bool:
    kind, ref = pointer.get("kind"), pointer.get("ref", "")
    if kind == "path":
        return (repo_root / ref).exists()
    if kind == "workflow":
        return (repo_root / ".github" / "workflows" / ref).exists()
    if kind == "pytest":
        r = subprocess.run(["python", "-m", "pytest", "--collect-only", "-q", ref],
                           cwd=repo_root, capture_output=True, text=True)
        return r.returncode == 0
    if kind == "symbol":
        r = subprocess.run(["grep", "-rqs", ref, str(repo_root / "src")], capture_output=True)
        return r.returncode == 0
    return False


def assess(manifest: dict[str, Any], *, repo_root: Path, capability: str = "C1") -> dict[str, Any]:
    """ML Test Score (Breck 2017). capability='C1' credits any test whose evidence verifies;
    capability='C0' additionally forces requires_mlops tests to 0 (an ad-hoc pipeline lacks the
    MLOps artifact), giving an objective C0-vs-C1 maturity delta. Final score = MIN over 4 sections."""
    repo_root = Path(repo_root)
    per_test: dict[str, Any] = {}
    verified: dict[str, bool] = {}
    section_scores = {s: 0.0 for s in _SECTIONS}
    for t in manifest.get("tests", []):
        masked = capability == "C0" and bool(t.get("requires_mlops", False))
        ok = (not masked) and any(_verify(p, repo_root) for p in t.get("evidence", []))
        verified[t["id"]] = ok
        credited = float(t["score"]) if ok else 0.0
        per_test[t["id"]] = {"section": t["section"], "claimed": t["score"], "credited": credited,
                             "requires_mlops": bool(t.get("requires_mlops", False))}
        if t["section"] in section_scores:
            section_scores[t["section"]] += credited
    ml_test_score = min(section_scores.values()) if section_scores else 0.0
    return {"section_scores": section_scores, "ml_test_score": ml_test_score, "capability": capability,
            "per_test": per_test, "verified": verified,
            "google_level": _google_level(ml_test_score),
            "azure_level": _azure_level(section_scores)}


def _google_level(score: float) -> int:
    # Triangulation heuristic from the ML Test Score (Breck Table: <1 weak ... >5 strong).
    return 0 if score < 1 else (1 if score < 3 else 2)


def _azure_level(section_scores: dict[str, float]) -> int:
    covered = sum(1 for v in section_scores.values() if v >= 1)
    return covered  # 0..4 sections with >=1 automated/manual test present
