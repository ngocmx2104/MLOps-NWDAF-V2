# tests/experiments/test_maturity_capability.py
from src.experiments.maturity import assess


def test_capability_c0_zeroes_mlops_tests(tmp_path):
    (tmp_path / "real.txt").write_text("x")
    manifest = {"tests": [
        {"id": "d1", "section": "data", "score": 1, "requires_mlops": False,
         "evidence": [{"kind": "path", "ref": "real.txt"}]},
        {"id": "i1", "section": "infrastructure", "score": 1, "requires_mlops": True,
         "evidence": [{"kind": "path", "ref": "real.txt"}]},
        {"id": "m1", "section": "model", "score": 1, "requires_mlops": False,
         "evidence": [{"kind": "path", "ref": "real.txt"}]},
        {"id": "mon1", "section": "monitoring", "score": 1, "requires_mlops": True,
         "evidence": [{"kind": "path", "ref": "real.txt"}]},
    ]}
    c1 = assess(manifest, repo_root=tmp_path, capability="C1")
    c0 = assess(manifest, repo_root=tmp_path, capability="C0")
    assert c1["section_scores"]["infrastructure"] == 1.0   # verifies under C1
    assert c0["section_scores"]["infrastructure"] == 0.0   # requires_mlops -> 0 under C0
    assert c0["section_scores"]["data"] == 1.0             # non-mlops test still counts
    assert c0["ml_test_score"] == 0.0 and c1["ml_test_score"] == 1.0
