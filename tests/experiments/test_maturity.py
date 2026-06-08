from src.experiments.maturity import assess


def test_assess_verifies_evidence(tmp_path):
    (tmp_path / "real.txt").write_text("x")
    manifest = {"tests": [
        {"id": "d1", "section": "data", "score": 1,
         "evidence": [{"kind": "path", "ref": "real.txt"}]},
        {"id": "d2", "section": "data", "score": 1,
         "evidence": [{"kind": "path", "ref": "MISSING.txt"}]},   # unverifiable -> 0
        {"id": "m1", "section": "model", "score": 0.5,
         "evidence": [{"kind": "path", "ref": "real.txt"}]},
        {"id": "i1", "section": "infrastructure", "score": 1,
         "evidence": [{"kind": "path", "ref": "real.txt"}]},
        {"id": "mon1", "section": "monitoring", "score": 1,
         "evidence": [{"kind": "path", "ref": "real.txt"}]},
    ]}
    rep = assess(manifest, repo_root=tmp_path)
    assert rep["section_scores"]["data"] == 1.0       # d1 counts, d2 dropped (unverified)
    assert rep["section_scores"]["model"] == 0.5
    assert rep["ml_test_score"] == 0.5                # MIN across sections
    assert rep["verified"]["d2"] is False
