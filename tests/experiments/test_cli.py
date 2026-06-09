import json

from src.experiments.cli import main


def test_assess_cli(tmp_path):
    (tmp_path / "real.txt").write_text("x")
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        "tests:\n"
        "  - id: d1\n    section: data\n    score: 1\n"
        "    evidence:\n      - {kind: path, ref: real.txt}\n")
    out = tmp_path / "report.json"
    rc = main(["assess", "--manifest", str(manifest), "--repo-root", str(tmp_path),
               "--output", str(out)])
    assert rc == 0
    assert "ml_test_score" in json.loads(out.read_text())


def test_assess_cli_missing_manifest(tmp_path):
    """Missing manifest should return rc=1, not raise FileNotFoundError."""
    rc = main(["assess", "--manifest", str(tmp_path / "nope.yaml"), "--repo-root", str(tmp_path)])
    assert rc == 1


def test_assess_cli_scores_full_manifest(tmp_path):
    """manifest with score:1 path evidence in all 4 sections -> ml_test_score == 1.0."""
    (tmp_path / "real.txt").write_text("x")
    manifest = tmp_path / "full.yaml"
    manifest.write_text(
        "tests:\n"
        "  - id: d1\n    section: data\n    score: 1\n"
        "    evidence:\n      - {kind: path, ref: real.txt}\n"
        "  - id: m1\n    section: model\n    score: 1\n"
        "    evidence:\n      - {kind: path, ref: real.txt}\n"
        "  - id: i1\n    section: infrastructure\n    score: 1\n"
        "    evidence:\n      - {kind: path, ref: real.txt}\n"
        "  - id: mo1\n    section: monitoring\n    score: 1\n"
        "    evidence:\n      - {kind: path, ref: real.txt}\n"
    )
    out = tmp_path / "report.json"
    rc = main(["assess", "--manifest", str(manifest), "--repo-root", str(tmp_path),
               "--output", str(out)])
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["ml_test_score"] == 1.0
