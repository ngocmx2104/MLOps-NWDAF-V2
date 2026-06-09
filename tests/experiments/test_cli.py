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
