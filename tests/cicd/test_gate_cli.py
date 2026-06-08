import json

from src.cicd.cli import main


def _write_result(p, roc):
    p.write_text(json.dumps({"model_name": "m", "model_version": None,
                             "metrics": {"roc_auc": roc, "pr_auc": roc}}))
    return p


def test_gate_cli_pass(tmp_path, capsys):
    res = _write_result(tmp_path / "training_result.json", 0.95)
    out = tmp_path / "gate.json"
    rc = main(["gate", "--metrics", str(res), "--backend", "noop",
               "--min-roc-auc", "0.7", "--output", str(out)])
    assert rc == 0
    assert json.loads(out.read_text())["passed"] is True


def test_gate_cli_fail_exit_1(tmp_path):
    res = _write_result(tmp_path / "training_result.json", 0.4)
    rc = main(["gate", "--metrics", str(res), "--backend", "noop",
               "--min-roc-auc", "0.7", "--output", str(tmp_path / "g.json")])
    assert rc == 1
