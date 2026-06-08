import json
import sys

from src.experiments.runner import run_n
from src.experiments.schema import RunConfig


def test_run_n_writes_jsonl_and_drops_warmup(tmp_path):
    rc = RunConfig(workload=[sys.executable, "-c", "print('seed={seed}')"],
                   n_runs=4, warmup_drop=1, seeds=[1, 2, 3, 4])
    out = run_n(rc, experiment_id="t", config_label="C1", output_dir=tmp_path)
    # 4 runs executed, all recorded raw; scored = 3 (warmup dropped)
    lines = (tmp_path / "runs.jsonl").read_text().splitlines()
    assert len(lines) == 4
    assert out["n_scored"] == 3
    assert all(r["resource"]["returncode"] == 0 for r in out["runs"])
    assert json.loads(lines[0])["seed"] == 1
