# tests/experiments/test_schema_records.py
import json

from src.experiments.records import append_jsonl, build_run_record, build_result_summary, write_json
from src.experiments.schema import RunConfig


def test_runconfig_seeds_and_warmup():
    rc = RunConfig(workload=["python", "-c", "print(1)"], n_runs=5, warmup_drop=1, seeds=[1, 2, 3, 4, 5])
    assert rc.n_runs == 5 and rc.warmup_drop == 1
    assert rc.scored_seeds() == [2, 3, 4, 5]  # first dropped as warmup


def test_records_roundtrip(tmp_path):
    rec = build_run_record(experiment_id="exp1", run_index=0, seed=42,
                           metrics={"roc_auc": 0.9}, resource={"wall_s": 1.2})
    p = append_jsonl(tmp_path / "runs.jsonl", rec)
    line = json.loads(p.read_text().splitlines()[0])
    assert line["seed"] == 42 and line["metrics"]["roc_auc"] == 0.9

    summ = build_result_summary(experiment_id="exp1", configs={"C1": {"roc_auc_mean": 0.9}})
    sp = write_json(tmp_path / "summary.json", summ)
    assert json.loads(sp.read_text())["experiment_id"] == "exp1"
