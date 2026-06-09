# tests/experiments/test_exp_common.py
import sys

from src.experiments.exp_common import ConfigSpec, read_training_result, run_experiment


def _fake_workload(tmp_path):
    # a workload that writes a training_result.json into its --output-dir, like the real CLI
    script = tmp_path / "fake_train.py"
    script.write_text(
        "import json,sys,os\n"
        "od=sys.argv[sys.argv.index('--output-dir')+1]\n"
        "seed=sys.argv[sys.argv.index('--seed')+1]\n"
        "os.makedirs(od,exist_ok=True)\n"
        "json.dump({'metrics':{'roc_auc':0.9,'pr_auc':0.8,'f1':0.7},'train_seconds':0.01,\n"
        "  'run_id':'r'+seed,'backend':'noop','model_path':od+'/m.joblib'},open(od+'/training_result.json','w'))\n"
        "open(od+'/m.joblib','w').write('x')\n")
    return [sys.executable, str(script), "--output-dir", "{output_dir}", "--seed", "{seed}"]


def test_run_experiment_collects_per_seed(tmp_path):
    spec = ConfigSpec(label="C1", workload=_fake_workload(tmp_path), env={})
    runs = run_experiment(spec, experiment_id="exp_t", seeds=[1, 2], output_root=tmp_path / "out")
    assert len(runs) == 2
    assert runs[0]["resource"]["returncode"] == 0
    assert runs[0]["result"]["metrics"]["roc_auc"] == 0.9       # parsed training_result.json
    assert runs[0]["resource"]["wall_s"] > 0                    # measured by harness
    assert (tmp_path / "out" / "exp_t" / "C1" / "runs.jsonl").exists()


def test_read_training_result_missing(tmp_path):
    assert read_training_result(tmp_path / "nope") is None       # no crash on missing
