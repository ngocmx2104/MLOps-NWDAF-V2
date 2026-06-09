# tests/experiments/test_exp1.py
import sys

from src.experiments.exp1_e2e import run_exp1


def test_run_exp1_summary(tmp_path, monkeypatch):
    # use the real training CLI as workload on a tiny labeled parquet fixture
    from tests.experiments._fixtures import write_tiny_labeled_parquet
    ds = write_tiny_labeled_parquet(tmp_path / "d5.parquet")
    workload = [sys.executable, "-m", "src.training", "train", "--dataset", str(ds),
                "--backend", "noop", "--output-dir", "{output_dir}", "--seed", "{seed}"]
    summary = run_exp1(dataset=str(ds), seeds=[1, 2], output_root=tmp_path / "out",
                       workload=workload)
    assert summary["experiment_id"] == "exp1_e2e"
    assert "roc_auc" in summary["model_perf"] and "mean" in summary["model_perf"]["roc_auc"]
    assert summary["n_runs"] == 2
    assert summary["cost_sensitivity"][0]["ratio"] == 1     # sensitivity curve over cost ratios
    assert (tmp_path / "out" / "exp1_e2e" / "exp1_summary.json").exists()
