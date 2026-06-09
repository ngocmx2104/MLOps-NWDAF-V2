# tests/experiments/test_exp5.py
import sys

from src.experiments.exp5_model_swap import run_exp5


def test_run_exp5_compares_models(tmp_path):
    from tests.experiments._fixtures import write_tiny_labeled_parquet
    ds = write_tiny_labeled_parquet(tmp_path / "d5.parquet")

    def wl(model):
        return [sys.executable, "-m", "src.training", "train", "--dataset", str(ds),
                "--model-type", model, "--backend", "noop",
                "--output-dir", "{output_dir}", "--seed", "{seed}"]

    summary = run_exp5(dataset=str(ds), seeds=[1, 2], output_root=tmp_path / "out",
                       iforest_workload=wl("iforest"), lstm_workload=wl("lstm_ae"))
    assert "iforest" in summary and "lstm_ae" in summary
    assert "roc_auc" in summary["iforest"]["model_perf"]      # both compared on same metric
    assert summary["swap_core_changes"] == 0                  # swap = config flag only
    assert (tmp_path / "out" / "exp5_model_swap" / "exp5_summary.json").exists()
