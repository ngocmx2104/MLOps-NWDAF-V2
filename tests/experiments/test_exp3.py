# tests/experiments/test_exp3.py
import sys

from src.experiments.exp3_framework import run_exp3


def test_run_exp3_overhead(tmp_path):
    from tests.experiments._fixtures import write_tiny_labeled_parquet
    ds = write_tiny_labeled_parquet(tmp_path / "d5.parquet")

    def workload():
        return [sys.executable, "-m", "src.training", "train", "--dataset", str(ds),
                "--output-dir", "{output_dir}", "--seed", "{seed}"]

    summary = run_exp3(dataset=str(ds), seeds=[1, 2], output_root=tmp_path / "out",
                       workload=workload(),
                       backends={"noop": {}, "mlflow": {"MLFLOW_TRACKING_URI": f"sqlite:///{tmp_path}/m.db"}})
    assert "noop" in summary["overhead"] and "mlflow" in summary["overhead"]
    # overhead delta computed vs noop baseline
    assert "delta_wall_s" in summary["overhead"]["mlflow"]
    assert summary["governance"]["mlflow"]["registry"] in (True, False)
