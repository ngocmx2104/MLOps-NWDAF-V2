from src.monitoring.cli import main as mon_main


def test_retrain_cli_smoke(reference_parquet, predictions_no_drift, tmp_path):
    """CT entrypoint runs the retrain cycle; no drift -> retrained False, exit 0."""
    rc = mon_main(["retrain",
                   "--predictions", str(predictions_no_drift),
                   "--reference", str(reference_parquet),
                   "--dataset", str(reference_parquet),
                   "--output-dir", str(tmp_path / "rc"),
                   "--backend", "noop"])
    assert rc == 0
