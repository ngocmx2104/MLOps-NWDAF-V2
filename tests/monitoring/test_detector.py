from src.monitoring.detector import DriftDetector
from src.monitoring.schema import MonitoringConfig


def test_detects_no_drift(reference_parquet, predictions_no_drift):
    det = DriftDetector(MonitoringConfig(min_features_alert=3, min_observed_rows=50))
    result = det.detect(reference_parquet, predictions_no_drift)
    assert result["drift_detected"] is False
    assert "evidently_drift_share" in result and "psi" in result
    assert result["evidently_drift_share"] <= 0.5  # Evidently cross-check also sees no drift


def test_detects_drift(reference_parquet, predictions_drift):
    det = DriftDetector(MonitoringConfig(min_features_alert=3, min_observed_rows=50))
    result = det.detect(reference_parquet, predictions_drift)
    assert result["drift_detected"] is True
    assert result["psi"]["drift_detected"] is True
    assert result["evidently_drift_share"] >= 0.5


def test_load_observed_reads_feature_values(predictions_drift):
    det = DriftDetector(MonitoringConfig())
    obs = det.load_observed(predictions_drift)
    assert len(obs) == 200 and "n_handover" in obs.columns
