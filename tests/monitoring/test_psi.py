import numpy as np
import pandas as pd

from src.monitoring.psi import PSIDriftMonitor, compute_psi
from src.monitoring.schema import MonitoringConfig


def test_psi_zero_for_same_distribution():
    rng = np.random.RandomState(0)
    x = rng.gamma(2.0, 1.0, 500)
    assert compute_psi(x, x) < 1e-9  # identical distribution -> PSI is exactly 0


def test_psi_high_for_shifted_distribution():
    rng = np.random.RandomState(0)
    ref = rng.gamma(2.0, 1.0, 500)
    obs = rng.gamma(2.0, 1.0, 500) * 4.0
    assert compute_psi(ref, obs) >= 0.25  # >= alert tier


def test_monitor_no_drift_when_identical():
    rng = np.random.RandomState(0)
    cols = ["n_handover", "pingpong_count", "entropy_cell_seq"]
    ref = pd.DataFrame({c: rng.gamma(2.0, 1.0, 300) for c in cols})
    mon = PSIDriftMonitor(reference_frame=ref, config=MonitoringConfig(min_features_alert=2))
    result = mon.evaluate(ref.copy())
    assert result["drift_detected"] is False
    assert result["per_feature"]["n_handover"]["psi_level"] == "ok"


def test_monitor_detects_drift_when_shifted():
    rng = np.random.RandomState(0)
    cols = ["n_handover", "pingpong_count", "entropy_cell_seq"]
    ref = pd.DataFrame({c: rng.gamma(2.0, 1.0, 300) for c in cols})
    obs = pd.DataFrame({c: rng.gamma(2.0, 1.0, 300) * 4.0 for c in cols})
    mon = PSIDriftMonitor(reference_frame=ref, config=MonitoringConfig(min_features_alert=2, min_observed_rows=50))
    result = mon.evaluate(obs)
    assert result["drift_detected"] is True
    assert result["alerted_features"] >= 2
    assert "ks_pvalue" in result["per_feature"]["n_handover"]
