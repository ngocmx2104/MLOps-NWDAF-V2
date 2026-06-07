import pandas as pd
import pytest


@pytest.fixture
def d1_like_df():
    """Minimal D1-like frame: one IMSI doing A->B->A ping-pong + one normal IMSI."""
    base = pd.Timestamp("2024-06-26T14:00:00", tz="UTC")
    rows = [
        # IMSI 111: A(10) -> B(20) -> A(10) within 30s gaps => 1 ping-pong, 3 handovers
        {"imsi": "111", "eci": "10", "event_ts": base},
        {"imsi": "111", "eci": "20", "event_ts": base + pd.Timedelta(seconds=10)},
        {"imsi": "111", "eci": "10", "event_ts": base + pd.Timedelta(seconds=20)},
        # IMSI 222: single handover, no ping-pong
        {"imsi": "222", "eci": "30", "event_ts": base + pd.Timedelta(seconds=5)},
    ]
    return pd.DataFrame(rows)
