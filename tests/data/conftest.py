import pytest


@pytest.fixture
def tiny_profile():
    """Minimal profile with the keys EBSSyntheticGenerator requires."""
    return {
        "handover": {
            "events_per_minute": 100,
            "imsi_per_minute": 40,
            "ho_count_per_imsi": {
                "buckets": {"1": 0.4, "2-3": 0.35, "4-7": 0.15,
                            "8-15": 0.07, "16-30": 0.02, "31+": 0.01},
            },
        },
        "topology": {
            "cell_pool": [str(100_000_000 + i) for i in range(20)],
            "imsi_prefix": "45204",
            "imsi_length": 15,
        },
        "event_type_mix": {"l_handover": 0.1, "l_service_request": 0.5, "l_tau": 0.4},
    }
