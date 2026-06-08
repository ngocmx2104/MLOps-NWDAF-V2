import json

from src.experiments.metrics.operational import dora_metrics, latency_percentiles


def test_latency_percentiles(tmp_path):
    p = tmp_path / "predictions.jsonl"
    with p.open("w") as f:
        for ms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            f.write(json.dumps({"event": "predict", "latency_ms": ms}) + "\n")
    pct = latency_percentiles(p)
    assert pct["p50_ms"] == 55.0 and pct["p95_ms"] >= 90.0 and pct["count"] == 10


def test_dora_metrics():
    # 3 deploys, 1 failed change, window 2 days
    events = [
        {"kind": "deploy", "ts": 0.0},
        {"kind": "deploy", "ts": 3600.0},
        {"kind": "deploy_failed", "ts": 4000.0},
        {"kind": "deploy", "ts": 7200.0},
    ]
    d = dora_metrics(events, window_days=2.0)
    assert d["deploy_count"] == 3
    assert d["deploy_frequency_per_day"] == 3 / 2.0
    assert d["change_fail_rate"] == 1 / 4  # 1 failed of 4 change attempts
