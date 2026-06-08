from fastapi.testclient import TestClient

from src.monitoring.metrics import add_metrics
from src.serving.app import create_app
from src.serving.runtime import ServingRuntime
from src.serving.schema import ServingConfig


def test_metrics_endpoint_exposes_counter(iforest_on_reference, tmp_path):
    cfg = ServingConfig(loader="path", model_type="iforest",
                        model_path=str(iforest_on_reference), output_root=str(tmp_path / "s"))
    app = add_metrics(create_app(ServingRuntime.build(cfg)))
    client = TestClient(app)
    feats = {c: 1.0 for c in ["n_handover", "n_unique_cells", "pingpong_count",
                              "pingpong_rate", "mean_inter_ho_s", "std_inter_ho_s", "entropy_cell_seq"]}
    assert client.post("/predict", json={"features": feats}).status_code == 200
    r = client.get("/metrics")
    assert r.status_code == 200
    # the counter actually incremented (name alone appears even at 0 via HELP/TYPE lines)
    assert "nwdaf_predict_requests_total 1.0" in r.text
    assert "nwdaf_predict_latency_ms" in r.text
