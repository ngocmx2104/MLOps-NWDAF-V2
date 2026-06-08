from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.runtime import ServingRuntime
from src.serving.schema import ServingConfig


def _client(model_path, tmp_path, **kw):
    cfg = ServingConfig(loader="path", model_type="iforest", model_path=str(model_path),
                        output_root=str(tmp_path / "s"), **kw)
    return TestClient(create_app(ServingRuntime.build(cfg)))


def test_health(iforest_model_path, tmp_path):
    c = _client(iforest_model_path, tmp_path)
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_model_info(iforest_model_path, tmp_path):
    c = _client(iforest_model_path, tmp_path)
    r = c.get("/model-info")
    assert r.status_code == 200 and r.json()["model_type"] == "iforest"


def test_predict_with_features(iforest_model_path, tmp_path, sample_rows):
    c = _client(iforest_model_path, tmp_path)
    r = c.post("/predict", json={"features": sample_rows[0]})
    assert r.status_code == 200
    body = r.json()
    assert "is_anomaly" in body and body["latency_ms"] >= 0.0


def test_predict_rejects_both_imsi_and_features(iforest_model_path, tmp_path, sample_rows):
    c = _client(iforest_model_path, tmp_path)
    r = c.post("/predict", json={"imsi": "111", "features": sample_rows[0]})
    assert r.status_code == 422  # pydantic validation error


def test_predict_with_imsi_uses_feast(iforest_model_path, tmp_path, materialized_repo, sample_rows):
    c = _client(iforest_model_path, tmp_path, feast_repo_path=str(materialized_repo))
    r = c.post("/predict", json={"imsi": "111"})
    assert r.status_code == 200 and "anomaly_score" in r.json()


def test_predict_rejects_neither(iforest_model_path, tmp_path):
    c = _client(iforest_model_path, tmp_path)
    r = c.post("/predict", json={})  # neither imsi nor features
    assert r.status_code == 422


def test_predict_imsi_without_feast_is_400(iforest_model_path, tmp_path):
    c = _client(iforest_model_path, tmp_path)  # no feast_repo_path configured
    r = c.post("/predict", json={"imsi": "111"})
    assert r.status_code == 400  # valid schema, runtime failure (ValueError -> 400)


def test_admin_reload_over_http(iforest_model_path, tmp_path):
    c = _client(iforest_model_path, tmp_path)
    r = c.post("/admin/reload")
    assert r.status_code == 200 and r.json()["model_type"] == "iforest"


def test_admin_rollback_over_http_path_loader(iforest_model_path, tmp_path):
    c = _client(iforest_model_path, tmp_path)
    r = c.post("/admin/rollback", params={"target_version": str(iforest_model_path)})
    assert r.status_code == 200 and r.json()["event"] == "rollback"
