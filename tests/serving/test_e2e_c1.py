"""End-to-end test: C1 registry-backed serving + rollback via TestClient (P5 DoD)."""
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.runtime import ServingRuntime
from src.serving.schema import ServingConfig


def test_c1_registry_serving_and_rollback(mlflow_registry, tmp_path, sample_rows):
    uri, first, latest = mlflow_registry
    cfg = ServingConfig(loader="registry", model_type="iforest", tracking_uri=uri,
                        output_root=str(tmp_path / "s"))
    rt = ServingRuntime.build(cfg)
    client = TestClient(create_app(rt))

    # C1 serves the staging (latest) version end-to-end
    r = client.post("/predict", json={"features": sample_rows[0]})
    assert r.status_code == 200 and r.json()["model_version"].endswith(f"/{latest}")

    # rollback to the first version (DoD: rollback to another version OK)
    rb = client.post("/admin/rollback", params={"target_version": first})
    assert rb.status_code == 200
    info = client.get("/model-info").json()
    assert info["model_version"].endswith(f"/{first}")
