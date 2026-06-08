from src.serving.runtime import ServingRuntime
from src.serving.schema import PredictRequest, ServingConfig


def test_predict_with_features_path_loader(iforest_model_path, tmp_path, sample_rows):
    cfg = ServingConfig(loader="path", model_type="iforest",
                        model_path=str(iforest_model_path), output_root=str(tmp_path / "s"))
    rt = ServingRuntime.build(cfg)
    resp = rt.predict(PredictRequest(features=sample_rows[0]))
    assert resp.latency_ms >= 0.0
    assert resp.model_type == "iforest"
    assert isinstance(resp.is_anomaly, bool)
    # prediction is logged
    assert (tmp_path / "s" / "predictions.jsonl").exists()


def test_rollback_registry_switches_version(mlflow_registry, tmp_path):
    uri, first, latest = mlflow_registry
    cfg = ServingConfig(loader="registry", model_type="iforest", tracking_uri=uri,
                        output_root=str(tmp_path / "s"))
    rt = ServingRuntime.build(cfg)
    assert rt.model.version.endswith(f"/{latest}")  # alias staging -> latest
    record = rt.rollback(target_version=first, reason="test")
    assert rt.model.version.endswith(f"/{first}")
    assert record["to_version"].endswith(f"/{first}")
    assert (tmp_path / "s" / "rollback_history.jsonl").exists()


def test_reload_rebuilds_model(iforest_model_path, tmp_path):
    cfg = ServingConfig(loader="path", model_type="iforest",
                        model_path=str(iforest_model_path), output_root=str(tmp_path / "s"))
    rt = ServingRuntime.build(cfg)
    info = rt.reload()
    assert info["model_type"] == "iforest"


def test_model_swap_same_runtime_api(iforest_model_path, lstm_model_path, tmp_path, sample_rows):
    """RQ4: the SAME runtime API serves either model via config (no code change)."""
    a = ServingRuntime.build(ServingConfig(loader="path", model_type="iforest",
                                           model_path=str(iforest_model_path),
                                           output_root=str(tmp_path / "a")))
    b = ServingRuntime.build(ServingConfig(loader="path", model_type="lstm_ae",
                                           model_path=str(lstm_model_path),
                                           output_root=str(tmp_path / "b")))
    ra = a.predict(PredictRequest(features=sample_rows[0]))
    rb = b.predict(PredictRequest(features=sample_rows[0]))
    assert ra.model_type == "iforest" and rb.model_type == "lstm_ae"


def test_rollback_path_loader(iforest_model_path, tmp_path):
    """C0 baseline: rollback for the path loader switches to a target FILE artifact."""
    cfg = ServingConfig(loader="path", model_type="iforest",
                        model_path=str(iforest_model_path), output_root=str(tmp_path / "s"))
    rt = ServingRuntime.build(cfg)
    record = rt.rollback(target_version=str(iforest_model_path), reason="c0 swap")
    assert record["event"] == "rollback" and rt.model.model_type == "iforest"
    assert (tmp_path / "s" / "rollback_history.jsonl").exists()
