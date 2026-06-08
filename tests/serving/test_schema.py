import pytest
from pydantic import ValidationError

from src.serving.schema import PredictRequest, ServingConfig, REGISTRY_NAMES


def test_predict_request_accepts_imsi_only():
    r = PredictRequest(imsi="111")
    assert r.imsi == "111" and r.features is None


def test_predict_request_accepts_features_only():
    feats = {c: 1.0 for c in ["n_handover", "n_unique_cells", "pingpong_count",
                              "pingpong_rate", "mean_inter_ho_s", "std_inter_ho_s", "entropy_cell_seq"]}
    r = PredictRequest(features=feats)
    assert r.features["n_handover"] == 1.0


def test_predict_request_rejects_both():
    with pytest.raises(ValidationError):
        PredictRequest(imsi="111", features={"n_handover": 1.0})


def test_predict_request_rejects_neither():
    with pytest.raises(ValidationError):
        PredictRequest()


def test_registry_names_cover_both_models():
    assert REGISTRY_NAMES["iforest"] == "iforest_pingpong"
    assert REGISTRY_NAMES["lstm_ae"] == "lstm_ae_pingpong"


def test_serving_config_defaults():
    cfg = ServingConfig()
    assert cfg.model_type == "iforest" and cfg.loader == "path"
    assert cfg.registry_alias == "staging" and cfg.port == 8080


def test_resolved_registry_name_fallback_and_override():
    assert ServingConfig(model_type="lstm_ae").resolved_registry_name() == "lstm_ae_pingpong"
    assert ServingConfig(registry_name="custom").resolved_registry_name() == "custom"


def test_resolved_registry_name_unknown_raises():
    with pytest.raises(ValueError, match="unknown model_type"):
        ServingConfig(model_type="xgboost").resolved_registry_name()


def test_to_dict_includes_feature_columns():
    d = ServingConfig().to_dict()
    assert d["loader"] == "path" and len(d["feature_columns"]) == 7
