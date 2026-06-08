import joblib
import pytest

from src.serving.predictor import LoadedModel
from src.training.schema import FEATURE_COLUMNS


def test_iforest_predict(iforest_model_path, sample_rows):
    model = joblib.load(iforest_model_path)
    lm = LoadedModel(model_type="iforest", version="v1", model_obj=model,
                     feature_columns=list(FEATURE_COLUMNS))
    out = lm.predict(sample_rows)
    assert len(out) == 1
    assert isinstance(out[0]["is_anomaly"], bool)
    assert isinstance(out[0]["anomaly_score"], float)


def test_lstm_predict(lstm_model_path, sample_rows):
    lm = LoadedModel(model_type="lstm_ae", version="v1", meta_path=str(lstm_model_path),
                     feature_columns=list(FEATURE_COLUMNS))
    out = lm.predict(sample_rows)
    assert len(out) == 1 and out[0]["is_anomaly"] in (True, False)


def test_predict_missing_feature_raises(iforest_model_path):
    model = joblib.load(iforest_model_path)
    lm = LoadedModel(model_type="iforest", version="v1", model_obj=model,
                     feature_columns=list(FEATURE_COLUMNS))
    with pytest.raises(KeyError):
        lm.predict([{"n_handover": 1.0}])  # missing the rest
