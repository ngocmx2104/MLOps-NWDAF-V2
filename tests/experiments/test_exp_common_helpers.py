# tests/experiments/test_exp_common_helpers.py
import json

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from src.experiments.exp_common import config_storage_bytes, inference_latency_jsonl, traceability_ok


def test_inference_latency_jsonl(tmp_path):
    model = IsolationForest(n_estimators=10, random_state=0).fit(np.random.RandomState(0).rand(50, 7))
    mp = tmp_path / "m.joblib"
    joblib.dump(model, mp)
    x = np.random.RandomState(1).rand(20, 7)
    out = inference_latency_jsonl(mp, x, tmp_path / "preds.jsonl")
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == 20 and all("latency_ms" in r for r in lines)


def test_config_storage_bytes(tmp_path):
    (tmp_path / "a").write_bytes(b"x" * 500)
    assert config_storage_bytes(tmp_path) >= 500
    assert config_storage_bytes(tmp_path / "missing") == 0     # absent dir -> 0


def test_traceability_ok():
    # C1 run record has a run_id + model_version (registry) -> traceable with require_registry=True
    c1 = {"result": {"run_id": "abc", "model_version": "1", "metrics": {"roc_auc": 0.9}},
          "resource": {"returncode": 0}}
    assert traceability_ok(c1, require_registry=True) is True
    # a record with no model_version cannot be traced to a registered model
    assert traceability_ok({"result": {}, "resource": {"returncode": 0}}, require_registry=True) is False
