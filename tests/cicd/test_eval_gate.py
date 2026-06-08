import pytest

from src.cicd.eval_gate import evaluate_gate, run_eval_gate
from src.cicd.schema import GateConfig


def test_evaluate_gate_pass():
    r = evaluate_gate({"roc_auc": 0.95, "pr_auc": 0.8}, GateConfig(min_roc_auc=0.7))
    assert r.passed and not r.reasons


def test_evaluate_gate_fail_roc():
    r = evaluate_gate({"roc_auc": 0.4}, GateConfig(min_roc_auc=0.7))
    assert not r.passed and any("roc_auc" in x for x in r.reasons)


def test_run_eval_gate_promotes_on_pass(mlflow_sqlite, tmp_path):
    from src.tracking.mlflow_tracker import MLflowTracker
    from src.tracking.schema import ExperimentConfig
    from mlflow.tracking import MlflowClient

    tr = MLflowTracker()
    tr.init_experiment(ExperimentConfig(experiment_name="t", run_name="r", backend="mlflow"))
    m = tmp_path / "m.joblib"
    m.write_bytes(b"x")
    uri_v = tr.register_model(str(m), "gate_model", alias="candidate")
    tr.end_experiment()

    res = run_eval_gate(metrics={"roc_auc": 0.99}, model_name="gate_model",
                        model_version=uri_v, tracker=MLflowTracker(),
                        cfg=GateConfig(min_roc_auc=0.7))
    assert res.passed
    assert str(MlflowClient().get_model_version_by_alias("gate_model", "staging").version) \
        == uri_v.rsplit("/", 1)[-1]


def test_run_eval_gate_no_promote_on_fail(mlflow_sqlite, tmp_path):
    from src.tracking.mlflow_tracker import MLflowTracker
    from src.tracking.schema import ExperimentConfig
    from mlflow.tracking import MlflowClient
    from mlflow.exceptions import MlflowException

    tr = MLflowTracker()
    tr.init_experiment(ExperimentConfig(experiment_name="t", run_name="r", backend="mlflow"))
    m = tmp_path / "m.joblib"
    m.write_bytes(b"x")
    uri_v = tr.register_model(str(m), "bad_model", alias="candidate")
    tr.end_experiment()

    res = run_eval_gate(metrics={"roc_auc": 0.30}, model_name="bad_model",
                        model_version=uri_v, tracker=MLflowTracker(),
                        cfg=GateConfig(min_roc_auc=0.7))
    assert not res.passed and res.promoted_to is None
    with pytest.raises(MlflowException):  # 'staging' alias never created for a rejected model
        MlflowClient().get_model_version_by_alias("bad_model", "staging")
