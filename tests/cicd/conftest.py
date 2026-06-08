import pytest


@pytest.fixture
def mlflow_sqlite(tmp_path, monkeypatch):
    import mlflow
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    return tmp_path
