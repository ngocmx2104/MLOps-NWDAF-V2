import pytest
from src.tracking import create_tracker
from src.tracking.noop import NoopTracker

def test_default_backend_is_noop(monkeypatch):
    monkeypatch.delenv("MLOPS_BACKEND", raising=False)
    assert isinstance(create_tracker(), NoopTracker)

def test_explicit_noop():
    assert isinstance(create_tracker("noop"), NoopTracker)

def test_env_var_selects_backend(monkeypatch):
    monkeypatch.setenv("MLOPS_BACKEND", "noop")
    assert isinstance(create_tracker(), NoopTracker)

def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown"):
        create_tracker("nope")


def test_case_insensitive_backend():
    assert isinstance(create_tracker("NOOP"), NoopTracker)
    assert isinstance(create_tracker("Noop"), NoopTracker)
