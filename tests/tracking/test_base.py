import inspect

import pytest

from src.tracking.base import BaseTracker


def test_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        BaseTracker()


def test_interface_methods_exist():
    for name in ["init_experiment", "log_params", "log_metrics", "log_dataset",
                 "log_artifact", "register_model", "end_experiment"]:
        assert hasattr(BaseTracker, name)
        assert inspect.isfunction(getattr(BaseTracker, name))


def test_partial_subclass_cannot_instantiate():
    class Partial(BaseTracker):
        def init_experiment(self, config):
            ...

    with pytest.raises(TypeError):
        Partial()
