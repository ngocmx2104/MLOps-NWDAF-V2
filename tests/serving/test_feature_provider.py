import pytest

from src.serving.feature_provider import FeastOnlineProvider
from src.training.schema import FEATURE_COLUMNS


def test_online_lookup_returns_feature_row(materialized_repo):
    provider = FeastOnlineProvider(materialized_repo)
    rows = provider.get(["111"])
    assert len(rows) == 1
    assert set(FEATURE_COLUMNS).issubset(rows[0].keys())
    assert rows[0]["n_handover"] == 3 and rows[0]["pingpong_count"] == 1


def test_online_lookup_unknown_imsi_raises(materialized_repo):
    provider = FeastOnlineProvider(materialized_repo)
    with pytest.raises(KeyError):
        provider.get(["999999"])  # not materialized -> null features
