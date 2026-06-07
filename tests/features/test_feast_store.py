from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.features.feast_store import (
    apply_and_materialize, get_online_features, get_training_features,
)

REPO_SRC = Path("src/features/feast_repo")


@pytest.fixture
def feast_repo(tmp_path, d1_like_df):
    """A temp Feast repo with a tiny handover feature parquet."""
    from src.features.builder import compute_ue_window_features
    from src.features.schema import WindowConfig

    repo = tmp_path / "feast_repo"
    repo.mkdir()
    (repo / "feature_store.yaml").write_text((REPO_SRC / "feature_store.yaml").read_text())
    feats = compute_ue_window_features(d1_like_df, WindowConfig())
    # Feast online needs non-null Int64 keys/features; fill timing NaNs for the demo
    feats["mean_inter_ho_s"] = feats["mean_inter_ho_s"].fillna(0.0)
    feats["std_inter_ho_s"] = feats["std_inter_ho_s"].fillna(0.0)
    src_parquet = repo / "data" / "handover_features.parquet"
    src_parquet.parent.mkdir(parents=True)
    feats.to_parquet(src_parquet, index=False)
    return repo, src_parquet, feats


def test_apply_materialize_and_online_retrieve(feast_repo):
    repo, src_parquet, feats = feast_repo
    store = apply_and_materialize(repo, src_parquet)
    online = get_online_features(store, ["111"])
    assert online["n_handover"].iloc[0] == 3
    assert online["pingpong_count"].iloc[0] == 1


def test_train_serve_consistency(feast_repo):
    """Online (serving) features must equal offline (training) features for same key+time."""
    repo, src_parquet, feats = feast_repo
    store = apply_and_materialize(repo, src_parquet)
    online = get_online_features(store, ["111"]).set_index("imsi")
    entity_df = pd.DataFrame({
        "imsi": ["111"],
        "event_timestamp": [datetime.now(timezone.utc)],
    })
    offline = get_training_features(store, entity_df).set_index("imsi")
    for f in ["n_handover", "n_unique_cells", "pingpong_count"]:
        assert online.loc["111", f] == offline.loc["111", f]
