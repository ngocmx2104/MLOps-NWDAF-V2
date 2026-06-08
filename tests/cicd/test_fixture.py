from pathlib import Path

import pandas as pd

from src.cicd.fixture import make_fixture_dataset
from src.features.schema import D2_FEATURE_COLUMNS


def test_fixture_has_features_and_labels(tmp_path):
    out = make_fixture_dataset(tmp_path / "f.parquet", n=200, seed=0)
    df = pd.read_parquet(out)
    for col in [c.name for c in D2_FEATURE_COLUMNS]:
        assert col in df.columns
    assert {"label", "weak_label"}.issubset(df.columns)
    assert 0 < df["label"].sum() < len(df)  # both classes present


def test_fixture_iforest_passes_gate(tmp_path):
    """The fixture must yield a model good enough that the gate PASSES (roc_auc>=0.7),
    otherwise the CD-pipeline smoke would always block deploy."""
    out = make_fixture_dataset(tmp_path / "f.parquet", n=300, seed=0)
    from src.training.pipeline import run_training
    from src.training.schema import TrainingConfig
    res = run_training(Path(out), model_type="iforest", backend="noop",
                       output_dir=tmp_path / "m",
                       cfg=TrainingConfig(use_labels_for_evaluation=True, label_column="label"))
    assert res["metrics"]["roc_auc"] >= 0.7
