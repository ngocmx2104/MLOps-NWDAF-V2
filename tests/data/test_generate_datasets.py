import json

from src.data.generate_datasets import EXPERIMENT_SEEDS, generate_feature_datasets


def test_feature_datasets_count_and_manifest(tmp_path):
    results = generate_feature_datasets(tmp_path, n_samples=50)
    # 10 seeds + 3 drift variants = 13 parquet files
    assert len(list(tmp_path.glob("features_*.parquet"))) == 13
    assert len(results) == 13
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["dataset"] == "E"
    assert manifest["n_seeds"] == len(EXPERIMENT_SEEDS)


def test_seeds_are_fixed():
    assert EXPERIMENT_SEEDS == [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
