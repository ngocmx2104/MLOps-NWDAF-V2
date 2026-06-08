"""FeastOnlineProvider — low-latency online feature lookup by IMSI (serving path).

Wraps src.features.feast_store.get_online_features and returns plain feature-dict
rows that LoadedModel.predict consumes. Raises KeyError if an IMSI has no online row.
"""
from __future__ import annotations

from pathlib import Path

from src.features.feast_store import get_online_features
from src.training.schema import FEATURE_COLUMNS

_FEATS = list(FEATURE_COLUMNS)


class FeastOnlineProvider:
    def __init__(self, repo_path: str | Path) -> None:
        # feast is a hard serving dependency (transitively imported via feast_store at
        # module load, and installed by the Docker image's [feast] extra). FeatureStore
        # is imported here rather than at class level only to keep __init__ self-contained.
        from feast import FeatureStore
        self.store = FeatureStore(repo_path=str(repo_path))

    def get(self, imsis: list[str]) -> list[dict[str, float]]:
        frame = get_online_features(self.store, imsis)
        rows: list[dict[str, float]] = []
        for imsi, record in zip(imsis, frame.to_dict("records")):
            if any(record.get(c) is None for c in _FEATS):
                raise KeyError(f"No online features for imsi={imsi!r}")
            rows.append({c: float(record[c]) for c in _FEATS})
        return rows
