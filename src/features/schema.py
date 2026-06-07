"""Feature schema and version definition for the handover use case.

This module is the single source of truth for:
- which features D2 contains
- the feature version identifier
- the window configuration used to aggregate handover events per subscriber

Reference: docs/DATASET_STRATEGY.md (D2 Feature Dataset)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Feature version -- bump when feature logic or schema changes
# ---------------------------------------------------------------------------
FEATURE_VERSION = "ho_features_v1"


@dataclass(frozen=True, slots=True)
class FeatureDef:
    """Definition of a single feature column."""

    name: str
    dtype: str
    description: str


@dataclass(frozen=True)
class WindowConfig:
    """Sliding / fixed-window parameters for subscriber-level aggregation."""

    window_seconds: int = 300
    pingpong_max_gap_seconds: int = 30

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# D2 feature schema -- locked for v1
# ---------------------------------------------------------------------------
D2_KEY_COLUMNS: tuple[FeatureDef, ...] = (
    FeatureDef("imsi", "string", "Subscriber identity (grouping key)"),
    FeatureDef("window_start", "datetime64[ns, UTC]", "Start of the aggregation window"),
)

D2_FEATURE_COLUMNS: tuple[FeatureDef, ...] = (
    FeatureDef("n_handover", "int64", "Total handover events in the window"),
    FeatureDef("n_unique_cells", "int64", "Number of distinct cell IDs visited"),
    FeatureDef("pingpong_count", "int64", "Ping-pong handover occurrences (A->B->A within gap threshold)"),
    FeatureDef("pingpong_rate", "float64", "Ratio of ping-pong count to total handovers"),
    FeatureDef("mean_inter_ho_s", "float64", "Mean inter-handover time in seconds"),
    FeatureDef("std_inter_ho_s", "float64", "Std dev of inter-handover time in seconds"),
    FeatureDef("entropy_cell_seq", "float64", "Shannon entropy of the cell-visit sequence"),
)

D2_PROVENANCE_COLUMNS: tuple[FeatureDef, ...] = (
    FeatureDef("feature_version", "string", "Feature schema version tag"),
    FeatureDef("source_snapshot_id", "string", "D1 snapshot ID used as input"),
)

# Convenience lists
D2_ALL_COLUMNS: tuple[str, ...] = tuple(
    c.name for c in D2_KEY_COLUMNS + D2_FEATURE_COLUMNS + D2_PROVENANCE_COLUMNS
)

D2_NUMERIC_FEATURE_NAMES: tuple[str, ...] = tuple(
    c.name for c in D2_FEATURE_COLUMNS
)

# ---------------------------------------------------------------------------
# D5 weak-label schema -- locked for v1
# ---------------------------------------------------------------------------
WEAK_LABEL_VERSION = "weak_label_v1"


@dataclass(frozen=True)
class WeakLabelConfig:
    """Thresholds for rule-based weak labeling."""

    min_pingpong_count: int = 1
    min_handover: int = 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


D5_LABEL_COLUMNS: tuple[FeatureDef, ...] = (
    FeatureDef("weak_label", "int64", "1 = anomalous (ping-pong), 0 = normal; rule-derived, NOT ground truth"),
)

D5_PROVENANCE_COLUMNS: tuple[FeatureDef, ...] = (
    FeatureDef("weak_label_version", "string", "Weak label rule version tag"),
    FeatureDef("source_feature_version", "string", "D2 feature version used as input"),
    FeatureDef("source_snapshot_id", "string", "D1 snapshot ID (transitive lineage)"),
)


DEFAULT_WINDOW_CONFIG = WindowConfig()
DEFAULT_WEAK_LABEL_CONFIG = WeakLabelConfig()
