from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GateConfig:
    """Eval-gate thresholds + registry alias names. min_pr_auc=0 disables the PR-AUC check."""
    min_roc_auc: float = 0.70
    min_pr_auc: float = 0.0
    candidate_alias: str = "candidate"
    staging_alias: str = "staging"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
