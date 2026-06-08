from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GateConfig:
    """Eval-gate thresholds + the target promotion alias. min_pr_auc=0 disables the PR-AUC
    check. The candidate alias is set at registration time (training schema CANDIDATE_STATE),
    so it is intentionally NOT a field here — the gate only controls the staging promotion."""
    min_roc_auc: float = 0.70
    min_pr_auc: float = 0.0
    staging_alias: str = "staging"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
