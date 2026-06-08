"""Eval-gate: decide if a trained model is good enough to deploy, and (on pass) promote it
from the 'candidate' alias to 'staging'. Shared by the CD pipeline (new model) AND the P6
auto-retrain loop (retrained model) so promotion governance lives in ONE place."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.cicd.schema import GateConfig
from src.tracking.base import BaseTracker


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    promoted_to: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "reasons": self.reasons,
                "metrics": self.metrics, "promoted_to": self.promoted_to}


def evaluate_gate(metrics: dict[str, Any], cfg: GateConfig | None = None) -> GateResult:
    cfg = cfg or GateConfig()
    roc = float(metrics.get("roc_auc", 0.0))
    pr = float(metrics.get("pr_auc", 0.0))
    reasons: list[str] = []
    if roc < cfg.min_roc_auc:
        reasons.append(f"roc_auc {roc:.4f} < min {cfg.min_roc_auc}")
    if cfg.min_pr_auc > 0.0 and pr < cfg.min_pr_auc:
        reasons.append(f"pr_auc {pr:.4f} < min {cfg.min_pr_auc}")
    return GateResult(passed=not reasons, reasons=reasons,
                      metrics={"roc_auc": roc, "pr_auc": pr})


def _version_of(model_version: str | None) -> str | None:
    if not model_version:
        return None
    # `or None` guards a trailing-slash URI ("models:/name/3/" -> "" -> None) so a blank
    # version never reaches promote_model.
    return model_version.rsplit("/", 1)[-1] or None  # "models:/name/3" -> "3"


def run_eval_gate(*, metrics: dict[str, Any], model_name: str | None,
                  model_version: str | None, tracker: BaseTracker,
                  cfg: GateConfig | None = None) -> GateResult:
    cfg = cfg or GateConfig()
    result = evaluate_gate(metrics, cfg)
    version = _version_of(model_version)
    if result.passed and model_name and version is not None:
        # promote_model returns None on the NoopTracker (C0 ablation): the gate still
        # evaluates, but no alias side-effect happens — intended for the C0/C1 contrast.
        result.promoted_to = tracker.promote_model(model_name, cfg.staging_alias, version)
    return result
