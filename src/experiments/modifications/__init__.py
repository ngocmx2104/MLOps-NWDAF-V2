"""NWDAF-specific modifications for Exp-6 modifiability (RQ4c).

Each modification targets a real file in the pipeline and makes a
concrete, meaningful edit that reflects a realistic operational change
to the 5G handover anomaly-detection system.

Modifications:
- add_feature       (section=data)      — add a new feature to D2_FEATURE_COLUMNS
                                          in src/features/schema.py
- change_pingpong_rule (section=label)  — tighten the weak-label threshold in
                                          src/features/weak_labels.py
- change_psi_threshold (section=monitoring) — lower psi_alert in
                                          src/monitoring/schema.py

test_targets for each mod:
- add_feature:           tests/experiments/test_schema_records.py
  (schema tests import src.experiments.schema; a feature schema change
   exercises cross-module compatibility; we expect 0 regressions because
   adding a feature def does not break existing code paths)
- change_pingpong_rule:  tests/experiments/test_schema_records.py
  (same rationale — threshold change is isolated, 0 expected regressions)
- change_psi_threshold:  tests/experiments/test_schema_records.py
  (same rationale — monitoring constant change, 0 expected regressions)
"""
from __future__ import annotations

from pathlib import Path

from src.experiments.exp6_modifiability import Modification

# ---------------------------------------------------------------------------
# Mod 1: add_feature (section=data)
# Adds a new FeatureDef entry to D2_FEATURE_COLUMNS in src/features/schema.py.
# Rationale: extending the feature set for richer handover characterisation
# (e.g. max_inter_ho_s captures the worst-case gap, relevant for anomaly scoring).
# ---------------------------------------------------------------------------
_ADD_FEATURE_ANCHOR = (
    '    FeatureDef("entropy_cell_seq", "float64", '
    '"Shannon entropy of the cell-visit sequence"),\n'
    ")"
)
_ADD_FEATURE_REPLACEMENT = (
    '    FeatureDef("entropy_cell_seq", "float64", '
    '"Shannon entropy of the cell-visit sequence"),\n'
    '    FeatureDef("max_inter_ho_s", "float64", '
    '"Max inter-handover time in seconds; captures worst-case gap"),\n'
    ")"
)


def _apply_add_feature(root: Path) -> None:
    path = root / "src/features/schema.py"
    text = path.read_text()
    assert _ADD_FEATURE_ANCHOR in text, (
        f"add_feature anchor not found in {path}; mod needs updating"
    )
    path.write_text(text.replace(_ADD_FEATURE_ANCHOR, _ADD_FEATURE_REPLACEMENT, 1))


MOD_ADD_FEATURE = Modification(
    mod_id="add_feature",
    section="data",
    apply=_apply_add_feature,
    test_targets=["tests/experiments/test_schema_records.py"],
)

# ---------------------------------------------------------------------------
# Mod 2: change_pingpong_rule (section=label)
# Tightens the weak-label threshold: min_pingpong_count 1 -> 2 and
# min_handover 3 -> 5 in the default WeakLabelConfig in src/features/schema.py.
# Rationale: reduce false positives by requiring stronger ping-pong evidence.
# ---------------------------------------------------------------------------
_PP_RULE_ANCHOR = (
    "    min_pingpong_count: int = 1\n"
    "    min_handover: int = 3"
)
_PP_RULE_REPLACEMENT = (
    "    min_pingpong_count: int = 2\n"
    "    min_handover: int = 5"
)


def _apply_change_pingpong_rule(root: Path) -> None:
    path = root / "src/features/schema.py"
    text = path.read_text()
    assert _PP_RULE_ANCHOR in text, (
        f"change_pingpong_rule anchor not found in {path}; mod needs updating"
    )
    path.write_text(text.replace(_PP_RULE_ANCHOR, _PP_RULE_REPLACEMENT, 1))


MOD_CHANGE_PINGPONG_RULE = Modification(
    mod_id="change_pingpong_rule",
    section="label",
    apply=_apply_change_pingpong_rule,
    test_targets=["tests/experiments/test_schema_records.py"],
)

# ---------------------------------------------------------------------------
# Mod 3: change_psi_threshold (section=monitoring)
# Lowers PSI_ALERT from 0.25 to 0.20 (equal to PSI_WARN) in
# src/monitoring/schema.py to be more sensitive to distribution drift.
# Rationale: tighten alerting in a high-availability 5G NWDAF environment.
# ---------------------------------------------------------------------------
_PSI_ANCHOR = "PSI_ALERT = 0.25"
_PSI_REPLACEMENT = "PSI_ALERT = 0.20"


def _apply_change_psi_threshold(root: Path) -> None:
    path = root / "src/monitoring/schema.py"
    text = path.read_text()
    assert _PSI_ANCHOR in text, (
        f"change_psi_threshold anchor not found in {path}; mod needs updating"
    )
    path.write_text(text.replace(_PSI_ANCHOR, _PSI_REPLACEMENT, 1))


MOD_CHANGE_PSI_THRESHOLD = Modification(
    mod_id="change_psi_threshold",
    section="monitoring",
    apply=_apply_change_psi_threshold,
    test_targets=["tests/experiments/test_schema_records.py"],
)

# ---------------------------------------------------------------------------
# All three mods as a list (for run_exp6)
# ---------------------------------------------------------------------------
NWDAF_MODS: list[Modification] = [
    MOD_ADD_FEATURE,
    MOD_CHANGE_PINGPONG_RULE,
    MOD_CHANGE_PSI_THRESHOLD,
]
