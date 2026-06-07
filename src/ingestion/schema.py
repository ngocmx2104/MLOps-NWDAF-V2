"""EBS positional schema -- single source of truth for raw EBS column layout.

This module defines the 52-field positional schema observed in actual raw EBS
log files.  The field order is determined by inspecting the raw data, NOT by
the ``EBS_fields_explained.xlsx`` spreadsheet (which is a semantic dictionary
only and has a different column order / field count).

Reference: docs/EBS_SCHEMA_SPEC.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class FieldDef:
    """Definition of a single EBS positional field."""

    position: int
    name: str
    confidence: str  # "high", "medium-high", "medium", "low-medium", "low"


# ---------------------------------------------------------------------------
# Positional schema v1 -- locked from EBS_SCHEMA_SPEC.md Section 4
# ---------------------------------------------------------------------------
EBS_POSITIONAL_FIELDS: tuple[FieldDef, ...] = (
    FieldDef(0,  "event_id",                    "high"),
    FieldDef(1,  "event_result",                "high"),
    FieldDef(2,  "duration",                    "high"),
    FieldDef(3,  "request_retries",             "high"),
    FieldDef(4,  "sub_type",                    "high"),
    FieldDef(5,  "msisdn",                      "high"),
    FieldDef(6,  "imsi",                        "high"),
    FieldDef(7,  "mtmsi",                       "high"),
    FieldDef(8,  "imeisv",                      "high"),
    FieldDef(9,  "mmegi",                       "high"),
    FieldDef(10, "mmec",                        "high"),
    FieldDef(11, "tac",                         "high"),
    FieldDef(12, "eci",                         "high"),
    FieldDef(13, "sgw",                         "medium-high"),
    FieldDef(14, "sgsn",                        "medium"),
    FieldDef(15, "l_cause_prot_type",           "medium-high"),
    FieldDef(16, "cause_code",                  "medium"),
    FieldDef(17, "sub_cause_code",              "medium"),
    FieldDef(18, "apn",                         "high"),
    FieldDef(19, "pdn_default_bearer_id",       "medium-high"),
    FieldDef(20, "pdn_paa",                     "medium-high"),
    FieldDef(21, "pdn_pgw",                     "medium-high"),
    FieldDef(22, "originating_cause_prot_type", "medium"),
    FieldDef(23, "originating_cause_code",      "medium"),
    FieldDef(24, "csg_id",                      "medium"),
    FieldDef(25, "old_mtmsi",                   "medium"),
    FieldDef(26, "old_tac",                     "medium"),
    FieldDef(27, "old_mmegi",                   "medium"),
    FieldDef(28, "old_mmec",                    "medium"),
    FieldDef(29, "old_eci",                     "medium"),
    FieldDef(30, "old_sgw",                     "medium"),
    FieldDef(31, "old_sgsn",                    "medium"),
    FieldDef(32, "msc",                         "medium"),
    FieldDef(33, "target_lac",                  "medium"),
    FieldDef(34, "lac",                         "low-medium"),
    FieldDef(35, "rac",                         "low-medium"),
    FieldDef(36, "ci",                          "low-medium"),
    FieldDef(37, "handover_node_role",          "high"),
    FieldDef(38, "handover_rat_change_type",    "high"),
    FieldDef(39, "handover_sgw_change_type",    "high"),
    FieldDef(40, "target_rnc_id",               "medium"),
    FieldDef(41, "target_macro_enodeb_id",      "low-medium"),
    FieldDef(42, "srvcc_type",                  "low"),
    FieldDef(43, "cs_fallback_service_type",    "low"),
    FieldDef(44, "csfb_triggered",              "low"),
    FieldDef(45, "l_service_req_trigger",        "low"),
    FieldDef(46, "combined_tau_type",           "low"),
    FieldDef(47, "detach_trigger",              "low"),
    FieldDef(48, "event_time",                  "high"),
    FieldDef(49, "paging_attempts",             "medium"),
    FieldDef(50, "ue_requested_apn",            "medium"),
    FieldDef(51, "date_hour",                   "high"),
)

SCHEMA_VERSION = "ebs_raw_positional_v1"
EXPECTED_FIELD_COUNT = len(EBS_POSITIONAL_FIELDS)  # 52

# Quick-access helpers
FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in EBS_POSITIONAL_FIELDS)
FIELD_NAME_TO_POS: dict[str, int] = {f.name: f.position for f in EBS_POSITIONAL_FIELDS}

# Columns required for the handover use case (must be non-null in most records)
USE_CASE_REQUIRED_FIELDS: tuple[str, ...] = (
    "event_id",
    "imsi",
    "event_time",
)

# All core + provenance columns that D1 canonical snapshot must contain
D1_CORE_COLUMNS: tuple[str, ...] = (
    "event_id",
    "event_result",
    "sub_type",
    "imsi",
    "tac",
    "eci",
    "event_time",
    "event_ts",
    "date_hour",
)

D1_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "dataset_snapshot_id",
    "schema_version",
    "source_file",
    "raw_field_count",
    "record_index",
)

D1_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "msisdn",
    "mtmsi",
    "imeisv",
    "mmegi",
    "mmec",
    "sgw",
    "handover_node_role",
    "handover_rat_change_type",
    "handover_sgw_change_type",
    "target_rnc_id",
    "duration",
    "request_retries",
    "ci",
    "old_eci",
    "old_tac",
)


def get_field_names() -> Sequence[str]:
    """Return the ordered list of 52 field names."""
    return FIELD_NAMES
