from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.experiments.schema import EXPERIMENT_RECORD_VERSION, RESULT_SUMMARY_VERSION


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return path


def append_jsonl(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    return path


def build_run_record(*, experiment_id: str, run_index: int, seed: int,
                     metrics: dict[str, Any] | None = None,
                     resource: dict[str, Any] | None = None,
                     config_label: str = "") -> dict[str, Any]:
    return {"record_version": EXPERIMENT_RECORD_VERSION, "recorded_at": utc_now_iso(),
            "experiment_id": experiment_id, "config_label": config_label,
            "run_index": run_index, "seed": seed,
            "metrics": metrics or {}, "resource": resource or {}}


def build_result_summary(*, experiment_id: str, configs: dict[str, Any],
                         notes: list[str] | None = None) -> dict[str, Any]:
    return {"summary_version": RESULT_SUMMARY_VERSION, "generated_at": utc_now_iso(),
            "experiment_id": experiment_id, "configs": configs, "notes": notes or []}
