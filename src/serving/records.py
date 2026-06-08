"""Lightweight JSONL records for serving (full monitoring is P6)."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Serialize appends: uvicorn serves /predict on multiple threads, so concurrent
# writers to the same JSONL must not interleave lines (would break parseability).
_WRITE_LOCK = threading.Lock()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
    with _WRITE_LOCK, path.open("a", encoding="utf-8") as f:
        f.write(line)


def build_prediction_record(response: dict[str, Any]) -> dict[str, Any]:
    return {"recorded_at": utc_now_iso(), "event": "predict", **response}


def build_deployment_record(*, model_type: str, model_version: str, loader: str) -> dict[str, Any]:
    return {"recorded_at": utc_now_iso(), "event": "deploy",
            "model_type": model_type, "model_version": model_version, "loader": loader}


def build_rollback_record(*, from_version: str, to_version: str, reason: str) -> dict[str, Any]:
    return {"recorded_at": utc_now_iso(), "event": "rollback",
            "from_version": from_version, "to_version": to_version, "reason": reason}
