# src/experiments/metrics/operational.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def latency_percentiles(predictions_jsonl: Path) -> dict[str, Any]:
    """Operational latency from serving predictions.jsonl (each record has latency_ms)."""
    lat = []
    for line in Path(predictions_jsonl).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "latency_ms" in rec:
            lat.append(float(rec["latency_ms"]))
    if not lat:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    arr = np.asarray(lat)
    return {"count": len(arr),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99))}


def dora_metrics(events: list[dict[str, Any]], *, window_days: float) -> dict[str, Any]:
    """DORA from a normalized event stream. Each event: {"kind": ..., "ts": seconds}.
    kinds: deploy | deploy_failed | recover_start | recover_end. Definitions are explicit so
    C0-vs-C1 (P8b) is comparable. lead_time/MTTR computed from paired ts when present."""
    deploys = [e for e in events if e["kind"] == "deploy"]
    failed = [e for e in events if e["kind"] == "deploy_failed"]
    changes = len(deploys) + len(failed)
    recover_pairs = [(e["ts"], r["ts"]) for e, r in zip(
        [e for e in events if e["kind"] == "recover_start"],
        [e for e in events if e["kind"] == "recover_end"])]
    mttr = float(np.mean([b - a for a, b in recover_pairs])) if recover_pairs else None
    return {"deploy_count": len(deploys),
            "deploy_frequency_per_day": len(deploys) / window_days if window_days else None,
            "change_fail_rate": (len(failed) / changes) if changes else 0.0,
            "mttr_seconds": mttr}


def events_from_logs(deployment_jsonl: Path | None, retrain_jsonl: Path | None) -> list[dict[str, Any]]:
    """Adapter: serving deployment_history.jsonl + monitoring retrain_history.jsonl -> normalized
    events. deploy = {event:deploy} or {event:retrain_deployed}; deploy_failed = retrain_rejected."""
    from datetime import datetime

    def _ts(iso: str) -> float:
        return datetime.fromisoformat(iso).timestamp()

    events: list[dict[str, Any]] = []
    for path, mapping in [(deployment_jsonl, {"deploy": "deploy"}),
                          (retrain_jsonl, {"retrain_deployed": "deploy", "retrain_rejected": "deploy_failed"})]:
        if not path or not Path(path).exists():
            continue
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            kind = mapping.get(rec.get("event"))
            if kind and rec.get("recorded_at"):
                events.append({"kind": kind, "ts": _ts(rec["recorded_at"])})
    return sorted(events, key=lambda e: e["ts"])
