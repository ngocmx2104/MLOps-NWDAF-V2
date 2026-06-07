"""Extract statistical profile from real EBS files.

Reads the 3 raw EBS files, computes distributions for all features
relevant to synthetic data generation, and saves as profile.json.

Usage:
    python -m src.data.real_profile
    python -m src.data.real_profile --output artifacts/synthetic/profile.json
"""
from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Defaults — real EBS file locations
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "data" / "raw_ebs"
_DEFAULT_EBS_FILES = sorted(_DATA_DIR.glob("*_ebs"))
_DEFAULT_OUTPUT = _PROJECT_ROOT / "artifacts" / "synthetic" / "profile.json"

# Positional field indices (from src/ingestion/schema.py)
_F = {
    "event_id": 0,
    "event_result": 1,
    "duration": 2,
    "sub_type": 4,
    "msisdn": 5,
    "imsi": 6,
    "mmegi": 9,
    "mmec": 10,
    "eci": 12,
    "event_time": 48,
    "date_hour": 51,
}


def _parse_ebs_lines(file_path: Path) -> list[list[str]]:
    """Parse a raw EBS file into list of field lists."""
    records = []
    with file_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.rstrip("\n\r")
            if stripped:
                records.append(stripped.split(";"))
    return records


def _safe_int(val: str) -> int | None:
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _percentiles(arr: np.ndarray) -> dict[str, float]:
    if len(arr) == 0:
        return {}
    return {
        f"p{p}": round(float(np.percentile(arr, p)), 2)
        for p in [5, 10, 25, 50, 75, 90, 95, 99]
    }


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------
def extract_profile(ebs_files: list[Path] | None = None) -> dict[str, Any]:
    """Extract statistical profile from real EBS files.

    Returns a dict ready to be serialised as profile.json.
    """
    ebs_files = ebs_files or _DEFAULT_EBS_FILES

    # ---- Parse all files ----
    all_records: list[list[str]] = []
    per_file_stats: list[dict[str, Any]] = []

    for fp in ebs_files:
        if not fp.exists():
            continue
        records = _parse_ebs_lines(fp)
        ho_records = [r for r in records if len(r) > _F["event_id"] and r[_F["event_id"]] == "l_handover"]
        per_file_stats.append({
            "file": fp.name,
            "total_lines": len(records),
            "handover_lines": len(ho_records),
        })
        all_records.extend(records)

    # ---- Event type distribution ----
    event_types = collections.Counter(
        r[_F["event_id"]] for r in all_records if len(r) > _F["event_id"]
    )
    total_events = sum(event_types.values())
    event_type_mix = {
        evt: round(cnt / total_events, 4)
        for evt, cnt in event_types.most_common()
    }

    # ---- Filter handover events ----
    ho = [r for r in all_records if len(r) > _F["event_time"] and r[_F["event_id"]] == "l_handover"]

    # ---- IMSI-level analysis ----
    imsi_events: dict[str, list[dict]] = collections.defaultdict(list)
    for r in ho:
        imsi = r[_F["imsi"]]
        et = _safe_int(r[_F["event_time"]])
        if et is None:
            continue
        imsi_events[imsi].append({
            "eci": r[_F["eci"]],
            "event_time_ms": et,
            "duration": _safe_int(r[_F["duration"]]) or 0,
            "sub_type": r[_F["sub_type"]],
            "result": r[_F["event_result"]],
        })

    # Sort each IMSI by time
    for imsi in imsi_events:
        imsi_events[imsi].sort(key=lambda e: e["event_time_ms"])

    # ---- HO count per IMSI ----
    ho_counts = np.array([len(evts) for evts in imsi_events.values()])
    ho_count_buckets = {
        "1": round(float(np.mean(ho_counts == 1)), 4),
        "2-3": round(float(np.mean((ho_counts >= 2) & (ho_counts <= 3))), 4),
        "4-7": round(float(np.mean((ho_counts >= 4) & (ho_counts <= 7))), 4),
        "8-15": round(float(np.mean((ho_counts >= 8) & (ho_counts <= 15))), 4),
        "16-30": round(float(np.mean((ho_counts >= 16) & (ho_counts <= 30))), 4),
        "31+": round(float(np.mean(ho_counts >= 31)), 4),
    }

    # ---- Cells per IMSI ----
    cells_per_imsi = np.array([
        len({e["eci"] for e in evts}) for evts in imsi_events.values()
    ])

    # ---- Duration ----
    durations = np.array([e["duration"] for evts in imsi_events.values() for e in evts])

    # ---- Sub-type (x2/s1) ----
    sub_types = collections.Counter(e["sub_type"] for evts in imsi_events.values() for e in evts)
    total_ho = sum(sub_types.values())

    # ---- Event result (success/reject) ----
    results = collections.Counter(e["result"] for evts in imsi_events.values() for e in evts)

    # ---- Ping-pong detection (A->B->A within 30s) ----
    pp_imsi_count = 0
    total_pp_sequences = 0
    pp_inter_ho_times: list[float] = []

    for imsi, events in imsi_events.items():
        cells = [e["eci"] for e in events]
        times = [e["event_time_ms"] for e in events]
        pp_count = 0
        for i in range(len(cells) - 2):
            if cells[i] == cells[i + 2] and cells[i] != cells[i + 1]:
                gap_s = (times[i + 2] - times[i]) / 1000.0
                if gap_s <= 30:
                    pp_count += 1
                    pp_inter_ho_times.append((times[i + 1] - times[i]) / 1000.0)
                    pp_inter_ho_times.append((times[i + 2] - times[i + 1]) / 1000.0)
        if pp_count > 0:
            pp_imsi_count += 1
            total_pp_sequences += pp_count

    pp_inter_arr = np.array(pp_inter_ho_times) if pp_inter_ho_times else np.array([])

    # ---- Inter-HO timing (all IMSIs with >=2 HOs) ----
    all_inter_ho: list[float] = []
    for events in imsi_events.values():
        if len(events) < 2:
            continue
        times = [e["event_time_ms"] for e in events]
        for i in range(1, len(times)):
            all_inter_ho.append((times[i] - times[i - 1]) / 1000.0)
    inter_ho_arr = np.array(all_inter_ho) if all_inter_ho else np.array([])

    # ---- Cell pool ----
    all_ecis = list({e["eci"] for evts in imsi_events.values() for e in evts})
    all_ecis.sort()

    # ---- IMSI prefix analysis ----
    imsi_list = list(imsi_events.keys())
    imsi_prefix = imsi_list[0][:5] if imsi_list else "45204"
    imsi_lengths = collections.Counter(len(i) for i in imsi_list)

    # ---- Topology ----
    mmegis = list({r[_F["mmegi"]] for r in ho if len(r) > _F["mmegi"] and r[_F["mmegi"]]})
    mmecs = list({r[_F["mmec"]] for r in ho if len(r) > _F["mmec"] and r[_F["mmec"]]})

    # ---- Event time range ----
    all_times = [e["event_time_ms"] for evts in imsi_events.values() for e in evts]
    time_arr = np.array(all_times)

    # ---- Build profile ----
    n_files = len(per_file_stats)
    n_minutes = n_files  # 1 file = 1 minute

    profile: dict[str, Any] = {
        "source": {
            "files": [s["file"] for s in per_file_stats],
            "n_files": n_files,
            "duration_minutes": n_minutes,
            "total_events": total_events,
            "total_handovers": len(ho),
            "per_file": per_file_stats,
        },
        "event_type_mix": event_type_mix,
        "handover": {
            "events_per_minute": round(len(ho) / max(1, n_minutes)),
            "imsi_per_minute": round(len(imsi_events) / max(1, n_minutes)),
            "unique_imsis": len(imsi_events),
            "x2_ratio": round(sub_types.get("x2", 0) / max(1, total_ho), 4),
            "s1_ratio": round(sub_types.get("s1", 0) / max(1, total_ho), 4),
            "success_rate": round(results.get("success", 0) / max(1, total_ho), 4),
            "ho_count_per_imsi": {
                "mean": round(float(ho_counts.mean()), 2),
                "std": round(float(ho_counts.std()), 2),
                "median": round(float(np.median(ho_counts)), 1),
                "max": int(ho_counts.max()),
                "min": int(ho_counts.min()),
                "buckets": ho_count_buckets,
            },
            "cells_per_imsi": {
                "mean": round(float(cells_per_imsi.mean()), 2),
                "std": round(float(cells_per_imsi.std()), 2),
                "median": round(float(np.median(cells_per_imsi)), 1),
                "max": int(cells_per_imsi.max()),
            },
            "duration_ms": {
                "mean": round(float(durations.mean()), 1),
                "std": round(float(durations.std()), 1),
                **_percentiles(durations),
            },
            "inter_ho_s": {
                "mean": round(float(inter_ho_arr.mean()), 2) if len(inter_ho_arr) else None,
                "std": round(float(inter_ho_arr.std()), 2) if len(inter_ho_arr) else None,
                **_percentiles(inter_ho_arr),
            },
            "pingpong": {
                "imsi_with_pp": pp_imsi_count,
                "imsi_pp_ratio": round(pp_imsi_count / max(1, len(imsi_events)), 4),
                "total_pp_sequences": total_pp_sequences,
                "pp_event_ratio": round(total_pp_sequences / max(1, len(ho)), 4),
                "pp_inter_ho_s": {
                    "mean": round(float(pp_inter_arr.mean()), 2) if len(pp_inter_arr) else None,
                    "std": round(float(pp_inter_arr.std()), 2) if len(pp_inter_arr) else None,
                    "min": round(float(pp_inter_arr.min()), 2) if len(pp_inter_arr) else None,
                    "max": round(float(pp_inter_arr.max()), 2) if len(pp_inter_arr) else None,
                },
            },
        },
        "topology": {
            "imsi_prefix": imsi_prefix,
            "imsi_length": max(imsi_lengths, key=imsi_lengths.get) if imsi_lengths else 15,
            "cell_pool": all_ecis,
            "cell_pool_size": len(all_ecis),
            "mmegi": mmegis,
            "mmec": mmecs,
        },
        "time_range": {
            "min_epoch_ms": int(time_arr.min()),
            "max_epoch_ms": int(time_arr.max()),
            "range_seconds": round((time_arr.max() - time_arr.min()) / 1000, 1),
        },
    }
    return profile


def save_profile(profile: dict[str, Any], output_path: Path | None = None) -> Path:
    """Save profile dict to JSON file."""
    output_path = output_path or _DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract real EBS statistical profile")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--ebs-dir", type=Path, default=_DATA_DIR)
    args = parser.parse_args()

    ebs_files = sorted(args.ebs_dir.glob("*_ebs")) if args.ebs_dir.exists() else _DEFAULT_EBS_FILES

    print(f"Extracting profile from {len(ebs_files)} EBS files...")
    profile = extract_profile(ebs_files)
    out = save_profile(profile, args.output)
    print(f"Profile saved to {out}")
    print(f"  Total events: {profile['source']['total_events']}")
    print(f"  Handovers: {profile['source']['total_handovers']}")
    print(f"  Unique IMSIs: {profile['handover']['unique_imsis']}")
    print(f"  Cell pool: {profile['topology']['cell_pool_size']}")
    print(f"  Ping-pong IMSIs: {profile['handover']['pingpong']['imsi_with_pp']}")
