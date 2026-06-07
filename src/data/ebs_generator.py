"""Raw EBS Synthetic Generator.

Generates raw EBS files (52-field, semicolon-delimited) that pass through
the full pipeline: Parse -> Feature Engineering -> Weak Labeling -> Training.

Only 8 fields are realistic (event_id, event_result, duration, sub_type,
imsi, eci, event_time, date_hour). The remaining 44 are placeholders.

Supports:
  - 3 subscriber behavior types: Normal, Mobile, PingPong
  - 3 drift patterns: sudden, gradual, recurring
  - Multi-source simulation via MMEGI parameter
  - Ground truth labels for each IMSI

Usage:
    python -m src.data.ebs_generator --scenario baseline --n-files 20
    python -m src.data.ebs_generator --scenario sudden_drift --n-files 30
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PROFILE = _PROJECT_ROOT / "artifacts" / "synthetic" / "profile.json"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "artifacts" / "synthetic" / "raw_ebs"
_N_FIELDS = 52


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DriftScenario:
    """Controls how subscriber behavior mix changes over files."""
    drift_type: Literal["none", "sudden", "gradual", "recurring"] = "none"
    drift_at_file: int = 15
    start_file: int = 10
    end_file: int = 30
    period_files: int = 10
    normal_pp_ratio: float = 0.013
    drift_pp_ratio: float = 0.08
    normal_mobile_ratio: float = 0.054
    drift_mobile_ratio: float = 0.054

    def pp_ratio_for_file(self, file_idx: int) -> float:
        """Return PingPong UE ratio for a given file index."""
        if self.drift_type == "none":
            return self.normal_pp_ratio
        if self.drift_type == "sudden":
            return self.drift_pp_ratio if file_idx >= self.drift_at_file else self.normal_pp_ratio
        if self.drift_type == "gradual":
            if file_idx < self.start_file:
                return self.normal_pp_ratio
            if file_idx >= self.end_file:
                return self.drift_pp_ratio
            progress = (file_idx - self.start_file) / max(1, self.end_file - self.start_file)
            return self.normal_pp_ratio + progress * (self.drift_pp_ratio - self.normal_pp_ratio)
        if self.drift_type == "recurring":
            cycle_pos = (file_idx % (self.period_files * 2))
            return self.drift_pp_ratio if cycle_pos >= self.period_files else self.normal_pp_ratio
        return self.normal_pp_ratio


@dataclass
class SourceConfig:
    """Configuration for a single data source (simulates one NF/MME)."""
    mmegi: str = "32772"
    mmec: str = "76"
    label: str = "urban"
    imsi_count: int = 3900
    ho_rate_multiplier: float = 1.0


# ---------------------------------------------------------------------------
# Subscriber behavior models
# ---------------------------------------------------------------------------

class _SubscriberModel:
    """Generate handover events for one IMSI in one minute window."""

    def __init__(self, rng: np.random.RandomState, cell_pool: list[str],
                 imsi: str, ho_count: int, behavior: str):
        self.rng = rng
        self.cell_pool = cell_pool
        self.imsi = imsi
        self.ho_count = ho_count
        self.behavior = behavior

    def generate(self, base_time_ms: int) -> list[dict[str, Any]]:
        if self.ho_count <= 0:
            return []

        if self.behavior == "pingpong":
            return self._gen_pingpong(base_time_ms)
        elif self.behavior == "mobile":
            return self._gen_mobile(base_time_ms)
        else:
            return self._gen_normal(base_time_ms)

    def _gen_normal(self, base_time_ms: int) -> list[dict]:
        # For ho_count>=2, always at least 2 cells (real median n_unique=2)
        if self.ho_count >= 2:
            n_cells = min(self.ho_count, max(2, int(self.rng.poisson(1.2) + 1)))
        else:
            n_cells = 1
        cells = self.rng.choice(self.cell_pool, size=n_cells, replace=False).tolist()
        events = []
        t = base_time_ms + int(self.rng.uniform(0, 30_000))
        for i in range(self.ho_count):
            # Ensure sequential cell diversity: alternate when possible
            if n_cells >= 2 and i > 0:
                prev_eci = events[-1]["eci"]
                other_cells = [c for c in cells if c != prev_eci]
                eci = other_cells[self.rng.randint(0, len(other_cells))] if other_cells else cells[0]
            else:
                eci = cells[self.rng.randint(0, len(cells))]
            events.append(self._make_event(t, eci))
            # Real inter-HO: mean=22s, median=14s, high variance
            # Use larger params to compensate for 60s window truncation
            t += int(self.rng.lognormal(mean=3.2, sigma=1.2) * 1000)
            t = min(t, base_time_ms + 59_999)
        return events

    def _gen_mobile(self, base_time_ms: int) -> list[dict]:
        n_cells = min(self.ho_count, self.rng.randint(3, 7))
        cells = self.rng.choice(self.cell_pool, size=n_cells, replace=False).tolist()
        events = []
        t = base_time_ms + int(self.rng.uniform(0, 10_000))
        cell_idx = 0
        for i in range(self.ho_count):
            eci = cells[cell_idx % len(cells)]
            cell_idx += 1
            events.append(self._make_event(t, eci))
            # Mobile: faster than normal but still variable
            t += int(self.rng.lognormal(mean=2.3, sigma=0.8) * 1000)
            t = min(t, base_time_ms + 59_999)
        return events

    def _gen_pingpong(self, base_time_ms: int) -> list[dict]:
        n_cells = self.rng.choice([2, 3], p=[0.7, 0.3])
        cells = self.rng.choice(self.cell_pool, size=n_cells, replace=False).tolist()
        events = []
        t = base_time_ms + int(self.rng.uniform(0, 5_000))
        cell_idx = 0
        for i in range(self.ho_count):
            eci = cells[cell_idx % len(cells)]
            cell_idx += 1
            events.append(self._make_event(t, eci))
            gap_ms = int(self.rng.lognormal(mean=0.5, sigma=0.8) * 1000)
            gap_ms = max(300, min(gap_ms, 10_000))
            t += gap_ms
            t = min(t, base_time_ms + 59_999)
        return events

    def _make_event(self, time_ms: int, eci: str) -> dict:
        duration = self._sample_duration()
        result = "success" if self.rng.random() < 0.982 else "reject"
        sub_type = "x2" if self.rng.random() < 0.898 else "s1"
        return {
            "imsi": self.imsi,
            "eci": eci,
            "event_time_ms": time_ms,
            "duration": duration,
            "result": result,
            "sub_type": sub_type,
        }

    def _sample_duration(self) -> int:
        # Log-normal matching real: mean~63, median~30, heavy right tail
        d = int(self.rng.lognormal(mean=3.2, sigma=1.0))
        return max(1, min(d, 15_000))


# ---------------------------------------------------------------------------
# EBS line formatter
# ---------------------------------------------------------------------------

def _format_ebs_line(event: dict, source: SourceConfig, fake_msisdn: str) -> str:
    """Format a handover event dict into a 52-field semicolon-delimited line."""
    fields = [""] * _N_FIELDS

    # 8 realistic fields
    fields[0] = "l_handover"
    fields[1] = event["result"]
    fields[2] = str(event["duration"])
    fields[4] = event["sub_type"]
    fields[6] = event["imsi"]
    fields[12] = event["eci"]
    fields[48] = str(event["event_time_ms"])

    # date_hour from timestamp
    dt = datetime.fromtimestamp(event["event_time_ms"] / 1000.0, tz=timezone.utc)
    fields[51] = dt.strftime("%Y-%m-%d-%H")

    # Placeholder fields (present but not used by pipeline)
    fields[3] = "0"                         # request_retries
    fields[5] = fake_msisdn                 # msisdn (PII demo)
    fields[7] = str(abs(hash(event["imsi"])) % 10**10)  # mtmsi
    fields[8] = str(abs(hash(event["imsi"] + "i")) % 10**15)  # imeisv
    fields[9] = source.mmegi
    fields[10] = source.mmec
    fields[11] = "12111"                    # tac
    fields[13] = "10.192.99.230"            # sgw
    fields[15] = "s1ap"                     # l_cause_prot_type
    fields[18] = "ims,v-internet"           # apn
    fields[32] = f"MS{source.label[:2].upper()}01"  # msc
    fields[37] = "intra" if event["sub_type"] == "x2" else ""
    fields[38] = "intra_lte"
    fields[39] = "FALSE"

    return ";".join(fields)


def _format_background_line(event_type: str, rng: np.random.RandomState,
                            time_ms: int, source: SourceConfig) -> str:
    """Format a non-handover background event."""
    fields = [""] * _N_FIELDS
    fields[0] = event_type
    fields[1] = "success"
    fields[2] = str(rng.randint(1, 200))
    fields[3] = "0"
    fields[4] = "normal"
    fields[5] = f"84{rng.randint(100000000, 999999999)}"
    fields[6] = f"45204{rng.randint(10**9, 10**10)}"
    fields[9] = source.mmegi
    fields[10] = source.mmec
    fields[11] = "12111"
    fields[12] = str(rng.randint(50000000, 250000000))
    fields[13] = "10.192.99.230"
    fields[48] = str(time_ms)
    dt = datetime.fromtimestamp(time_ms / 1000.0, tz=timezone.utc)
    fields[51] = dt.strftime("%Y-%m-%d-%H")
    return ";".join(fields)


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

class EBSSyntheticGenerator:
    """Generate synthetic raw EBS files matching real data distributions."""

    def __init__(self, profile: dict[str, Any], random_state: int = 42):
        self.profile = profile
        self.rng = np.random.RandomState(random_state)

        ho = profile["handover"]
        self.ho_per_minute = ho["events_per_minute"]
        self.imsi_per_minute = ho["imsi_per_minute"]
        self.ho_buckets = ho["ho_count_per_imsi"]["buckets"]

        topo = profile["topology"]
        self.cell_pool = topo["cell_pool"]
        self.imsi_prefix = topo["imsi_prefix"]
        self.imsi_length = topo["imsi_length"]

        self.event_type_mix = profile["event_type_mix"]

    def _generate_imsi(self) -> str:
        suffix_len = self.imsi_length - len(self.imsi_prefix)
        suffix = str(self.rng.randint(10**(suffix_len-1), 10**suffix_len))
        return self.imsi_prefix + suffix

    def _sample_ho_count(self, behavior: str) -> int:
        """Sample HO count for a subscriber based on behavior type."""
        if behavior == "normal":
            # Bucket "1" (40.9%) and "2-3" (34.5%) — geometric-like
            return int(self.rng.geometric(p=0.4)) + 0  # mean ~2.5
        elif behavior == "mobile":
            return self.rng.randint(4, 8)  # 4-7
        else:  # pingpong
            # Log-normal + offset to get 8-80 range
            count = int(self.rng.lognormal(mean=2.5, sigma=0.6)) + 8
            return min(count, 80)

    def generate_minute(
        self,
        base_time_ms: int,
        source: SourceConfig,
        drift_scenario: DriftScenario,
        file_idx: int,
    ) -> tuple[list[str], dict[str, str]]:
        """Generate all events for one minute (one file).

        Returns:
            (lines, ground_truth) where ground_truth maps IMSI -> behavior type
        """
        pp_ratio = drift_scenario.pp_ratio_for_file(file_idx)
        mobile_ratio = drift_scenario.normal_mobile_ratio

        n_imsi = int(self.imsi_per_minute * source.ho_rate_multiplier)

        # Assign behavior types
        n_pp = max(1, int(n_imsi * pp_ratio))
        n_mobile = int(n_imsi * mobile_ratio)
        n_normal = n_imsi - n_pp - n_mobile

        imsis_and_behaviors: list[tuple[str, str]] = []
        for _ in range(n_normal):
            imsis_and_behaviors.append((self._generate_imsi(), "normal"))
        for _ in range(n_mobile):
            imsis_and_behaviors.append((self._generate_imsi(), "mobile"))
        for _ in range(n_pp):
            imsis_and_behaviors.append((self._generate_imsi(), "pingpong"))

        self.rng.shuffle(imsis_and_behaviors)

        # Generate handover events
        ho_events: list[dict] = []
        ground_truth: dict[str, str] = {}

        for imsi, behavior in imsis_and_behaviors:
            ground_truth[imsi] = behavior
            ho_count = self._sample_ho_count(behavior)
            model = _SubscriberModel(
                rng=self.rng,
                cell_pool=self.cell_pool,
                imsi=imsi,
                ho_count=ho_count,
                behavior=behavior,
            )
            ho_events.extend(model.generate(base_time_ms))

        # Format handover lines
        lines: list[str] = []
        for evt in ho_events:
            fake_msisdn = f"84{abs(hash(evt['imsi'])) % 10**9}"
            lines.append(_format_ebs_line(evt, source, fake_msisdn))

        # Generate background events (non-handover)
        bg_types = {k: v for k, v in self.event_type_mix.items() if k != "l_handover"}
        # Scale: real data has ~70K background per 8K handovers
        n_background = int(len(ho_events) * 8.5)
        bg_type_names = list(bg_types.keys())
        bg_type_probs = np.array(list(bg_types.values()))
        bg_type_probs = bg_type_probs / bg_type_probs.sum()

        for _ in range(n_background):
            evt_type = self.rng.choice(bg_type_names, p=bg_type_probs)
            t = base_time_ms + self.rng.randint(0, 60_000)
            lines.append(_format_background_line(evt_type, self.rng, t, source))

        # Sort all lines by event_time (field 48)
        def _sort_key(line: str) -> int:
            parts = line.split(";")
            try:
                return int(parts[48])
            except (IndexError, ValueError):
                return 0

        lines.sort(key=_sort_key)

        return lines, ground_truth

    def generate_dataset(
        self,
        output_dir: Path,
        n_files: int,
        scenario: DriftScenario,
        sources: list[SourceConfig] | None = None,
        base_timestamp_ms: int | None = None,
    ) -> dict[str, Any]:
        """Generate a complete dataset of raw EBS files.

        Returns metadata dict with file paths and ground truth.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if sources is None:
            sources = [SourceConfig()]

        if base_timestamp_ms is None:
            # Start at 2024-06-26 15:00 UTC+7 (after real data ends)
            base_timestamp_ms = 1719385200000  # 2024-06-26T08:00:00Z

        all_ground_truth: dict[str, dict[str, str]] = {}
        file_manifest: list[dict[str, Any]] = []

        for file_idx in range(n_files):
            file_time_ms = base_timestamp_ms + file_idx * 60_000  # 1 file per minute
            dt = datetime.fromtimestamp(file_time_ms / 1000.0, tz=timezone.utc)
            ts_start = dt.strftime("%Y%m%d.%H%M")
            dt_end = datetime.fromtimestamp((file_time_ms + 60_000) / 1000.0, tz=timezone.utc)
            ts_end = dt_end.strftime("%Y%m%d.%H%M")
            file_name = f"A{ts_start}+0700-{ts_end}+0700_{900 + file_idx}_ebs"
            file_path = output_dir / file_name

            all_lines: list[str] = []
            file_gt: dict[str, str] = {}

            for source in sources:
                lines, gt = self.generate_minute(
                    base_time_ms=file_time_ms,
                    source=source,
                    drift_scenario=scenario,
                    file_idx=file_idx,
                )
                all_lines.extend(lines)
                file_gt.update(gt)

            # Sort merged lines by event_time
            def _sort_key(line: str) -> int:
                parts = line.split(";")
                try:
                    return int(parts[48])
                except (IndexError, ValueError):
                    return 0

            all_lines.sort(key=_sort_key)

            file_path.write_text("\n".join(all_lines) + "\n", encoding="utf-8")
            all_ground_truth[file_name] = file_gt

            pp_ratio = scenario.pp_ratio_for_file(file_idx)
            ho_count = sum(1 for line in all_lines if line.startswith("l_handover;"))
            file_manifest.append({
                "file": file_name,
                "file_idx": file_idx,
                "total_lines": len(all_lines),
                "handover_lines": ho_count,
                "pp_ratio": round(pp_ratio, 4),
                "drift_active": pp_ratio > scenario.normal_pp_ratio * 1.5,
            })

        # Save ground truth
        gt_path = output_dir / "ground_truth.json"
        gt_path.write_text(json.dumps(all_ground_truth, ensure_ascii=False), encoding="utf-8")

        # Save manifest
        manifest = {
            "scenario": scenario.drift_type,
            "n_files": n_files,
            "sources": [{"mmegi": s.mmegi, "label": s.label} for s in sources],
            "files": file_manifest,
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        return manifest


# ---------------------------------------------------------------------------
# Predefined scenarios
# ---------------------------------------------------------------------------

SCENARIOS = {
    "baseline": DriftScenario(drift_type="none"),
    "sudden_drift": DriftScenario(drift_type="sudden", drift_at_file=15),
    "gradual_drift": DriftScenario(drift_type="gradual", start_file=10, end_file=30),
    "recurring_drift": DriftScenario(drift_type="recurring", period_files=10),
}

MULTI_SOURCES = [
    SourceConfig(mmegi="32772", mmec="76", label="urban", imsi_count=3900, ho_rate_multiplier=1.0),
    SourceConfig(mmegi="32773", mmec="77", label="suburban", imsi_count=2000, ho_rate_multiplier=0.4),
    SourceConfig(mmegi="32774", mmec="78", label="highway", imsi_count=1500, ho_rate_multiplier=0.6),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _generate_all(profile_path: Path, output_root: Path, seed: int) -> None:
    """Generate all datasets: B (baseline), C1-C3 (drift), D (multi-source)."""
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    gen = EBSSyntheticGenerator(profile, random_state=seed)

    datasets = [
        ("baseline", 20, SCENARIOS["baseline"], None),
        ("sudden_drift", 30, SCENARIOS["sudden_drift"], None),
        ("gradual_drift", 30, SCENARIOS["gradual_drift"], None),
        ("recurring_drift", 60, SCENARIOS["recurring_drift"], None),
        ("multisource", 10, SCENARIOS["baseline"], MULTI_SOURCES),
    ]

    for name, n_files, scenario, sources in datasets:
        out = output_root / name
        print(f"Generating {name}: {n_files} files...")
        manifest = gen.generate_dataset(out, n_files, scenario, sources)
        total_ho = sum(f["handover_lines"] for f in manifest["files"])
        total_lines = sum(f["total_lines"] for f in manifest["files"])
        print(f"  -> {total_lines} lines, {total_ho} handovers")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic raw EBS files")
    parser.add_argument("--profile", type=Path, default=_DEFAULT_PROFILE)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)

    sub = parser.add_subparsers(dest="command")

    # Single scenario
    single = sub.add_parser("single", help="Generate a single scenario")
    single.add_argument("--scenario", choices=list(SCENARIOS.keys()), required=True)
    single.add_argument("--n-files", type=int, default=20)

    # All scenarios
    sub.add_parser("all", help="Generate all datasets (B, C1-C3, D)")

    args = parser.parse_args()

    profile = json.loads(args.profile.read_text(encoding="utf-8"))

    if args.command == "all":
        _generate_all(args.profile, args.output_dir, args.seed)
    elif args.command == "single":
        gen = EBSSyntheticGenerator(profile, random_state=args.seed)
        scenario = SCENARIOS[args.scenario]
        sources = MULTI_SOURCES if args.scenario == "multisource" else None
        out = args.output_dir / args.scenario
        print(f"Generating {args.scenario}: {args.n_files} files...")
        manifest = gen.generate_dataset(out, args.n_files, scenario, sources)
        total_ho = sum(f["handover_lines"] for f in manifest["files"])
        print(f"  -> {total_ho} handovers across {args.n_files} files")
    else:
        parser.print_help()
