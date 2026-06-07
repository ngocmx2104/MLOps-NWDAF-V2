"""Generate all synthetic datasets for Exp 1-4.

Datasets:
  B  : Raw EBS baseline (20 files, no drift)           → Exp-1, Exp-3 training
  C1 : Raw EBS sudden drift (30 files)                 → Exp-3
  C2 : Raw EBS gradual drift (30 files)                → Exp-3
  C3 : Raw EBS recurring drift (60 files)              → Exp-3
  D  : Raw EBS multi-source (10 files, 3 MMEGIs)       → Exp-1
  E  : Feature-level calibrated (10 seeds × 1000 rows) → Exp-2, Exp-4

Usage:
    python -m src.data.generate_datasets all          # Generate everything
    python -m src.data.generate_datasets features      # Dataset E only
    python -m src.data.generate_datasets raw --quick   # Quick test (3 files each)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.synthetic_generator import generate_and_save

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT = _PROJECT_ROOT / "artifacts" / "synthetic"
_DEFAULT_PROFILE = _DEFAULT_OUTPUT / "profile.json"

# 10 fixed seeds for reproducibility (N=10 experiment protocol)
EXPERIMENT_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]


def generate_feature_datasets(
    output_dir: Path,
    n_samples: int = 1000,
    anomaly_rate: float = 0.05,
) -> list[dict[str, Any]]:
    """Generate Dataset E: feature-level data for Exp-2 and Exp-4.

    Creates 10 parquet files (one per seed) with calibrated feature
    distributions and no drift. Also creates drift variants for validation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    # -- E-baseline: 10 seeds, no drift ----
    for seed in EXPERIMENT_SEEDS:
        out_path = output_dir / f"features_seed{seed}.parquet"
        meta = generate_and_save(
            output_path=out_path,
            n_samples=n_samples,
            random_state=seed,
            anomaly_rate=anomaly_rate,
        )
        results.append(meta)

    # -- E-drift: one file per drift type (for Exp-3 feature-level validation) ----
    for drift_type in ["sudden", "gradual", "recurring"]:
        out_path = output_dir / f"features_drift_{drift_type}.parquet"
        meta = generate_and_save(
            output_path=out_path,
            n_samples=n_samples,
            random_state=42,
            anomaly_rate=anomaly_rate,
            drift_type=drift_type,
            drift_start=500,
            drift_magnitude=2.0,
        )
        results.append(meta)

    # Save manifest
    manifest = {
        "dataset": "E",
        "description": "Feature-level calibrated synthetic data",
        "n_seeds": len(EXPERIMENT_SEEDS),
        "seeds": EXPERIMENT_SEEDS,
        "n_samples_per_file": n_samples,
        "anomaly_rate": anomaly_rate,
        "files": results,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return results


def generate_raw_ebs_datasets(
    output_dir: Path,
    profile_path: Path = _DEFAULT_PROFILE,
    seed: int = 42,
    quick: bool = False,
) -> dict[str, Any]:
    """Generate raw EBS datasets B, C1-C3, D.

    Args:
        output_dir: Root output directory for raw EBS files.
        profile_path: Path to profile.json from real_profile.py.
        seed: Random seed.
        quick: If True, generate 3 files per scenario (for testing).
    """
    from src.data.ebs_generator import (
        EBSSyntheticGenerator,
        MULTI_SOURCES,
        SCENARIOS,
    )

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    gen = EBSSyntheticGenerator(profile, random_state=seed)

    if quick:
        file_counts = {"baseline": 3, "sudden_drift": 3, "gradual_drift": 3,
                       "recurring_drift": 6, "multisource": 2}
    else:
        file_counts = {"baseline": 20, "sudden_drift": 30, "gradual_drift": 30,
                       "recurring_drift": 60, "multisource": 10}

    datasets = [
        ("baseline", file_counts["baseline"], SCENARIOS["baseline"], None),
        ("sudden_drift", file_counts["sudden_drift"], SCENARIOS["sudden_drift"], None),
        ("gradual_drift", file_counts["gradual_drift"], SCENARIOS["gradual_drift"], None),
        ("recurring_drift", file_counts["recurring_drift"], SCENARIOS["recurring_drift"], None),
        ("multisource", file_counts["multisource"], SCENARIOS["baseline"], MULTI_SOURCES),
    ]

    results = {}
    for name, n_files, scenario, sources in datasets:
        out = output_dir / name
        print(f"  Generating {name}: {n_files} files...")
        manifest = gen.generate_dataset(out, n_files, scenario, sources)
        total_ho = sum(f["handover_lines"] for f in manifest["files"])
        total_lines = sum(f["total_lines"] for f in manifest["files"])
        print(f"    -> {total_lines:,} lines, {total_ho:,} handovers")
        results[name] = manifest

    return results


def generate_all(
    output_root: Path = _DEFAULT_OUTPUT,
    profile_path: Path = _DEFAULT_PROFILE,
    seed: int = 42,
    quick: bool = False,
) -> None:
    """Generate all datasets for the experiment pipeline."""
    print("=" * 60)
    print("Generating all synthetic datasets")
    print("=" * 60)

    # Dataset E: feature-level
    print("\n[1/2] Dataset E: Feature-level (calibrated)")
    feat_dir = output_root / "features"
    feat_results = generate_feature_datasets(feat_dir, n_samples=1000)
    print(f"  -> {len(feat_results)} files generated")

    # Datasets B, C1-C3, D: raw EBS
    print(f"\n[2/2] Datasets B, C1-C3, D: Raw EBS {'(quick mode)' if quick else ''}")
    raw_dir = output_root / "raw_ebs"
    generate_raw_ebs_datasets(raw_dir, profile_path, seed, quick)

    print("\n" + "=" * 60)
    print("All datasets generated successfully!")
    print(f"Output: {output_root}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate all synthetic datasets")
    parser.add_argument("command", choices=["all", "features", "raw"],
                        help="What to generate")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--profile", type=Path, default=_DEFAULT_PROFILE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Generate small datasets for testing")
    args = parser.parse_args()

    if args.command == "all":
        generate_all(args.output, args.profile, args.seed, args.quick)
    elif args.command == "features":
        print("Generating Dataset E: Feature-level")
        results = generate_feature_datasets(args.output / "features")
        print(f"  -> {len(results)} files")
    elif args.command == "raw":
        print("Generating Datasets B, C1-C3, D: Raw EBS")
        generate_raw_ebs_datasets(
            args.output / "raw_ebs", args.profile, args.seed, args.quick
        )
