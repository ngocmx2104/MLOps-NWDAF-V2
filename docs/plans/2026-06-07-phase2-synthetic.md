# Phase 2 — Synthetic Data Generation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Build the synthetic-data layer: extract a statistical profile from real EBS, generate (a) feature-level calibrated datasets (Dataset E) for Exp-2/4 and (b) raw-EBS datasets with controlled drift (B/C/D) for Exp-1/3 — all versioned via DVC, reproducible by seed.

**Architecture:** Port the proven generators from `MLOps_Project/src/data/` (drop-in, `src.` imports), re-implemented WITH pytest tests. `synthetic_generator` is profile-independent (calibration baked as constants → easy unit tests). `ebs_generator` needs a `profile.json` (from `real_profile` over the real EBS already in `data/raw_ebs/`). Add DVC stages `profile` + `features` (fast, run now) and define `raw_data` (slow full gen, deferred to pre-experiment).

**Tech Stack:** numpy, pandas, DVC.

**Source to port (read each):**
- `/Users/ngocmx/Thạc Sĩ/MLOps_Project/src/data/synthetic_generator.py`
- `/Users/ngocmx/Thạc Sĩ/MLOps_Project/src/data/real_profile.py`
- `/Users/ngocmx/Thạc Sĩ/MLOps_Project/src/data/ebs_generator.py`
- `/Users/ngocmx/Thạc Sĩ/MLOps_Project/src/data/generate_datasets.py`

**Branch:** `phase2-synthetic` (from `main`).

**Files created:**
```
src/data/__init__.py
src/data/synthetic_generator.py   # feature-level gen + drift (port verbatim)
src/data/real_profile.py          # profile extraction (port + path fix)
src/data/ebs_generator.py         # raw EBS gen (port verbatim)
src/data/generate_datasets.py     # orchestrator (port verbatim)
tests/data/__init__.py, conftest.py, test_synthetic_generator.py, test_real_profile.py,
  test_ebs_generator.py, test_generate_datasets.py
dvc.yaml                          # extend: profile, features, raw_data stages
```

---

### Task 1: Port feature-level generator + tests

**Files:** `src/data/__init__.py` (empty), `src/data/synthetic_generator.py`, `tests/data/__init__.py` (empty), `tests/data/test_synthetic_generator.py`.

- [ ] **Step 1: Port** `MLOps_Project/src/data/synthetic_generator.py` VERBATIM → `src/data/synthetic_generator.py` (no cross-package deps). Provides `DriftConfig`, `generate_synthetic_data(n_samples, random_state, anomaly_rate, drift_config) -> DataFrame`, `generate_and_save(output_path, n_samples, random_state, anomaly_rate, drift_type, drift_start, drift_magnitude) -> dict`.

- [ ] **Step 2: Tests** — `tests/data/test_synthetic_generator.py`:
```python
import pandas as pd

from src.data.synthetic_generator import generate_and_save, generate_synthetic_data

FEATURES = ["n_handover", "n_unique_cells", "pingpong_count", "pingpong_rate",
            "mean_inter_ho_s", "std_inter_ho_s", "entropy_cell_seq"]


def test_has_7_features_and_labels():
    df = generate_synthetic_data(n_samples=200, random_state=42, anomaly_rate=0.05)
    assert all(f in df.columns for f in FEATURES)
    assert len(df) == 200
    assert df["label"].sum() == int(200 * 0.05)
    assert {"imsi", "window_start", "label"} <= set(df.columns)


def test_deterministic_by_seed():
    a = generate_synthetic_data(n_samples=100, random_state=7, anomaly_rate=0.05)
    b = generate_synthetic_data(n_samples=100, random_state=7, anomaly_rate=0.05)
    pd.testing.assert_frame_equal(a, b)


def test_generate_and_save_with_drift(tmp_path):
    out = tmp_path / "f.parquet"
    meta = generate_and_save(out, n_samples=300, random_state=42,
                             anomaly_rate=0.05, drift_type="sudden", drift_start=150)
    assert out.exists()
    assert meta["n_samples"] == 300
    assert meta["drift_type"] == "sudden"
    assert meta["n_drifted_samples"] > 0
    df = pd.read_parquet(out)
    assert df["has_drift"].sum() > 0


def test_no_drift_default(tmp_path):
    meta = generate_and_save(tmp_path / "g.parquet", n_samples=100, random_state=1)
    assert meta["drift_type"] == "none"
    assert meta["n_drifted_samples"] == 0
```

- [ ] **Step 3: Run** `source .venv/bin/activate && pytest tests/data/test_synthetic_generator.py -v` (4 passed) + `ruff check src/data tests/data`.
- [ ] **Step 4: Commit** `git add src/data/__init__.py src/data/synthetic_generator.py tests/data/__init__.py tests/data/test_synthetic_generator.py` → `feat(data): port feature-level synthetic generator with drift` + trailer.

---

### Task 2: Port real-profile extractor (with path fix) + test

**Files:** `src/data/real_profile.py`, `tests/data/test_real_profile.py`.

- [ ] **Step 1: Port** `MLOps_Project/src/data/real_profile.py` → `src/data/real_profile.py` with ONE change: replace
  `_DATA_DIR = _PROJECT_ROOT / "mlops_source_and_thesis" / "Data EBS"`
  with
  `_DATA_DIR = _PROJECT_ROOT / "data" / "raw_ebs"`
  and replace the hardcoded `_DEFAULT_EBS_FILES = [...]` list with
  `_DEFAULT_EBS_FILES = sorted(_DATA_DIR.glob("*_ebs"))`.
  Keep everything else (extract_profile, save_profile, the `__main__` block which already globs `--ebs-dir`).

- [ ] **Step 2: Test** — `tests/data/test_real_profile.py`:
```python
from pathlib import Path

import pytest

from src.data.real_profile import extract_profile

REAL_EBS = sorted(Path("data/raw_ebs").glob("*_ebs"))


@pytest.mark.skipif(not REAL_EBS, reason="real EBS files not present (run dvc pull)")
def test_extract_profile_keys():
    p = extract_profile(REAL_EBS)
    assert p["source"]["total_handovers"] > 1000
    assert p["topology"]["cell_pool_size"] > 0
    assert "pingpong" in p["handover"]
    assert p["handover"]["unique_imsis"] > 0
    assert "event_type_mix" in p


@pytest.mark.skipif(not REAL_EBS, reason="real EBS files not present")
def test_extract_profile_handover_stats():
    p = extract_profile(REAL_EBS)
    assert "ho_count_per_imsi" in p["handover"]
    assert "buckets" in p["handover"]["ho_count_per_imsi"]
    assert isinstance(p["topology"]["cell_pool"], list)
```

- [ ] **Step 3: Run** `pytest tests/data/test_real_profile.py -v` (2 passed — they run because `data/raw_ebs` exists from Phase 1) + `ruff check src/data tests/data`.
- [ ] **Step 4: Commit** `git add src/data/real_profile.py tests/data/test_real_profile.py` → `feat(data): port real-EBS profile extractor (data/raw_ebs)` + trailer.

---

### Task 3: Port raw-EBS generator + tests (incl. parser integration)

**Files:** `src/data/ebs_generator.py`, `tests/data/conftest.py`, `tests/data/test_ebs_generator.py`.

- [ ] **Step 1: Port** `MLOps_Project/src/data/ebs_generator.py` VERBATIM → `src/data/ebs_generator.py`. Provides `DriftScenario`, `SourceConfig`, `EBSSyntheticGenerator(profile, random_state).generate_dataset(output_dir, n_files, scenario, sources, base_timestamp_ms) -> manifest`, `SCENARIOS`, `MULTI_SOURCES`.

- [ ] **Step 2: Fixture** — `tests/data/conftest.py`:
```python
import pytest


@pytest.fixture
def tiny_profile():
    """Minimal profile with the keys EBSSyntheticGenerator requires."""
    return {
        "handover": {
            "events_per_minute": 100,
            "imsi_per_minute": 40,
            "ho_count_per_imsi": {
                "buckets": {"1": 0.4, "2-3": 0.35, "4-7": 0.15,
                            "8-15": 0.07, "16-30": 0.02, "31+": 0.01},
            },
        },
        "topology": {
            "cell_pool": [str(100_000_000 + i) for i in range(20)],
            "imsi_prefix": "45204",
            "imsi_length": 15,
        },
        "event_type_mix": {"l_handover": 0.1, "l_service_request": 0.5, "l_tau": 0.4},
    }
```

- [ ] **Step 3: Tests** — `tests/data/test_ebs_generator.py`:
```python
from src.data.ebs_generator import EBSSyntheticGenerator, SCENARIOS


def test_generate_dataset_manifest(tiny_profile, tmp_path):
    gen = EBSSyntheticGenerator(tiny_profile, random_state=42)
    manifest = gen.generate_dataset(tmp_path / "sudden", n_files=2,
                                    scenario=SCENARIOS["sudden_drift"])
    assert manifest["n_files"] == 2
    assert manifest["scenario"] == "sudden"
    assert len(manifest["files"]) == 2
    assert all("drift_active" in f and "handover_lines" in f for f in manifest["files"])
    assert len(list((tmp_path / "sudden").glob("*_ebs"))) == 2
    assert (tmp_path / "sudden" / "manifest.json").exists()
    assert (tmp_path / "sudden" / "ground_truth.json").exists()


def test_sudden_drift_flag_flips(tiny_profile, tmp_path):
    gen = EBSSyntheticGenerator(tiny_profile, random_state=42)
    manifest = gen.generate_dataset(tmp_path / "s", n_files=20,
                                    scenario=SCENARIOS["sudden_drift"])
    early = [f for f in manifest["files"] if f["file_idx"] < 15]
    late = [f for f in manifest["files"] if f["file_idx"] >= 15]
    assert not any(f["drift_active"] for f in early)
    assert all(f["drift_active"] for f in late)


def test_generated_ebs_parses_with_phase1_parser(tiny_profile, tmp_path):
    from src.ingestion.parser import normalize_timestamps, parse_ebs_files
    gen = EBSSyntheticGenerator(tiny_profile, random_state=42)
    gen.generate_dataset(tmp_path / "b", n_files=1, scenario=SCENARIOS["baseline"])
    files = list((tmp_path / "b").glob("*_ebs"))
    df = normalize_timestamps(parse_ebs_files(files))
    assert (df["raw_field_count"] == 52).all()
    assert (df["event_id"] == "l_handover").any()
    assert df["event_ts"].notna().all()
```

- [ ] **Step 4: Run** `pytest tests/data/test_ebs_generator.py -v` (3 passed) + `ruff check src/data tests/data`.
- [ ] **Step 5: Commit** `git add src/data/ebs_generator.py tests/data/conftest.py tests/data/test_ebs_generator.py` → `feat(data): port raw-EBS synthetic generator (drift scenarios + multi-source)` + trailer.

---

### Task 4: Port dataset orchestrator + test

**Files:** `src/data/generate_datasets.py`, `tests/data/test_generate_datasets.py`.

- [ ] **Step 1: Port** `MLOps_Project/src/data/generate_datasets.py` VERBATIM → `src/data/generate_datasets.py`. Provides `generate_feature_datasets`, `generate_raw_ebs_datasets`, `generate_all`, `EXPERIMENT_SEEDS`, and the `__main__` CLI (`all|features|raw`, `--quick`).

- [ ] **Step 2: Test** — `tests/data/test_generate_datasets.py`:
```python
import json

from src.data.generate_datasets import EXPERIMENT_SEEDS, generate_feature_datasets


def test_feature_datasets_count_and_manifest(tmp_path):
    results = generate_feature_datasets(tmp_path, n_samples=50)
    # 10 seeds + 3 drift variants = 13 parquet files
    assert len(list(tmp_path.glob("features_*.parquet"))) == 13
    assert len(results) == 13
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["dataset"] == "E"
    assert manifest["n_seeds"] == len(EXPERIMENT_SEEDS)


def test_seeds_are_fixed():
    assert EXPERIMENT_SEEDS == [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
```

- [ ] **Step 3: Run** `pytest tests/data/test_generate_datasets.py -v` (2 passed) + `ruff check src/data tests/data`.
- [ ] **Step 4: Commit** `git add src/data/generate_datasets.py tests/data/test_generate_datasets.py` → `feat(data): port dataset orchestrator (E features + B/C/D raw)` + trailer.

---

### Task 5: DVC stages (profile + features now; raw_data deferred) + generate real artifacts

**Files:** modify `dvc.yaml`, update `dvc.lock`.

- [ ] **Step 1: Extend `dvc.yaml`** — add three stages (keep the existing `snapshot` stage):
```yaml
  profile:
    cmd: python -m src.data.real_profile --ebs-dir data/raw_ebs --output artifacts/synthetic/profile.json
    deps:
      - data/raw_ebs
      - src/data/real_profile.py
    outs:
      - artifacts/synthetic/profile.json

  features:
    cmd: python -m src.data.generate_datasets features --output artifacts/synthetic
    deps:
      - src/data/synthetic_generator.py
      - src/data/generate_datasets.py
    outs:
      - artifacts/synthetic/features

  raw_data:
    cmd: python -m src.data.generate_datasets raw --output artifacts/synthetic --profile artifacts/synthetic/profile.json
    deps:
      - artifacts/synthetic/profile.json
      - src/data/ebs_generator.py
      - src/data/generate_datasets.py
    outs:
      - artifacts/synthetic/raw_ebs
```

- [ ] **Step 2: Run the fast stages** — `source .venv/bin/activate && dvc repro profile features`
  Expected: `profile` builds `artifacts/synthetic/profile.json` from the real EBS (3 files); `features` builds 13 parquets under `artifacts/synthetic/features/` + manifest. Updates `dvc.lock`.
  > DO NOT run `dvc repro raw_data` — it generates 150 raw files (~60 min). It is defined for later (run before Exp-1/Exp-3 in Phase 8).

- [ ] **Step 3: Verify the generated artifacts**
```bash
python -c "import json; p=json.load(open('artifacts/synthetic/profile.json')); print('handovers', p['source']['total_handovers'], 'imsis', p['handover']['unique_imsis'], 'cells', p['topology']['cell_pool_size'], 'pp_imsis', p['handover']['pingpong']['imsi_with_pp'])"
ls -1 artifacts/synthetic/features/ | head
python -c "import pandas as pd; df=pd.read_parquet('artifacts/synthetic/features/features_seed42.parquet'); print('rows', len(df), 'anomaly', df['label'].sum(), 'cols', list(df.columns))"
python -c "import pandas as pd; df=pd.read_parquet('artifacts/synthetic/features/features_drift_sudden.parquet'); print('drifted', df['has_drift'].sum())"
```
Expected: profile shows ~24,347 handovers / ~8,393 IMSIs / cell pool > 0 / ping-pong IMSIs > 0; 13 feature parquets; seed42 file = 1000 rows, ~50 anomalies, 7 features + label; drift_sudden file has drifted rows.

- [ ] **Step 4: Commit DVC artifacts** — `git add dvc.yaml dvc.lock` then `feat(data): DVC stages — profile + feature datasets (raw_data deferred)` + trailer. Confirm `git status`: `artifacts/synthetic/` is gitignored (only dvc.yaml/dvc.lock committed). If DVC added `artifacts/` entries to a `.gitignore`, include that file in the commit.

---

### Task 6: Phase 2 verification (DoD)

- [ ] **Step 1: Full suite** — `pytest -q && ruff check .` → all green (Phase 0–1 tests + ~11 new data tests).
- [ ] **Step 2: Reproducibility** — `dvc repro profile features` again → both report "didn't change, skipping".
- [ ] **Step 3: DoD checklist:** profile.json calibrated from real EBS; Dataset E (13 feature parquets, drift variants) generated + manifest; raw-EBS generator validated by unit tests (drift_active flips at scenario boundary; output parses with Phase-1 parser); `raw_data` stage defined for full generation later. All tests green.

> **Deferred (documented, not blocking Phase 2):** full raw datasets B/C1-C3/D via `dvc repro raw_data` (~60 min) — run before Exp-1/Exp-3 in Phase 8.
