# Phase 3 — Features + Feast Feature Store — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Build the feature subsystem — port the D2 feature builder (7 handover features), weak-labeling, schema, quality — and add a **Feast feature store** (component C3) with offline+online stores, proving train/serve feature consistency on the real handover data.

**Architecture:** Ports from `MLOps_Project/src/features/` (drop-in, `src.` imports) + tests. NEW: a Feast `local` provider repo (File offline + SQLite online) over the D2 feature parquet built from the real D1 snapshot (Phase 1). Feast is the genuinely new C3 component the thesis claims; its value here is **train/serve consistency** (NWDAF AnLF real-time inference vs MTLF batch training) + point-in-time retrieval.

**Tech Stack:** pandas, **Feast 0.63** (installs on Python 3.14 — confirmed), DVC.

**Source to port (read each):**
- `/Users/ngocmx/Thạc Sĩ/MLOps_Project/src/features/schema.py`
- `/Users/ngocmx/Thạc Sĩ/MLOps_Project/src/features/quality.py`
- `/Users/ngocmx/Thạc Sĩ/MLOps_Project/src/features/builder.py`
- `/Users/ngocmx/Thạc Sĩ/MLOps_Project/src/features/weak_labels.py`

**Branch:** `phase3-features-feast` (from `main`).

**Prereq:** `pip install -e ".[feast]"` (Feast not yet installed; `[dev]` only so far).

**Files created:**
```
src/features/__init__.py, schema.py, quality.py, builder.py, weak_labels.py, cli.py, __main__.py
src/features/feast_store.py                 # NEW: Feast helpers (build defs, apply, materialize, retrieve)
src/features/feast_repo/feature_store.yaml  # NEW: Feast config (local + sqlite)
src/features/feast_repo/.gitignore          # ignore registry.db / online_store.db / data/
tests/features/__init__.py, conftest.py, test_builder.py, test_weak_labels.py,
  test_quality.py, test_feast_store.py
dvc.yaml                                    # extend: d2_features, weak_labels stages
```

---

### Task 1: Install Feast + port feature schema & quality

**Files:** `src/features/__init__.py` (empty), `src/features/schema.py`, `src/features/quality.py`, `tests/features/__init__.py` (empty), `tests/features/test_quality.py`.

- [ ] **Step 1: Install Feast extra** — `source .venv/bin/activate && pip install -e ".[feast]"`. Verify `python -c "import feast; print(feast.__version__)"` prints a version (≈0.63). If install fails, STOP and report BLOCKED with the error.
- [ ] **Step 2: Port** `schema.py` and `quality.py` VERBATIM from `MLOps_Project/src/features/` → `src/features/`. schema defines `WindowConfig`, `WeakLabelConfig`, `FEATURE_VERSION`, `WEAK_LABEL_VERSION`, `D2_NUMERIC_FEATURE_NAMES`, etc. quality defines `run_d2_quality_checks` (5), `run_d5_quality_checks` (4), `format_feature_quality_report`, `QualityCheckResult`.
- [ ] **Step 3: Tests** — `tests/features/test_quality.py`:
```python
import pandas as pd

from src.features.quality import run_d2_quality_checks, run_d5_quality_checks


def _d2_df():
    return pd.DataFrame({
        "imsi": ["1", "2"], "window_start": pd.to_datetime(["2024-06-26", "2024-06-26"], utc=True),
        "n_handover": [3, 1], "n_unique_cells": [2, 1], "pingpong_count": [1, 0],
        "pingpong_rate": [0.33, 0.0], "mean_inter_ho_s": [10.0, float("nan")],
        "std_inter_ho_s": [2.0, float("nan")], "entropy_cell_seq": [0.9, 0.0],
        "feature_version": ["ho_features_v1"] * 2, "source_snapshot_id": ["D1_X"] * 2,
    })


def test_d2_checks_pass_on_valid():
    results = run_d2_quality_checks(_d2_df())
    assert len(results) == 5
    assert all(r.passed for r in results)


def test_d5_checks_detect_binary_labels():
    df = _d2_df()
    df["weak_label"] = [1, 0]
    df["weak_label_version"] = "weak_label_v1"
    df["source_feature_version"] = "ho_features_v1"
    results = run_d5_quality_checks(df)
    assert len(results) == 4
    assert all(r.passed for r in results)
```
- [ ] **Step 4: Run** `pytest tests/features/test_quality.py -v` (2 passed) + `ruff check src/features tests/features`.
- [ ] **Step 5: Commit** `git add pyproject.toml src/features/__init__.py src/features/schema.py src/features/quality.py tests/features/__init__.py tests/features/test_quality.py` → `feat(features): install Feast + port feature schema & quality checks` + trailer. (pyproject unchanged but include if pip wrote anything; otherwise omit.)

---

### Task 2: Port feature builder (7 features) + tests

**Files:** `src/features/builder.py`, `tests/features/conftest.py`, `tests/features/test_builder.py`.

- [ ] **Step 1: Port** `builder.py` VERBATIM → `src/features/builder.py`. Provides `compute_ue_window_features(df_d1, cfg) -> DataFrame` and `build_d2_feature_dataset(d1_parquet_path, output_dir, *, window_config, feature_version) -> dict`.
- [ ] **Step 2: Fixture** — `tests/features/conftest.py`:
```python
import pandas as pd
import pytest


@pytest.fixture
def d1_like_df():
    """Minimal D1-like frame: one IMSI doing A->B->A ping-pong + one normal IMSI."""
    base = pd.Timestamp("2024-06-26T14:00:00", tz="UTC")
    rows = [
        # IMSI 111: A(10) -> B(20) -> A(10) within 30s gaps => 1 ping-pong, 3 handovers
        {"imsi": "111", "eci": "10", "event_ts": base},
        {"imsi": "111", "eci": "20", "event_ts": base + pd.Timedelta(seconds=10)},
        {"imsi": "111", "eci": "10", "event_ts": base + pd.Timedelta(seconds=20)},
        # IMSI 222: single handover, no ping-pong
        {"imsi": "222", "eci": "30", "event_ts": base + pd.Timedelta(seconds=5)},
    ]
    return pd.DataFrame(rows)
```
- [ ] **Step 3: Tests** — `tests/features/test_builder.py`:
```python
from src.features.builder import compute_ue_window_features
from src.features.schema import WindowConfig

FEATURES = ["n_handover", "n_unique_cells", "pingpong_count", "pingpong_rate",
            "mean_inter_ho_s", "std_inter_ho_s", "entropy_cell_seq"]


def test_computes_7_features(d1_like_df):
    out = compute_ue_window_features(d1_like_df, WindowConfig())
    assert all(f in out.columns for f in FEATURES)
    assert {"imsi", "window_start"} <= set(out.columns)


def test_pingpong_detected(d1_like_df):
    out = compute_ue_window_features(d1_like_df, WindowConfig())
    imsi_111 = out[out["imsi"] == "111"].iloc[0]
    assert imsi_111["n_handover"] == 3
    assert imsi_111["n_unique_cells"] == 2
    assert imsi_111["pingpong_count"] == 1


def test_single_handover_has_nan_timing(d1_like_df):
    out = compute_ue_window_features(d1_like_df, WindowConfig())
    imsi_222 = out[out["imsi"] == "222"].iloc[0]
    assert imsi_222["n_handover"] == 1
    assert imsi_222["pingpong_count"] == 0
    assert pd.isna(imsi_222["mean_inter_ho_s"])
```
(add `import pandas as pd` at top.)
- [ ] **Step 4: Run** `pytest tests/features/test_builder.py -v` (3 passed) + ruff.
- [ ] **Step 5: Commit** `git add src/features/builder.py tests/features/conftest.py tests/features/test_builder.py` → `feat(features): port D2 feature builder (7 handover features)` + trailer.

---

### Task 3: Port weak-labeling + tests

**Files:** `src/features/weak_labels.py`, `tests/features/test_weak_labels.py`.

- [ ] **Step 1: Port** `weak_labels.py` VERBATIM → `src/features/weak_labels.py`. Provides `apply_weak_labels(df_d2, cfg) -> DataFrame` and `build_d5_weak_label_dataset(...)`.
- [ ] **Step 2: Tests** — `tests/features/test_weak_labels.py`:
```python
import pandas as pd

from src.features.schema import WeakLabelConfig
from src.features.weak_labels import apply_weak_labels


def test_weak_label_rule():
    df = pd.DataFrame({
        "pingpong_count": [2, 0, 1, 5],
        "n_handover": [5, 1, 2, 10],
    })
    out = apply_weak_labels(df, WeakLabelConfig())  # min_pp=1, min_ho=3
    # row0: pp2>=1 & ho5>=3 -> 1; row1: 0 -> 0; row2: ho2<3 -> 0; row3: 1
    assert out["weak_label"].tolist() == [1, 0, 0, 1]


def test_weak_label_is_binary():
    df = pd.DataFrame({"pingpong_count": [0, 3], "n_handover": [0, 4]})
    out = apply_weak_labels(df, WeakLabelConfig())
    assert set(out["weak_label"].unique()) <= {0, 1}
```
- [ ] **Step 3: Run** `pytest tests/features/test_weak_labels.py -v` (2 passed) + ruff.
- [ ] **Step 4: Commit** `git add src/features/weak_labels.py tests/features/test_weak_labels.py` → `feat(features): port rule-based weak labeling` + trailer.

---

### Task 4: Feast feature store (NEW — C3) + tests

**Files:** `src/features/feast_store.py`, `src/features/feast_repo/feature_store.yaml`, `src/features/feast_repo/.gitignore`, `tests/features/test_feast_store.py`.

- [ ] **Step 1: Feast config** — `src/features/feast_repo/feature_store.yaml`:
```yaml
project: nwdaf_handover
provider: local
registry: registry.db
online_store:
  type: sqlite
  path: online_store.db
entity_key_serialization_version: 3
```
And `src/features/feast_repo/.gitignore`:
```
registry.db
online_store.db
data/
```

- [ ] **Step 2: Feast helpers** — `src/features/feast_store.py`:
```python
"""Feast feature store helpers (component C3) for handover features.

Provides programmatic apply/materialize/retrieve over a `local` Feast repo
(File offline + SQLite online), proving train/serve feature consistency.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from feast import Entity, FeatureStore, FeatureView, Field, FileSource
from feast.types import Float64, Int64

FEATURE_NAMES = [
    "n_handover", "n_unique_cells", "pingpong_count", "pingpong_rate",
    "mean_inter_ho_s", "std_inter_ho_s", "entropy_cell_seq",
]
_DTYPES = {
    "n_handover": Int64, "n_unique_cells": Int64, "pingpong_count": Int64,
    "pingpong_rate": Float64, "mean_inter_ho_s": Float64,
    "std_inter_ho_s": Float64, "entropy_cell_seq": Float64,
}
FEATURE_REFS = [f"handover_features:{n}" for n in FEATURE_NAMES]


def build_definitions(source_parquet: str) -> list:
    """Build the Feast Entity + FeatureView for a given source parquet."""
    imsi = Entity(name="imsi", join_keys=["imsi"])
    source = FileSource(
        name="handover_feature_source",
        path=str(source_parquet),
        timestamp_field="window_start",
    )
    fv = FeatureView(
        name="handover_features",
        entities=[imsi],
        ttl=timedelta(days=3650),
        schema=[Field(name=n, dtype=_DTYPES[n]) for n in FEATURE_NAMES],
        online=True,
        source=source,
    )
    return [imsi, fv]


def apply_and_materialize(repo_path: Path, source_parquet: Path,
                          end_date: datetime | None = None) -> FeatureStore:
    """Apply definitions and materialize features into the online store."""
    store = FeatureStore(repo_path=str(repo_path))
    store.apply(build_definitions(str(source_parquet)))
    end = end_date or datetime.now(timezone.utc)
    store.materialize(start_date=datetime(2020, 1, 1, tzinfo=timezone.utc), end_date=end)
    return store


def get_online_features(store: FeatureStore, imsis: list[str]) -> pd.DataFrame:
    """Serving path: low-latency online features for given IMSIs."""
    rows = [{"imsi": i} for i in imsis]
    return store.get_online_features(features=FEATURE_REFS, entity_rows=rows).to_df()


def get_training_features(store: FeatureStore, entity_df: pd.DataFrame) -> pd.DataFrame:
    """Training path: point-in-time-correct historical features."""
    return store.get_historical_features(entity_df=entity_df, features=FEATURE_REFS).to_df()
```

- [ ] **Step 3: Tests** — `tests/features/test_feast_store.py` (proves apply/materialize/retrieve + **train/serve consistency**):
```python
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.features.feast_store import (
    apply_and_materialize, get_online_features, get_training_features,
)

REPO_SRC = Path("src/features/feast_repo")


@pytest.fixture
def feast_repo(tmp_path, d1_like_df):
    """A temp Feast repo with a tiny handover feature parquet."""
    from src.features.builder import compute_ue_window_features
    from src.features.schema import WindowConfig

    repo = tmp_path / "feast_repo"
    repo.mkdir()
    (repo / "feature_store.yaml").write_text((REPO_SRC / "feature_store.yaml").read_text())
    feats = compute_ue_window_features(d1_like_df, WindowConfig())
    # Feast online needs non-null Int64 keys/features; fill timing NaNs for the demo
    feats["mean_inter_ho_s"] = feats["mean_inter_ho_s"].fillna(0.0)
    feats["std_inter_ho_s"] = feats["std_inter_ho_s"].fillna(0.0)
    src_parquet = repo / "data" / "handover_features.parquet"
    src_parquet.parent.mkdir(parents=True)
    feats.to_parquet(src_parquet, index=False)
    return repo, src_parquet, feats


def test_apply_materialize_and_online_retrieve(feast_repo):
    repo, src_parquet, feats = feast_repo
    store = apply_and_materialize(repo, src_parquet)
    online = get_online_features(store, ["111"])
    assert online["n_handover"].iloc[0] == 3
    assert online["pingpong_count"].iloc[0] == 1


def test_train_serve_consistency(feast_repo):
    """Online (serving) features must equal offline (training) features for same key+time."""
    repo, src_parquet, feats = feast_repo
    store = apply_and_materialize(repo, src_parquet)
    online = get_online_features(store, ["111"]).set_index("imsi")
    entity_df = pd.DataFrame({
        "imsi": ["111"],
        "event_timestamp": [datetime.now(timezone.utc)],
    })
    offline = get_training_features(store, entity_df).set_index("imsi")
    for f in ["n_handover", "n_unique_cells", "pingpong_count"]:
        assert online.loc["111", f] == offline.loc["111", f]
```

- [ ] **Step 4: Run** `pytest tests/features/test_feast_store.py -v` (2 passed). If Feast emits deprecation warnings, that's fine; if a test errors on the API, adapt minimally (the controller researched Feast 0.51+ API: `Entity(join_keys=...)`, `Field(name, dtype)`, `store.apply([...])`, `store.materialize(start,end)`, `get_online_features(features, entity_rows).to_df()`, `get_historical_features(entity_df, features).to_df()`). Note any adaptation. Then `ruff check src/features tests/features`.
- [ ] **Step 5: Commit** `git add src/features/feast_store.py src/features/feast_repo/ tests/features/test_feast_store.py` → `feat(features): Feast feature store (C3) with train/serve consistency` + trailer.

---

### Task 5: Feature CLI + DVC stages (build D2/D5 from real D1)

**Files:** `src/features/cli.py`, `src/features/__main__.py`, modify `dvc.yaml`.

- [ ] **Step 1: CLI** — `src/features/cli.py` with subcommands (argparse), imports at top (no sys.path guard — editable install):
  - `build-d2`: glob `artifacts/snapshots/D1_*.parquet` (or `--d1 <path>`), call `build_d2_feature_dataset(d1, output_dir=artifacts/features)`. Print summary.
  - `weak-label`: glob `artifacts/features/D2_*.parquet` (or `--d2 <path>`), call `build_d5_weak_label_dataset(d2, output_dir=artifacts/features)`. Print label distribution.
  Create `src/features/__main__.py` calling `main()`.
- [ ] **Step 2: Extend `dvc.yaml`** — add:
```yaml
  d2_features:
    cmd: python -m src.features.cli build-d2 --output artifacts/features
    deps:
      - artifacts/snapshots
      - src/features/builder.py
      - src/features/schema.py
    outs:
      - artifacts/features
  weak_labels:
    cmd: python -m src.features.cli weak-label --output artifacts/features_labeled
    deps:
      - artifacts/features
      - src/features/weak_labels.py
    outs:
      - artifacts/features_labeled
```
- [ ] **Step 3: Run** `dvc repro d2_features weak_labels` (ensure `artifacts/snapshots` exists first; if not, `dvc repro snapshot`). Verify a real D2 parquet (tens of thousands of rows from the real handover snapshot) + D5 with a sane positive rate (~20%). REPORT the row counts + positive rate.
- [ ] **Step 4: Smoke the CLI** in `tests/features/test_cli_smoke.py` (build D2 from a tiny D1 parquet via subprocess) — optional but recommended; mirror the ingestion CLI smoke pattern.
- [ ] **Step 5: Commit** `git add src/features/cli.py src/features/__main__.py dvc.yaml dvc.lock tests/features/test_cli_smoke.py` → `feat(features): CLI + DVC stages (D2 features, D5 weak labels)` + trailer. Confirm artifacts not staged in git.

---

### Task 6: Phase 3 verification (DoD)

- [ ] **Step 1:** `pytest -q && ruff check .` → all green (Phase 0–2 + ~11 new feature tests incl. Feast).
- [ ] **Step 2:** `dvc repro` (no changes) → all stages skip (idempotent).
- [ ] **Step 3: DoD checklist:** 7-feature builder + weak labels ported & tested; **Feast feature store applies, materializes, and serves online features; train/serve consistency test passes**; D2/D5 DVC stages produce versioned real feature datasets; all tests green.

> **Note (carry to P4):** add `data/models/` to `.gitignore` when model artifacts appear (deferred from Phase 1 review).
