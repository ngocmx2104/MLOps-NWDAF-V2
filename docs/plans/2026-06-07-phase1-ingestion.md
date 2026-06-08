# Phase 1 — Data Ingestion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the EBS data-ingestion subsystem — positional parser, schema validation, data-quality checks, D1 canonical snapshot — plus a DVC pipeline stage that produces a versioned snapshot. Establishes RQ1 evidence and feeds Exp-1.

**Architecture:** Port the proven logic from the old repo `MLOps_Project/src/ingestion/` (it already uses the `src.` layout, so ports are near drop-in), re-implemented cleanly WITH pytest tests (the old repo had none — adding tests raises the ML Test Score, Phase 8). Add a `dvc.yaml` stage `snapshot` that turns raw EBS → versioned D1 parquet + metadata sidecar.

**Tech Stack:** pandas, pyarrow, DVC. Tests use tiny in-memory EBS fixtures (no real-data dependency for unit tests).

**Source to port (read these exact files):**
- `/Users/ngocmx/Thạc Sĩ/MLOps_Project/src/ingestion/schema.py`
- `/Users/ngocmx/Thạc Sĩ/MLOps_Project/src/ingestion/parser.py`
- `/Users/ngocmx/Thạc Sĩ/MLOps_Project/src/ingestion/quality.py`
- `/Users/ngocmx/Thạc Sĩ/MLOps_Project/src/ingestion/snapshot.py`
- `/Users/ngocmx/Thạc Sĩ/MLOps_Project/src/ingestion/cli.py`

**Branch:** `phase1-ingestion` (create from `main`).

**File structure created this phase:**
```
src/ingestion/__init__.py
src/ingestion/schema.py      # 52-field positional schema (port)
src/ingestion/parser.py      # parse_ebs_files + normalize_timestamps (port)
src/ingestion/quality.py     # 8 quality checks (port)
src/ingestion/snapshot.py    # build_d1_snapshot (port)
src/ingestion/cli.py         # parse-raw / build-snapshot / check-quality (port)
src/ingestion/__main__.py    # python -m src.ingestion
tests/ingestion/__init__.py
tests/ingestion/conftest.py  # tiny EBS fixtures (NEW)
tests/ingestion/test_schema.py, test_parser.py, test_quality.py, test_snapshot.py (NEW)
dvc.yaml                     # snapshot stage (NEW)
data/raw_ebs/                # real EBS files copied in (gitignored, DVC-tracked)
```

---

### Task 1: Port ingestion schema

**Files:** Create `src/ingestion/__init__.py` (empty), `src/ingestion/schema.py`, `tests/ingestion/__init__.py` (empty), `tests/ingestion/test_schema.py`.

- [ ] **Step 1: Port schema.py** — copy `MLOps_Project/src/ingestion/schema.py` VERBATIM to `src/ingestion/schema.py` (it has no cross-package imports; drop-in). It defines `EBS_POSITIONAL_FIELDS` (52 `FieldDef`), `EXPECTED_FIELD_COUNT=52`, `FIELD_NAMES`, `FIELD_NAME_TO_POS`, `SCHEMA_VERSION`, `USE_CASE_REQUIRED_FIELDS`, `D1_CORE_COLUMNS`, `D1_PROVENANCE_COLUMNS`, `D1_OPTIONAL_COLUMNS`, `get_field_names()`.

- [ ] **Step 2: Write tests** — `tests/ingestion/test_schema.py`:
```python
from src.ingestion.schema import (
    EXPECTED_FIELD_COUNT, FIELD_NAMES, FIELD_NAME_TO_POS, get_field_names,
)


def test_field_count_is_52():
    assert EXPECTED_FIELD_COUNT == 52
    assert len(FIELD_NAMES) == 52


def test_key_field_positions():
    assert FIELD_NAME_TO_POS["event_id"] == 0
    assert FIELD_NAME_TO_POS["imsi"] == 6
    assert FIELD_NAME_TO_POS["event_time"] == 48


def test_get_field_names_matches():
    assert tuple(get_field_names()) == FIELD_NAMES
```

- [ ] **Step 3: Run** — `source .venv/bin/activate && pytest tests/ingestion/test_schema.py -v` → 3 passed. Then `ruff check src/ingestion tests/ingestion`.

- [ ] **Step 4: Commit** — `git add src/ingestion/__init__.py src/ingestion/schema.py tests/ingestion/__init__.py tests/ingestion/test_schema.py` then commit `feat(ingestion): port EBS 52-field positional schema` + blank line + `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 2: Port parser + add fixtures

**Files:** Create `src/ingestion/parser.py`, `tests/ingestion/conftest.py`, `tests/ingestion/test_parser.py`.

- [ ] **Step 1: Port parser.py** — copy `MLOps_Project/src/ingestion/parser.py` VERBATIM to `src/ingestion/parser.py` (imports `from src.ingestion.schema import ...` — already correct). Provides `parse_ebs_files(file_paths, *, warn_field_count=True)`, `iter_parsed_rows`, `normalize_timestamps(df)`.

- [ ] **Step 2: Create fixture** — `tests/ingestion/conftest.py`:
```python
import pytest


def make_ebs_line(event_id="l_handover", imsi="111", event_time="1719381600000", eci="10"):
    """Build one 52-field semicolon-delimited EBS line (positions per schema)."""
    fields = [""] * 52
    fields[0] = event_id       # event_id
    fields[6] = imsi           # imsi
    fields[12] = eci           # eci (cell)
    fields[48] = event_time    # event_time (epoch ms)
    fields[51] = "2024062614"  # date_hour
    return ";".join(fields)


@pytest.fixture
def tiny_ebs_file(tmp_path):
    """A 3-line raw EBS file: 2 handover events (same IMSI, 2 cells) + 1 non-handover."""
    lines = [
        make_ebs_line("l_handover", "111", "1719381600000", "10"),
        make_ebs_line("l_handover", "111", "1719381630000", "20"),
        make_ebs_line("l_service_request", "222", "1719381660000", "30"),
    ]
    p = tmp_path / "A20240626.1400+0700-20240626.1401+0700_840_ebs"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
```

- [ ] **Step 3: Write tests** — `tests/ingestion/test_parser.py`:
```python
import pandas as pd

from src.ingestion.parser import normalize_timestamps, parse_ebs_files
from src.ingestion.schema import EXPECTED_FIELD_COUNT


def test_parse_basic(tiny_ebs_file):
    df = parse_ebs_files([tiny_ebs_file])
    assert len(df) == 3
    assert df["event_id"].tolist() == ["l_handover", "l_handover", "l_service_request"]
    assert df["imsi"].tolist() == ["111", "111", "222"]
    assert (df["raw_field_count"] == EXPECTED_FIELD_COUNT).all()
    assert df["source_file"].iloc[0] == tiny_ebs_file.name
    assert (df["schema_version"] == "ebs_raw_positional_v1").all()


def test_empty_values_become_none(tiny_ebs_file):
    df = parse_ebs_files([tiny_ebs_file])
    # msisdn (pos 5) was left empty in the fixture
    assert df["msisdn"].isna().all()


def test_normalize_timestamps(tiny_ebs_file):
    df = normalize_timestamps(parse_ebs_files([tiny_ebs_file]))
    assert pd.api.types.is_datetime64_any_dtype(df["event_ts"])
    assert df["event_ts"].notna().all()
    assert str(df["event_ts"].dt.tz) == "UTC"


def test_missing_file_skipped_but_others_parsed(tiny_ebs_file, tmp_path):
    df = parse_ebs_files([tmp_path / "nope_ebs", tiny_ebs_file])
    assert len(df) == 3
```

- [ ] **Step 4: Run** — `pytest tests/ingestion/test_parser.py -v` → 4 passed; `ruff check src/ingestion tests/ingestion`.

- [ ] **Step 5: Commit** — `git add src/ingestion/parser.py tests/ingestion/conftest.py tests/ingestion/test_parser.py` then `feat(ingestion): port EBS positional parser + timestamp normalization` + trailer.

---

### Task 3: Port quality checks

**Files:** Create `src/ingestion/quality.py`, `tests/ingestion/test_quality.py`.

- [ ] **Step 1: Port quality.py** — copy `MLOps_Project/src/ingestion/quality.py` VERBATIM to `src/ingestion/quality.py`. Provides `QualityCheckResult`, the per-check functions, `run_raw_quality_checks`, `run_d1_quality_checks`, `format_quality_report`, `quality_results_to_dicts`.

- [ ] **Step 2: Write tests** — `tests/ingestion/test_quality.py`:
```python
from src.ingestion.parser import normalize_timestamps, parse_ebs_files
from src.ingestion.quality import (
    check_event_filter, check_field_count, run_d1_quality_checks,
    run_raw_quality_checks, quality_results_to_dicts,
)


def test_field_count_check_passes(tiny_ebs_file):
    df = parse_ebs_files([tiny_ebs_file])
    res = check_field_count(df)
    assert res.passed
    assert res.detail["pct_matching"] == 100.0


def test_raw_checks_run(tiny_ebs_file):
    df = parse_ebs_files([tiny_ebs_file])
    results = run_raw_quality_checks(df)
    assert len(results) == 3
    assert all(hasattr(r, "passed") for r in results)


def test_event_filter_on_handover_only(tiny_ebs_file):
    df = normalize_timestamps(parse_ebs_files([tiny_ebs_file]))
    ho = df[df["event_id"] == "l_handover"].copy()
    assert check_event_filter(ho).passed  # 100% are l_handover


def test_results_serialisable(tiny_ebs_file):
    df = parse_ebs_files([tiny_ebs_file])
    dicts = quality_results_to_dicts(run_raw_quality_checks(df))
    assert isinstance(dicts, list) and "check_id" in dicts[0]
```

- [ ] **Step 3: Run** — `pytest tests/ingestion/test_quality.py -v` → 4 passed; `ruff check src/ingestion tests/ingestion`.

- [ ] **Step 4: Commit** — `git add src/ingestion/quality.py tests/ingestion/test_quality.py` then `feat(ingestion): port EBS/D1 data-quality checks` + trailer.

---

### Task 4: Port D1 snapshot builder

**Files:** Create `src/ingestion/snapshot.py`, `tests/ingestion/test_snapshot.py`.

- [ ] **Step 1: Port snapshot.py** — copy `MLOps_Project/src/ingestion/snapshot.py` VERBATIM to `src/ingestion/snapshot.py`. Provides `build_d1_snapshot(source_files, output_dir, *, event_filter="l_handover", snapshot_id=None) -> dict` (parse → raw QC → normalize → filter → metadata → select D1 cols → D1 QC → write parquet + JSON sidecar).

- [ ] **Step 2: Write tests** — `tests/ingestion/test_snapshot.py`:
```python
import json
from pathlib import Path

import pandas as pd

from src.ingestion.snapshot import build_d1_snapshot


def test_build_snapshot_filters_handover(tiny_ebs_file, tmp_path):
    out = tmp_path / "snap"
    result = build_d1_snapshot([tiny_ebs_file], out, snapshot_id="D1_TEST")
    assert result["raw_row_count"] == 3
    assert result["d1_row_count"] == 2  # only the 2 l_handover rows
    df = pd.read_parquet(result["parquet_path"])
    assert (df["event_id"] == "l_handover").all()
    assert "dataset_snapshot_id" in df.columns and "event_ts" in df.columns


def test_snapshot_writes_metadata_sidecar(tiny_ebs_file, tmp_path):
    result = build_d1_snapshot([tiny_ebs_file], tmp_path / "snap", snapshot_id="D1_TEST")
    meta = json.loads(Path(result["metadata_path"]).read_text())
    assert meta["dataset_snapshot_id"] == "D1_TEST"
    assert meta["event_filter"] == "l_handover"
    assert meta["raw_row_count"] == 3 and meta["filtered_row_count"] == 2
    assert "raw_quality_checks" in meta and "d1_quality_checks" in meta


def test_snapshot_returns_quality_flag(tiny_ebs_file, tmp_path):
    result = build_d1_snapshot([tiny_ebs_file], tmp_path / "snap")
    assert isinstance(result["all_quality_checks_passed"], bool)
```

- [ ] **Step 3: Run** — `pytest tests/ingestion/test_snapshot.py -v` → 3 passed; `ruff check src/ingestion tests/ingestion`.

- [ ] **Step 4: Commit** — `git add src/ingestion/snapshot.py tests/ingestion/test_snapshot.py` then `feat(ingestion): port D1 canonical snapshot builder` + trailer.

---

### Task 5: Port CLI + `__main__`

**Files:** Create `src/ingestion/cli.py`, `src/ingestion/__main__.py`.

- [ ] **Step 1: Port cli.py** — copy `MLOps_Project/src/ingestion/cli.py` to `src/ingestion/cli.py`, but CHANGE the default data paths: set `_DATA_DIR = _project_root / "data" / "raw_ebs"` and `DEFAULT_EBS_FILES = sorted(_DATA_DIR.glob("*_ebs"))` (instead of the old hardcoded 3 paths under `mlops_source_and_thesis/Data EBS`). Keep the three subcommands: `build-snapshot`, `parse-raw`, `check-quality`. Change default `--output-dir` to `_project_root / "artifacts" / "snapshots"`.

- [ ] **Step 2: Create `src/ingestion/__main__.py`:**
```python
from src.ingestion.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke test the CLI via fixture** — add to `tests/ingestion/test_cli_smoke.py`:
```python
import subprocess
import sys
from pathlib import Path

from tests.ingestion.conftest import make_ebs_line


def test_cli_build_snapshot_smoke(tmp_path):
    raw = tmp_path / "A20240626.1400+0700-20240626.1401+0700_840_ebs"
    raw.write_text("\n".join([
        make_ebs_line("l_handover", "111", "1719381600000", "10"),
        make_ebs_line("l_handover", "111", "1719381630000", "20"),
    ]) + "\n", encoding="utf-8")
    out = tmp_path / "snap"
    r = subprocess.run(
        [sys.executable, "-m", "src.ingestion.cli", "build-snapshot",
         "--output-dir", str(out), "--snapshot-id", "D1_SMOKE", str(raw)],
        capture_output=True, text=True, cwd=Path.cwd(),
    )
    assert r.returncode == 0, r.stderr
    assert (out / "D1_SMOKE.parquet").exists()
    assert (out / "D1_SMOKE_metadata.json").exists()
```
> Note: importing `make_ebs_line` from `tests.ingestion.conftest` requires `tests/__init__.py` and `tests/ingestion/__init__.py` to exist (they do). If import fails, inline the helper instead.

- [ ] **Step 4: Run** — `pytest tests/ingestion/test_cli_smoke.py -v` → 1 passed; `ruff check src/ingestion tests/ingestion`.

- [ ] **Step 5: Commit** — `git add src/ingestion/cli.py src/ingestion/__main__.py tests/ingestion/test_cli_smoke.py` then `feat(ingestion): CLI (build-snapshot/parse-raw/check-quality) + __main__` + trailer.

---

### Task 6: Real EBS data + DVC `snapshot` stage (versioned output)

**Files:** Create `data/raw_ebs/` (real files, DVC-tracked), `dvc.yaml`, `dvc.lock`, `data/raw_ebs.dvc` (or stage deps), `params.yaml`.

- [ ] **Step 1: Copy real EBS files into the repo**

Run (the source has spaces in the dir name — quote it):
```bash
mkdir -p data/raw_ebs
cp "/Users/ngocmx/Thạc Sĩ/MLOps_Project/mlops_source_and_thesis/Data EBS/"*_ebs data/raw_ebs/
ls -1 data/raw_ebs/
```
Expected: 3 files `A20240626.14*_ebs`. If the source directory does not exist, STOP and report BLOCKED (do not fabricate data).

- [ ] **Step 2: Track raw data with DVC**

Run: `dvc add data/raw_ebs`
Expected: creates `data/raw_ebs.dvc` and adds `data/raw_ebs` to `data/.gitignore`. (Confirm `data/` is already in root `.gitignore`; the DVC pointer `data/raw_ebs.dvc` IS committed to git.)

- [ ] **Step 3: Create `dvc.yaml` with the `snapshot` stage**
```yaml
stages:
  snapshot:
    cmd: python -m src.ingestion.cli build-snapshot --output-dir artifacts/snapshots data/raw_ebs/*_ebs
    deps:
      - data/raw_ebs
      - src/ingestion
    outs:
      - artifacts/snapshots
```
> `artifacts/` is gitignored; DVC manages the `snapshot` output. The shell glob `data/raw_ebs/*_ebs` is expanded by the shell DVC invokes.

- [ ] **Step 4: Run the pipeline**

Run: `source .venv/bin/activate && dvc repro snapshot`
Expected: stage runs, prints "D1 Canonical Snapshot Build Complete", writes `artifacts/snapshots/D1_EBS_HO_*.parquet` + `*_metadata.json`, creates/updates `dvc.lock`.

- [ ] **Step 5: Verify the snapshot**
```bash
ls -1 artifacts/snapshots/
python -c "import pandas as pd, glob; p=glob.glob('artifacts/snapshots/D1_*.parquet')[0]; df=pd.read_parquet(p); print('rows', len(df)); print('all handover', (df['event_id']=='l_handover').all()); print('cols', list(df.columns)[:8])"
```
Expected: a real D1 parquet with thousands of handover rows, all `event_id == l_handover`, provenance columns present. Also open one `*_metadata.json` and confirm `raw_row_count` >> `filtered_row_count` and quality-check sections present.

- [ ] **Step 6: Commit the DVC artifacts (NOT the data)**

`git add dvc.yaml dvc.lock data/raw_ebs.dvc data/.gitignore` then commit `feat(ingestion): DVC snapshot stage — versioned D1 from real EBS` + trailer. Confirm `git status` shows NO large `*_ebs` files staged (they are DVC-tracked, gitignored).

---

### Task 7: Phase 1 verification (DoD)

- [ ] **Step 1: Full suite** — `pytest -q && ruff check .` → all green (Phase-0 tests + ~15 new ingestion tests).
- [ ] **Step 2: Reproducibility** — `dvc repro` (no changes) → reports "Stage 'snapshot' didn't change, skipping" (proves deterministic/cached).
- [ ] **Step 3: DoD checklist:** real EBS parsed → versioned D1 snapshot via `dvc repro`; data-quality metrics in metadata sidecar; all tests green; `dvc status` clean.
