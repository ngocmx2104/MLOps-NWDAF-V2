# Phase 8a — Measurement Foundation + ML Test Score — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây hạ tầng đo lường tái dùng (`src/experiments/`): harness N-run subprocess, 6 collector metric, thống kê (Wilcoxon+bootstrap CI), ML Test Score assessor (evidence + auto-verify), provenance — tất cả unit-test trên input synthetic; KHÔNG sinh số liệu kết quả (đó là P8b).

**Architecture:** Mỗi collector là hàm thuần `(inputs) -> dict` test độc lập. Harness chạy workload như subprocess, đo timing/RSS ngoài (`psutil`), đọc artifact pipeline emit. ML Test Score = manifest evidence + assessor verify từng pointer. Theo convention repo: `schema.py` + `cli.py` + `__main__.py`.

**Tech Stack:** Python 3.14, numpy/pandas/scipy/scikit-learn, psutil, PyYAML, pytest/ruff. Reuse `src/monitoring` (PSI/KS), `src/tracking` (provenance), `src/serving` JSONL schemas.

**Spec:** `docs/superpowers/specs/2026-06-09-phase8a-foundation-design.md`. **Conventions:** TDD; targeted `git add <files>` (KHÔNG `-A`); mọi commit kết bằng dòng trống + trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; provenance qua `BaseTracker`; KHÔNG push/merge trừ khi HV yêu cầu.

---

## File Structure

| File | Trách nhiệm | Task |
|---|---|---|
| `src/experiments/__init__.py` · `schema.py` | `ExperimentConfig`, `RunConfig` | T1 |
| `src/experiments/records.py` | raw JSONL + run-record + result-summary (port) | T1 |
| `src/experiments/metrics/model_perf.py` | P/R/F1/ROC-AUC/PR-AUC/MCC | T2 |
| `src/experiments/metrics/business.py` | expected_cost, detection rate, early-lead | T3 |
| `src/experiments/metrics/resource.py` | subprocess timing/RSS/CPU, storage, overhead Δ | T4 |
| `src/experiments/metrics/drift_quality.py` | wrap PSI/KS + data-quality | T5 |
| `src/experiments/metrics/operational.py` | latency percentiles + DORA from events | T6 |
| `src/experiments/runner.py` | N-run subprocess loop + seeds + warmup-drop | T7 |
| `src/experiments/stats.py` | Wilcoxon + bootstrap CI + summarize | T8 |
| `src/experiments/maturity.py` + `mltest_manifest.yaml` | ML Test Score assessor + manifest | T9 |
| `src/training/{pipeline,data}.py` | provenance §9.6 (log ctx) | T10 |
| `src/experiments/cli.py` · `__main__.py` | CLI (assess, summarize) | T11 |
| `tests/experiments/**` | unit tests | every task |

All `metrics/` modules live under `src/experiments/metrics/` (with `__init__.py`).

---

## Task 1: Scaffold — schema + records

**Files:**
- Create: `src/experiments/__init__.py`, `src/experiments/schema.py`, `src/experiments/records.py`
- Test: `tests/experiments/__init__.py` (empty), `tests/experiments/test_schema_records.py`

- [ ] **Step 1: Write failing test**

```python
# tests/experiments/test_schema_records.py
import json

from src.experiments.records import append_jsonl, build_run_record, build_result_summary, write_json
from src.experiments.schema import ExperimentConfig, RunConfig


def test_runconfig_seeds_and_warmup():
    rc = RunConfig(workload=["python", "-c", "print(1)"], n_runs=5, warmup_drop=1, seeds=[1, 2, 3, 4, 5])
    assert rc.n_runs == 5 and rc.warmup_drop == 1
    assert rc.scored_seeds() == [2, 3, 4, 5]  # first dropped as warmup


def test_records_roundtrip(tmp_path):
    rec = build_run_record(experiment_id="exp1", run_index=0, seed=42,
                           metrics={"roc_auc": 0.9}, resource={"wall_s": 1.2})
    p = append_jsonl(tmp_path / "runs.jsonl", rec)
    line = json.loads(p.read_text().splitlines()[0])
    assert line["seed"] == 42 and line["metrics"]["roc_auc"] == 0.9

    summ = build_result_summary(experiment_id="exp1", configs={"C1": {"roc_auc_mean": 0.9}})
    sp = write_json(tmp_path / "summary.json", summ)
    assert json.loads(sp.read_text())["experiment_id"] == "exp1"
```

- [ ] **Step 2: Run test, verify FAIL** — `pytest tests/experiments/test_schema_records.py -q` → `ModuleNotFoundError: src.experiments`.

- [ ] **Step 3: Implement**

`src/experiments/__init__.py`:
```python
"""Phase 8a — measurement foundation: harness, metric collectors, stats, ML Test Score."""
```

`src/experiments/schema.py`:
```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

EXPERIMENT_RECORD_VERSION = "experiment_record_v1"
RESULT_SUMMARY_VERSION = "result_summary_v1"
DEFAULT_EXPERIMENT_ROOT = "artifacts/experiments"


@dataclass(frozen=True)
class RunConfig:
    """One workload run repeated n_runs times with fixed seeds; warmup_drop leading runs
    are excluded from statistics (THESIS_SPEC §4.4 protocol)."""
    workload: list[str]                 # subprocess argv; "{seed}" tokens get substituted
    n_runs: int = 10
    warmup_drop: int = 1
    seeds: list[int] = field(default_factory=lambda: list(range(42, 52)))
    env: dict[str, str] = field(default_factory=dict)

    def scored_seeds(self) -> list[int]:
        return self.seeds[self.warmup_drop:self.n_runs]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    experiment_root: str = DEFAULT_EXPERIMENT_ROOT
    configs: dict[str, RunConfig] = field(default_factory=dict)  # e.g. {"C0": rc0, "C1": rc1}

    def to_dict(self) -> dict[str, Any]:
        return {"experiment_id": self.experiment_id, "experiment_root": self.experiment_root,
                "configs": {k: v.to_dict() for k, v in self.configs.items()}}
```

`src/experiments/records.py`:
```python
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
```

- [ ] **Step 4: Run test, verify PASS** — `pytest tests/experiments/test_schema_records.py -q` → 2 passed.

- [ ] **Step 5: Commit**
```bash
git add src/experiments/__init__.py src/experiments/schema.py src/experiments/records.py tests/experiments/__init__.py tests/experiments/test_schema_records.py
git commit -m "feat(experiments): P8a scaffold — RunConfig/ExperimentConfig + JSONL records"
```

---

## Task 2: model_perf collector

**Files:**
- Create: `src/experiments/metrics/__init__.py`, `src/experiments/metrics/model_perf.py`
- Test: `tests/experiments/test_model_perf.py`

- [ ] **Step 1: Write failing test**
```python
# tests/experiments/test_model_perf.py
import numpy as np

from src.experiments.metrics.model_perf import compute_model_metrics


def test_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_model_metrics(y_true, y_score, threshold=0.5)
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_single_class_guard():
    m = compute_model_metrics(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]), threshold=0.5)
    assert m["roc_auc"] is None and m["pr_auc"] is None  # undefined with one class
```

- [ ] **Step 2: Run, verify FAIL** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/experiments/metrics/__init__.py`:
```python
"""Metric collectors for the 6 metric groups (Phase 8a)."""
```

`src/experiments/metrics/model_perf.py`:
```python
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)


def compute_model_metrics(y_true, y_score, *, threshold: float = 0.5) -> dict[str, Any]:
    """Model-performance group. roc_auc/pr_auc are None when y_true has a single class
    (undefined). y_pred is derived by thresholding y_score."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    out: dict[str, Any] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true.tolist())) > 1 else None,
    }
    if len(set(y_true.tolist())) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out
```

- [ ] **Step 4: Run, verify PASS** — 2 passed.

- [ ] **Step 5: Commit**
```bash
git add src/experiments/metrics/__init__.py src/experiments/metrics/model_perf.py tests/experiments/test_model_perf.py
git commit -m "feat(experiments): model-performance collector (P/R/F1/ROC-AUC/PR-AUC/MCC)"
```

---

## Task 3: business collector (expected cost)

**Files:**
- Create: `src/experiments/metrics/business.py`
- Test: `tests/experiments/test_business.py`

- [ ] **Step 1: Write failing test**
```python
# tests/experiments/test_business.py
import numpy as np

from src.experiments.metrics.business import business_metrics


def test_expected_cost():
    # y_true vs y_pred: 1 FP (pred1,true0), 1 FN (pred0,true1)
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    m = business_metrics(y_true, y_pred, c_fp=2.0, c_fn=10.0, window_hours=4.0)
    assert m["fp"] == 1 and m["fn"] == 1
    assert m["expected_cost"] == 1 * 2.0 + 1 * 10.0    # 12.0
    assert m["detections_per_hour"] == 2 / 4.0         # 2 positive predictions / 4h
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement**
```python
# src/experiments/metrics/business.py
from __future__ import annotations

from typing import Any

import numpy as np


def business_metrics(y_true, y_pred, *, c_fp: float, c_fn: float,
                     window_hours: float | None = None) -> dict[str, Any]:
    """Business-impact group. expected_cost = FP*C(FP) + FN*C(FN) (Elkan 2001 cost-sensitive).
    C(FP)/C(FN) are parameters; their VALUES + justification are set by the caller (P8b/thesis)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    out: dict[str, Any] = {
        "fp": fp, "fn": fn, "tp": tp,
        "expected_cost": fp * c_fp + fn * c_fn,
        "cost_params": {"c_fp": c_fp, "c_fn": c_fn},
    }
    if window_hours:
        out["detections_per_hour"] = int(np.sum(y_pred == 1)) / window_hours
    return out
```

- [ ] **Step 4: Run, verify PASS.**

- [ ] **Step 5: Commit**
```bash
git add src/experiments/metrics/business.py tests/experiments/test_business.py
git commit -m "feat(experiments): business-impact collector (cost-sensitive expected cost)"
```

---

## Task 4: resource collector (subprocess timing/RSS)

**Files:**
- Create: `src/experiments/metrics/resource.py`
- Test: `tests/experiments/test_resource.py`

- [ ] **Step 1: Write failing test**
```python
# tests/experiments/test_resource.py
import sys

from src.experiments.metrics.resource import measure_subprocess, storage_bytes


def test_measure_subprocess_allocates(tmp_path):
    # allocate ~20MB then exit 0
    code = "x = bytearray(20*1024*1024); import time; time.sleep(0.05)"
    res = measure_subprocess([sys.executable, "-c", code])
    assert res["returncode"] == 0
    assert res["wall_s"] > 0
    assert res["peak_rss_mb"] > 10  # saw the allocation


def test_storage_bytes(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 1000)
    assert storage_bytes(tmp_path) >= 1000
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement** (fresh impl; psutil is a project dep per HANDOFF §3)
```python
# src/experiments/metrics/resource.py
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import psutil


def measure_subprocess(cmd: list[str], *, env: dict[str, str] | None = None,
                       sample_interval: float = 0.02, timeout: float | None = None) -> dict[str, Any]:
    """Cost/Resource group. Run cmd as a subprocess, sampling peak RSS + CPU via psutil
    while it lives, and wall time via perf_counter. Isolated process => honest measurement."""
    full_env = {**dict(__import__("os").environ), **(env or {})}
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, env=full_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p = psutil.Process(proc.pid)
    peak_rss = 0
    cpu_samples: list[float] = []
    try:
        while proc.poll() is None:
            try:
                peak_rss = max(peak_rss, p.memory_info().rss)
                cpu_samples.append(p.cpu_percent(interval=None))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(sample_interval)
    finally:
        proc.wait(timeout=timeout)
    wall_s = time.perf_counter() - t0
    return {"returncode": proc.returncode, "wall_s": wall_s,
            "peak_rss_mb": peak_rss / (1024 * 1024),
            "cpu_pct_mean": (sum(cpu_samples) / len(cpu_samples)) if cpu_samples else 0.0}


def storage_bytes(path: Path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def tracking_overhead(noop_wall_s: float, backend_wall_s: float,
                      noop_rss_mb: float, backend_rss_mb: float) -> dict[str, float]:
    """Δ(backend - noop): the tracking-layer overhead (lõi Exp-2/3, P8b feeds the numbers)."""
    return {"delta_wall_s": backend_wall_s - noop_wall_s,
            "delta_rss_mb": backend_rss_mb - noop_rss_mb}
```

- [ ] **Step 4: Run, verify PASS** — `pytest tests/experiments/test_resource.py -q` (may take ~1s).

- [ ] **Step 5: Commit**
```bash
git add src/experiments/metrics/resource.py tests/experiments/test_resource.py
git commit -m "feat(experiments): cost/resource collector (subprocess RSS/CPU/wall + overhead Δ)"
```

---

## Task 5: drift_quality collector (wrap monitoring)

**Files:**
- Create: `src/experiments/metrics/drift_quality.py`
- Test: `tests/experiments/test_drift_quality.py`

- [ ] **Step 1: Write failing test**
```python
# tests/experiments/test_drift_quality.py
import numpy as np
import pandas as pd

from src.experiments.metrics.drift_quality import data_quality, drift_metrics


def test_drift_metrics_detects_shift():
    rng = np.random.RandomState(0)
    ref = pd.DataFrame({"f1": rng.normal(0, 1, 500), "f2": rng.normal(0, 1, 500)})
    obs = pd.DataFrame({"f1": rng.normal(5, 1, 500), "f2": rng.normal(5, 1, 500)})  # big shift
    m = drift_metrics(ref, obs, min_features_alert=1)
    assert m["drift_detected"] is True
    assert m["per_feature"]["f1"]["psi"] > 0.25


def test_data_quality():
    df = pd.DataFrame({"a": [1, 1, None], "b": [1, 1, 1]})
    q = data_quality(df)
    assert q["null_rate"] > 0 and q["duplicate_rate"] >= 0 and q["n_rows"] == 3
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement** (wrap `src/monitoring`)
```python
# src/experiments/metrics/drift_quality.py
from __future__ import annotations

from typing import Any

import pandas as pd

from src.monitoring.psi import PSIDriftMonitor
from src.monitoring.schema import MonitoringConfig


def drift_metrics(reference: pd.DataFrame, observed: pd.DataFrame, *,
                  min_features_alert: int = 3) -> dict[str, Any]:
    """Drift group (PSI 3-tier + KS) via the verified monitoring numerics."""
    cfg = MonitoringConfig(min_features_alert=min_features_alert, min_observed_rows=1)
    monitor = PSIDriftMonitor(reference_frame=reference, config=cfg)
    return monitor.evaluate(observed)


def data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Data-quality group: null/duplicate rates + row count."""
    n = len(df)
    null_rate = float(df.isna().mean().mean()) if n else 0.0
    dup_rate = float(df.duplicated().mean()) if n else 0.0
    return {"n_rows": n, "n_cols": df.shape[1], "null_rate": null_rate, "duplicate_rate": dup_rate}
```

- [ ] **Step 4: Run, verify PASS.**

- [ ] **Step 5: Commit**
```bash
git add src/experiments/metrics/drift_quality.py tests/experiments/test_drift_quality.py
git commit -m "feat(experiments): drift+data-quality collector (wraps monitoring PSI/KS)"
```

---

## Task 6: operational collector (latency + DORA)

**Files:**
- Create: `src/experiments/metrics/operational.py`
- Test: `tests/experiments/test_operational.py`

- [ ] **Step 1: Write failing test**
```python
# tests/experiments/test_operational.py
import json

from src.experiments.metrics.operational import dora_metrics, latency_percentiles


def test_latency_percentiles(tmp_path):
    p = tmp_path / "predictions.jsonl"
    with p.open("w") as f:
        for ms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            f.write(json.dumps({"event": "predict", "latency_ms": ms}) + "\n")
    pct = latency_percentiles(p)
    assert pct["p50_ms"] == 55.0 and pct["p95_ms"] >= 90.0 and pct["count"] == 10


def test_dora_metrics():
    # 3 deploys, 1 failed change, window 2 days
    events = [
        {"kind": "deploy", "ts": 0.0},
        {"kind": "deploy", "ts": 3600.0},
        {"kind": "deploy_failed", "ts": 4000.0},
        {"kind": "deploy", "ts": 7200.0},
    ]
    d = dora_metrics(events, window_days=2.0)
    assert d["deploy_count"] == 3
    assert d["deploy_frequency_per_day"] == 3 / 2.0
    assert d["change_fail_rate"] == 1 / 4  # 1 failed of 4 change attempts
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement**
```python
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
```

- [ ] **Step 4: Run, verify PASS.**

- [ ] **Step 5: Commit**
```bash
git add src/experiments/metrics/operational.py tests/experiments/test_operational.py
git commit -m "feat(experiments): operational collector (latency percentiles + DORA from events)"
```

---

## Task 7: runner (N-run subprocess loop)

**Files:**
- Create: `src/experiments/runner.py`
- Test: `tests/experiments/test_runner.py`

- [ ] **Step 1: Write failing test**
```python
# tests/experiments/test_runner.py
import json
import sys

from src.experiments.runner import run_n
from src.experiments.schema import RunConfig


def test_run_n_writes_jsonl_and_drops_warmup(tmp_path):
    rc = RunConfig(workload=[sys.executable, "-c", "print('seed={seed}')"],
                   n_runs=4, warmup_drop=1, seeds=[1, 2, 3, 4])
    out = run_n(rc, experiment_id="t", config_label="C1", output_dir=tmp_path)
    # 4 runs executed, all recorded raw; scored = 3 (warmup dropped)
    lines = (tmp_path / "runs.jsonl").read_text().splitlines()
    assert len(lines) == 4
    assert out["n_scored"] == 3
    assert all(r["resource"]["returncode"] == 0 for r in out["runs"])
    assert json.loads(lines[0])["seed"] == 1
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement**
```python
# src/experiments/runner.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.experiments.metrics.resource import measure_subprocess
from src.experiments.records import append_jsonl, build_run_record


def _bind(workload: list[str], seed: int) -> list[str]:
    return [tok.replace("{seed}", str(seed)) for tok in workload]


def run_n(rc, *, experiment_id: str, config_label: str, output_dir: Path) -> dict[str, Any]:
    """Run the workload n_runs times (one subprocess each, fixed seeds), measuring resource;
    write every run to runs.jsonl. Warmup runs are recorded but flagged scored=False."""
    output_dir = Path(output_dir)
    runs: list[dict[str, Any]] = []
    for i, seed in enumerate(rc.seeds[:rc.n_runs]):
        res = measure_subprocess(_bind(rc.workload, seed), env=rc.env)
        scored = i >= rc.warmup_drop
        rec = build_run_record(experiment_id=experiment_id, run_index=i, seed=seed,
                               resource=res, config_label=config_label)
        rec["scored"] = scored
        append_jsonl(output_dir / "runs.jsonl", rec)
        runs.append(rec)
    return {"experiment_id": experiment_id, "config_label": config_label,
            "n_total": len(runs), "n_scored": sum(1 for r in runs if r["scored"]),
            "runs": runs}
```

- [ ] **Step 4: Run, verify PASS.**

- [ ] **Step 5: Commit**
```bash
git add src/experiments/runner.py tests/experiments/test_runner.py
git commit -m "feat(experiments): N-run subprocess harness (fixed seeds + warmup drop + raw JSONL)"
```

---

## Task 8: stats (Wilcoxon + bootstrap CI)

**Files:**
- Create: `src/experiments/stats.py`
- Test: `tests/experiments/test_stats.py`

- [ ] **Step 1: Write failing test**
```python
# tests/experiments/test_stats.py
from src.experiments.stats import bootstrap_ci, summarize, wilcoxon_compare


def test_wilcoxon_detects_difference():
    c0 = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.1, 0.9, 1.0, 1.0]
    c1 = [2.0, 2.1, 1.9, 2.05, 1.95, 2.0, 2.1, 1.9, 2.0, 2.0]
    r = wilcoxon_compare(c0, c1)
    assert r["p_value"] < 0.05 and r["n"] == 10


def test_bootstrap_ci_and_summary():
    s = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["mean"] == 3.0 and s["n"] == 5
    lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], seed=0)["ci95"]
    assert lo <= 3.0 <= hi
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement**
```python
# src/experiments/stats.py
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import wilcoxon


def summarize(samples: list[float]) -> dict[str, Any]:
    arr = np.asarray(samples, dtype=float)
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()), "max": float(arr.max())}


def bootstrap_ci(samples: list[float], *, ci: float = 0.95, n_boot: int = 10000,
                 seed: int = 12345) -> dict[str, Any]:
    rng = np.random.RandomState(seed)
    arr = np.asarray(samples, dtype=float)
    means = arr[rng.randint(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return {"ci": ci, "ci95": (lo, hi)}


def wilcoxon_compare(c0: list[float], c1: list[float]) -> dict[str, Any]:
    """Paired Wilcoxon signed-rank (C0 vs C1), alpha=0.05 (THESIS_SPEC §4.4)."""
    stat, p = wilcoxon(np.asarray(c0, dtype=float), np.asarray(c1, dtype=float))
    return {"n": len(c0), "statistic": float(stat), "p_value": float(p),
            "significant_0_05": bool(p < 0.05),
            "c0": summarize(c0), "c1": summarize(c1)}
```

- [ ] **Step 4: Run, verify PASS.**

- [ ] **Step 5: Commit**
```bash
git add src/experiments/stats.py tests/experiments/test_stats.py
git commit -m "feat(experiments): stats — Wilcoxon signed-rank + bootstrap CI + summarize"
```

---

## Task 9: ML Test Score assessor + manifest

**Files:**
- Create: `src/experiments/maturity.py`, `src/experiments/mltest_manifest.yaml`
- Test: `tests/experiments/test_maturity.py`

- [ ] **Step 1: Write failing test**
```python
# tests/experiments/test_maturity.py
from src.experiments.maturity import assess


def test_assess_verifies_evidence(tmp_path):
    (tmp_path / "real.txt").write_text("x")
    manifest = {"tests": [
        {"id": "d1", "section": "data", "score": 1,
         "evidence": [{"kind": "path", "ref": "real.txt"}]},
        {"id": "d2", "section": "data", "score": 1,
         "evidence": [{"kind": "path", "ref": "MISSING.txt"}]},   # unverifiable -> 0
        {"id": "m1", "section": "model", "score": 0.5,
         "evidence": [{"kind": "path", "ref": "real.txt"}]},
        {"id": "i1", "section": "infrastructure", "score": 1,
         "evidence": [{"kind": "path", "ref": "real.txt"}]},
        {"id": "mon1", "section": "monitoring", "score": 1,
         "evidence": [{"kind": "path", "ref": "real.txt"}]},
    ]}
    rep = assess(manifest, repo_root=tmp_path)
    assert rep["section_scores"]["data"] == 1.0       # d1 counts, d2 dropped (unverified)
    assert rep["section_scores"]["model"] == 0.5
    assert rep["ml_test_score"] == 0.5                # MIN across sections
    assert rep["verified"]["d2"] is False
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement**
```python
# src/experiments/maturity.py
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

_SECTIONS = ("data", "model", "infrastructure", "monitoring")


def _verify(pointer: dict[str, Any], repo_root: Path) -> bool:
    kind, ref = pointer.get("kind"), pointer.get("ref", "")
    if kind == "path":
        return (repo_root / ref).exists()
    if kind == "workflow":
        return (repo_root / ".github" / "workflows" / ref).exists()
    if kind == "pytest":
        r = subprocess.run(["python", "-m", "pytest", "--collect-only", "-q", ref],
                           cwd=repo_root, capture_output=True, text=True)
        return r.returncode == 0
    if kind == "symbol":
        r = subprocess.run(["grep", "-rqs", ref, str(repo_root / "src")], capture_output=True)
        return r.returncode == 0
    return False


def assess(manifest: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    """ML Test Score (Breck 2017): credit a test's score ONLY if at least one evidence pointer
    verifies. Final score = MIN over the 4 sections. Also derive Google/Azure present/absent."""
    repo_root = Path(repo_root)
    per_test: dict[str, Any] = {}
    verified: dict[str, bool] = {}
    section_scores = {s: 0.0 for s in _SECTIONS}
    for t in manifest.get("tests", []):
        ok = any(_verify(p, repo_root) for p in t.get("evidence", []))
        verified[t["id"]] = ok
        credited = float(t["score"]) if ok else 0.0
        per_test[t["id"]] = {"section": t["section"], "claimed": t["score"], "credited": credited}
        if t["section"] in section_scores:
            section_scores[t["section"]] += credited
    ml_test_score = min(section_scores.values()) if section_scores else 0.0
    return {"section_scores": section_scores, "ml_test_score": ml_test_score,
            "per_test": per_test, "verified": verified,
            "google_level": _google_level(ml_test_score),
            "azure_level": _azure_level(section_scores)}


def _google_level(score: float) -> int:
    # Triangulation heuristic from the ML Test Score (Breck Table: <1 weak ... >5 strong).
    return 0 if score < 1 else (1 if score < 3 else 2)


def _azure_level(section_scores: dict[str, float]) -> int:
    covered = sum(1 for v in section_scores.values() if v >= 1)
    return covered  # 0..4 sections with >=1 automated/manual test present
```

`src/experiments/mltest_manifest.yaml` (skeleton — P8b fills real per-test evidence as artifacts accrue):
```yaml
# ML Test Score evidence manifest (Breck et al. 2017). score: 0 | 0.5 (manual) | 1 (automated).
# Each test credited ONLY if >=1 evidence pointer verifies (see src/experiments/maturity.py).
tests:
  - id: data_3_feature_code_tested
    section: data
    score: 1
    evidence:
      - {kind: path, ref: "src/features/quality.py"}
  - id: infra_1_training_reproducible
    section: infrastructure
    score: 1
    evidence:
      - {kind: pytest, ref: "tests/training/test_pipeline_mlflow.py"}
  - id: infra_5_ci_full_suite
    section: infrastructure
    score: 1
    evidence:
      - {kind: workflow, ref: "ci.yml"}
  - id: monitor_1_drift_detection
    section: monitoring
    score: 1
    evidence:
      - {kind: pytest, ref: "tests/monitoring/test_retrain.py"}
  - id: model_2_offline_metrics
    section: model
    score: 1
    evidence:
      - {kind: symbol, ref: "compute_model_metrics"}
```

- [ ] **Step 4: Run, verify PASS** — `pytest tests/experiments/test_maturity.py -q`.

- [ ] **Step 5: Commit**
```bash
git add src/experiments/maturity.py src/experiments/mltest_manifest.yaml tests/experiments/test_maturity.py
git commit -m "feat(experiments): ML Test Score assessor (evidence manifest + auto-verify + MIN)"
```

---

## Task 10: Provenance (§9.6) — log TrainingContext

**Files:**
- Modify: `src/training/pipeline.py`
- Test: `tests/training/test_provenance.py`

- [ ] **Step 1: Write failing test**
```python
# tests/training/test_provenance.py
from src.training.pipeline import run_training
from src.training.schema import TrainingConfig


def test_training_logs_lineage_tags(labeled_features, tmp_path, monkeypatch):
    """run_training should log dataset lineage (dataset_id/feature_version/source_snapshot_id)
    as params via the tracker (provenance, HANDOFF §9.6)."""
    captured = {}
    import src.training.pipeline as pipe

    real_create = pipe.create_tracker

    def spy_create(backend=None):
        tr = real_create(backend)
        orig = tr.log_params
        tr.log_params = lambda params: (captured.update(params), orig(params))[1]
        return tr

    monkeypatch.setattr(pipe, "create_tracker", spy_create)
    path, _ = labeled_features
    run_training(path, model_type="iforest", backend="noop", output_dir=tmp_path / "m",
                 cfg=TrainingConfig(use_labels_for_evaluation=True, label_column="label"))
    assert "dataset_id" in captured and "feature_version" in captured and "source_snapshot_id" in captured
```

(`labeled_features` fixture is in `tests/training/conftest.py`.)

- [ ] **Step 2: Run, verify FAIL** — lineage keys absent.

- [ ] **Step 3: Implement** — in `src/training/pipeline.py`, the iforest branch already gets `ctx` from `load_training_dataset(dataset_path, cfg)` (currently unused). Log it. Change the `tracker.log_params({...})` call (currently `{**cfg.to_dict(), "model_type": model_type}`) to merge lineage when `ctx` is available:

```python
    params = {**cfg.to_dict(), "model_type": model_type}
    if model_type == "iforest":
        params.update({"dataset_id": ctx.dataset_id, "feature_version": ctx.feature_version,
                       "source_snapshot_id": ctx.source_snapshot_id,
                       "dataset_layer": ctx.dataset_layer, "row_count": ctx.row_count})
    tracker.log_params(params)
```

(`ctx` is bound only in the iforest branch — the `if model_type == "iforest"` guard ensures it's referenced only when defined; LSTM lineage is deferred to P8b with §9.7. The existing `tracker.log_params(...)` line already sits AFTER the model-type branch, so just replace that one line with the block above — no reordering needed.)

- [ ] **Step 4: Run, verify PASS** — `pytest tests/training/test_provenance.py tests/training/ -q` (no regressions).

- [ ] **Step 5: Commit**
```bash
git add src/training/pipeline.py tests/training/test_provenance.py
git commit -m "feat(training): log dataset lineage params for provenance (HANDOFF §9.6)"
```

---

## Task 11: CLI + whole-phase verify

**Files:**
- Create: `src/experiments/cli.py`, `src/experiments/__main__.py`
- Test: `tests/experiments/test_cli.py`

- [ ] **Step 1: Write failing test**
```python
# tests/experiments/test_cli.py
import json

from src.experiments.cli import main


def test_assess_cli(tmp_path):
    (tmp_path / "real.txt").write_text("x")
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        "tests:\n"
        "  - id: d1\n    section: data\n    score: 1\n"
        "    evidence:\n      - {kind: path, ref: real.txt}\n")
    out = tmp_path / "report.json"
    rc = main(["assess", "--manifest", str(manifest), "--repo-root", str(tmp_path),
               "--output", str(out)])
    assert rc == 0
    assert "ml_test_score" in json.loads(out.read_text())
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement**
```python
# src/experiments/cli.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.experiments.maturity import assess
from src.experiments.records import write_json


def cmd_assess(args: argparse.Namespace) -> int:
    manifest = yaml.safe_load(Path(args.manifest).read_text())
    report = assess(manifest, repo_root=Path(args.repo_root))
    if args.output:
        write_json(Path(args.output), report)
    print(f"[ml-test-score] {report['ml_test_score']} "
          f"(sections={report['section_scores']}, google L{report['google_level']})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("assess", help="Run the ML Test Score assessor over an evidence manifest.")
    a.add_argument("--manifest", required=True)
    a.add_argument("--repo-root", dest="repo_root", default=".")
    a.add_argument("--output", default=None)
    a.set_defaults(func=cmd_assess)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
```

`src/experiments/__main__.py`:
```python
import sys

from src.experiments.cli import main

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run, verify PASS.**

- [ ] **Step 5: Whole-phase verify**

Run:
```bash
source .venv/bin/activate
pytest -q && ruff check .
python -m src.experiments assess --manifest src/experiments/mltest_manifest.yaml --repo-root . --output /tmp/mlts.json
```
Expected: full suite green (133 + new P8a tests), ruff clean, assessor prints a real ML Test Score from the live repo manifest (every credited point verified).

- [ ] **Step 6: Update docs + commit**

Mark P8a done in `docs/HANDOFF.md` (§0/§6) + `docs/IMPLEMENTATION_PLAN.md`; note §9.6 resolved; next = P8b.
```bash
git add src/experiments/cli.py src/experiments/__main__.py tests/experiments/test_cli.py docs/HANDOFF.md docs/IMPLEMENTATION_PLAN.md
git commit -m "feat(experiments): assess CLI + P8a whole-phase verify; mark P8a done"
```

- [ ] **Step 7: Final whole-phase review + finish branch** — dispatch final reviewer (opus) over `main..HEAD`; apply fixes; then `superpowers:finishing-a-development-branch` (merge `phase8a` → main local, only when HV chooses).

---

## Definition of Done (kiểm chứng được)

1. `src/experiments/` đầy đủ (schema/records/runner/stats/maturity/metrics×5/cli) + `python -m src.experiments assess` chạy.
2. 6 nhóm collector có unit-test xanh trên synthetic (model_perf, business, resource, drift_quality, operational+DORA).
3. runner: subprocess N-run, fixed seeds, warmup-drop, raw JSONL ✔ test.
4. stats: Wilcoxon + bootstrap CI + summarize ✔ test.
5. ML Test Score assessor: evidence auto-verify + MIN + Google/Azure ✔ test (pointer thật/giả); manifest skeleton chạy trên repo thật.
6. Provenance §9.6: lineage params logged ✔ test.
7. 133 test cũ + mới xanh; ruff sạch; KHÔNG sinh số liệu kết quả (P8a chỉ hạ tầng).
8. Truy vết RQ/metric đầy đủ; ranh giới P8a/P8b rõ.

> **Quy tắc vàng:** P8a là hạ tầng đo — KHÔNG bịa số liệu; collector gắn 1 nhóm metric; ML Test Score chỉ tính điểm verify được. Số thật sinh ở P8b.
