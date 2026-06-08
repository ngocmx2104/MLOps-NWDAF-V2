# Phase 6 — Monitoring + auto-retrain (C8) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Build the monitoring subsystem (C8): covariate-drift detection (PSI 3-tier custom + Evidently industry cross-check) over the serving `predictions.jsonl`, and a **drift → auto-retrain → reload** closed loop — the RQ4/Exp-4 capability that lets the pipeline recover from concept drift.

**Architecture:** A standalone `src/monitoring/` package reads the serving substrate (`predictions.jsonl`, which carries each request's `feature_values`) and a reference distribution (the training feature parquet). `DriftDetector` runs PSI per-feature (3-tier ok/warn/alert, primary measured detector) and Evidently `DataDriftPreset` (standard cross-validation), producing one drift verdict. `AutoRetrainTrigger` (cooldown) gates `run_retrain_cycle`, which retrains on a **sliding window** (`apply_sliding_window` — closing the P4 carry-forward) via the existing `run_training`, applies a basic eval gate (new ROC-AUC ≥ threshold), and calls `ServingRuntime.reload()` to deploy. A minimal Prometheus `/metrics` endpoint (request count + latency) makes serving scrape-ready.

**Tech Stack:** numpy/scipy (PSI + KS), **Evidently 0.7.21** (new API: `from evidently import Report` + `from evidently.presets import DataDriftPreset`), pandas, `prometheus_client`, FastAPI (existing serving app), reuses `src.training.run_training` + `src.serving.ServingRuntime`.

**Research framing (locked in brainstorming — anti-over-engineering):** Monitoring is **C1's capability**, NOT a backend abstraction (there is no meaningful "noop monitor" — absence of monitoring IS C0). The controlled comparison is the drift→retrain loop **ON (C1) vs OFF (C0)**, measured in Exp-4 (P8): detection latency, drift recall/FPR, recovery time. `run_retrain_cycle` returns whether it retrained so the P8 harness can drive both arms. PSI = the implemented detector (measured); Evidently = standard-tool cross-validation (answers the "manual/cảm tính" critique). Drift is on **`feature_values`** (covariate) because labels are weak (no supervised DDM/ADWIN — per CLAUDE.md §4).

**Explicitly OUT of P6 (deferred to P7, recorded in HANDOFF §9 #12):** real Prometheus server + Grafana + dashboards + `docker-compose.monitoring.yml` (observability infrastructure belongs with P7's server-standup). **Dropped entirely:** a periodic `loop` daemon (Exp-4 feeds windows as a script, not a live daemon) and pushing drift gauges to Prometheus (the drift artifact is `drift_history.jsonl` + measured metrics, not a gauge).

**Source to read (port the proven numerics, write clean):**
- `/Users/ngocmx/Thạc Sĩ/MLOps_Project/src/serving/monitoring.py:186-314` — `_compute_psi`, `PSIDriftMonitor` (3-criteria PSI+KS), `AutoRetrainTrigger` (cooldown). Port these numerics verbatim; do NOT port `MonitoringState`/`SystemMetricsCollector`/`DriftMonitor` (mean-shift) — superseded/cut.
- Current repo: `src/serving/records.py` (`append_jsonl`), `src/serving/runtime.py` (`ServingRuntime.reload`), `src/training/pipeline.py` (`run_training` returns `{"model_path","model_version","metrics":{...roc_auc...}}`), `src/training/data.py` (`apply_sliding_window`), `src/training/schema.py` (`FEATURE_COLUMNS`), `src/serving/app.py` (the FastAPI app to instrument).

**Branch:** `phase6-monitoring` (from `main`).

**Verified APIs (this venv):** Evidently 0.7.21 — `Report([DataDriftPreset()]).run(current_data=cur, reference_data=ref)` returns a Snapshot; `snapshot.dict()["metrics"]` is a list where the drift-count metric has `value={"count":N,"share":S}`. `prometheus_client` installed. `run_training(...)` + `ServingRuntime.reload()` verified in P4/P5.

---

### Task 1: schema + PSI drift (3-tier + KS)

**Files:** Create `src/monitoring/__init__.py` (empty), `src/monitoring/schema.py`, `src/monitoring/psi.py`, `tests/monitoring/__init__.py` (empty), `tests/monitoring/test_psi.py`.

- [ ] **Step 1: Write the failing test** — `tests/monitoring/test_psi.py`:

```python
import numpy as np
import pandas as pd

from src.monitoring.psi import PSIDriftMonitor, compute_psi
from src.monitoring.schema import MonitoringConfig


def test_psi_zero_for_same_distribution():
    rng = np.random.RandomState(0)
    x = rng.gamma(2.0, 1.0, 500)
    assert compute_psi(x, x) < 0.01


def test_psi_high_for_shifted_distribution():
    rng = np.random.RandomState(0)
    ref = rng.gamma(2.0, 1.0, 500)
    obs = rng.gamma(2.0, 1.0, 500) * 4.0
    assert compute_psi(ref, obs) >= 0.25  # >= alert tier


def test_monitor_no_drift_when_identical():
    rng = np.random.RandomState(0)
    cols = ["n_handover", "pingpong_count", "entropy_cell_seq"]
    ref = pd.DataFrame({c: rng.gamma(2.0, 1.0, 300) for c in cols})
    mon = PSIDriftMonitor(reference_frame=ref, config=MonitoringConfig(min_features_alert=2))
    result = mon.evaluate(ref.copy())
    assert result["drift_detected"] is False
    assert result["per_feature"]["n_handover"]["psi_level"] == "ok"


def test_monitor_detects_drift_when_shifted():
    rng = np.random.RandomState(0)
    cols = ["n_handover", "pingpong_count", "entropy_cell_seq"]
    ref = pd.DataFrame({c: rng.gamma(2.0, 1.0, 300) for c in cols})
    obs = pd.DataFrame({c: rng.gamma(2.0, 1.0, 300) * 4.0 for c in cols})
    mon = PSIDriftMonitor(reference_frame=ref, config=MonitoringConfig(min_features_alert=2, min_observed_rows=50))
    result = mon.evaluate(obs)
    assert result["drift_detected"] is True
    assert result["alerted_features"] >= 2
    # KS cross-statistic is present per feature
    assert "ks_pvalue" in result["per_feature"]["n_handover"]
```

- [ ] **Step 2: Run** `pytest tests/monitoring/test_psi.py -v` → expect FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write `src/monitoring/__init__.py`** (empty) and **`src/monitoring/schema.py`:**

```python
"""Monitoring subsystem (component C8) — config + record schemas.

Monitoring is C1's capability (no backend abstraction); the C0/C1 contrast is the
drift->retrain loop ON vs OFF, measured in Exp-4 (P8).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

PSI_WARN = 0.20
PSI_ALERT = 0.25
PSI_BINS = 10


@dataclass(frozen=True)
class MonitoringConfig:
    psi_warn: float = PSI_WARN
    psi_alert: float = PSI_ALERT
    psi_bins: int = PSI_BINS
    min_features_alert: int = 3       # drift_detected when >= this many features hit 'alert'
    min_observed_rows: int = 50       # below this, report 'waiting-min-window'
    evidently_drift_threshold: float = 0.5  # evidently drift share >= this -> flags (cross-check)
    cooldown_seconds: float = 3600.0  # min gap between retrains
    retrain_min_auc: float = 0.5      # eval gate: deploy only if new val ROC-AUC >= this
    window_days: int = 5              # sliding window for retraining data

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

- [ ] **Step 4: Write `src/monitoring/psi.py`** (port the numerics from `MLOps_Project/src/serving/monitoring.py`):

```python
"""PSI-based covariate drift detection (3-tier ok/warn/alert) + KS cross-statistic.

Ported from the verified MLOps_Project monitoring numerics. PSI is the primary
*measured* detector for Exp-4; the KS p-value is reported alongside as a check.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.monitoring.schema import MonitoringConfig


def compute_psi(reference: np.ndarray, observed: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1-D arrays."""
    eps = 1e-6
    breakpoints = np.linspace(float(np.min(reference)), float(np.max(reference)), bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    ref_counts = np.histogram(reference, bins=breakpoints)[0].astype(float)
    obs_counts = np.histogram(observed, bins=breakpoints)[0].astype(float)
    ref_pct = np.clip(ref_counts / (ref_counts.sum() + eps), eps, None)
    obs_pct = np.clip(obs_counts / (obs_counts.sum() + eps), eps, None)
    return float(np.sum((obs_pct - ref_pct) * np.log(obs_pct / ref_pct)))


@dataclass(frozen=True)
class PSIDriftMonitor:
    reference_frame: pd.DataFrame | None = None
    config: MonitoringConfig = field(default_factory=MonitoringConfig)

    def evaluate(self, observed_frame: pd.DataFrame | None) -> dict[str, Any]:
        cfg = self.config
        if self.reference_frame is None:
            return {"status": "reference-unavailable", "drift_detected": False, "per_feature": {}}
        if observed_frame is None or observed_frame.empty:
            return {"status": "insufficient-live-window", "drift_detected": False, "per_feature": {}}
        if observed_frame.shape[0] < cfg.min_observed_rows:
            return {"status": "waiting-min-window", "drift_detected": False,
                    "observed_rows": int(observed_frame.shape[0]),
                    "required_rows": cfg.min_observed_rows, "per_feature": {}}

        per_feature: dict[str, Any] = {}
        alerted = 0
        for col in observed_frame.columns:
            if col not in self.reference_frame.columns:
                continue
            ref = self.reference_frame[col].dropna().to_numpy()
            obs = observed_frame[col].dropna().to_numpy()
            if len(ref) == 0 or len(obs) == 0:
                continue
            psi_val = compute_psi(ref, obs, cfg.psi_bins)
            ks_stat, ks_p = ks_2samp(ref, obs)
            level = "alert" if psi_val >= cfg.psi_alert else ("warn" if psi_val >= cfg.psi_warn else "ok")
            per_feature[col] = {"psi": round(psi_val, 4), "psi_level": level,
                                "ks_statistic": round(float(ks_stat), 4),
                                "ks_pvalue": round(float(ks_p), 6)}
            if psi_val >= cfg.psi_alert:
                alerted += 1

        return {"status": "evaluated", "drift_detected": alerted >= cfg.min_features_alert,
                "alerted_features": alerted, "total_features": len(per_feature),
                "psi_alert_threshold": cfg.psi_alert, "psi_warn_threshold": cfg.psi_warn,
                "min_features_for_alert": cfg.min_features_alert, "per_feature": per_feature}
```

- [ ] **Step 5: Run** `pytest tests/monitoring/test_psi.py -v` (expect 4 passed) + `ruff check src/monitoring tests/monitoring` (clean).

- [ ] **Step 6: Commit**

```bash
git add src/monitoring/__init__.py src/monitoring/schema.py src/monitoring/psi.py tests/monitoring/__init__.py tests/monitoring/test_psi.py
git commit -m "feat(monitoring): MonitoringConfig + PSI 3-tier drift detector (C8)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Evidently drift cross-check

**Files:** Create `src/monitoring/evidently_drift.py`, `tests/monitoring/test_evidently_drift.py`.

- [ ] **Step 1: Write the failing test** — `tests/monitoring/test_evidently_drift.py`:

```python
import numpy as np
import pandas as pd

from src.monitoring.evidently_drift import evidently_drift_share

COLS = ["n_handover", "pingpong_count", "entropy_cell_seq"]


def test_no_drift_low_share():
    rng = np.random.RandomState(0)
    ref = pd.DataFrame({c: rng.gamma(2.0, 1.0, 400) for c in COLS})
    cur = pd.DataFrame({c: rng.gamma(2.0, 1.0, 400) for c in COLS})
    assert evidently_drift_share(ref, cur) <= 0.5


def test_drift_high_share():
    rng = np.random.RandomState(0)
    ref = pd.DataFrame({c: rng.gamma(2.0, 1.0, 400) for c in COLS})
    cur = pd.DataFrame({c: rng.gamma(2.0, 1.0, 400) * 4.0 for c in COLS})
    assert evidently_drift_share(ref, cur) >= 0.5
```

- [ ] **Step 2: Run** `pytest tests/monitoring/test_evidently_drift.py -v` → expect FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write `src/monitoring/evidently_drift.py`** (Evidently 0.7.x API — verified):

```python
"""Evidently drift cross-check (industry-standard validation of the custom PSI detector).

Returns the share of drifted columns from Evidently's DataDriftPreset. Used to
cross-validate PSI (answers the 'manual/cảm tính' critique); PSI remains the primary
measured detector. Evidently 0.7.x API: Report([DataDriftPreset]).run(current, reference).
"""
from __future__ import annotations

import pandas as pd


def evidently_drift_share(reference: pd.DataFrame, current: pd.DataFrame) -> float:
    """Fraction of columns Evidently flags as drifted (0.0–1.0)."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    snapshot = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
    for metric in snapshot.dict().get("metrics", []):
        value = metric.get("value")
        if isinstance(value, dict) and "share" in value:
            return float(value["share"])
    return 0.0
```

- [ ] **Step 4: Run** `pytest tests/monitoring/test_evidently_drift.py -v` (expect 2 passed; Evidently prints warnings — harmless) + `ruff check src/monitoring tests/monitoring` (clean).
  (If the installed Evidently exposes the drift share under a different key/shape, adapt the extraction to find the drift-share value from `snapshot.dict()` and NOTE it — do NOT weaken the asserts; the no-drift vs drift contrast MUST hold.)

- [ ] **Step 5: Commit**

```bash
git add src/monitoring/evidently_drift.py tests/monitoring/test_evidently_drift.py
git commit -m "feat(monitoring): Evidently drift-share cross-check (standard-tool validation)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: DriftDetector (reads predictions.jsonl) + fixtures

**Files:** Create `src/monitoring/detector.py`, `tests/monitoring/conftest.py`, `tests/monitoring/test_detector.py`.

- [ ] **Step 1: Write the shared fixtures** — `tests/monitoring/conftest.py`:

```python
"""Monitoring-test fixtures: a reference feature parquet + serving predictions.jsonl
(drift / no-drift) + a trained model. All under tmp_path."""
import json

import numpy as np
import pandas as pd
import pytest

from src.training.pipeline import run_training
from src.training.schema import FEATURE_COLUMNS, TrainingConfig

FEATURES = list(FEATURE_COLUMNS)


def _frame(rng, n, scale=1.0):
    df = pd.DataFrame({f: rng.gamma(2.0, 1.0, n) * scale for f in FEATURES})
    df["imsi"] = [str(i) for i in range(n)]
    df["window_start"] = pd.date_range("2024-06-26", periods=n, freq="5min", tz="UTC")
    return df


@pytest.fixture
def reference_parquet(tmp_path):
    rng = np.random.RandomState(0)
    df = _frame(rng, 400)
    df["label"] = 0
    anom = rng.choice(400, size=20, replace=False)
    df.loc[anom, FEATURES] *= 5.0
    df.loc[anom, "label"] = 1
    path = tmp_path / "reference.parquet"
    df.to_parquet(path, index=False)
    return path


def _write_predictions(path, frame):
    """Write a serving-style predictions.jsonl (each line has feature_values)."""
    with path.open("w", encoding="utf-8") as f:
        for _, row in frame.iterrows():
            rec = {"recorded_at": "2024-06-27T00:00:00Z", "event": "predict",
                   "is_anomaly": False, "anomaly_score": 0.1, "model_type": "iforest",
                   "model_version": "v1", "latency_ms": 1.0,
                   "feature_values": {f: float(row[f]) for f in FEATURES}}
            f.write(json.dumps(rec) + "\n")


@pytest.fixture
def predictions_no_drift(reference_parquet, tmp_path):
    # Resample rows from the reference itself -> SAME distribution -> guaranteed no drift
    # (avoids a false PSI signal from the reference's injected anomaly tail).
    ref = pd.read_parquet(reference_parquet)
    sample = ref.sample(n=200, replace=True, random_state=1)
    path = tmp_path / "pred_nodrift.jsonl"
    _write_predictions(path, sample)
    return path


@pytest.fixture
def predictions_drift(tmp_path):
    rng = np.random.RandomState(2)
    path = tmp_path / "pred_drift.jsonl"
    _write_predictions(path, _frame(rng, 200, scale=4.0))  # shifted -> drift
    return path


@pytest.fixture
def iforest_on_reference(reference_parquet, tmp_path):
    out = run_training(reference_parquet, model_type="iforest", backend="noop",
                       output_dir=tmp_path / "m",
                       cfg=TrainingConfig(use_labels_for_evaluation=True, label_column="label"))
    return out["model_path"]
```

- [ ] **Step 2: Write the failing test** — `tests/monitoring/test_detector.py`:

```python
from src.monitoring.detector import DriftDetector
from src.monitoring.schema import MonitoringConfig


def test_detects_no_drift(reference_parquet, predictions_no_drift):
    det = DriftDetector(MonitoringConfig(min_features_alert=3, min_observed_rows=50))
    result = det.detect(reference_parquet, predictions_no_drift)
    assert result["drift_detected"] is False
    assert "evidently_drift_share" in result and "psi" in result


def test_detects_drift(reference_parquet, predictions_drift):
    det = DriftDetector(MonitoringConfig(min_features_alert=3, min_observed_rows=50))
    result = det.detect(reference_parquet, predictions_drift)
    assert result["drift_detected"] is True
    # both detectors agree on a clear drift case
    assert result["psi"]["drift_detected"] is True
    assert result["evidently_drift_share"] >= 0.5


def test_load_observed_reads_feature_values(predictions_drift):
    det = DriftDetector(MonitoringConfig())
    obs = det.load_observed(predictions_drift)
    assert len(obs) == 200 and "n_handover" in obs.columns
```

- [ ] **Step 3: Run** `pytest tests/monitoring/test_detector.py -v` → expect FAIL (ModuleNotFoundError).

- [ ] **Step 4: Write `src/monitoring/detector.py`:**

```python
"""DriftDetector — covariate drift over serving predictions vs a training reference.

Combines PSI (primary measured detector) with Evidently (standard cross-check).
`drift_detected` follows PSI (the detector measured in Exp-4); the Evidently share is
reported alongside for validation. Observed features come from the serving
predictions.jsonl (each record carries feature_values).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.monitoring.evidently_drift import evidently_drift_share
from src.monitoring.psi import PSIDriftMonitor
from src.monitoring.schema import MonitoringConfig
from src.training.schema import FEATURE_COLUMNS

_FEATS = list(FEATURE_COLUMNS)


class DriftDetector:
    def __init__(self, config: MonitoringConfig | None = None) -> None:
        self.config = config or MonitoringConfig()

    def load_observed(self, predictions_path: str | Path) -> pd.DataFrame:
        rows: list[dict[str, float]] = []
        with Path(predictions_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                feats = rec.get("feature_values")
                if feats:
                    rows.append({c: float(feats[c]) for c in _FEATS if c in feats})
        return pd.DataFrame(rows, columns=_FEATS)

    def load_reference(self, reference_path: str | Path) -> pd.DataFrame:
        df = pd.read_parquet(reference_path)
        return df[[c for c in _FEATS if c in df.columns]].copy()

    def detect(self, reference_path: str | Path, predictions_path: str | Path) -> dict[str, Any]:
        reference = self.load_reference(reference_path)
        observed = self.load_observed(predictions_path)
        psi_result = PSIDriftMonitor(reference_frame=reference, config=self.config).evaluate(observed)
        share = (evidently_drift_share(reference, observed)
                 if not observed.empty and len(observed) >= self.config.min_observed_rows else 0.0)
        return {"drift_detected": bool(psi_result["drift_detected"]),
                "psi": psi_result, "evidently_drift_share": share,
                "observed_rows": int(len(observed))}
```

- [ ] **Step 5: Run** `pytest tests/monitoring/test_detector.py -v` (expect 3 passed) + `ruff check src/monitoring tests/monitoring` (clean).

- [ ] **Step 6: Commit**

```bash
git add src/monitoring/detector.py tests/monitoring/conftest.py tests/monitoring/test_detector.py
git commit -m "feat(monitoring): DriftDetector over predictions.jsonl (PSI + Evidently)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: AutoRetrainTrigger + run_retrain_cycle (the closed loop)

**Files:** Create `src/monitoring/retrain.py`, `tests/monitoring/test_retrain.py`.

- [ ] **Step 1: Write the failing test** — `tests/monitoring/test_retrain.py`:

```python
import pytest

from src.monitoring.retrain import AutoRetrainTrigger, run_retrain_cycle
from src.monitoring.schema import MonitoringConfig
from src.serving.runtime import ServingRuntime
from src.serving.schema import ServingConfig


def test_trigger_respects_cooldown():
    trig = AutoRetrainTrigger(cooldown_seconds=10_000.0)
    assert trig.should_retrain({"drift_detected": True}) is True
    trig.record_retrain()
    assert trig.should_retrain({"drift_detected": True}) is False  # within cooldown
    assert trig.should_retrain({"drift_detected": False}) is False


def test_no_drift_skips_retrain(reference_parquet, predictions_no_drift, tmp_path):
    out = run_retrain_cycle(
        predictions_path=predictions_no_drift, reference_path=reference_parquet,
        dataset_path=reference_parquet, output_dir=tmp_path / "rc",
        config=MonitoringConfig(min_features_alert=3, min_observed_rows=50))
    assert out["retrained"] is False
    assert out["drift"]["drift_detected"] is False


def test_drift_triggers_retrain_and_reload(reference_parquet, predictions_drift, tmp_path, monkeypatch):
    """C1 closed loop: drift -> retrain (mlflow registry) -> ServingRuntime.reload() serves new version."""
    import mlflow
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    # an initial registered model + a registry-backed runtime serving alias 'staging'
    from src.training.pipeline import run_training
    from src.training.schema import TrainingConfig
    run_training(reference_parquet, model_type="iforest", backend="mlflow",
                 output_dir=tmp_path / "m0",
                 cfg=TrainingConfig(use_labels_for_evaluation=True, label_column="label"))
    runtime = ServingRuntime.build(ServingConfig(loader="registry", model_type="iforest",
                                                 tracking_uri=uri, output_root=str(tmp_path / "s")))
    before = runtime.model.version
    out = run_retrain_cycle(
        predictions_path=predictions_drift, reference_path=reference_parquet,
        dataset_path=reference_parquet, output_dir=tmp_path / "rc",
        config=MonitoringConfig(min_features_alert=3, min_observed_rows=50, cooldown_seconds=0.0),
        model_type="iforest", backend="mlflow", runtime=runtime)
    assert out["retrained"] is True
    assert out["new_auc"] >= 0.0
    assert runtime.model.version != before  # reload() deployed the freshly retrained version


def test_eval_gate_blocks_bad_model(reference_parquet, predictions_drift, tmp_path):
    """A retrain whose ROC-AUC is below the gate is NOT deployed."""
    out = run_retrain_cycle(
        predictions_path=predictions_drift, reference_path=reference_parquet,
        dataset_path=reference_parquet, output_dir=tmp_path / "rc",
        config=MonitoringConfig(min_features_alert=3, min_observed_rows=50,
                                cooldown_seconds=0.0, retrain_min_auc=1.1),  # impossible gate
        model_type="iforest", backend="noop")
    assert out["retrained"] is False and out["gate_failed"] is True
```

- [ ] **Step 2: Run** `pytest tests/monitoring/test_retrain.py -v` → expect FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write `src/monitoring/retrain.py`:**

```python
"""Auto-retrain closed loop: detect drift -> (cooldown) -> retrain on a sliding window
-> eval gate -> ServingRuntime.reload(). Returns whether a retrain was deployed so the
P8 Exp-4 harness can compare loop ON (C1) vs OFF (C0).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.monitoring.detector import DriftDetector
from src.monitoring.schema import MonitoringConfig
from src.serving.records import append_jsonl, utc_now_iso
from src.training.data import apply_sliding_window
from src.training.pipeline import run_training
from src.training.schema import TrainingConfig


@dataclass
class AutoRetrainTrigger:
    cooldown_seconds: float = 3600.0
    _last_retrain_time: float = 0.0
    _total_retrain_count: int = 0

    def should_retrain(self, drift_result: dict[str, Any]) -> bool:
        if not drift_result.get("drift_detected", False):
            return False
        return (time.time() - self._last_retrain_time) >= self.cooldown_seconds

    def record_retrain(self) -> None:
        self._last_retrain_time = time.time()
        self._total_retrain_count += 1


def run_retrain_cycle(*, predictions_path: Path, reference_path: Path, dataset_path: Path,
                      output_dir: Path, config: MonitoringConfig | None = None,
                      model_type: str = "iforest", backend: str = "noop",
                      runtime: Any = None,
                      trigger: AutoRetrainTrigger | None = None) -> dict[str, Any]:
    # NOTE: when backend="mlflow", the caller sets MLFLOW_TRACKING_URI in the env (the
    # standard MLflow pattern) so the retrained model registers where the serving
    # runtime reads from. run_training picks up that global URI.
    config = config or MonitoringConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trigger = trigger or AutoRetrainTrigger(cooldown_seconds=config.cooldown_seconds)

    # 1. detect drift over the serving predictions
    drift = DriftDetector(config).detect(reference_path, predictions_path)
    append_jsonl(output_dir / "drift_history.jsonl",
                 {"recorded_at": utc_now_iso(), "event": "drift_check", **_drift_summary(drift)})
    if not trigger.should_retrain(drift):
        return {"retrained": False, "drift": drift}

    # 2. retrain on a sliding window of the dataset (MTLF strategy)
    df = pd.read_parquet(dataset_path)
    windowed = apply_sliding_window(df, window_days=config.window_days)
    window_path = output_dir / "retrain_window.parquet"
    windowed.to_parquet(window_path, index=False)
    result = run_training(window_path, model_type=model_type, backend=backend,
                          output_dir=output_dir / "model",
                          cfg=TrainingConfig(use_labels_for_evaluation=True, label_column="label"))
    new_auc = float(result["metrics"].get("roc_auc", 0.0))

    # 3. eval gate — do not deploy a model below the bar
    if new_auc < config.retrain_min_auc:
        append_jsonl(output_dir / "retrain_history.jsonl",
                     {"recorded_at": utc_now_iso(), "event": "retrain_rejected",
                      "new_auc": new_auc, "gate": config.retrain_min_auc})
        return {"retrained": False, "gate_failed": True, "new_auc": new_auc, "drift": drift}

    # 4. deploy — reload serving (registry-alias loaders pick up the new staging version)
    trigger.record_retrain()
    if runtime is not None:
        runtime.reload()
    append_jsonl(output_dir / "retrain_history.jsonl",
                 {"recorded_at": utc_now_iso(), "event": "retrain_deployed",
                  "new_auc": new_auc, "model_version": result.get("model_version"),
                  "model_path": result.get("model_path")})
    return {"retrained": True, "new_auc": new_auc, "model_version": result.get("model_version"),
            "model_path": result.get("model_path"), "drift": drift}


def _drift_summary(drift: dict[str, Any]) -> dict[str, Any]:
    return {"drift_detected": drift["drift_detected"],
            "evidently_drift_share": drift["evidently_drift_share"],
            "alerted_features": drift["psi"].get("alerted_features"),
            "observed_rows": drift["observed_rows"]}
```

- [ ] **Step 4: Run** `pytest tests/monitoring/test_retrain.py -v` (expect 4 passed) + `ruff check src/monitoring tests/monitoring` (clean). (The drift+reload test trains two mlflow models — a few seconds — that is expected, not a hang.)

- [ ] **Step 5: Commit**

```bash
git add src/monitoring/retrain.py tests/monitoring/test_retrain.py
git commit -m "feat(monitoring): AutoRetrainTrigger + drift->retrain->reload closed loop (RQ4)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Prometheus /metrics endpoint (minimal, scrape-ready)

**Files:** Create `src/monitoring/metrics.py`, `tests/monitoring/test_metrics.py`. Modify `src/serving/cli.py` (one-line wiring).

- [ ] **Step 1: Write the failing test** — `tests/monitoring/test_metrics.py`:

```python
from fastapi.testclient import TestClient

from src.monitoring.metrics import add_metrics
from src.serving.app import create_app
from src.serving.runtime import ServingRuntime
from src.serving.schema import ServingConfig


def test_metrics_endpoint_exposes_counter(iforest_on_reference, tmp_path):
    cfg = ServingConfig(loader="path", model_type="iforest",
                        model_path=str(iforest_on_reference), output_root=str(tmp_path / "s"))
    app = add_metrics(create_app(ServingRuntime.build(cfg)))
    client = TestClient(app)
    feats = {c: 1.0 for c in ["n_handover", "n_unique_cells", "pingpong_count",
                              "pingpong_rate", "mean_inter_ho_s", "std_inter_ho_s", "entropy_cell_seq"]}
    client.post("/predict", json={"features": feats})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "nwdaf_predict_requests_total" in r.text
    assert "nwdaf_predict_latency_ms" in r.text
```

- [ ] **Step 2: Run** `pytest tests/monitoring/test_metrics.py -v` → expect FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write `src/monitoring/metrics.py`:**

```python
"""Minimal Prometheus instrumentation for the serving app (C8 ops half).

Adds a scrape-ready /metrics endpoint exposing request count + latency. A real
Prometheus server + Grafana that scrape this are P7 (observability infra). We use a
private CollectorRegistry per app so tests don't pollute the global default.
"""
from __future__ import annotations

from time import perf_counter

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest


def add_metrics(app: FastAPI) -> FastAPI:
    registry = CollectorRegistry()
    req_count = Counter("nwdaf_predict_requests_total", "Total /predict requests", registry=registry)
    latency = Histogram("nwdaf_predict_latency_ms", "Predict latency (ms)", registry=registry)

    @app.middleware("http")
    async def _instrument(request, call_next):
        t0 = perf_counter()
        response = await call_next(request)
        if request.url.path == "/predict" and request.method == "POST":
            req_count.inc()
            latency.observe((perf_counter() - t0) * 1000.0)
        return response

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app
```

- [ ] **Step 4: Wire it into the served app** — in `src/serving/cli.py`, inside `_cmd_serve`, after `app = create_app(...)` add metrics instrumentation. Change:

```python
def _cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from src.monitoring.metrics import add_metrics
    from src.serving.app import create_app
    from src.serving.runtime import ServingRuntime
    config = build_config_from_args(args)
    app = add_metrics(create_app(ServingRuntime.build(config)))
    uvicorn.run(app, host=config.host, port=config.port)
```

(Only the two lines — the `add_metrics` import and wrapping `create_app(...)` — change; everything else in `_cmd_serve` stays. This keeps `/predict` itself untouched.)

- [ ] **Step 5: Run** `pytest tests/monitoring/test_metrics.py -v` (expect 1 passed) + `pytest tests/serving/test_cli.py -v` (still 3 passed — the wiring didn't break CLI parsing) + `ruff check src/monitoring src/serving tests/monitoring` (clean).

- [ ] **Step 6: Commit**

```bash
git add src/monitoring/metrics.py tests/monitoring/test_metrics.py src/serving/cli.py
git commit -m "feat(monitoring): scrape-ready Prometheus /metrics on serving app (C8 ops)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: CLI (`check`)

**Files:** Create `src/monitoring/cli.py`, `src/monitoring/__main__.py`, `tests/monitoring/test_cli.py`.

- [ ] **Step 1: Write the failing test** — `tests/monitoring/test_cli.py`:

```python
from src.monitoring.cli import build_parser, cmd_check


def test_parser_has_check_subcommand():
    args = build_parser().parse_args(["check", "--reference", "r.parquet", "--predictions", "p.jsonl"])
    assert args.command == "check" and args.reference == "r.parquet"


def test_cmd_check_reports_drift(reference_parquet, predictions_drift, capsys):
    import json
    args = build_parser().parse_args(
        ["check", "--reference", str(reference_parquet), "--predictions", str(predictions_drift),
         "--min-features-alert", "3"])
    cmd_check(args)
    data = json.loads(capsys.readouterr().out)
    assert data["drift_detected"] is True
```

- [ ] **Step 2: Run** `pytest tests/monitoring/test_cli.py -v` → expect FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write `src/monitoring/cli.py`** (argparse idiom matching `src/serving/cli.py`):

```python
"""Monitoring CLI: `python -m src.monitoring check ...` runs a one-shot drift check
over a serving predictions.jsonl against a training reference parquet."""
from __future__ import annotations

import argparse
import json

from src.monitoring.detector import DriftDetector
from src.monitoring.schema import MonitoringConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.monitoring")
    sub = parser.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("check", help="One-shot drift check (predictions vs reference)")
    sp.add_argument("--reference", required=True, help="Training reference feature parquet")
    sp.add_argument("--predictions", required=True, help="Serving predictions.jsonl")
    sp.add_argument("--min-features-alert", type=int, default=MonitoringConfig.min_features_alert)
    sp.add_argument("--min-observed-rows", type=int, default=MonitoringConfig.min_observed_rows)
    sp.set_defaults(func=cmd_check)
    return parser


def cmd_check(args: argparse.Namespace) -> None:
    config = MonitoringConfig(min_features_alert=args.min_features_alert,
                              min_observed_rows=args.min_observed_rows)
    result = DriftDetector(config).detect(args.reference, args.predictions)
    summary = {"drift_detected": result["drift_detected"],
               "evidently_drift_share": result["evidently_drift_share"],
               "alerted_features": result["psi"].get("alerted_features"),
               "observed_rows": result["observed_rows"]}
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write `src/monitoring/__main__.py`:**

```python
from src.monitoring.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run** `pytest tests/monitoring/test_cli.py -v` (expect 2 passed) + `python -m src.monitoring check --help` (prints usage) + `ruff check src/monitoring tests/monitoring` (clean).

- [ ] **Step 6: Commit**

```bash
git add src/monitoring/cli.py src/monitoring/__main__.py tests/monitoring/test_cli.py
git commit -m "feat(monitoring): CLI check (one-shot drift report)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Verify (DoD)

**Files:** none (verification).

- [ ] **Step 1:** Full suite + lint.
Run: `pytest -q` → Expected: all green (Phase 0–5 + ~16 new monitoring tests).
Run: `ruff check .` → Expected: All checks passed.

- [ ] **Step 2: DoD checklist** — verify and report PASS/FAIL with one line of evidence each:
  1. **PSI 3-tier drift** detects covariate drift, no false alarm on stable data (`test_psi.py`).
  2. **Evidently cross-check** agrees on a clear drift case (`test_evidently_drift.py`, `test_detector.py::test_detects_drift`).
  3. **DriftDetector reads the serving `predictions.jsonl`** substrate (`test_detector.py::test_load_observed_reads_feature_values`).
  4. **drift → auto-retrain → reload closed loop** works end-to-end; reload deploys the new version (`test_retrain.py::test_drift_triggers_retrain_and_reload`).
  5. **Eval gate** blocks a sub-threshold model (`test_retrain.py::test_eval_gate_blocks_bad_model`).
  6. **Sliding-window retrain** wired (`apply_sliding_window` used in `run_retrain_cycle` — closes P4 carry-forward §9 #8).
  7. **Prometheus `/metrics`** scrape-ready on serving (`test_metrics.py`).
  8. **C1-capability framing:** `run_retrain_cycle` returns `retrained` so Exp-4 can run loop ON (C1) vs OFF (C0). All tests green; ruff clean.

> **Carry to P7/P8:** P7 stands up the real Prometheus server + Grafana + dashboards scraping `/metrics` (HANDOFF §9 #12). P8 Exp-4 drives `run_retrain_cycle` over the P2 sudden/gradual/recurring drift scenarios and measures detection latency / drift recall-FPR / recovery time, contrasting loop ON (C1) vs OFF (C0).
