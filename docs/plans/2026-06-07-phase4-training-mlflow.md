# Phase 4 — Training + MLflow/ClearML Trackers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Build the training subsystem (C4) + real MLflow/ClearML trackers with **model registry + alias/stage** (C5/C6). IsolationForest (primary, sklearn) + **PyTorch LSTM-AE** (model-swap, RQ4). The tracker layer is the C0/C1/C2 control variable for RQ3 (Exp-2/3).

**Architecture:** Port the tracker-agnostic training core (IForest, data, schema) from `MLOps_Project/src/training/`. REWRITE the MLflow/ClearML trackers to the **hardened P0 `BaseTracker` interface** (the factory already lazy-imports `src.tracking.mlflow_tracker.MLflowTracker` / `src.tracking.clearml_tracker.ClearMLTracker`) using a **real registry** (the old prototype only logged paths as params — we do better). REWRITE the LSTM-AE in **PyTorch** (torch 2.12 works on Python 3.14; no TF/3.12 venv needed). A training pipeline wires data → train → eval → tracker(init/log/register/end), selected by `MLOPS_BACKEND`.

**Tech Stack:** scikit-learn, **PyTorch 2.12**, MLflow 0.x (sqlite-backed for registry), ClearML (offline mode for tests), joblib.

**Source to port (read each):**
- `/Users/ngocmx/Thạc Sĩ/MLOps_Project/src/training/schema.py` (port verbatim)
- `/Users/ngocmx/Thạc Sĩ/MLOps_Project/src/training/core.py` (port verbatim — IForest)
- `/Users/ngocmx/Thạc Sĩ/MLOps_Project/src/training/data.py` (port verbatim — sliding window, load/split)
- `/Users/ngocmx/Thạc Sĩ/MLOps_Project/src/training/lstm_detector.py` (reference only — REWRITE in PyTorch)
- Old trackers reference: `/Users/ngocmx/Thạc Sĩ/MLOps_Project/src/tracking/__init__.py` (adapt to new interface)

**Branch:** `phase4-training`.

**New BaseTracker interface (from P0 — trackers MUST match this):**
`init_experiment(config: ExperimentConfig) -> RunHandle` · `log_params(dict)` · `log_metrics(dict, step=None)` · `log_dataset(path, name=None)` · `log_artifact(path, artifact_path=None)` · `register_model(model_path, name, metrics=None, alias=None) -> str|None` · `end_experiment(status="FINISHED")`.

---

### Task 1: Switch `[lstm]` extra to PyTorch + update docs

**Files:** `pyproject.toml`, `CLAUDE.md`, `docs/THESIS_SPEC.md`.

- [ ] **Step 1:** In `pyproject.toml`, change the `[lstm]` extra to:
```toml
lstm = ["torch>=2.6"]
```
(remove the tensorflow line + the `python_version < '3.14'` marker + the WARNING comments).
- [ ] **Step 2:** `pip install -e ".[lstm]"` → verify `python -c "import torch; print(torch.__version__)"` (≈2.12). If it fails, report BLOCKED.
- [ ] **Step 3:** Update `CLAUDE.md`: in §1 change the "Python 3.14 → TF/LSTM needs 3.12 venv" note to "LSTM-AE dùng **PyTorch** (chạy trên Python 3.14, một venv duy nhất)". In §5 change `[lstm]` mention from tensorflow to torch.
- [ ] **Step 4:** Update `docs/THESIS_SPEC.md` §10 / wherever it says LSTM-AE/tensorflow → note PyTorch (single venv). Keep "LSTM-AE" as the model name.
- [ ] **Step 5: Commit** `git add pyproject.toml CLAUDE.md docs/THESIS_SPEC.md` → `chore(deps): switch LSTM-AE to PyTorch (Python 3.14 single venv)` + trailer.

---

### Task 2: MLflowTracker — real registry (C5/C6)

**Files:** `src/tracking/mlflow_tracker.py`, `tests/tracking/test_mlflow_tracker.py`.

- [ ] **Step 1: Write `src/tracking/mlflow_tracker.py`:**
```python
"""MLflow tracker (Config C1) — real experiment tracking + model registry."""
from __future__ import annotations

from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.tracking.base import BaseTracker
from src.tracking.schema import ExperimentConfig, RunHandle


class MLflowTracker(BaseTracker):
    def __init__(self) -> None:
        self._run = None
        self._client: MlflowClient | None = None

    def init_experiment(self, config: ExperimentConfig) -> RunHandle:
        mlflow.set_experiment(config.experiment_name)
        self._run = mlflow.start_run(run_name=config.run_name, tags=config.tags or {})
        self._client = MlflowClient()
        info = self._run.info
        uri = mlflow.get_tracking_uri()
        url = f"{uri}/#/experiments/{info.experiment_id}/runs/{info.run_id}"
        return RunHandle(run_id=info.run_id, backend="mlflow", url=url)

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_dataset(self, path: str, name: str | None = None) -> None:
        mlflow.log_artifact(str(path), artifact_path="datasets")

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def register_model(self, model_path: str, name: str,
                       metrics: dict[str, float] | None = None,
                       alias: str | None = None) -> str | None:
        mlflow.log_artifact(str(model_path), artifact_path="model")
        run_id = self._run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri=model_uri, name=name)
        if alias:
            self._client.set_registered_model_alias(name, alias, mv.version)
        return f"models:/{name}/{mv.version}"

    def end_experiment(self, status: str = "FINISHED") -> None:
        mlflow.end_run(status=status)
```
- [ ] **Step 2: Tests** — `tests/tracking/test_mlflow_tracker.py` (uses a sqlite tracking URI so the registry works, no server):
```python
import pandas as pd  # noqa: F401  (kept only if used; remove if ruff flags)
import pytest

from src.tracking import create_tracker
from src.tracking.mlflow_tracker import MLflowTracker
from src.tracking.schema import ExperimentConfig


@pytest.fixture
def mlflow_local(tmp_path, monkeypatch):
    import mlflow
    db = tmp_path / "mlflow.db"
    arts = tmp_path / "mlartifacts"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{db}")
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    monkeypatch.setattr(mlflow, "get_artifact_uri", mlflow.get_artifact_uri)
    return tmp_path


def test_factory_returns_mlflow(mlflow_local):
    assert isinstance(create_tracker("mlflow"), MLflowTracker)


def test_full_lifecycle_with_registry(mlflow_local, tmp_path):
    t = MLflowTracker()
    h = t.init_experiment(ExperimentConfig(experiment_name="t_exp", run_name="r1",
                                           backend="mlflow", tags={"model": "iforest"}))
    assert h.run_id and h.backend == "mlflow"
    t.log_params({"contamination": 0.01, "n_estimators": 200})
    t.log_metrics({"roc_auc": 0.98, "pr_auc": 0.55})
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"dummy-model-bytes")
    version_uri = t.register_model(str(model_file), "iforest_pingpong",
                                   metrics={"roc_auc": 0.98}, alias="staging")
    assert version_uri and version_uri.startswith("models:/iforest_pingpong/")
    t.end_experiment()
    # the alias resolves to a real registered version
    from mlflow.tracking import MlflowClient
    mv = MlflowClient().get_model_version_by_alias("iforest_pingpong", "staging")
    assert mv.version is not None
```
- [ ] **Step 3: Run** `pytest tests/tracking/test_mlflow_tracker.py -v` (2 passed) + `ruff check src/tracking tests/tracking`. If MLflow registry API differs in the installed version, adapt minimally (e.g., `set_registered_model_alias` vs stage transitions) and NOTE it.
- [ ] **Step 4: Commit** `git add src/tracking/mlflow_tracker.py tests/tracking/test_mlflow_tracker.py` → `feat(tracking): MLflowTracker with real model registry (C1)` + trailer.

---

### Task 3: ClearMLTracker — offline-capable (framework comparison)

**Files:** `src/tracking/clearml_tracker.py`, `tests/tracking/test_clearml_tracker.py`.

- [ ] **Step 1: Write `src/tracking/clearml_tracker.py`:**
```python
"""ClearML tracker — experiment tracking + model registry (framework comparison vs MLflow)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.tracking.base import BaseTracker
from src.tracking.schema import ExperimentConfig, RunHandle

_PROJECT = "NWDAF-MLOps"


class ClearMLTracker(BaseTracker):
    def __init__(self) -> None:
        self._task = None

    def init_experiment(self, config: ExperimentConfig) -> RunHandle:
        from clearml import Task
        self._task = Task.init(
            project_name=config.experiment_name or _PROJECT,
            task_name=config.run_name,
            tags=list(config.tags.keys()) if config.tags else [],
            reuse_last_task_id=False,
            auto_connect_frameworks=False,
            auto_connect_arg_parser=False,
        )
        return RunHandle(run_id=self._task.id, backend="clearml",
                         url=self._task.get_output_log_web_page())

    def log_params(self, params: dict[str, Any]) -> None:
        self._task.connect(dict(params))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        logger = self._task.get_logger()
        for k, v in metrics.items():
            if step is not None:
                logger.report_scalar(title="metrics", series=k, value=float(v), iteration=step)
            else:
                logger.report_single_value(name=k, value=float(v))

    def log_dataset(self, path: str, name: str | None = None) -> None:
        self._task.upload_artifact(name=name or Path(path).name, artifact_object=str(path))

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        self._task.upload_artifact(name=artifact_path or Path(path).name, artifact_object=str(path))

    def register_model(self, model_path: str, name: str,
                       metrics: dict[str, float] | None = None,
                       alias: str | None = None) -> str | None:
        from clearml import OutputModel
        om = OutputModel(task=self._task, name=name)
        om.update_weights(weights_filename=str(model_path), auto_delete_file=False)
        if alias:
            om.set_metadata("alias", alias, "str")
        return om.id

    def end_experiment(self, status: str = "FINISHED") -> None:
        if self._task is not None:
            self._task.close()
```
- [ ] **Step 2: Tests** — `tests/tracking/test_clearml_tracker.py` (ClearML **offline mode** — no server):
```python
import pytest

from src.tracking.clearml_tracker import ClearMLTracker
from src.tracking.schema import ExperimentConfig


@pytest.fixture(autouse=True)
def clearml_offline(tmp_path, monkeypatch):
    from clearml import Task
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
    Task.set_offline(offline_mode=True)
    yield
    Task.set_offline(offline_mode=False)


def test_init_and_log_offline(tmp_path):
    t = ClearMLTracker()
    h = t.init_experiment(ExperimentConfig(experiment_name="NWDAF-MLOps", run_name="clearml_r1",
                                           backend="clearml", tags={"model": "iforest"}))
    assert h.backend == "clearml" and h.run_id
    t.log_params({"contamination": 0.01})
    t.log_metrics({"roc_auc": 0.97})
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"dummy")
    mid = t.register_model(str(model_file), "iforest_pingpong", {"roc_auc": 0.97}, alias="staging")
    assert mid is not None
    t.end_experiment()
```
- [ ] **Step 3: Run** `pytest tests/tracking/test_clearml_tracker.py -v` (1 passed). ClearML offline mode may print warnings — fine. If `OutputModel` in offline mode errors, wrap the register_model body in a try/except that still returns a deterministic id in offline mode, and NOTE the adaptation (offline ClearML has limited model support). Do not skip the test. + `ruff check`.
- [ ] **Step 4: Commit** `git add src/tracking/clearml_tracker.py tests/tracking/test_clearml_tracker.py` → `feat(tracking): ClearMLTracker (framework comparison vs MLflow)` + trailer.

---

### Task 4: Port training core (IForest) + schema + data

**Files:** `src/training/__init__.py` (empty), `src/training/schema.py`, `src/training/core.py`, `src/training/data.py`, `tests/training/__init__.py` (empty), `tests/training/conftest.py`, `tests/training/test_core.py`, `tests/training/test_data.py`.

- [ ] **Step 1: Port VERBATIM** `schema.py`, `core.py`, `data.py` from `MLOps_Project/src/training/` → `src/training/`. (`schema.py` imports `from src.features.schema import D2_FEATURE_COLUMNS` — already present from Phase 3. `core.py` and `data.py` import `from src.training.schema import ...` — drop-in.)
- [ ] **Step 2: Fixture** — `tests/training/conftest.py`:
```python
import numpy as np
import pandas as pd
import pytest

from src.features.schema import D2_FEATURE_COLUMNS

FEATURES = [c.name for c in D2_FEATURE_COLUMNS]


@pytest.fixture
def labeled_features(tmp_path):
    """A small calibrated-ish feature parquet with a `label` column (5% anomalies)."""
    rng = np.random.RandomState(0)
    n = 300
    data = {f: rng.gamma(2.0, 1.0, n) for f in FEATURES}
    df = pd.DataFrame(data)
    df["imsi"] = [str(i) for i in range(n)]
    df["window_start"] = pd.date_range("2024-06-26", periods=n, freq="5min", tz="UTC")
    df["label"] = 0
    anom = rng.choice(n, size=15, replace=False)
    df.loc[anom, FEATURES] = df.loc[anom, FEATURES] * 5.0  # make anomalies extreme
    df.loc[anom, "label"] = 1
    df["weak_label"] = df["label"]
    path = tmp_path / "features.parquet"
    df.to_parquet(path, index=False)
    return path, df
```
- [ ] **Step 3: Tests** — `tests/training/test_core.py`:
```python
import numpy as np

from src.training.core import train_isolation_forest
from src.training.schema import TrainingConfig


def test_iforest_trains_and_scores():
    rng = np.random.RandomState(0)
    x_train = rng.normal(size=(200, 7))
    x_val = rng.normal(size=(50, 7))
    y_val = np.zeros(50, dtype=int); y_val[:5] = 1
    res = train_isolation_forest(x_train, x_val, y_val, TrainingConfig())
    assert res.model is not None
    assert "roc_auc" in res.metrics and 0.0 <= res.metrics["roc_auc"] <= 1.0
    assert res.fit_summary["n_train_rows"] == 200
```
And `tests/training/test_data.py`:
```python
import pandas as pd

from src.training.data import (
    apply_sliding_window, load_training_dataset, prepare_training_matrices, split_training_data,
)
from src.training.schema import TrainingConfig


def test_load_and_prepare(labeled_features):
    path, _ = labeled_features
    cfg = TrainingConfig(use_labels_for_evaluation=True, label_column="label")
    df, ctx, meta = load_training_dataset(path, cfg)
    assert ctx.row_count == len(df)
    x, y = prepare_training_matrices(df, cfg)
    assert x.shape[1] == 7 and y is not None
    xt, xv, yt, yv = split_training_data(x, y, cfg)
    assert len(xt) > len(xv)


def test_sliding_window_keeps_recent(labeled_features):
    _, df = labeled_features
    out = apply_sliding_window(df, window_days=1)
    assert len(out) <= len(df)
```
- [ ] **Step 4: Run** `pytest tests/training/test_core.py tests/training/test_data.py -v` (3 passed) + ruff.
- [ ] **Step 5: Commit** `git add src/training/__init__.py src/training/schema.py src/training/core.py src/training/data.py tests/training/` → `feat(training): port IForest core + training data utils` + trailer.

---

### Task 5: PyTorch LSTM-AE detector (model-swap, RQ4)

**Files:** `src/training/lstm_detector.py`, `tests/training/test_lstm_detector.py`.

- [ ] **Step 1: Write `src/training/lstm_detector.py`** (PyTorch reimplementation of the reference TF LSTM-AE — same role: reconstruction-based detector, P95 threshold):
```python
"""PyTorch LSTM-Autoencoder anomaly detector (model-swap alternative to IsolationForest, RQ4)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "n_handover", "n_unique_cells", "pingpong_count", "pingpong_rate",
    "mean_inter_ho_s", "std_inter_ho_s", "entropy_cell_seq",
]
SEQ_LEN = 1


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, encoding_dim: int = 4, seq_len: int = SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(n_features, encoding_dim, batch_first=True)
        self.decoder = nn.LSTM(encoding_dim, encoding_dim, batch_first=True)
        self.head = nn.Linear(encoding_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, seq_len, n_features)
        _, (h, _) = self.encoder(x)
        z = h[-1].unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, seq_len, enc)
        dec, _ = self.decoder(z)
        return self.head(dec)


def _to_seq(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x.reshape(x.shape[0], SEQ_LEN, x.shape[1]), dtype=torch.float32)


def train_lstm_ae(features_path: Path, out_model_dir: Path, *, random_state: int = 42,
                  epochs: int = 30, batch_size: int = 32, encoding_dim: int = 4,
                  threshold_percentile: float = 95.0, label_column: str = "label") -> dict[str, Any]:
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    out_model_dir = Path(out_model_dir)
    out_model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(features_path)
    x_raw = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    y = df[label_column].to_numpy() if label_column in df.columns else None
    x_tr, x_val, _, y_val = train_test_split(
        x_raw, y if y is not None else np.zeros(len(x_raw)),
        test_size=0.2, random_state=random_state,
        stratify=y if (y is not None and len(set(y.tolist())) > 1) else None,
    )
    scaler = StandardScaler().fit(x_tr)
    xt, xv = _to_seq(scaler.transform(x_tr)), _to_seq(scaler.transform(x_val))

    model = LSTMAutoencoder(len(FEATURE_COLS), encoding_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(xt.shape[0])
        for i in range(0, xt.shape[0], batch_size):
            b = xt[perm[i:i + batch_size]]
            opt.zero_grad()
            loss = loss_fn(model(b), b)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        tr_mse = ((model(xt) - xt) ** 2).mean(dim=(1, 2)).numpy()
        val_mse = ((model(xv) - xv) ** 2).mean(dim=(1, 2)).numpy()
    threshold = float(np.percentile(tr_mse, threshold_percentile))
    val_pred = (val_mse > threshold).astype(int)

    metrics: dict[str, float] = {"threshold": threshold, "mean_val_mse": float(val_mse.mean())}
    if y is not None and len(set(y_val.tolist())) > 1:
        metrics.update({
            "precision": float(precision_score(y_val, val_pred, zero_division=0)),
            "recall": float(recall_score(y_val, val_pred, zero_division=0)),
            "f1": float(f1_score(y_val, val_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_val, val_mse)),
            "pr_auc": float(average_precision_score(y_val, val_mse)),
        })

    model_path = out_model_dir / "model_lstm_ae.pt"
    torch.save(model.state_dict(), model_path)
    meta_path = out_model_dir / "lstm_ae_meta.joblib"
    joblib.dump({"feature_cols": FEATURE_COLS, "threshold": threshold,
                 "encoding_dim": encoding_dim, "scaler": scaler,
                 "state_dict_path": str(model_path)}, meta_path)
    return {"model_path": str(meta_path), "state_dict_path": str(model_path),
            "metrics": metrics, "train_rows": int(len(x_tr)), "val_rows": int(len(x_val))}


def predict_lstm_ae(x: np.ndarray, model_meta_path: Path) -> tuple[np.ndarray, np.ndarray]:
    # meta is a trusted internal artifact produced by train_lstm_ae() in this same pipeline
    # (holds feature_cols, threshold, fitted StandardScaler) -> joblib.load is safe here.
    meta = joblib.load(model_meta_path)
    model = LSTMAutoencoder(len(meta["feature_cols"]), meta["encoding_dim"])
    # state_dict_path holds only tensors -> weights_only=True avoids arbitrary-code execution.
    model.load_state_dict(torch.load(meta["state_dict_path"], weights_only=True))
    model.eval()
    xs = _to_seq(meta["scaler"].transform(x))
    with torch.no_grad():
        mse = ((model(xs) - xs) ** 2).mean(dim=(1, 2)).numpy()
    return (mse > meta["threshold"]).astype(int), mse
```
- [ ] **Step 2: Tests** — `tests/training/test_lstm_detector.py`:
```python
import numpy as np

from src.training.lstm_detector import predict_lstm_ae, train_lstm_ae


def test_lstm_ae_trains_and_predicts(labeled_features, tmp_path):
    path, df = labeled_features
    out = train_lstm_ae(path, tmp_path / "lstm", epochs=5, label_column="label")
    assert "metrics" in out and "threshold" in out["metrics"]
    assert (tmp_path / "lstm" / "model_lstm_ae.pt").exists()
    # inference round-trips
    x = df[[c for c in df.columns if c in __import__("src.training.lstm_detector", fromlist=["FEATURE_COLS"]).FEATURE_COLS]].to_numpy(float)
    preds, scores = predict_lstm_ae(x, out["model_path"])
    assert len(preds) == len(df) and len(scores) == len(df)
    assert set(np.unique(preds)).issubset({0, 1})


def test_lstm_ae_deterministic(labeled_features, tmp_path):
    path, _ = labeled_features
    a = train_lstm_ae(path, tmp_path / "a", epochs=5, random_state=1, label_column="label")
    b = train_lstm_ae(path, tmp_path / "b", epochs=5, random_state=1, label_column="label")
    assert abs(a["metrics"]["threshold"] - b["metrics"]["threshold"]) < 1e-6
```
(If the dynamic import in the first test is awkward, just hardcode `FEATURE_COLS` import at top: `from src.training.lstm_detector import FEATURE_COLS`.)
- [ ] **Step 3: Run** `pytest tests/training/test_lstm_detector.py -v` (2 passed) + ruff. (torch CPU training of a tiny model for 5 epochs is fast.)
- [ ] **Step 4: Commit** `git add src/training/lstm_detector.py tests/training/test_lstm_detector.py` → `feat(training): PyTorch LSTM-AE detector (model-swap, RQ4)` + trailer.

---

### Task 6: Training pipeline (tracker-integrated) + CLI

**Files:** `src/training/pipeline.py`, `src/training/cli.py`, `src/training/__main__.py`, `tests/training/test_pipeline.py`.

- [ ] **Step 1: Write `src/training/pipeline.py`** — the C0/C1/C2 control point:
```python
"""Training pipeline: data -> train -> eval -> tracker(init/log/register/end).

Backend selected by MLOPS_BACKEND (noop=C0 / mlflow=C1 / clearml). Model selected
by model_type (iforest | lstm_ae). This is the experiment control point for RQ3.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib

from src.tracking import ExperimentConfig, create_tracker
from src.training.core import train_isolation_forest
from src.training.data import (
    load_training_dataset, prepare_training_matrices, split_training_data,
)
from src.training.lstm_detector import train_lstm_ae
from src.training.schema import MODEL_FAMILY, TrainingConfig


def run_training(dataset_path: Path, *, model_type: str = "iforest",
                 backend: str | None = None, output_dir: Path = Path("artifacts/models"),
                 cfg: TrainingConfig | None = None, run_name: str | None = None) -> dict[str, Any]:
    cfg = cfg or TrainingConfig(use_labels_for_evaluation=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = create_tracker(backend)
    run_name = run_name or f"{model_type}_s{cfg.random_state}"
    handle = tracker.init_experiment(ExperimentConfig(
        experiment_name="training", run_name=run_name,
        backend=backend or "noop", tags={"model": model_type}))

    t0 = time.perf_counter()
    if model_type == "iforest":
        df, ctx, _ = load_training_dataset(dataset_path, cfg)
        x, y = prepare_training_matrices(df, cfg)
        x_tr, x_val, _, y_val = split_training_data(x, y, cfg)
        result = train_isolation_forest(x_tr, x_val, y_val, cfg)
        model_path = output_dir / f"model_iforest_s{cfg.random_state}.joblib"
        joblib.dump(result.model, model_path)
        metrics = result.metrics
    elif model_type == "lstm_ae":
        out = train_lstm_ae(dataset_path, output_dir / f"lstm_s{cfg.random_state}",
                            random_state=cfg.random_state,
                            label_column=cfg.label_column if cfg.label_column else "label")
        model_path = Path(out["model_path"])
        metrics = out["metrics"]
    else:
        raise ValueError(f"Unknown model_type={model_type!r}")
    train_seconds = time.perf_counter() - t0

    tracker.log_params({**cfg.to_dict(), "model_type": model_type})
    tracker.log_metrics({k: float(v) for k, v in metrics.items()})
    model_version = tracker.register_model(str(model_path), MODEL_FAMILY,
                                           metrics={k: float(v) for k, v in metrics.items()
                                                    if isinstance(v, (int, float))},
                                           alias="staging")
    tracker.end_experiment()
    return {"run_id": handle.run_id, "backend": handle.backend, "model_type": model_type,
            "model_path": str(model_path), "model_version": model_version,
            "metrics": metrics, "train_seconds": train_seconds}
```
- [ ] **Step 2:** `src/training/cli.py` (argparse, modeled on `src/ingestion/cli.py`): `train` subcommand with `--dataset`, `--model-type {iforest,lstm_ae}`, `--backend`, `--output-dir`, `--seed`. Calls `run_training`. Print summary. `src/training/__main__.py` calls `main()`.
- [ ] **Step 3: Tests** — `tests/training/test_pipeline.py` (C0 noop end-to-end + model-swap, no server needed):
```python
from src.training.pipeline import run_training
from src.training.schema import TrainingConfig


def test_pipeline_iforest_noop(labeled_features, tmp_path):
    path, _ = labeled_features
    out = run_training(path, model_type="iforest", backend="noop",
                       output_dir=tmp_path / "m",
                       cfg=TrainingConfig(use_labels_for_evaluation=True, label_column="label"))
    assert out["backend"] == "noop"
    assert out["metrics"]["roc_auc"] >= 0.0
    assert out["model_version"] is None  # noop registers nothing


def test_model_swap_iforest_to_lstm(labeled_features, tmp_path):
    """Same pipeline, swap model_type — proves RQ4 flexibility (no pipeline change)."""
    path, _ = labeled_features
    cfg = TrainingConfig(use_labels_for_evaluation=True, label_column="label")
    a = run_training(path, model_type="iforest", backend="noop", output_dir=tmp_path / "a", cfg=cfg)
    b = run_training(path, model_type="lstm_ae", backend="noop", output_dir=tmp_path / "b", cfg=cfg)
    assert a["model_type"] == "iforest" and b["model_type"] == "lstm_ae"
    assert "metrics" in a and "metrics" in b  # both ran through the SAME pipeline
```
- [ ] **Step 4: Run** `pytest tests/training/test_pipeline.py -v` (2 passed) + ruff.
- [ ] **Step 5: Commit** `git add src/training/pipeline.py src/training/cli.py src/training/__main__.py tests/training/test_pipeline.py` → `feat(training): tracker-integrated training pipeline + CLI (C0/C1/C2 control point)` + trailer.

---

### Task 7: Verify (DoD)

- [ ] **Step 1:** `pytest -q && ruff check .` → all green (Phase 0–3 + ~12 new training/tracking tests).
- [ ] **Step 2:** Smoke C1 (MLflow) end-to-end via the pipeline against a temp sqlite MLflow (set `MLFLOW_TRACKING_URI=sqlite:///<tmp>/m.db`): `run_training(<E parquet>, model_type="iforest", backend="mlflow", ...)` → returns a `models:/iforest_pingpong/<v>` version. (Can be a test or a manual one-liner — report the result.)
- [ ] **Step 3: DoD checklist:** MLflowTracker + ClearMLTracker implement the hardened interface with a real registry; IForest + PyTorch LSTM-AE both train via the SAME pipeline (model-swap works); `run_training` is the C0/C1/C2 control point (backend via MLOPS_BACKEND); all tests green; LSTM runs on Python 3.14 (PyTorch).

> **Carry to P8:** the model registry + alias enables Exp-2 (C0/C1 ablation, MLflow vs ClearML overhead) and Exp-5 (model-swap). Real MLflow/ClearML *servers* (docker) come in P7; P4 uses sqlite-MLflow + ClearML-offline for tests.
