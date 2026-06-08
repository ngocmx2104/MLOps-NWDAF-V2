# Phase 5 — Serving (FastAPI + Docker) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Build the serving subsystem (C7): a FastAPI `/predict` that loads a trained model and returns an anomaly verdict + latency, with `/health`, `/model-info`, hot-reload, rollback, and Docker — the NWDAF AnLF real-time inference path for ping-pong handover.

**Architecture:** A `ModelLoader` abstraction mirrors P4's `BaseTracker`: `PathModelLoader` (C0 — load a model file by hand) vs `RegistryModelLoader` (C1 — load from the MLflow registry by alias/version, enabling alias-based rollback). A `LoadedModel` dispatches inference by `model_type` (IsolationForest via `score_samples`/`predict`; LSTM-AE by **reusing** `src.training.lstm_detector.predict_lstm_ae`). `/predict` accepts an `imsi` (→ Feast online feature lookup, the C1/governed path) **or** raw `features` (fallback/test/C0). The loader backend is the controlled C0/C1 variable for the deployment dimension of RQ3; the `model_type` swap proves RQ4 at serving time.

**Tech Stack:** FastAPI 0.136 + uvicorn + pydantic v2, scikit-learn (IForest), PyTorch (LSTM-AE, reused), MLflow 3.13 (sqlite-backed registry for tests, `download_artifacts` + `get_model_version_by_alias`), Feast (online sqlite store), joblib, Docker.

**RQ traceability:** RQ2 (completes the C7 component → end-to-end pipeline; train/serve consistency via Feast; real-time AnLF). RQ3 (C0 path-load + manual vs C1 registry-load + alias-rollback → operational metrics deploy/rollback time + ML Test Score serving infra present/absent). RQ4 (serve iforest **or** lstm_ae through the same service, no code change). Feeds Exp-1 (E2E), Exp-2 (C0/C1 ablation), Exp-5 (model-swap).

**Scope decisions (locked in brainstorming):**
- **iforest is the robust registry + Docker path** (artifact is a single self-contained joblib). **lstm_ae serving uses PathModelLoader locally** for the model-swap demo (RQ4): its artifact is a meta joblib that references the `.pt` by an absolute training path (a P4-deferred limitation, HANDOFF §9 #7). Making the LSTM artifact self-contained for remote-registry/Docker is **deferred to P7/P8** — noted, not built here.
- **Drift/PSI/Evidently, Prometheus metrics, auto-retrain are P6** — NOT in this plan. `/predict` stays lean; we only log predictions to JSONL.
- **No serving-level framework benchmark** (MLflow vs ClearML serving) — out of scope; registry loader uses MLflow (the chosen primary).

**Source to read (port concepts, write clean — do NOT copy the old http.server layer):**
- `/Users/ngocmx/Thạc Sĩ/MLOps_Project/src/serving/{loader,runtime,pipeline,records,schema}.py` — concepts only (ResolvedModel, runtime predict/reload/rollback, JSONL records). The old HTTP layer was stdlib `http.server`; **we use FastAPI**. The old loader used a bespoke JSON registry; **we use the MLflow registry**.
- `/Users/ngocmx/Thạc Sĩ/MLOps_Project/docker-compose.yml` — Docker reference.
- Current repo: `src/features/feast_store.py` (`get_online_features`), `src/training/lstm_detector.py` (`predict_lstm_ae`, `FEATURE_COLS`), `src/training/schema.py` (`FEATURE_COLUMNS`, `MODEL_FAMILY`), `src/training/pipeline.py` (`run_training`, `_REGISTRY_NAMES`), `src/ingestion/cli.py` (CLI idiom).

**Branch:** `phase5-serving` (from `main`).

**Verified APIs (in this venv):** mlflow 3.13 (`mlflow.artifacts.download_artifacts`, `MlflowClient.get_model_version_by_alias`); `IsolationForest.score_samples`/`.predict`; `predict_lstm_ae(x: np.ndarray, model_meta_path: Path) -> (preds, scores)`; FastAPI 0.136 / pydantic 2.13 / `fastapi.testclient.TestClient` OK; `run_training(..., backend="mlflow")` registers `models:/iforest_pingpong/<v>` with alias `staging`.

---

### Task 1: Scaffold + schema (pydantic request/response + ServingConfig)

**Files:**
- Create: `src/serving/__init__.py` (empty)
- Create: `src/serving/schema.py`
- Create: `tests/serving/__init__.py` (empty)
- Create: `tests/serving/test_schema.py`

- [ ] **Step 1: Write the failing test** — `tests/serving/test_schema.py`:

```python
import pytest
from pydantic import ValidationError

from src.serving.schema import PredictRequest, ServingConfig, REGISTRY_NAMES


def test_predict_request_accepts_imsi_only():
    r = PredictRequest(imsi="111")
    assert r.imsi == "111" and r.features is None


def test_predict_request_accepts_features_only():
    feats = {c: 1.0 for c in ["n_handover", "n_unique_cells", "pingpong_count",
                              "pingpong_rate", "mean_inter_ho_s", "std_inter_ho_s", "entropy_cell_seq"]}
    r = PredictRequest(features=feats)
    assert r.features["n_handover"] == 1.0


def test_predict_request_rejects_both():
    with pytest.raises(ValidationError):
        PredictRequest(imsi="111", features={"n_handover": 1.0})


def test_predict_request_rejects_neither():
    with pytest.raises(ValidationError):
        PredictRequest()


def test_registry_names_cover_both_models():
    assert REGISTRY_NAMES["iforest"] == "iforest_pingpong"
    assert REGISTRY_NAMES["lstm_ae"] == "lstm_ae_pingpong"


def test_serving_config_defaults():
    cfg = ServingConfig()
    assert cfg.model_type == "iforest" and cfg.loader == "path"
    assert cfg.registry_alias == "staging" and cfg.port == 8080
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/serving/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: src.serving.schema`.

- [ ] **Step 3: Write minimal implementation** — `src/serving/__init__.py` (empty) and `src/serving/schema.py`:

```python
"""Serving subsystem (component C7) — request/response + runtime config schemas."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel, model_validator

from src.training.schema import FEATURE_COLUMNS, MODEL_FAMILY

# Registry name per model family. iforest reuses the canonical MODEL_FAMILY; the
# lstm name mirrors src.training.pipeline._REGISTRY_NAMES (a drift-guard test in
# test_model_loader.py asserts they stay in sync).
REGISTRY_NAMES: dict[str, str] = {"iforest": MODEL_FAMILY, "lstm_ae": "lstm_ae_pingpong"}

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_ALIAS = "staging"
DEFAULT_OUTPUT_ROOT = "artifacts/serving"


class PredictRequest(BaseModel):
    """A prediction request: supply EXACTLY one of imsi (Feast online lookup) or features."""
    imsi: str | None = None
    features: dict[str, float] | None = None
    request_id: str | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "PredictRequest":
        if (self.imsi is None) == (self.features is None):
            raise ValueError("provide exactly one of: imsi, features")
        return self


class PredictResponse(BaseModel):
    request_id: str
    is_anomaly: bool
    anomaly_score: float
    model_type: str
    model_version: str
    latency_ms: float
    feature_values: dict[str, float]


class ModelInfo(BaseModel):
    model_type: str
    model_version: str
    loader: str
    feature_columns: list[str]


@dataclass(frozen=True)
class ServingConfig:
    model_type: str = "iforest"          # iforest | lstm_ae
    loader: str = "path"                 # path (C0) | registry (C1)
    model_path: str | None = None        # for path loader
    registry_name: str | None = None     # for registry loader (default: REGISTRY_NAMES[model_type])
    registry_alias: str = DEFAULT_ALIAS
    tracking_uri: str | None = None      # mlflow sqlite/server uri
    feast_repo_path: str | None = None   # required for imsi (online) requests
    output_root: str = DEFAULT_OUTPUT_ROOT
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT

    def resolved_registry_name(self) -> str:
        return self.registry_name or REGISTRY_NAMES[self.model_type]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_columns"] = list(FEATURE_COLUMNS)
        return payload
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/serving/test_schema.py -v` → Expected: 6 passed.
Run: `ruff check src/serving tests/serving` → Expected: All checks passed.

- [ ] **Step 5: Commit**

```bash
git add src/serving/__init__.py src/serving/schema.py tests/serving/__init__.py tests/serving/test_schema.py
git commit -m "feat(serving): request/response + ServingConfig schemas (C7 scaffold)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Predictor — LoadedModel with multi-model dispatch + shared fixtures

**Files:**
- Create: `src/serving/predictor.py`
- Create: `tests/serving/conftest.py` (shared fixtures used by Tasks 2–8)
- Create: `tests/serving/test_predictor.py`

- [ ] **Step 1: Write the shared fixtures** — `tests/serving/conftest.py`:

```python
"""Shared serving-test fixtures: a small labeled parquet + trained models + a
sqlite-MLflow registry + a materialized Feast repo. All artifacts live under tmp_path."""
import numpy as np
import pandas as pd
import pytest

from src.training.pipeline import run_training
from src.training.schema import FEATURE_COLUMNS, TrainingConfig

FEATURES = list(FEATURE_COLUMNS)


@pytest.fixture
def labeled_parquet(tmp_path):
    """A 300-row feature parquet with imsi/window_start/label (5% extreme anomalies)."""
    rng = np.random.RandomState(0)
    n = 300
    df = pd.DataFrame({f: rng.gamma(2.0, 1.0, n) for f in FEATURES})
    df["imsi"] = [str(i) for i in range(n)]
    df["window_start"] = pd.date_range("2024-06-26", periods=n, freq="5min", tz="UTC")
    df["label"] = 0
    anom = rng.choice(n, size=15, replace=False)
    df.loc[anom, FEATURES] = df.loc[anom, FEATURES] * 5.0
    df.loc[anom, "label"] = 1
    path = tmp_path / "features.parquet"
    df.to_parquet(path, index=False)
    return path, df


@pytest.fixture
def _cfg():
    return TrainingConfig(use_labels_for_evaluation=True, label_column="label")


@pytest.fixture
def iforest_model_path(labeled_parquet, tmp_path, _cfg):
    path, _ = labeled_parquet
    out = run_training(path, model_type="iforest", backend="noop",
                       output_dir=tmp_path / "m_if", cfg=_cfg)
    return out["model_path"]  # a self-contained joblib (bare IsolationForest)


@pytest.fixture
def lstm_model_path(labeled_parquet, tmp_path, _cfg):
    path, _ = labeled_parquet
    out = run_training(path, model_type="lstm_ae", backend="noop",
                       output_dir=tmp_path / "m_lstm", cfg=_cfg)
    return out["model_path"]  # a joblib meta bundle (references the .pt by abs path)


@pytest.fixture
def mlflow_registry(labeled_parquet, tmp_path, monkeypatch, _cfg):
    """A sqlite MLflow with TWO registered iforest versions; alias 'staging' on the latest.
    Returns (tracking_uri, first_version_number, latest_version_number)."""
    import mlflow
    from mlflow.tracking import MlflowClient
    path, _ = labeled_parquet
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    run_training(path, model_type="iforest", backend="mlflow",
                 output_dir=tmp_path / "r1", cfg=_cfg, run_name="v_first")
    run_training(path, model_type="iforest", backend="mlflow",
                 output_dir=tmp_path / "r2", cfg=_cfg, run_name="v_latest")
    versions = MlflowClient().search_model_versions("name='iforest_pingpong'")
    nums = sorted(int(mv.version) for mv in versions)
    return uri, str(nums[0]), str(nums[-1])


@pytest.fixture
def sample_rows():
    """One normal-ish feature row as the API would receive it."""
    return [{f: 1.0 for f in FEATURES}]
```

- [ ] **Step 2: Write the failing test** — `tests/serving/test_predictor.py`:

```python
import joblib

from src.serving.predictor import LoadedModel
from src.training.schema import FEATURE_COLUMNS


def test_iforest_predict(iforest_model_path, sample_rows):
    model = joblib.load(iforest_model_path)
    lm = LoadedModel(model_type="iforest", version="v1", model_obj=model,
                     feature_columns=list(FEATURE_COLUMNS))
    out = lm.predict(sample_rows)
    assert len(out) == 1
    assert isinstance(out[0]["is_anomaly"], bool)
    assert isinstance(out[0]["anomaly_score"], float)


def test_lstm_predict(lstm_model_path, sample_rows):
    lm = LoadedModel(model_type="lstm_ae", version="v1", meta_path=str(lstm_model_path),
                     feature_columns=list(FEATURE_COLUMNS))
    out = lm.predict(sample_rows)
    assert len(out) == 1 and out[0]["is_anomaly"] in (True, False)


def test_predict_missing_feature_raises(iforest_model_path):
    model = joblib.load(iforest_model_path)
    lm = LoadedModel(model_type="iforest", version="v1", model_obj=model,
                     feature_columns=list(FEATURE_COLUMNS))
    try:
        lm.predict([{"n_handover": 1.0}])  # missing the rest
        raised = False
    except KeyError:
        raised = True
    assert raised
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/serving/test_predictor.py -v`
Expected: FAIL with `ModuleNotFoundError: src.serving.predictor`.

- [ ] **Step 4: Write minimal implementation** — `src/serving/predictor.py`:

```python
"""LoadedModel: a model + its inference dispatch (tracker/loader-agnostic).

iforest -> sklearn IsolationForest (score_samples/predict).
lstm_ae -> reuse src.training.lstm_detector.predict_lstm_ae (train/serve consistency).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.training.schema import FEATURE_COLUMNS


@dataclass
class LoadedModel:
    model_type: str
    version: str
    model_obj: Any = None            # sklearn model (iforest)
    meta_path: str | None = None     # joblib meta bundle path (lstm_ae)
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    def _matrix(self, rows: list[dict[str, float]]) -> np.ndarray:
        return np.asarray(
            [[float(r[c]) for c in self.feature_columns] for r in rows], dtype=float
        )

    def predict(self, rows: list[dict[str, float]]) -> list[dict[str, Any]]:
        x = self._matrix(rows)  # raises KeyError if a feature column is missing
        if self.model_type == "iforest":
            scores = -self.model_obj.score_samples(x)
            preds = (self.model_obj.predict(x) == -1).astype(int)
        elif self.model_type == "lstm_ae":
            from src.training.lstm_detector import predict_lstm_ae
            preds, scores = predict_lstm_ae(x, Path(self.meta_path))
        else:
            raise ValueError(f"Unknown model_type={self.model_type!r}")
        return [{"is_anomaly": bool(p), "anomaly_score": float(s)}
                for p, s in zip(preds, scores)]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/serving/test_predictor.py -v` → Expected: 3 passed.
Run: `ruff check src/serving tests/serving` → Expected: All checks passed.

- [ ] **Step 6: Commit**

```bash
git add src/serving/predictor.py tests/serving/conftest.py tests/serving/test_predictor.py
git commit -m "feat(serving): LoadedModel multi-model predict dispatch + test fixtures

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: ModelLoader abstraction (Path + Registry + factory)

**Files:**
- Create: `src/serving/model_loader.py`
- Create: `tests/serving/test_model_loader.py`

- [ ] **Step 1: Write the failing test** — `tests/serving/test_model_loader.py`:

```python
import pytest

from src.serving.model_loader import (
    PathModelLoader, RegistryModelLoader, create_loader,
)
from src.serving.schema import REGISTRY_NAMES, ServingConfig


def test_path_loader_iforest(iforest_model_path):
    lm = PathModelLoader(iforest_model_path, "iforest").load()
    assert lm.model_type == "iforest" and lm.model_obj is not None


def test_path_loader_lstm(lstm_model_path):
    lm = PathModelLoader(lstm_model_path, "lstm_ae").load()
    assert lm.model_type == "lstm_ae" and lm.meta_path is not None


def test_path_loader_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PathModelLoader(tmp_path / "nope.joblib", "iforest").load()


def test_registry_loader_iforest(mlflow_registry):
    uri, _first, latest = mlflow_registry
    lm = RegistryModelLoader("iforest_pingpong", "iforest", alias="staging",
                             tracking_uri=uri).load()
    assert lm.model_type == "iforest" and lm.model_obj is not None
    assert lm.version.endswith(f"/{latest}")  # alias 'staging' -> latest version


def test_registry_loader_by_version(mlflow_registry):
    uri, first, _latest = mlflow_registry
    lm = RegistryModelLoader("iforest_pingpong", "iforest", version=first,
                             tracking_uri=uri).load()
    assert lm.version.endswith(f"/{first}")


def test_factory_path(iforest_model_path):
    cfg = ServingConfig(loader="path", model_type="iforest", model_path=str(iforest_model_path))
    assert create_loader(cfg).load().model_type == "iforest"


def test_factory_registry(mlflow_registry):
    uri, _f, _l = mlflow_registry
    cfg = ServingConfig(loader="registry", model_type="iforest", tracking_uri=uri)
    assert create_loader(cfg).load().model_obj is not None


def test_registry_names_match_training():
    """Guard: serving's REGISTRY_NAMES must stay in sync with training's private map."""
    from src.training.pipeline import _REGISTRY_NAMES
    assert REGISTRY_NAMES == _REGISTRY_NAMES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/serving/test_model_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: src.serving.model_loader`.

- [ ] **Step 3: Write minimal implementation** — `src/serving/model_loader.py`:

```python
"""ModelLoader abstraction — the C0/C1 deployment control variable.

PathModelLoader (C0)     -> load a model artifact by local path (ad-hoc deploy).
RegistryModelLoader (C1) -> load from the MLflow registry by alias or version
                            (governed deploy + alias-based rollback).
create_loader(config) selects one, mirroring src.tracking.create_tracker.

Security note: joblib.load below deserializes model artifacts that are produced by
THIS project's own training pipeline (src.training.run_training -> joblib.dump) and
served either from a deployer-controlled local path or our own MLflow registry. They
are trusted internal artifacts (single-node thesis scope), not untrusted input -> the
pickle-execution risk does not apply here. Do NOT point a loader at an untrusted file.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib

from src.serving.predictor import LoadedModel
from src.serving.schema import ServingConfig
from src.training.schema import FEATURE_COLUMNS

_FEATS = list(FEATURE_COLUMNS)


class ModelLoader(ABC):
    @abstractmethod
    def load(self) -> LoadedModel: ...


def _load_artifact(model_file: Path, model_type: str, version: str) -> LoadedModel:
    if model_type == "iforest":
        return LoadedModel(model_type="iforest", version=version,
                           model_obj=joblib.load(model_file), feature_columns=_FEATS)
    if model_type == "lstm_ae":
        # NOTE single-node: the meta bundle references its .pt by an absolute training
        # path. Self-contained packaging for remote/Docker is deferred to P7/P8.
        return LoadedModel(model_type="lstm_ae", version=version,
                           meta_path=str(model_file), feature_columns=_FEATS)
    raise ValueError(f"Unknown model_type={model_type!r}")


class PathModelLoader(ModelLoader):
    def __init__(self, model_path: str | Path, model_type: str) -> None:
        self.model_path = Path(model_path)
        self.model_type = model_type

    def load(self) -> LoadedModel:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
        return _load_artifact(self.model_path, self.model_type, version=self.model_path.stem)


class RegistryModelLoader(ModelLoader):
    def __init__(self, name: str, model_type: str, *, alias: str | None = None,
                 version: str | None = None, tracking_uri: str | None = None) -> None:
        if not alias and not version:
            raise ValueError("RegistryModelLoader needs an alias or a version")
        self.name = name
        self.model_type = model_type
        self.alias = alias
        self.version = version
        self.tracking_uri = tracking_uri

    def load(self) -> LoadedModel:
        import mlflow
        from mlflow.tracking import MlflowClient
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient()
        if self.version:
            version_num = str(self.version)
            uri = f"models:/{self.name}/{version_num}"
        else:
            mv = client.get_model_version_by_alias(self.name, self.alias)
            version_num = str(mv.version)
            uri = f"models:/{self.name}@{self.alias}"
        local_dir = mlflow.artifacts.download_artifacts(artifact_uri=uri)
        files = sorted(Path(local_dir).rglob("*.joblib"))
        if not files:
            raise FileNotFoundError(f"No model artifact (*.joblib) under {uri}")
        return _load_artifact(files[0], self.model_type,
                              version=f"models:/{self.name}/{version_num}")


def create_loader(config: ServingConfig) -> ModelLoader:
    if config.loader == "path":
        if not config.model_path:
            raise ValueError("path loader requires config.model_path")
        return PathModelLoader(config.model_path, config.model_type)
    if config.loader == "registry":
        return RegistryModelLoader(config.resolved_registry_name(), config.model_type,
                                   alias=config.registry_alias, tracking_uri=config.tracking_uri)
    raise ValueError(f"Unknown loader={config.loader!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/serving/test_model_loader.py -v` → Expected: 8 passed.
(If `download_artifacts` for the `models:/name@alias` URI behaves differently in the installed MLflow, adapt minimally — e.g. resolve the version first then `download_artifacts(artifact_uri=f"models:/{name}/{version}")` — and NOTE it. Do not weaken assertions.)
Run: `ruff check src/serving tests/serving` → Expected: All checks passed.

- [ ] **Step 5: Commit**

```bash
git add src/serving/model_loader.py tests/serving/test_model_loader.py
git commit -m "feat(serving): ModelLoader abstraction (path=C0 / registry=C1) + factory

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: FeastOnlineProvider (online feature lookup by IMSI)

**Files:**
- Create: `src/serving/feature_provider.py`
- Create: `tests/serving/test_feature_provider.py`

- [ ] **Step 1a: Add the shared `materialized_repo` fixture to `tests/serving/conftest.py`** (it is reused by Task 6's app tests, so it lives in conftest from the start). First add `from pathlib import Path` to the import block at the TOP of conftest.py (with the existing imports, to avoid ruff E402), then append this constant + fixture:

```python
_REPO_SRC = Path("src/features/feast_repo")


@pytest.fixture
def materialized_repo(tmp_path):
    """A temp Feast repo materialized with two IMSIs (proven P3 pattern)."""
    from src.features.feast_store import apply_and_materialize
    repo = tmp_path / "feast_repo"
    repo.mkdir()
    (repo / "feature_store.yaml").write_text((_REPO_SRC / "feature_store.yaml").read_text())
    df = pd.DataFrame({
        "imsi": ["111", "222"],
        "window_start": pd.to_datetime(["2024-06-26T00:00:00Z", "2024-06-26T00:05:00Z"]),
        "n_handover": [3, 7], "n_unique_cells": [2, 2], "pingpong_count": [1, 4],
        "pingpong_rate": [0.33, 0.57], "mean_inter_ho_s": [10.0, 5.0],
        "std_inter_ho_s": [1.0, 0.5], "entropy_cell_seq": [0.9, 0.8],
    })
    src_parquet = repo / "data" / "handover_features.parquet"
    src_parquet.parent.mkdir(parents=True)
    df.to_parquet(src_parquet, index=False)
    apply_and_materialize(repo, src_parquet)
    return repo
```

- [ ] **Step 1b: Write the failing test** — `tests/serving/test_feature_provider.py` (uses the shared fixture):

```python
import pytest

from src.serving.feature_provider import FeastOnlineProvider
from src.training.schema import FEATURE_COLUMNS


def test_online_lookup_returns_feature_row(materialized_repo):
    provider = FeastOnlineProvider(materialized_repo)
    rows = provider.get(["111"])
    assert len(rows) == 1
    assert set(FEATURE_COLUMNS).issubset(rows[0].keys())
    assert rows[0]["n_handover"] == 3 and rows[0]["pingpong_count"] == 1


def test_online_lookup_unknown_imsi_raises(materialized_repo):
    provider = FeastOnlineProvider(materialized_repo)
    with pytest.raises(KeyError):
        provider.get(["999999"])  # not materialized -> null features
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/serving/test_feature_provider.py -v`
Expected: FAIL with `ModuleNotFoundError: src.serving.feature_provider`.

- [ ] **Step 3: Write minimal implementation** — `src/serving/feature_provider.py`:

```python
"""FeastOnlineProvider — low-latency online feature lookup by IMSI (serving path).

Wraps src.features.feast_store.get_online_features and returns plain feature-dict
rows that LoadedModel.predict consumes. Raises KeyError if an IMSI has no online row.
"""
from __future__ import annotations

from pathlib import Path

from src.features.feast_store import get_online_features
from src.training.schema import FEATURE_COLUMNS

_FEATS = list(FEATURE_COLUMNS)


class FeastOnlineProvider:
    def __init__(self, repo_path: str | Path) -> None:
        # Lazy import: only the imsi/online path needs feast, so the C0 features-only
        # path (and a minimal image without the feast extra) can import serving freely.
        from feast import FeatureStore
        self.store = FeatureStore(repo_path=str(repo_path))

    def get(self, imsis: list[str]) -> list[dict[str, float]]:
        frame = get_online_features(self.store, imsis)
        rows: list[dict[str, float]] = []
        for imsi, record in zip(imsis, frame.to_dict("records")):
            if any(record.get(c) is None for c in _FEATS):
                raise KeyError(f"No online features for imsi={imsi!r}")
            rows.append({c: float(record[c]) for c in _FEATS})
        return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/serving/test_feature_provider.py -v` → Expected: 2 passed.
Run: `ruff check src/serving tests/serving` → Expected: All checks passed.

- [ ] **Step 5: Commit**

```bash
git add src/serving/feature_provider.py tests/serving/conftest.py tests/serving/test_feature_provider.py
git commit -m "feat(serving): FeastOnlineProvider (online feature lookup by IMSI)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Runtime + records (predict latency, reload, rollback)

**Files:**
- Create: `src/serving/records.py`
- Create: `src/serving/runtime.py`
- Create: `tests/serving/test_runtime.py`

- [ ] **Step 1: Write the failing test** — `tests/serving/test_runtime.py`:

```python
from src.serving.runtime import ServingRuntime
from src.serving.schema import PredictRequest, ServingConfig


def test_predict_with_features_path_loader(iforest_model_path, tmp_path, sample_rows):
    cfg = ServingConfig(loader="path", model_type="iforest",
                        model_path=str(iforest_model_path), output_root=str(tmp_path / "s"))
    rt = ServingRuntime.build(cfg)
    resp = rt.predict(PredictRequest(features=sample_rows[0]))
    assert resp.latency_ms >= 0.0
    assert resp.model_type == "iforest"
    assert isinstance(resp.is_anomaly, bool)
    # prediction is logged
    assert (tmp_path / "s" / "predictions.jsonl").exists()


def test_rollback_registry_switches_version(mlflow_registry, tmp_path):
    uri, first, latest = mlflow_registry
    cfg = ServingConfig(loader="registry", model_type="iforest", tracking_uri=uri,
                        output_root=str(tmp_path / "s"))
    rt = ServingRuntime.build(cfg)
    assert rt.model.version.endswith(f"/{latest}")  # alias staging -> latest
    record = rt.rollback(target_version=first, reason="test")
    assert rt.model.version.endswith(f"/{first}")
    assert record["to_version"].endswith(f"/{first}")
    assert (tmp_path / "s" / "rollback_history.jsonl").exists()


def test_reload_rebuilds_model(iforest_model_path, tmp_path):
    cfg = ServingConfig(loader="path", model_type="iforest",
                        model_path=str(iforest_model_path), output_root=str(tmp_path / "s"))
    rt = ServingRuntime.build(cfg)
    info = rt.reload()
    assert info["model_type"] == "iforest"


def test_model_swap_same_runtime_api(iforest_model_path, lstm_model_path, tmp_path, sample_rows):
    """RQ4: the SAME runtime API serves either model via config (no code change)."""
    a = ServingRuntime.build(ServingConfig(loader="path", model_type="iforest",
                                           model_path=str(iforest_model_path),
                                           output_root=str(tmp_path / "a")))
    b = ServingRuntime.build(ServingConfig(loader="path", model_type="lstm_ae",
                                           model_path=str(lstm_model_path),
                                           output_root=str(tmp_path / "b")))
    ra = a.predict(PredictRequest(features=sample_rows[0]))
    rb = b.predict(PredictRequest(features=sample_rows[0]))
    assert ra.model_type == "iforest" and rb.model_type == "lstm_ae"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/serving/test_runtime.py -v`
Expected: FAIL with `ModuleNotFoundError: src.serving.runtime`.

- [ ] **Step 3a: Write `src/serving/records.py`:**

```python
"""Lightweight JSONL records for serving (full monitoring is P6)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def build_prediction_record(response: dict[str, Any]) -> dict[str, Any]:
    return {"recorded_at": utc_now_iso(), "event": "predict", **response}


def build_deployment_record(*, model_type: str, model_version: str, loader: str) -> dict[str, Any]:
    return {"recorded_at": utc_now_iso(), "event": "deploy",
            "model_type": model_type, "model_version": model_version, "loader": loader}


def build_rollback_record(*, from_version: str, to_version: str, reason: str) -> dict[str, Any]:
    return {"recorded_at": utc_now_iso(), "event": "rollback",
            "from_version": from_version, "to_version": to_version, "reason": reason}
```

- [ ] **Step 3b: Write `src/serving/runtime.py`:**

```python
"""ServingRuntime — holds the active model + feature provider; predict/reload/rollback.

Backend (path=C0 / registry=C1) is the controlled deployment variable for RQ3;
model_type (iforest/lstm_ae) swaps the served model for RQ4. Latency is measured
with perf_counter (operational metric).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from src.serving.feature_provider import FeastOnlineProvider
from src.serving.model_loader import RegistryModelLoader, create_loader
from src.serving.predictor import LoadedModel
from src.serving.records import (
    append_jsonl, build_deployment_record, build_prediction_record, build_rollback_record,
)
from src.serving.schema import ModelInfo, PredictRequest, PredictResponse, ServingConfig


@dataclass
class ServingRuntime:
    config: ServingConfig
    model: LoadedModel
    provider: FeastOnlineProvider | None
    output_root: Path

    @classmethod
    def build(cls, config: ServingConfig) -> "ServingRuntime":
        model = create_loader(config).load()
        provider = FeastOnlineProvider(config.feast_repo_path) if config.feast_repo_path else None
        rt = cls(config=config, model=model, provider=provider,
                 output_root=Path(config.output_root))
        append_jsonl(rt.output_root / "deployment_history.jsonl",
                     build_deployment_record(model_type=model.model_type,
                                             model_version=model.version, loader=config.loader))
        return rt

    def _rows_for(self, request: PredictRequest) -> list[dict[str, float]]:
        if request.imsi is not None:
            if self.provider is None:
                raise ValueError("imsi requests require config.feast_repo_path")
            return self.provider.get([request.imsi])
        return [request.features]

    def predict(self, request: PredictRequest) -> PredictResponse:
        t0 = perf_counter()
        rows = self._rows_for(request)
        out = self.model.predict(rows)[0]
        latency_ms = (perf_counter() - t0) * 1000.0
        resp = PredictResponse(
            request_id=request.request_id or f"pred-{uuid.uuid4().hex[:12]}",
            is_anomaly=out["is_anomaly"], anomaly_score=out["anomaly_score"],
            model_type=self.model.model_type, model_version=self.model.version,
            latency_ms=latency_ms, feature_values=rows[0])
        append_jsonl(self.output_root / "predictions.jsonl",
                     build_prediction_record(resp.model_dump()))
        return resp

    def reload(self) -> dict[str, Any]:
        self.model = create_loader(self.config).load()
        return self.status()

    def rollback(self, target_version: str, reason: str = "manual rollback") -> dict[str, Any]:
        previous = self.model.version
        if self.config.loader == "registry":
            loader = RegistryModelLoader(self.config.resolved_registry_name(),
                                         self.config.model_type, version=target_version,
                                         tracking_uri=self.config.tracking_uri)
        else:
            from src.serving.model_loader import PathModelLoader
            loader = PathModelLoader(target_version, self.config.model_type)
        self.model = loader.load()
        record = build_rollback_record(from_version=previous, to_version=self.model.version,
                                       reason=reason)
        append_jsonl(self.output_root / "rollback_history.jsonl", record)
        return record

    def status(self) -> dict[str, Any]:
        return ModelInfo(model_type=self.model.model_type, model_version=self.model.version,
                         loader=self.config.loader,
                         feature_columns=self.model.feature_columns).model_dump()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/serving/test_runtime.py -v` → Expected: 4 passed.
Run: `ruff check src/serving tests/serving` → Expected: All checks passed.

- [ ] **Step 5: Commit**

```bash
git add src/serving/records.py src/serving/runtime.py tests/serving/test_runtime.py
git commit -m "feat(serving): ServingRuntime predict(latency)/reload/rollback + JSONL records

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: FastAPI app + endpoints

**Files:**
- Create: `src/serving/app.py`
- Create: `tests/serving/test_app.py`

- [ ] **Step 1: Write the failing test** — `tests/serving/test_app.py`:

```python
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.runtime import ServingRuntime
from src.serving.schema import ServingConfig


def _client(model_path, tmp_path, **kw):
    cfg = ServingConfig(loader="path", model_type="iforest", model_path=str(model_path),
                        output_root=str(tmp_path / "s"), **kw)
    return TestClient(create_app(ServingRuntime.build(cfg)))


def test_health(iforest_model_path, tmp_path):
    c = _client(iforest_model_path, tmp_path)
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_model_info(iforest_model_path, tmp_path):
    c = _client(iforest_model_path, tmp_path)
    r = c.get("/model-info")
    assert r.status_code == 200 and r.json()["model_type"] == "iforest"


def test_predict_with_features(iforest_model_path, tmp_path, sample_rows):
    c = _client(iforest_model_path, tmp_path)
    r = c.post("/predict", json={"features": sample_rows[0]})
    assert r.status_code == 200
    body = r.json()
    assert "is_anomaly" in body and body["latency_ms"] >= 0.0


def test_predict_rejects_both_imsi_and_features(iforest_model_path, tmp_path, sample_rows):
    c = _client(iforest_model_path, tmp_path)
    r = c.post("/predict", json={"imsi": "111", "features": sample_rows[0]})
    assert r.status_code == 422  # pydantic validation error


def test_predict_with_imsi_uses_feast(iforest_model_path, tmp_path, materialized_repo, sample_rows):
    c = _client(iforest_model_path, tmp_path, feast_repo_path=str(materialized_repo))
    r = c.post("/predict", json={"imsi": "111"})
    assert r.status_code == 200 and "anomaly_score" in r.json()
```

(The `materialized_repo` fixture already lives in `tests/serving/conftest.py` from Task 4, so it is shared automatically.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/serving/test_app.py -v`
Expected: FAIL with `ModuleNotFoundError: src.serving.app`.

- [ ] **Step 3: Write `src/serving/app.py`:**

```python
"""FastAPI app for serving (C7): /health, /model-info, /predict, /admin/reload, /admin/rollback."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from src.serving.runtime import ServingRuntime
from src.serving.schema import ModelInfo, PredictRequest, PredictResponse


def create_app(runtime: ServingRuntime) -> FastAPI:
    app = FastAPI(title="NWDAF ping-pong handover detector", version="1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model_version": runtime.model.version}

    @app.get("/model-info", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        return ModelInfo(**runtime.status())

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        try:
            return runtime.predict(request)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/admin/reload", response_model=ModelInfo)
    def reload() -> ModelInfo:
        return ModelInfo(**runtime.reload())

    @app.post("/admin/rollback")
    def rollback(target_version: str, reason: str = "manual rollback") -> dict:
        try:
            return runtime.rollback(target_version=target_version, reason=reason)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/serving/test_app.py -v` → Expected: 5 passed.
Run: `pytest tests/serving -q` → Expected: all serving tests pass.
Run: `ruff check src/serving tests/serving` → Expected: All checks passed.

- [ ] **Step 5: Commit**

```bash
git add src/serving/app.py tests/serving/test_app.py
git commit -m "feat(serving): FastAPI app (/predict /health /model-info /admin reload+rollback)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: CLI + Docker

**Files:**
- Create: `src/serving/cli.py`
- Create: `src/serving/__main__.py`
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.dockerignore`
- Create: `tests/serving/test_cli.py`

- [ ] **Step 1: Write the failing test** — `tests/serving/test_cli.py`:

```python
from src.serving.cli import build_config_from_args, build_parser


def test_parser_has_serve_subcommand():
    parser = build_parser()
    args = parser.parse_args(["serve", "--model-path", "m.joblib"])
    assert args.command == "serve" and args.model_path == "m.joblib"


def test_build_config_defaults_to_path_loader():
    parser = build_parser()
    args = parser.parse_args(["serve", "--model-path", "m.joblib"])
    cfg = build_config_from_args(args)
    assert cfg.loader == "path" and cfg.model_type == "iforest"
    assert cfg.model_path == "m.joblib" and cfg.port == 8080


def test_build_config_registry():
    parser = build_parser()
    args = parser.parse_args(["serve", "--loader", "registry", "--model-type", "iforest",
                              "--tracking-uri", "sqlite:///x.db", "--port", "9000"])
    cfg = build_config_from_args(args)
    assert cfg.loader == "registry" and cfg.tracking_uri == "sqlite:///x.db" and cfg.port == 9000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/serving/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: src.serving.cli`.

- [ ] **Step 3a: Write `src/serving/cli.py`** (argparse idiom matching `src/ingestion/cli.py`; reads env fallbacks so a container is configurable):

```python
"""Serving CLI: `python -m src.serving serve ...` runs the FastAPI app via uvicorn.

Config resolves from CLI args, then env vars (MLOPS_SERVING_*), then defaults — so a
Docker container can be configured entirely through the environment.
"""
from __future__ import annotations

import argparse
import os

from src.serving.schema import DEFAULT_HOST, DEFAULT_OUTPUT_ROOT, DEFAULT_PORT, ServingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.serving")
    sub = parser.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("serve", help="Run the FastAPI serving app")
    sp.add_argument("--model-type", default=os.environ.get("MLOPS_SERVING_MODEL_TYPE", "iforest"),
                    choices=["iforest", "lstm_ae"])
    sp.add_argument("--loader", default=os.environ.get("MLOPS_SERVING_LOADER", "path"),
                    choices=["path", "registry"])
    sp.add_argument("--model-path", default=os.environ.get("MLOPS_SERVING_MODEL_PATH"))
    sp.add_argument("--registry-name", default=os.environ.get("MLOPS_SERVING_REGISTRY_NAME"))
    sp.add_argument("--registry-alias", default=os.environ.get("MLOPS_SERVING_REGISTRY_ALIAS", "staging"))
    sp.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    sp.add_argument("--feast-repo-path", default=os.environ.get("MLOPS_SERVING_FEAST_REPO"))
    sp.add_argument("--output-root", default=os.environ.get("MLOPS_SERVING_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
    sp.add_argument("--host", default=os.environ.get("MLOPS_SERVING_HOST", DEFAULT_HOST))
    sp.add_argument("--port", type=int, default=int(os.environ.get("MLOPS_SERVING_PORT", DEFAULT_PORT)))
    sp.set_defaults(func=_cmd_serve)
    return parser


def build_config_from_args(args: argparse.Namespace) -> ServingConfig:
    return ServingConfig(
        model_type=args.model_type, loader=args.loader, model_path=args.model_path,
        registry_name=args.registry_name, registry_alias=args.registry_alias,
        tracking_uri=args.tracking_uri, feast_repo_path=args.feast_repo_path,
        output_root=args.output_root, host=args.host, port=args.port)


def _cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from src.serving.app import create_app
    from src.serving.runtime import ServingRuntime
    config = build_config_from_args(args)
    app = create_app(ServingRuntime.build(config))
    uvicorn.run(app, host=config.host, port=config.port)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3b: Write `src/serving/__main__.py`:**

```python
from src.serving.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 3c: Write `Dockerfile`** (iforest serving — no torch needed; lighter image):

```dockerfile
FROM python:3.14-slim

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src

# Install the package with the feast extra (online lookup); torch/[lstm] is NOT
# installed -> iforest serving only (the self-contained, Docker-friendly path).
RUN pip install --no-cache-dir -e ".[feast]"

ENV MLOPS_SERVING_HOST=0.0.0.0 \
    MLOPS_SERVING_PORT=8080
EXPOSE 8080

CMD ["python", "-m", "src.serving", "serve"]
```

- [ ] **Step 3d: Write `.dockerignore`:**

```
.venv/
__pycache__/
*.pyc
.git/
artifacts/
mlruns/
data/
tests/
docs/
.dvc/
```

- [ ] **Step 3e: Write `docker-compose.yml`** (mounts a host model dir; serves it):

```yaml
services:
  serving:
    build: .
    image: nwdaf-serving:latest
    ports:
      - "8080:8080"
    environment:
      MLOPS_SERVING_LOADER: path
      MLOPS_SERVING_MODEL_TYPE: iforest
      MLOPS_SERVING_MODEL_PATH: /app/models/model_iforest.joblib
      MLOPS_SERVING_OUTPUT_ROOT: /app/serving_out
    volumes:
      - ./artifacts/models:/app/models:ro
      - ./artifacts/serving:/app/serving_out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/serving/test_cli.py -v` → Expected: 3 passed.
Run: `python -m src.serving serve --help` → Expected: usage with `serve` + all flags.
Run: `ruff check src/serving tests/serving` → Expected: All checks passed.

- [ ] **Step 5: Commit**

```bash
git add src/serving/cli.py src/serving/__main__.py Dockerfile docker-compose.yml .dockerignore tests/serving/test_cli.py
git commit -m "feat(serving): CLI (uvicorn serve) + Docker (iforest image + compose)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Verify (DoD)

**Files:** none (verification + smoke).

- [ ] **Step 1:** Full suite + lint.

Run: `pytest -q` → Expected: all green (Phase 0–4 + ~25 new serving tests).
Run: `ruff check .` → Expected: All checks passed.

- [ ] **Step 2: C1 end-to-end smoke** (registry-backed serving + rollback, via TestClient — write as `tests/serving/test_e2e_c1.py`, committed):

```python
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.runtime import ServingRuntime
from src.serving.schema import ServingConfig


def test_c1_registry_serving_and_rollback(mlflow_registry, tmp_path, sample_rows):
    uri, first, latest = mlflow_registry
    cfg = ServingConfig(loader="registry", model_type="iforest", tracking_uri=uri,
                        output_root=str(tmp_path / "s"))
    rt = ServingRuntime.build(cfg)
    client = TestClient(create_app(rt))

    # C1 serves the staging (latest) version end-to-end
    r = client.post("/predict", json={"features": sample_rows[0]})
    assert r.status_code == 200 and r.json()["model_version"].endswith(f"/{latest}")

    # rollback to the first version (DoD: rollback to another version OK)
    rb = client.post("/admin/rollback", params={"target_version": first})
    assert rb.status_code == 200
    info = client.get("/model-info").json()
    assert info["model_version"].endswith(f"/{first}")
```

Run: `pytest tests/serving/test_e2e_c1.py -v` → Expected: 1 passed. Then commit:

```bash
git add tests/serving/test_e2e_c1.py
git commit -m "test(serving): C1 registry serving + rollback end-to-end (P5 DoD)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 3: Docker build smoke** (manual — report result; do not block on environment):

Run: `docker build -t nwdaf-serving:latest .`
Then a quick container smoke (requires a host model file at `artifacts/models/model_iforest.joblib` — produce one with `python -m src.training train --dataset <E parquet> --model-type iforest --output-dir artifacts/models`, or copy any iforest joblib and set the env path):
`docker run --rm -p 8080:8080 -v "$PWD/artifacts/models:/app/models:ro" -e MLOPS_SERVING_MODEL_PATH=/app/models/<file>.joblib nwdaf-serving:latest` → then `curl localhost:8080/health` returns `{"status":"ok",...}`.
Report the outcome (image built? `/health` OK?). If the `python:3.14-slim` base is unavailable in the environment, NOTE it and record the Dockerfile as written (the image base can be pinned at P7 when the CI/CD + servers are stood up).

- [ ] **Step 4: DoD checklist** — verify and report PASS/FAIL with one line of evidence each:
  1. `/predict` returns an anomaly verdict + **latency_ms** (cite `test_app.py::test_predict_with_features`).
  2. `/predict` fetches **online features from Feast by IMSI** (cite `test_app.py::test_predict_with_imsi_uses_feast`).
  3. Serves a **registered** model (C1) (cite `test_e2e_c1.py`).
  4. **Rollback** to another version works (cite `test_e2e_c1.py` / `test_runtime.py::test_rollback_registry_switches_version`).
  5. **Model-swap** iforest↔lstm_ae through the same service API (RQ4) (cite `test_runtime.py::test_model_swap_same_runtime_api`).
  6. **C0/C1 control variable**: loader is the only changed axis (path vs registry) — `create_loader` selects it (cite `test_model_loader.py` factory tests).
  7. **Container** builds + `/health` OK (cite Step 3 result).
  8. All tests green; ruff clean (cite full-suite count).

> **Carry to P6/P7/P8:** P6 wires drift (PSI/Evidently) + Prometheus into `/predict` + the prediction JSONL and auto-retrain→reload. P7 makes the LSTM artifact self-contained (log `.pt` + relative resolve) so lstm serving works from a remote registry / Docker, and stands up real MLflow/ClearML servers. P8 measures deploy-time/rollback-time (C0 vs C1) + serving ML Test Score items.
```
