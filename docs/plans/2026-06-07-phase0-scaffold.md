# Phase 0 — Scaffold & Tracking Abstraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dựng bộ khung repo cài được + tooling + CI + DVC, và lớp trừu tượng tracking `BaseTracker`/`NoopTracker`/`create_tracker()` có test — nền tảng cho mọi phase sau.

**Architecture:** Package `src/` cài editable; mỗi subpackage có `schema.py` + CLI. Lớp tracking theo Factory Pattern: `create_tracker()` đọc env `MLOPS_BACKEND` → Noop (P0) / MLflow, ClearML (P4). Core pipeline chỉ phụ thuộc `BaseTracker`, không gọi thẳng framework.

**Tech Stack:** Python 3.11, pyproject/setuptools, pytest, ruff, pre-commit, GitHub Actions, DVC.

---

### Task 1: Package skeleton + pyproject (cài editable được)

**Files:**
- Create: `pyproject.toml`
- Create: `src/__init__.py`, `src/tracking/__init__.py`
- Create: `tests/__init__.py`, `README.md`

- [ ] **Step 1: Tạo `pyproject.toml`**

```toml
[project]
name = "nwdaf-mlops"
version = "0.1.0"
description = "MLOps pipeline for 5G NWDAF ping-pong handover anomaly detection"
requires-python = ">=3.11"
dependencies = [
  "numpy>=2.0", "pandas>=2.2", "pyarrow>=18", "scipy>=1.14",
  "scikit-learn>=1.6", "joblib>=1.4",
  "mlflow>=2.18", "evidently>=0.4", "dvc[s3]>=3.55",
  "fastapi>=0.115", "uvicorn[standard]>=0.34", "pydantic>=2.10",
  "prometheus-client>=0.21", "psutil>=6.1", "python-dotenv>=1.0",
]
[project.optional-dependencies]
feast = ["feast>=0.40"]
lstm = ["tensorflow>=2.16"]
clearml = ["clearml>=1.16"]
dev = ["pytest>=8", "pytest-cov>=5", "ruff>=0.6", "pre-commit>=3.8"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["src*"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

> Note khi execute: `feast` để extra riêng vì hay xung đột dep với mlflow/tf; cài `pip install -e ".[dev,feast]"`. Nếu resolver xung đột, pin cụ thể lúc đó và ghi vào `pyproject`.

- [ ] **Step 2: Tạo các file `__init__.py` rỗng + README tối thiểu**

`src/__init__.py`, `src/tracking/__init__.py`, `tests/__init__.py` = file rỗng.
`README.md`:
```markdown
# NWDAF MLOps Thesis (Ver2)
Pipeline MLOps end-to-end cho phát hiện ping-pong handover (5G NWDAF). Xem `CLAUDE.md`, `docs/THESIS_SPEC.md`, `docs/IMPLEMENTATION_PLAN.md`.
Setup: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev,feast]"`
```

- [ ] **Step 3: Tạo venv + cài editable, xác minh**

Run: `python3.11 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`
Expected: cài thành công, `python -c "import src; print('ok')"` in `ok`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src/__init__.py src/tracking/__init__.py tests/__init__.py README.md
git commit -m "chore: scaffold package + pyproject"
```

---

### Task 2: Lint/format + pytest smoke + .gitignore

**Files:**
- Create: `.gitignore`, `tests/test_smoke.py`

- [ ] **Step 1: Tạo `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
artifacts/
mlruns/
mlartifacts/
*.joblib
.dvc/cache/
.dvc/tmp/
.env
reports/
```

- [ ] **Step 2: Viết test smoke (sẽ pass ngay — xác lập vòng pytest)**

`tests/test_smoke.py`:
```python
def test_package_imports():
    import src
    assert src is not None
```

- [ ] **Step 3: Chạy pytest + ruff, xác minh**

Run: `pytest && ruff check .`
Expected: pytest `1 passed`; ruff `All checks passed!`.

- [ ] **Step 4: Commit**

```bash
git add .gitignore tests/test_smoke.py
git commit -m "chore: add gitignore, ruff/pytest smoke"
```

---

### Task 3: GitHub Actions CI (lint + test mọi push)

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Tạo workflow**

```yaml
name: CI
on: [push, pull_request]
jobs:
  lint-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: pytest
```

- [ ] **Step 2: Commit + push, xác minh CI**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add lint+test workflow"
git push
```
Expected: tab Actions trên remote hiện job `lint-test` → xanh.

---

### Task 4: Khởi tạo DVC

**Files:**
- Create: `.dvc/` (do `dvc init` tạo), `.dvcignore`

- [ ] **Step 1: Init DVC**

Run: `dvc init`
Expected: tạo `.dvc/`, `.dvcignore`; `dvc status` in `There are no data or pipelines tracked...`.

- [ ] **Step 2: Cấu hình remote placeholder (local, đổi sau)**

Run: `dvc remote add -d localremote .dvcstore`
Expected: ghi vào `.dvc/config`. (Remote thật — S3 — cấu hình ở P1 khi có dữ liệu.)

- [ ] **Step 3: Commit**

```bash
git add .dvc/config .dvcignore .gitignore
git commit -m "chore: init DVC with local remote placeholder"
```

---

### Task 5: Tracking schema (config dataclasses)

**Files:**
- Create: `src/tracking/schema.py`
- Test: `tests/tracking/__init__.py`, `tests/tracking/test_schema.py`

- [ ] **Step 1: Viết test thất bại**

`tests/tracking/test_schema.py`:
```python
from src.tracking.schema import ExperimentConfig, RunHandle

def test_experiment_config_defaults():
    cfg = ExperimentConfig(experiment_name="exp", run_name="run1")
    assert cfg.backend == "noop"
    assert cfg.tags == {}

def test_run_handle():
    h = RunHandle(run_id=None, backend="noop")
    assert h.run_id is None and h.backend == "noop" and h.url is None
```
(tạo `tests/tracking/__init__.py` rỗng)

- [ ] **Step 2: Chạy test, xác minh FAIL**

Run: `pytest tests/tracking/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: src.tracking.schema`.

- [ ] **Step 3: Implement schema**

`src/tracking/schema.py`:
```python
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    experiment_name: str
    run_name: str
    backend: str = "noop"
    tags: dict = field(default_factory=dict)


@dataclass
class RunHandle:
    run_id: str | None
    backend: str
    url: str | None = None
```

- [ ] **Step 4: Chạy test, xác minh PASS**

Run: `pytest tests/tracking/test_schema.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/tracking/schema.py tests/tracking/
git commit -m "feat(tracking): ExperimentConfig + RunHandle schema"
```

---

### Task 6: `BaseTracker` interface (ABC)

**Files:**
- Create: `src/tracking/base.py`
- Test: `tests/tracking/test_base.py`

- [ ] **Step 1: Viết test thất bại (ABC không instantiate được; có đủ 6 method)**

`tests/tracking/test_base.py`:
```python
import inspect
import pytest
from src.tracking.base import BaseTracker

def test_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        BaseTracker()

def test_interface_methods_exist():
    for name in ["init_experiment", "log_params", "log_metrics",
                 "log_dataset", "register_model", "end_experiment"]:
        assert hasattr(BaseTracker, name)
        assert inspect.isfunction(getattr(BaseTracker, name))
```

- [ ] **Step 2: Chạy test, xác minh FAIL**

Run: `pytest tests/tracking/test_base.py -v`
Expected: FAIL — `ModuleNotFoundError: src.tracking.base`.

- [ ] **Step 3: Implement interface**

`src/tracking/base.py`:
```python
from abc import ABC, abstractmethod

from src.tracking.schema import ExperimentConfig, RunHandle


class BaseTracker(ABC):
    """Backend-agnostic experiment/registry interface. Core pipeline depends only on this."""

    @abstractmethod
    def init_experiment(self, config: ExperimentConfig) -> RunHandle: ...

    @abstractmethod
    def log_params(self, params: dict) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: dict, step: int | None = None) -> None: ...

    @abstractmethod
    def log_dataset(self, path: str, name: str | None = None) -> None: ...

    @abstractmethod
    def register_model(self, model_path: str, name: str,
                       metrics: dict | None = None) -> str | None: ...

    @abstractmethod
    def end_experiment(self, status: str = "FINISHED") -> None: ...
```

- [ ] **Step 4: Chạy test, xác minh PASS**

Run: `pytest tests/tracking/test_base.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/tracking/base.py tests/tracking/test_base.py
git commit -m "feat(tracking): BaseTracker abstract interface"
```

---

### Task 7: `NoopTracker` (cấu hình C0 — không tracking)

**Files:**
- Create: `src/tracking/noop.py`
- Test: `tests/tracking/test_noop.py`

- [ ] **Step 1: Viết test thất bại**

`tests/tracking/test_noop.py`:
```python
from src.tracking.noop import NoopTracker
from src.tracking.base import BaseTracker
from src.tracking.schema import ExperimentConfig

def test_noop_is_basetracker():
    assert issubclass(NoopTracker, BaseTracker)

def test_noop_lifecycle_safe_and_silent():
    t = NoopTracker()
    h = t.init_experiment(ExperimentConfig(experiment_name="e", run_name="r"))
    assert h.backend == "noop" and h.run_id is None
    t.log_params({"contamination": 0.01})
    t.log_metrics({"roc_auc": 0.9}, step=1)
    t.log_dataset("/tmp/x.parquet", name="d1")
    assert t.register_model("/tmp/m.joblib", "iforest", {"roc_auc": 0.9}) is None
    t.end_experiment("FINISHED")  # must not raise
```

- [ ] **Step 2: Chạy test, xác minh FAIL**

Run: `pytest tests/tracking/test_noop.py -v`
Expected: FAIL — `ModuleNotFoundError: src.tracking.noop`.

- [ ] **Step 3: Implement NoopTracker**

`src/tracking/noop.py`:
```python
from src.tracking.base import BaseTracker
from src.tracking.schema import ExperimentConfig, RunHandle


class NoopTracker(BaseTracker):
    """C0 baseline: ablated MLOps layer — accepts all calls, persists nothing."""

    def init_experiment(self, config: ExperimentConfig) -> RunHandle:
        self._config = config
        return RunHandle(run_id=None, backend="noop")

    def log_params(self, params: dict) -> None:
        pass

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        pass

    def log_dataset(self, path: str, name: str | None = None) -> None:
        pass

    def register_model(self, model_path: str, name: str,
                       metrics: dict | None = None) -> str | None:
        return None

    def end_experiment(self, status: str = "FINISHED") -> None:
        pass
```

- [ ] **Step 4: Chạy test, xác minh PASS**

Run: `pytest tests/tracking/test_noop.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/tracking/noop.py tests/tracking/test_noop.py
git commit -m "feat(tracking): NoopTracker (C0 baseline)"
```

---

### Task 8: `create_tracker()` factory (đọc `MLOPS_BACKEND`)

**Files:**
- Create: `src/tracking/factory.py`
- Modify: `src/tracking/__init__.py` (re-export)
- Test: `tests/tracking/test_factory.py`

- [ ] **Step 1: Viết test thất bại**

`tests/tracking/test_factory.py`:
```python
import pytest
from src.tracking import create_tracker
from src.tracking.noop import NoopTracker

def test_default_backend_is_noop(monkeypatch):
    monkeypatch.delenv("MLOPS_BACKEND", raising=False)
    assert isinstance(create_tracker(), NoopTracker)

def test_explicit_noop():
    assert isinstance(create_tracker("noop"), NoopTracker)

def test_env_var_selects_backend(monkeypatch):
    monkeypatch.setenv("MLOPS_BACKEND", "noop")
    assert isinstance(create_tracker(), NoopTracker)

def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown"):
        create_tracker("nope")
```

- [ ] **Step 2: Chạy test, xác minh FAIL**

Run: `pytest tests/tracking/test_factory.py -v`
Expected: FAIL — `ImportError: cannot import name 'create_tracker'`.

- [ ] **Step 3: Implement factory + re-export**

`src/tracking/factory.py`:
```python
import os

from src.tracking.base import BaseTracker
from src.tracking.noop import NoopTracker


def create_tracker(backend: str | None = None) -> BaseTracker:
    """Return a tracker for the given backend (or env MLOPS_BACKEND, default 'noop')."""
    backend = (backend or os.environ.get("MLOPS_BACKEND", "noop")).lower()
    if backend == "noop":
        return NoopTracker()
    if backend == "mlflow":
        from src.tracking.mlflow_tracker import MLflowTracker  # implemented in P4
        return MLflowTracker()
    if backend == "clearml":
        from src.tracking.clearml_tracker import ClearMLTracker  # implemented in P4
        return ClearMLTracker()
    raise ValueError(f"Unknown MLOPS_BACKEND={backend!r} (expected noop|mlflow|clearml)")
```

`src/tracking/__init__.py`:
```python
from src.tracking.base import BaseTracker
from src.tracking.factory import create_tracker
from src.tracking.noop import NoopTracker
from src.tracking.schema import ExperimentConfig, RunHandle

__all__ = ["BaseTracker", "NoopTracker", "create_tracker", "ExperimentConfig", "RunHandle"]
```

- [ ] **Step 4: Chạy test, xác minh PASS**

Run: `pytest tests/tracking/test_factory.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/tracking/factory.py src/tracking/__init__.py tests/tracking/test_factory.py
git commit -m "feat(tracking): create_tracker factory (MLOPS_BACKEND)"
```

---

### Task 9: Verify toàn bộ Phase 0 (DoD) + push

- [ ] **Step 1: Chạy full suite + lint**

Run: `pytest && ruff check .`
Expected: tất cả pass (≥9 tests), ruff sạch.

- [ ] **Step 2: Xác minh factory qua env (smoke thủ công)**

Run: `MLOPS_BACKEND=noop python -c "from src.tracking import create_tracker; print(type(create_tracker()).__name__)"`
Expected: in `NoopTracker`.

- [ ] **Step 3: Push + xác minh CI xanh**

```bash
git push
```
Expected: GitHub Actions `lint-test` xanh.

**✅ DoD Phase 0:** `pip install -e .` OK · `pytest` xanh · `ruff` sạch · CI xanh · `create_tracker('noop')` hoạt động · DVC init xong.
