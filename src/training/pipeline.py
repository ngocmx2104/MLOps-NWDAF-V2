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

# Distinct registry name per model family so a model-swap (RQ4) does NOT overwrite
# another model's "staging" alias in a real backend. iforest keeps the canonical
# MODEL_FAMILY name (preserves the registry contract used elsewhere).
_REGISTRY_NAMES = {"iforest": MODEL_FAMILY, "lstm_ae": "lstm_ae_pingpong"}


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
    tracker.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
    model_version = tracker.register_model(str(model_path), _REGISTRY_NAMES[model_type],
                                           metrics={k: float(v) for k, v in metrics.items()
                                                    if isinstance(v, (int, float))},
                                           alias="staging")
    tracker.end_experiment()
    return {"run_id": handle.run_id, "backend": handle.backend, "model_type": model_type,
            "model_path": str(model_path), "model_version": model_version,
            "metrics": metrics, "train_seconds": train_seconds}
