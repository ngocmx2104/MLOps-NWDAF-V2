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
