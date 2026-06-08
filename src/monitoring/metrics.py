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
    # millisecond-scale buckets (default Prometheus buckets are seconds -> a ~15ms
    # request would land only in +Inf and make percentile queries useless)
    latency = Histogram("nwdaf_predict_latency_ms", "Predict latency (ms)", registry=registry,
                        buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000])

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
