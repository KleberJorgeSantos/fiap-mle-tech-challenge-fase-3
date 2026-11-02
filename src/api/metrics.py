"""Prometheus instrumentation of the API (plain ``prometheus_client``).

Metric families:

* ``http_requests_total``            -- request count by method, endpoint and status
* ``http_request_duration_seconds``  -- end-to-end request latency (histogram)
* ``http_requests_in_progress``      -- concurrent requests (gauge)
* ``model_inference_seconds``        -- time spent inside the model only, by backend
* ``predictions_total``              -- predicted urgency distribution
* ``model_info``                     -- loaded model version and backend
"""

import time

from fastapi import FastAPI, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, Info, generate_latest
from starlette.responses import Response
from starlette.routing import Match

# Sub-millisecond buckets: ONNX inference of a single report takes well under 1 ms.
_LATENCY_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)

REQUESTS = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=_LATENCY_BUCKETS,
)
IN_PROGRESS = Gauge("http_requests_in_progress", "HTTP requests being processed")
INFERENCE_LATENCY = Histogram(
    "model_inference_seconds",
    "Model inference time in seconds (excludes HTTP/serialization)",
    ["backend"],
    buckets=_LATENCY_BUCKETS,
)
PREDICTIONS = Counter("predictions_total", "Predictions by urgency label", ["label"])
MODEL_INFO = Info("model", "Loaded model metadata")

# Scrapes and health probes would dominate the request rate panels.
_EXCLUDED_PATHS = {"/metrics", "/health"}


def _endpoint_of(request: Request) -> str:
    """Route template (``/predict``) instead of the raw path, to bound label cardinality.

    FastAPI API routes store themselves in ``scope["route"]``; plain Starlette routes
    (Swagger's ``/docs``, ``/openapi.json``) do not, so they are matched against the
    route table. Unknown paths (404s) collapse into ``unmatched``: one label, not one
    per random URL a scanner tries.
    """
    route = request.scope.get("route")
    if route is not None:
        return route.path
    for candidate in request.app.router.routes:
        match, _ = candidate.matches(request.scope)
        if match == Match.FULL:
            return getattr(candidate, "path", "unmatched")
    return "unmatched"


def setup_metrics(app: FastAPI) -> None:
    """Register the metrics middleware and the ``/metrics`` endpoint."""

    @app.middleware("http")
    async def prometheus_middleware(request: Request, call_next):
        if request.url.path in _EXCLUDED_PATHS:
            return await call_next(request)

        IN_PROGRESS.inc()
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            elapsed = time.perf_counter() - start
            endpoint = _endpoint_of(request)
            REQUESTS.labels(request.method, endpoint, str(status)).inc()
            REQUEST_LATENCY.labels(request.method, endpoint).observe(elapsed)
            IN_PROGRESS.dec()

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
