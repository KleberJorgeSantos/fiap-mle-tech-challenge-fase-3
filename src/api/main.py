"""FastAPI inference service for medical report urgency triage.

Run locally:
    uvicorn src.api.main:app --reload
"""

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status

from src.api.metrics import INFERENCE_LATENCY, MODEL_INFO, PREDICTIONS, setup_metrics
from src.api.schemas import BatchIn, BatchOut, HealthOut, Prediction, PredictionOut, ReportIn
from src.config import get_settings
from src.data.text import clean_text
from src.logging_config import setup_logging
from src.models.predictor import Predictor, load_metadata, load_predictor

logger = logging.getLogger(__name__)

# Loaded model and its metadata; replaced atomically by load_model().
_state: dict[str, Any] = {"predictor": None, "metadata": {}}


def load_model() -> None:
    """Load (or reload) the configured backend from ``MODEL_DIR`` and warm it up."""
    settings = get_settings()
    try:
        predictor = load_predictor(settings.model_backend, settings.model_dir)
        predictor.warmup()
        metadata = load_metadata(settings.model_dir)
    except FileNotFoundError as exc:
        logger.error("Model not loaded (%s). Run `make train` first.", exc)
        return
    except Exception:
        # Serve /health as "degraded" instead of crash-looping the container.
        logger.exception("Failed to load model artifacts")
        return
    _state.update(predictor=predictor, metadata=metadata)
    MODEL_INFO.info({"version": metadata["model_version"], "backend": predictor.backend})
    logger.info("Model %s loaded with backend=%s", metadata["model_version"], predictor.backend)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    setup_logging()
    load_model()
    yield


app = FastAPI(
    title="Medical Report Triage API",
    description="Classifies medical reports into urgency levels: normal / atencao / urgente.",
    version="1.0.0",
    lifespan=lifespan,
)
setup_metrics(app)


def _require_model() -> Predictor:
    predictor = _state["predictor"]
    if predictor is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded")
    return predictor


def _prepare(texts: list[str]) -> list[str]:
    """Apply the training-time cleaning and enforce the size limit."""
    max_len = get_settings().max_text_length
    cleaned = []
    for text in texts:
        if len(text) > max_len:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, f"Text longer than {max_len} characters"
            )
        value = clean_text(text)
        if not any(ch.isalnum() for ch in value):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Text has no words")
        cleaned.append(value)
    return cleaned


def _classify(predictor: Predictor, texts: list[str]) -> list[Prediction]:
    start = time.perf_counter()
    proba = predictor.predict_proba(texts)
    INFERENCE_LATENCY.labels(predictor.backend).observe(time.perf_counter() - start)

    predictions = []
    for row in proba:
        best = int(row.argmax())
        label = predictor.classes[best]
        PREDICTIONS.labels(label).inc()
        predictions.append(
            Prediction(
                label=label,
                confidence=round(float(row[best]), 4),
                probabilities={
                    c: round(float(p), 4) for c, p in zip(predictor.classes, row, strict=True)
                },
            )
        )
    return predictions


@app.get("/health", response_model=HealthOut, tags=["infra"])
def health() -> HealthOut:
    """Liveness/readiness: reports whether a model is loaded."""
    predictor = _state["predictor"]
    return HealthOut(
        status="ok" if predictor else "degraded",
        model_loaded=predictor is not None,
        backend=predictor.backend if predictor else get_settings().model_backend,
        model_version=_state["metadata"].get("model_version"),
    )


@app.post("/predict", response_model=PredictionOut, tags=["inference"])
def predict(report: ReportIn) -> PredictionOut:
    """Classify the urgency of one medical report."""
    start = time.perf_counter()
    predictor = _require_model()
    prediction = _classify(predictor, _prepare([report.text]))[0]
    return PredictionOut(
        **prediction.model_dump(),
        model_version=_state["metadata"]["model_version"],
        backend=predictor.backend,
        latency_ms=round((time.perf_counter() - start) * 1000, 3),
    )


@app.post("/predict/batch", response_model=BatchOut, tags=["inference"])
def predict_batch(batch: BatchIn) -> BatchOut:
    """Classify up to 256 reports in one call (one vectorized model run)."""
    start = time.perf_counter()
    predictor = _require_model()
    predictions = _classify(predictor, _prepare(batch.texts))
    return BatchOut(
        predictions=predictions,
        model_version=_state["metadata"]["model_version"],
        backend=predictor.backend,
        latency_ms=round((time.perf_counter() - start) * 1000, 3),
    )


@app.post("/admin/reload", response_model=HealthOut, tags=["infra"])
def reload_model() -> HealthOut:
    """Reload artifacts from disk, e.g. after the Airflow DAG publishes a new model."""
    load_model()
    return health()
