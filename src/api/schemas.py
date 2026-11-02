"""Request/response contracts of the triage API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

UrgencyLabel = Literal["normal", "atencao", "urgente"]

_EXAMPLE_TEXT = (
    "Acute myocardial infarction with ST-segment elevation. The patient presented "
    "with severe chest pain radiating to the left arm and shortness of breath."
)


class ReportIn(BaseModel):
    """A single medical report to classify."""

    model_config = ConfigDict(json_schema_extra={"example": {"text": _EXAMPLE_TEXT}})

    text: str = Field(..., min_length=1, description="Free-text medical report")


class BatchIn(BaseModel):
    """Several reports classified in a single call."""

    model_config = ConfigDict(json_schema_extra={"example": {"texts": [_EXAMPLE_TEXT]}})

    texts: list[str] = Field(..., min_length=1, max_length=256)


class Prediction(BaseModel):
    """Urgency classification of one report."""

    model_config = ConfigDict(protected_namespaces=())

    label: UrgencyLabel
    confidence: float = Field(..., ge=0, le=1)
    probabilities: dict[str, float]


class PredictionOut(Prediction):
    """Response of ``POST /predict``."""

    model_version: str
    backend: str
    latency_ms: float


class BatchOut(BaseModel):
    """Response of ``POST /predict/batch``."""

    model_config = ConfigDict(protected_namespaces=())

    predictions: list[Prediction]
    model_version: str
    backend: str
    latency_ms: float


class HealthOut(BaseModel):
    """Response of ``GET /health``."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    model_loaded: bool
    backend: str
    model_version: str | None = None
