"""Experiment tracking and model registry with MLflow.

Each published model becomes one MLflow run with its hyper-parameters, test
metrics, ONNX parity and artifacts. The ONNX model is registered in the Model
Registry and the ``champion`` alias points to the version in production.

Tracking is observability, not a gate: if no tracking URI is configured or the
server is unreachable, the pipeline logs a warning and carries on.
"""

import logging
import urllib.request
from pathlib import Path
from typing import Any

from src.config import (
    METADATA_FILE,
    ONNX_MODEL_FILE,
    SKLEARN_MODEL_FILE,
    get_settings,
)

logger = logging.getLogger(__name__)

CHAMPION_ALIAS = "champion"  # version in production
CHALLENGER_ALIAS = "challenger"  # latest candidate that lost the comparison


def _flatten(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """``{"tfidf": {"min_df": 2}}`` -> ``{"tfidf.min_df": 2}`` (MLflow params are flat)."""
    flat: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def _metrics_of(metadata: dict[str, Any], model_dir: Path) -> dict[str, float]:
    metrics = metadata["metrics"]
    values = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "recall_urgente": metrics["recall_urgente"],
        "onnx_label_agreement": metadata["onnx_parity"]["label_agreement"],
        "onnx_max_proba_diff": metadata["onnx_parity"]["max_proba_diff"],
        "onnx_size_mb": (model_dir / ONNX_MODEL_FILE).stat().st_size / 1e6,
        "sklearn_size_mb": (model_dir / SKLEARN_MODEL_FILE).stat().st_size / 1e6,
    }
    values.update({f"f1_{label}": f1 for label, f1 in metrics["f1_per_class"].items()})
    return values


def _server_reachable(uri: str) -> bool:
    """Fail fast: the MLflow client would otherwise retry for minutes."""
    if not uri.startswith("http"):
        return True
    try:
        with urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=3):
            return True
    except OSError:
        return False


def log_training_run(
    model_dir: Path,
    metadata: dict[str, Any],
    tracking_uri: str | None = None,
    alias: str = CHAMPION_ALIAS,
    extra_tags: dict[str, str] | None = None,
) -> str | None:
    """Log a model as an MLflow run, register it and point ``alias`` to that version.

    ``champion`` marks the published model; ``challenger`` a candidate that was
    registered for traceability but not published.

    Returns:
        The MLflow run id, or ``None`` when tracking is disabled or unavailable.
    """
    settings = get_settings()
    uri = tracking_uri or settings.mlflow_tracking_uri
    if not uri:
        logger.info("MLflow tracking disabled (MLFLOW_TRACKING_URI not set)")
        return None
    if not _server_reachable(uri):
        logger.warning("MLflow server not reachable at %s; skipping tracking", uri)
        return None

    import mlflow
    import onnx
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    version = metadata["model_version"]
    with mlflow.start_run(run_name=f"triage-{version}") as run:
        mlflow.set_tags(
            {
                "model_version": version,
                "dataset_sha256": metadata.get("dataset_sha256") or "unknown",
                "classes": ",".join(metadata["classes"]),
                **(extra_tags or {}),
            }
        )
        mlflow.log_params(_flatten(metadata["params"]))
        mlflow.log_metrics(_metrics_of(metadata, model_dir))
        mlflow.log_artifact(str(model_dir / METADATA_FILE))
        mlflow.log_artifact(str(model_dir / SKLEARN_MODEL_FILE), artifact_path="sklearn")
        info = mlflow.onnx.log_model(
            onnx.load(str(model_dir / ONNX_MODEL_FILE)),
            name="model",
            registered_model_name=settings.mlflow_registered_model,
        )

    client = MlflowClient()
    client.set_registered_model_alias(
        settings.mlflow_registered_model, alias, info.registered_model_version
    )
    logger.info(
        "MLflow run %s logged; %s v%s is now '%s'",
        run.info.run_id,
        settings.mlflow_registered_model,
        info.registered_model_version,
        alias,
    )
    return run.info.run_id
