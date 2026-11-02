"""Training pipeline steps.

Each step reads and writes files and returns plain paths/dicts, so the same
functions are used by the CLI (``python -m src.pipelines.train``) and by the
Airflow DAG (where return values travel between tasks through XCom).

A new model is written to a *staging* folder and only copied to the serving
``models/`` folder by :func:`publish_model`, after passing the quality gate,
the ONNX parity check and the champion x challenger comparison. A candidate
that loses the comparison is only registered as ``challenger`` in MLflow.
"""

import json
import logging
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import (
    LABELS,
    METADATA_FILE,
    ONNX_MODEL_FILE,
    SKLEARN_MODEL_FILE,
    TARGET_COLUMN,
    TEXT_COLUMN,
    load_params,
)
from src.data.preprocessing import file_sha256, map_labels, split_dataset, validate_dataset
from src.evaluation.metrics import check_quality_gate, evaluate
from src.models.export_onnx import check_parity, export_to_onnx
from src.models.train import save_model, train_model

logger = logging.getLogger(__name__)


def ingest_data(data_dir: Path) -> str:
    """Download the raw corpus (no-op if already cached)."""
    from src.data.download import download_dataset

    return str(download_dataset(data_dir / "raw"))


def validate_data(raw_path: str, data_dir: Path, params: dict[str, Any] | None = None) -> str:
    """Map labels, clean texts and enforce the data contract."""
    params = params or load_params()
    frame = map_labels(pd.read_csv(raw_path), params)
    validate_dataset(frame, min_rows=params["data"]["min_rows"])
    target = data_dir / "processed" / "dataset.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return str(target)


def preprocess(dataset_path: str, data_dir: Path, params: dict[str, Any] | None = None) -> dict:
    """Stratified train/test split."""
    params = params or load_params()
    train, test = split_dataset(
        pd.read_csv(dataset_path),
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
    )
    out_dir = data_dir / "processed"
    paths = {"train": str(out_dir / "train.csv"), "test": str(out_dir / "test.csv")}
    train.to_csv(paths["train"], index=False)
    test.to_csv(paths["test"], index=False)
    logger.info("Split: %d train / %d test", len(train), len(test))
    return paths


def train(train_path: str, staging_dir: Path, params: dict[str, Any] | None = None) -> str:
    """Fit the classifier and save it to the staging folder."""
    params = params or load_params()
    pipeline = train_model(pd.read_csv(train_path), params["model"])
    return str(save_model(pipeline, staging_dir / SKLEARN_MODEL_FILE))


def evaluate_model(
    model_path: str, test_path: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Score the candidate on the test split and apply the quality gate."""
    params = params or load_params()
    pipeline = joblib.load(model_path)
    test = pd.read_csv(test_path)
    metrics = evaluate(test[TARGET_COLUMN].tolist(), pipeline.predict(test[TEXT_COLUMN].tolist()))
    check_quality_gate(metrics, params["quality_gate"]["min_macro_f1"])
    return metrics


def export_onnx(
    model_path: str, test_path: str, params: dict[str, Any] | None = None
) -> dict[str, float]:
    """Convert the candidate to ONNX and validate parity against sklearn."""
    params = params or load_params()
    pipeline = joblib.load(model_path)
    onnx_path = export_to_onnx(
        pipeline, Path(model_path).parent / ONNX_MODEL_FILE, opset=params["onnx"]["opset"]
    )
    texts = pd.read_csv(test_path)[TEXT_COLUMN].tolist()
    return check_parity(pipeline, onnx_path, texts, min_parity=params["onnx"]["min_parity"])


def compare_with_champion(
    metrics: dict[str, Any], model_dir: Path, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Champion x challenger: decide whether the candidate may replace production.

    The champion is the model currently in ``model_dir`` (what the API serves); its
    metrics were measured on the same deterministic test split as the candidate's.
    """
    from src.evaluation.promotion import decide_promotion
    from src.models.predictor import load_metadata

    params = params or load_params()
    try:
        champion = load_metadata(model_dir)
    except FileNotFoundError:
        champion = None

    decision = decide_promotion(
        metrics,
        champion["metrics"] if champion else None,
        params["promotion"]["metrics"],
        params["promotion"]["tolerance"],
    )
    decision["champion_version"] = champion["model_version"] if champion else None
    return decision


def _write_metadata(
    staging_dir: Path,
    metrics: dict[str, Any],
    parity: dict[str, float],
    raw_path: str | None,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Write ``metadata.json`` next to the staged artifacts and return it."""
    pipeline = joblib.load(staging_dir / SKLEARN_MODEL_FILE)
    trained_at = datetime.now(UTC)
    metadata = {
        "model_version": trained_at.strftime("%Y%m%d%H%M%S"),
        "trained_at": trained_at.isoformat(),
        "classes": [str(c) for c in pipeline.classes_],
        "labels_order": LABELS,
        "metrics": metrics,
        "onnx_parity": parity,
        "dataset_sha256": file_sha256(Path(raw_path)) if raw_path else None,
        "params": params["model"],
    }
    (staging_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def register_challenger(
    staging_dir: Path,
    metrics: dict[str, Any],
    parity: dict[str, float],
    decision: dict[str, Any],
    raw_path: str | None = None,
    params: dict[str, Any] | None = None,
) -> str | None:
    """Keep a rejected candidate traceable: MLflow run + ``challenger`` alias, no publish.

    Production (``models/``, S3, API) is untouched. Best effort, like tracking.
    """
    from src.tracking.mlflow_tracking import CHALLENGER_ALIAS, log_training_run

    params = params or load_params()
    metadata = _write_metadata(staging_dir, metrics, parity, raw_path, params)
    logger.warning(
        "Model %s NOT published (champion %s kept): %s",
        metadata["model_version"],
        decision.get("champion_version"),
        decision["reason"],
    )
    try:
        return log_training_run(
            staging_dir,
            metadata,
            alias=CHALLENGER_ALIAS,
            extra_tags={
                "promotion": "rejected",
                "promotion_reason": decision["reason"],
                "champion_version": str(decision.get("champion_version")),
            },
        )
    except Exception:
        logger.warning("MLflow tracking of the challenger failed", exc_info=True)
        return None


def publish_model(
    staging_dir: Path,
    model_dir: Path,
    metrics: dict[str, Any],
    parity: dict[str, float],
    raw_path: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """Write metadata and promote the staged artifacts to the serving folder."""
    params = params or load_params()
    metadata = _write_metadata(staging_dir, metrics, parity, raw_path, params)

    model_dir.mkdir(parents=True, exist_ok=True)
    # Metadata goes last: the API reads it to know which version is on disk.
    for name in (SKLEARN_MODEL_FILE, ONNX_MODEL_FILE, METADATA_FILE):
        # Copy to a temp file then rename, so a reader never sees a half-written model.
        # copyfile (content only): copy2's chmod/utime fail on Windows bind mounts.
        tmp = model_dir / f".{name}.tmp"
        shutil.copyfile(staging_dir / name, tmp)
        os.replace(tmp, model_dir / name)
    logger.info("Published model %s to %s", metadata["model_version"], model_dir)
    return metadata["model_version"]


def track_experiment(model_dir: Path) -> str | None:
    """Log the published model to MLflow (best effort: never fails the pipeline)."""
    from src.models.predictor import load_metadata
    from src.tracking.mlflow_tracking import log_training_run

    try:
        return log_training_run(
            model_dir, load_metadata(model_dir), extra_tags={"promotion": "champion"}
        )
    except Exception:
        logger.warning("MLflow tracking failed; model is published anyway", exc_info=True)
        return None


def publish_to_cloud(model_dir: Path) -> str | None:
    """Ship the published model to S3 and trigger the CD (no-op when not configured).

    Unlike tracking, a configured-but-failing upload raises: the cloud would silently
    keep serving the previous model, so the Airflow task must fail and retry.
    """
    from src.deploy.cloud import publish_model_to_s3, trigger_cd
    from src.models.predictor import load_metadata

    version = load_metadata(model_dir)["model_version"]
    uri = publish_model_to_s3(model_dir, version)
    if uri:
        trigger_cd(version)
    return uri
