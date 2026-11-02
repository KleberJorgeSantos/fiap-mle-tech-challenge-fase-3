"""End-to-end training pipeline (script equivalent of the Airflow DAG).

Usage:
    python -m src.pipelines.train
"""

import logging

from src.config import get_settings
from src.logging_config import setup_logging
from src.pipelines import steps

logger = logging.getLogger(__name__)


def run() -> str:
    """ingest -> validate -> preprocess -> train -> evaluate -> export ONNX -> publish -> track -> cloud."""
    settings = get_settings()
    data_dir = settings.data_dir
    staging_dir = data_dir / "staging"

    raw_path = steps.ingest_data(data_dir)
    dataset_path = steps.validate_data(raw_path, data_dir)
    splits = steps.preprocess(dataset_path, data_dir)
    model_path = steps.train(splits["train"], staging_dir)
    metrics = steps.evaluate_model(model_path, splits["test"])
    parity = steps.export_onnx(model_path, splits["test"])
    version = steps.publish_model(staging_dir, settings.model_dir, metrics, parity, raw_path)
    steps.track_experiment(settings.model_dir)
    steps.publish_to_cloud(settings.model_dir)
    logger.info("Pipeline finished: version=%s macro_f1=%.4f", version, metrics["macro_f1"])
    return version


if __name__ == "__main__":
    setup_logging()
    run()
