"""Weekly (re)training DAG for the medical report triage classifier.

    ingest_data >> validate_data >> preprocess >> train_model
        >> evaluate_model (quality gate) >> export_onnx (parity check)
        >> publish_model >> track_experiment (MLflow) >> reload_api

Every task is a thin wrapper around ``src.pipelines.steps``: the same code runs
in the CLI (``python -m src.pipelines.train``), in CI and here. Tasks exchange
file paths and small metric dicts through XCom. If the quality gate or the
ONNX parity check fails, the run stops and the model in production is untouched.
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)


def _dirs() -> tuple[Path, Path, Path]:
    from src.config import get_settings

    settings = get_settings()
    return settings.data_dir, settings.data_dir / "staging", settings.model_dir


def _ingest_data() -> str:
    from src.pipelines import steps

    data_dir, _, _ = _dirs()
    return steps.ingest_data(data_dir)


def _validate_data(ti) -> str:
    from src.pipelines import steps

    data_dir, _, _ = _dirs()
    return steps.validate_data(ti.xcom_pull(task_ids="ingest_data"), data_dir)


def _preprocess(ti) -> dict:
    from src.pipelines import steps

    data_dir, _, _ = _dirs()
    return steps.preprocess(ti.xcom_pull(task_ids="validate_data"), data_dir)


def _train_model(ti) -> str:
    from src.pipelines import steps

    _, staging_dir, _ = _dirs()
    return steps.train(ti.xcom_pull(task_ids="preprocess")["train"], staging_dir)


def _evaluate_model(ti) -> dict:
    from src.pipelines import steps

    splits = ti.xcom_pull(task_ids="preprocess")
    return steps.evaluate_model(ti.xcom_pull(task_ids="train_model"), splits["test"])


def _export_onnx(ti) -> dict:
    from src.pipelines import steps

    splits = ti.xcom_pull(task_ids="preprocess")
    return steps.export_onnx(ti.xcom_pull(task_ids="train_model"), splits["test"])


def _publish_model(ti) -> str:
    from src.pipelines import steps

    _, staging_dir, model_dir = _dirs()
    return steps.publish_model(
        staging_dir,
        model_dir,
        metrics=ti.xcom_pull(task_ids="evaluate_model"),
        parity=ti.xcom_pull(task_ids="export_onnx"),
        raw_path=ti.xcom_pull(task_ids="ingest_data"),
    )


def _track_experiment() -> str | None:
    from src.pipelines import steps

    _, _, model_dir = _dirs()
    return steps.track_experiment(model_dir)


def _reload_api() -> None:
    """Ask the running API to load the new artifacts (no-op if it is offline)."""
    import urllib.request

    url = os.environ.get("TRIAGE_API_URL", "http://localhost:8000") + "/admin/reload"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="POST"), timeout=10) as r:
            logger.info("API reloaded: %s", r.read().decode())
    except OSError as exc:
        logger.warning("API not reachable at %s (%s); it will load the model on restart", url, exc)


default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "retry_exponential_backoff": True,
    "execution_timeout": timedelta(minutes=30),
}

with DAG(
    dag_id="triage_training_pipeline",
    description="Ingest -> validate -> train -> evaluate -> ONNX -> publish the triage model",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    tags=["fiap", "nlp", "triage", "training"],
    doc_md=__doc__,
) as dag:
    ingest = PythonOperator(task_id="ingest_data", python_callable=_ingest_data)
    validate = PythonOperator(task_id="validate_data", python_callable=_validate_data)
    prep = PythonOperator(task_id="preprocess", python_callable=_preprocess)
    train = PythonOperator(task_id="train_model", python_callable=_train_model)
    evaluate = PythonOperator(task_id="evaluate_model", python_callable=_evaluate_model)
    export = PythonOperator(task_id="export_onnx", python_callable=_export_onnx)
    publish = PythonOperator(task_id="publish_model", python_callable=_publish_model)
    track = PythonOperator(task_id="track_experiment", python_callable=_track_experiment)
    reload_api = PythonOperator(task_id="reload_api", python_callable=_reload_api, retries=0)

    ingest >> validate >> prep >> train >> evaluate >> export >> publish >> track >> reload_api
