"""DAG integrity checks. Skipped locally when Airflow is not installed; the
``dag-check`` CI job installs Airflow and runs them."""

from pathlib import Path

import pytest

pytest.importorskip("airflow.models")

from airflow.models import DagBag  # noqa: E402

DAGS_DIR = Path(__file__).resolve().parent.parent / "dags"
EXPECTED_ORDER = [
    "ingest_data",
    "validate_data",
    "preprocess",
    "train_model",
    "evaluate_model",
    "export_onnx",
    "publish_model",
    "track_experiment",
    "publish_to_cloud",
    "reload_api",
]


@pytest.fixture(scope="module")
def dagbag() -> DagBag:
    return DagBag(dag_folder=str(DAGS_DIR), include_examples=False)


def test_dag_imports_without_errors(dagbag) -> None:
    assert dagbag.import_errors == {}


def test_training_dag_has_expected_linear_flow(dagbag) -> None:
    dag = dagbag.get_dag("triage_training_pipeline")
    assert dag is not None
    assert [task.task_id for task in dag.topological_sort()] == EXPECTED_ORDER
    for upstream, downstream in zip(EXPECTED_ORDER, EXPECTED_ORDER[1:], strict=False):
        assert downstream in dag.get_task(upstream).downstream_task_ids


def test_training_dag_schedule_and_retries(dagbag) -> None:
    dag = dagbag.get_dag("triage_training_pipeline")
    assert dag.catchup is False
    assert dag.default_args["retries"] >= 1
