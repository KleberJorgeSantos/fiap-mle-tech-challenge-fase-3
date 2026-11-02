"""DAG integrity checks. Skipped locally when Airflow is not installed; the
``dag-check`` CI job installs Airflow and runs them."""

from pathlib import Path

import pytest

pytest.importorskip("airflow.models")

from airflow.models import DagBag  # noqa: E402
from airflow.operators.python import BranchPythonOperator  # noqa: E402

DAGS_DIR = Path(__file__).resolve().parent.parent / "dags"
TRAINING_CHAIN = [
    "ingest_data",
    "validate_data",
    "preprocess",
    "train_model",
    "evaluate_model",
    "export_onnx",
    "compare_with_champion",
]
PUBLISH_BRANCH = ["publish_model", "track_experiment", "publish_to_cloud", "reload_api"]


def _assert_chain(dag, task_ids: list[str]) -> None:
    for upstream, downstream in zip(task_ids, task_ids[1:], strict=False):
        assert dag.get_task(upstream).downstream_task_ids == {downstream}


@pytest.fixture(scope="module")
def dag():
    dagbag = DagBag(dag_folder=str(DAGS_DIR), include_examples=False)
    assert dagbag.import_errors == {}
    return dagbag.get_dag("triage_training_pipeline")


def test_training_chain_up_to_the_comparison(dag) -> None:
    _assert_chain(dag, TRAINING_CHAIN)


def test_comparison_branches_to_publish_or_challenger(dag) -> None:
    compare = dag.get_task("compare_with_champion")
    assert isinstance(compare, BranchPythonOperator)
    assert compare.downstream_task_ids == {"publish_model", "register_challenger"}
    assert dag.get_task("register_challenger").downstream_task_ids == set()


def test_publish_branch_order(dag) -> None:
    _assert_chain(dag, PUBLISH_BRANCH)
    assert len(dag.tasks) == len(TRAINING_CHAIN) + len(PUBLISH_BRANCH) + 1


def test_training_dag_schedule_and_retries(dag) -> None:
    assert dag.catchup is False
    assert dag.default_args["retries"] >= 1
