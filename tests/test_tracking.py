import pytest

from src.models.predictor import load_metadata
from src.pipelines import steps
from src.tracking import mlflow_tracking
from src.tracking.mlflow_tracking import CHAMPION_ALIAS, _flatten, log_training_run


@pytest.fixture
def file_store_uri(tmp_path, monkeypatch) -> str:
    # The filesystem backend is in maintenance mode in MLflow 3; fine for an offline test.
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return (tmp_path / "mlruns").as_uri()


def test_flatten_nested_params() -> None:
    assert _flatten({"tfidf": {"min_df": 2, "ngram": [1, 2]}, "C": 4}) == {
        "tfidf.min_df": 2,
        "tfidf.ngram": [1, 2],
        "C": 4,
    }


def test_tracking_disabled_without_uri(model_dir, use_settings) -> None:
    use_settings(MLFLOW_TRACKING_URI="")
    assert log_training_run(model_dir, load_metadata(model_dir)) is None


def test_tracking_skipped_when_server_is_down(model_dir) -> None:
    # Port 9 (discard) has nothing listening: must fail fast, not retry for minutes.
    uri = "http://127.0.0.1:9"
    assert log_training_run(model_dir, load_metadata(model_dir), tracking_uri=uri) is None


def test_logs_run_and_promotes_champion(model_dir, file_store_uri) -> None:
    import mlflow
    from mlflow import MlflowClient

    metadata = load_metadata(model_dir)
    run_id = log_training_run(model_dir, metadata, tracking_uri=file_store_uri)

    assert run_id is not None
    run = mlflow.get_run(run_id)
    assert run.data.metrics["macro_f1"] == metadata["metrics"]["macro_f1"]
    assert run.data.metrics["onnx_label_agreement"] >= 0.99
    assert run.data.params["classifier.C"] == str(metadata["params"]["classifier"]["C"])
    assert run.data.tags["model_version"] == metadata["model_version"]

    champion = MlflowClient().get_model_version_by_alias("triage-classifier", CHAMPION_ALIAS)
    assert champion.run_id == run_id


def test_track_experiment_never_fails_the_pipeline(model_dir, monkeypatch) -> None:
    def boom(*_args, **_kwargs):
        raise RuntimeError("tracking server exploded")

    monkeypatch.setattr(mlflow_tracking, "log_training_run", boom)
    assert steps.track_experiment(model_dir) is None
