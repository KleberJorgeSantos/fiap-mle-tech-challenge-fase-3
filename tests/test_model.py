import json

import numpy as np
import pandas as pd
import pytest

from src.config import LABELS, METADATA_FILE, ONNX_MODEL_FILE, SKLEARN_MODEL_FILE, TEXT_COLUMN
from src.evaluation.metrics import QualityGateError, check_quality_gate, evaluate
from src.models.export_onnx import OnnxParityError, check_parity
from src.models.predictor import OnnxPredictor, SklearnPredictor, load_predictor


def test_pipeline_publishes_all_artifacts(model_dir) -> None:
    for name in (SKLEARN_MODEL_FILE, ONNX_MODEL_FILE, METADATA_FILE):
        assert (model_dir / name).exists()


def test_metadata_records_metrics_and_lineage(model_dir) -> None:
    metadata = json.loads((model_dir / METADATA_FILE).read_text())
    assert set(metadata["classes"]) == set(LABELS)
    assert metadata["metrics"]["macro_f1"] > 0.8
    assert len(metadata["dataset_sha256"]) == 64


def test_onnx_matches_sklearn(trained_artifacts) -> None:
    assert trained_artifacts["parity"]["label_agreement"] >= 0.99


def test_backends_return_same_labels(model_dir, trained_artifacts) -> None:
    texts = pd.read_csv(trained_artifacts["splits"]["test"])[TEXT_COLUMN].tolist()
    sklearn_model = load_predictor("sklearn", model_dir)
    onnx_model = load_predictor("onnx", model_dir)
    assert isinstance(sklearn_model, SklearnPredictor)
    assert isinstance(onnx_model, OnnxPredictor)
    assert sklearn_model.predict(texts) == onnx_model.predict(texts)
    np.testing.assert_allclose(
        sklearn_model.predict_proba(texts), onnx_model.predict_proba(texts), atol=1e-3
    )


def test_load_predictor_rejects_unknown_backend(model_dir) -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        load_predictor("torch", model_dir)


def test_check_parity_raises_when_models_diverge(model_dir, trained_artifacts) -> None:
    import joblib

    pipeline = joblib.load(model_dir / SKLEARN_MODEL_FILE)
    texts = pd.read_csv(trained_artifacts["splits"]["test"])[TEXT_COLUMN].tolist()
    # Reversed class order makes ONNX argmax point to other labels.
    pipeline.named_steps["clf"].coef_ = -pipeline.named_steps["clf"].coef_
    with pytest.raises(OnnxParityError):
        check_parity(pipeline, model_dir / ONNX_MODEL_FILE, texts)


def test_evaluate_reports_expected_keys() -> None:
    metrics = evaluate(["normal", "urgente", "atencao"], ["normal", "urgente", "normal"])
    assert metrics["recall_urgente"] == 1.0
    assert set(metrics["f1_per_class"]) == set(LABELS)


def test_quality_gate_blocks_weak_models() -> None:
    with pytest.raises(QualityGateError):
        check_quality_gate({"macro_f1": 0.3}, min_macro_f1=0.6)
    check_quality_gate({"macro_f1": 0.7}, min_macro_f1=0.6)


def test_onnx_graph_has_no_locale_dependent_ops(model_dir) -> None:
    """StringNormalizer needs the en_US.UTF-8 locale, missing in slim images."""
    import onnx

    graph = onnx.load(str(model_dir / ONNX_MODEL_FILE)).graph
    assert "StringNormalizer" not in {node.op_type for node in graph.node}


def test_save_model_ignores_reducers_injected_into_the_python_pickler(
    model_dir, tmp_path, monkeypatch
) -> None:
    """Simulates Airflow's dill: reducers injected into the pure-Python pickler tables
    (the stock one and joblib's copy) must not leak into the saved model."""
    import pickle

    import joblib
    from joblib import numpy_pickle

    from src.models.train import save_model

    def poisoned(*_args, **_kwargs):
        raise AssertionError("pure-Python pickler dispatch must not be used")

    for table in (pickle._Pickler.dispatch, numpy_pickle.NumpyPickler.dispatch):
        monkeypatch.setitem(table, dict, poisoned)

    target = save_model(joblib.load(model_dir / SKLEARN_MODEL_FILE), tmp_path / "m.joblib")

    restored = joblib.load(target)
    assert list(restored.classes_) == sorted(LABELS)
