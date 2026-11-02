import pytest
from fastapi.testclient import TestClient

from src.api import main as api

URGENT_TEXT = "Myocardial infarction with coronary artery occlusion and cardiac arrhythmia"


@pytest.fixture
def client(model_dir, use_settings):
    use_settings(MODEL_DIR=str(model_dir), MODEL_BACKEND="onnx")
    with TestClient(api.app) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(tmp_path, use_settings):
    use_settings(MODEL_DIR=str(tmp_path))
    api._state.update(predictor=None, metadata={})
    with TestClient(api.app) as test_client:
        yield test_client


def test_health_reports_ok_with_model(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["backend"] == "onnx"


def test_health_reports_degraded_without_model(client_without_model) -> None:
    assert client_without_model.get("/health").json()["status"] == "degraded"


def test_predict_returns_503_without_model(client_without_model) -> None:
    response = client_without_model.post("/predict", json={"text": URGENT_TEXT})
    assert response.status_code == 503


def test_predict_classifies_urgent_report(client) -> None:
    response = client.post("/predict", json={"text": URGENT_TEXT})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "urgente"
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
    assert body["latency_ms"] > 0


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "!!! ???"}, {}, {"text": 123}])
def test_predict_rejects_invalid_payloads(client, payload) -> None:
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_too_long_text(client, use_settings) -> None:
    use_settings(MAX_TEXT_LENGTH="10")
    assert client.post("/predict", json={"text": "tumor " * 5}).status_code == 422


def test_predict_batch(client) -> None:
    texts = [URGENT_TEXT, "Gastric ulcer and liver hepatitis", "Tumor biopsy carcinoma"]
    body = client.post("/predict/batch", json={"texts": texts}).json()
    assert [p["label"] for p in body["predictions"]] == ["urgente", "atencao", "atencao"]


def test_sklearn_backend_gives_same_answer(model_dir, use_settings) -> None:
    use_settings(MODEL_DIR=str(model_dir), MODEL_BACKEND="sklearn")
    with TestClient(api.app) as sk_client:
        body = sk_client.post("/predict", json={"text": URGENT_TEXT}).json()
    assert body["backend"] == "sklearn"
    assert body["label"] == "urgente"


def test_reload_keeps_serving(client) -> None:
    assert client.post("/admin/reload").json()["model_loaded"] is True


def test_metrics_endpoint_exposes_custom_metrics(client) -> None:
    client.post("/predict", json={"text": URGENT_TEXT})
    client.post("/predict", json={"text": ""})
    text = client.get("/metrics").text
    assert 'http_requests_total{endpoint="/predict",method="POST",status="200"}' in text
    assert 'status="422"' in text
    assert "http_request_duration_seconds_bucket" in text
    assert 'model_inference_seconds_count{backend="onnx"}' in text
    assert 'predictions_total{label="urgente"}' in text
