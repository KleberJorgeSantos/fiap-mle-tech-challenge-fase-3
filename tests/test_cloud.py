import json

import pytest

from src.deploy.cloud import DISPATCH_EVENT, publish_model_to_s3, trigger_cd
from src.pipelines import steps


class FakeS3:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str]] = []

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.uploads.append((bucket, key))


class FakeResponse:
    status = 204

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        return None


@pytest.fixture
def cloud_settings(use_settings):
    use_settings(
        MODELS_BUCKET="triage-models",
        GITHUB_REPOSITORY="owner/repo",
        GITHUB_DISPATCH_TOKEN="token-123",
    )


def test_publish_skipped_without_bucket(model_dir, use_settings) -> None:
    use_settings(MODELS_BUCKET="")
    assert publish_model_to_s3(model_dir, "v1", s3_client=FakeS3()) is None


def test_publish_writes_version_and_champion(model_dir, cloud_settings) -> None:
    s3 = FakeS3()
    uri = publish_model_to_s3(model_dir, "20260101", s3_client=s3)

    assert uri == "s3://triage-models/models/champion/"
    keys = [key for _, key in s3.uploads]
    assert "models/20260101/model.onnx" in keys
    assert "models/champion/model.onnx" in keys
    # Champion last, metadata last within it: readers never see new metadata with old files.
    assert keys[-1] == "models/champion/metadata.json"
    assert {bucket for bucket, _ in s3.uploads} == {"triage-models"}


def test_trigger_cd_skipped_without_github_settings(use_settings) -> None:
    use_settings(GITHUB_REPOSITORY="", GITHUB_DISPATCH_TOKEN="")
    assert trigger_cd("v1", opener=lambda *_a, **_k: pytest.fail("must not call GitHub")) is False


def test_trigger_cd_sends_repository_dispatch(cloud_settings) -> None:
    captured = {}

    def opener(request, timeout):
        captured["request"] = request
        return FakeResponse()

    assert trigger_cd("20260101", opener=opener) is True
    request = captured["request"]
    assert request.full_url == "https://api.github.com/repos/owner/repo/dispatches"
    assert request.get_header("Authorization") == "Bearer token-123"
    body = json.loads(request.data)
    assert body == {"event_type": DISPATCH_EVENT, "client_payload": {"model_version": "20260101"}}


def test_publish_to_cloud_is_noop_when_not_configured(model_dir, use_settings) -> None:
    use_settings(MODELS_BUCKET="")
    assert steps.publish_to_cloud(model_dir) is None


def test_publish_to_cloud_uploads_then_triggers(model_dir, cloud_settings, monkeypatch) -> None:
    from src.deploy import cloud

    calls = []
    monkeypatch.setattr(
        cloud, "publish_model_to_s3", lambda d, v: calls.append(("s3", v)) or "s3://x/"
    )
    monkeypatch.setattr(cloud, "trigger_cd", lambda v: calls.append(("cd", v)) or True)

    assert steps.publish_to_cloud(model_dir) == "s3://x/"
    assert [name for name, _ in calls] == ["s3", "cd"]
