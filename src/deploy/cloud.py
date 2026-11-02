"""Hand-off of an approved model from the (local) training platform to the cloud.

    publish_model_to_s3:  models/<version>/ (immutable history) + models/champion/ (current)
    trigger_cd:           GitHub repository_dispatch "model-published" -> CD builds the
                          image with the champion model, pushes it to ECR, deploys to EC2

Both are opt-in: without ``MODELS_BUCKET`` / GitHub settings they log and return,
so the local pipeline keeps working with no AWS account.
"""

import json
import logging
import urllib.request
from pathlib import Path

from src.config import METADATA_FILE, ONNX_MODEL_FILE, SKLEARN_MODEL_FILE, get_settings

logger = logging.getLogger(__name__)

MODELS_PREFIX = "models"
CHAMPION = "champion"
DISPATCH_EVENT = "model-published"
_ARTIFACTS = (SKLEARN_MODEL_FILE, ONNX_MODEL_FILE, METADATA_FILE)


def publish_model_to_s3(model_dir: Path, version: str, s3_client=None) -> str | None:
    """Upload the published artifacts to S3.

    The champion is written last and ``metadata.json`` last within it, so a reader
    that sees the new metadata also sees the new model files.

    Returns:
        ``s3://`` URI of the champion prefix, or ``None`` when no bucket is configured.
    """
    settings = get_settings()
    if not settings.models_bucket:
        logger.info("MODELS_BUCKET not set: skipping S3 publication")
        return None

    if s3_client is None:
        import boto3

        s3_client = boto3.client("s3", region_name=settings.aws_region)

    bucket = settings.models_bucket
    for prefix in (f"{MODELS_PREFIX}/{version}", f"{MODELS_PREFIX}/{CHAMPION}"):
        for name in _ARTIFACTS:
            s3_client.upload_file(str(model_dir / name), bucket, f"{prefix}/{name}")
    uri = f"s3://{bucket}/{MODELS_PREFIX}/{CHAMPION}/"
    logger.info("Model %s published to %s", version, uri)
    return uri


def trigger_cd(version: str, opener=urllib.request.urlopen) -> bool:
    """Send a ``repository_dispatch`` so the CD workflow ships the new champion.

    Needs a token with permission to create repository dispatch events
    (fine-grained PAT: "Contents: Read and write" on the repository).
    """
    settings = get_settings()
    if not (settings.github_repository and settings.github_dispatch_token):
        logger.info("GITHUB_REPOSITORY/GITHUB_DISPATCH_TOKEN not set: skipping CD trigger")
        return False

    request = urllib.request.Request(
        f"https://api.github.com/repos/{settings.github_repository}/dispatches",
        method="POST",
        data=json.dumps(
            {"event_type": DISPATCH_EVENT, "client_payload": {"model_version": version}}
        ).encode(),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {settings.github_dispatch_token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with opener(request, timeout=15) as response:
        # GitHub answers 204 No Content on success
        logger.info("CD triggered for model %s (HTTP %s)", version, response.status)
    return True
