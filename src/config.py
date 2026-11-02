"""Central project configuration.

Two sources with distinct responsibilities:

* ``Settings`` -- environment variables (paths, backend, log level) loaded from ``.env``.
* ``params.yaml`` -- pipeline hyper-parameters, read by :func:`load_params`.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
PARAMS_PATH: Path = PROJECT_ROOT / "params.yaml"

# Urgency classes, ordered from least to most urgent. Single source of truth.
LABELS: list[str] = ["normal", "atencao", "urgente"]

TEXT_COLUMN: str = "text"
TARGET_COLUMN: str = "label"

SKLEARN_MODEL_FILE: str = "model.joblib"
ONNX_MODEL_FILE: str = "model.onnx"
METADATA_FILE: str = "metadata.json"


class Settings(BaseSettings):
    """Environment variables with safe defaults for local development."""

    # protected_namespaces=() allows field names starting with `model_`.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    model_backend: Literal["onnx", "sklearn"] = "onnx"
    model_dir: Path = PROJECT_ROOT / "models"
    data_dir: Path = PROJECT_ROOT / "data"
    reports_dir: Path = PROJECT_ROOT / "reports"
    max_text_length: int = 20_000
    log_level: str = "INFO"

    # Experiment tracking: disabled when no tracking URI is configured.
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "triage-nlp"
    mlflow_registered_model: str = "triage-classifier"

    # Cloud delivery: disabled unless a models bucket is configured.
    models_bucket: str | None = None
    aws_region: str = "us-east-1"
    github_repository: str | None = None  # owner/name, receives the repository_dispatch
    github_dispatch_token: str | None = None

    # TLS-inspecting networks re-sign HTTPS with their own CA, unknown to certifi.
    ca_bundle: Path | None = None

    @property
    def request_verify(self) -> str | bool:
        """``verify`` value for ``requests``: custom bundle or default."""
        return str(self.ca_bundle) if self.ca_bundle else True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached :class:`Settings` singleton."""
    return Settings()


@lru_cache(maxsize=1)
def load_params(path: Path | None = None) -> dict[str, Any]:
    """Load ``params.yaml`` as a dictionary."""
    target = path or PARAMS_PATH
    with target.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
