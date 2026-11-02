"""Download the Medical Abstracts TC Corpus from GitHub (public, no login)."""

import io
import logging
from pathlib import Path

import pandas as pd
import requests
import truststore

from src.config import get_settings, load_params

logger = logging.getLogger(__name__)

RAW_FILE_NAME = "medical_abstracts.csv"


def download_dataset(output_dir: Path | None = None, force: bool = False) -> Path:
    """Fetch the train/test CSVs and store them as a single raw CSV.

    Args:
        output_dir: Destination folder. Defaults to ``<DATA_DIR>/raw``.
        force: Download again even if the file already exists.

    Returns:
        Path of the combined raw CSV.
    """
    settings = get_settings()
    params = load_params()["data"]
    output_dir = output_dir or settings.data_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / RAW_FILE_NAME

    # Validate HTTPS with the OS trust store (works behind TLS-inspecting proxies).
    truststore.inject_into_ssl()

    if target.exists() and not force:
        logger.info("Raw dataset already present at %s", target)
        return target

    frames = []
    for file_name in params["files"]:
        url = f"{params['base_url']}/{file_name}"
        logger.info("Downloading %s", url)
        response = requests.get(url, timeout=60, verify=settings.request_verify)
        response.raise_for_status()
        frames.append(pd.read_csv(io.StringIO(response.text)))

    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(target, index=False)
    logger.info("Saved %d rows to %s", len(raw), target)
    return target
