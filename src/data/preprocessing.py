"""Label mapping, cleaning, validation and train/test split."""

import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import LABELS, TARGET_COLUMN, TEXT_COLUMN
from src.data.text import clean_text

logger = logging.getLogger(__name__)


def map_labels(raw: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Convert the raw corpus into the ``text`` / ``label`` schema.

    Args:
        raw: Raw corpus with the original text and disease-category columns.
        params: Full ``params.yaml`` content (uses ``data`` and ``label_mapping``).

    Returns:
        DataFrame with cleaned ``text`` and urgency ``label`` columns.
    """
    data_params = params["data"]
    mapping = {int(k): v for k, v in params["label_mapping"].items()}

    frame = pd.DataFrame(
        {
            TEXT_COLUMN: raw[data_params["text_column"]].map(clean_text),
            TARGET_COLUMN: raw[data_params["label_column"]].astype(int).map(mapping),
        }
    )
    unmapped = frame[TARGET_COLUMN].isna().sum()
    if unmapped:
        logger.warning("Dropping %d rows with unmapped labels", unmapped)
    frame = frame.dropna(subset=[TARGET_COLUMN])
    frame = frame[frame[TEXT_COLUMN].str.len() > 0]
    return frame.drop_duplicates(subset=[TEXT_COLUMN]).reset_index(drop=True)


def validate_dataset(frame: pd.DataFrame, min_rows: int) -> dict[str, int]:
    """Data-contract checks run before training.

    Raises:
        ValueError: If the schema, size or label set is not what the model expects.

    Returns:
        Number of samples per label.
    """
    missing = {TEXT_COLUMN, TARGET_COLUMN} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(frame) < min_rows:
        raise ValueError(f"Dataset has {len(frame)} rows, minimum is {min_rows}")
    unknown = set(frame[TARGET_COLUMN].unique()) - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown labels: {sorted(unknown)}")
    absent = set(LABELS) - set(frame[TARGET_COLUMN].unique())
    if absent:
        raise ValueError(f"Labels without samples: {sorted(absent)}")
    if frame[TEXT_COLUMN].isna().any():
        raise ValueError("Null texts found")

    distribution = frame[TARGET_COLUMN].value_counts().to_dict()
    logger.info("Dataset OK: %d rows, distribution=%s", len(frame), distribution)
    return {str(k): int(v) for k, v in distribution.items()}


def split_dataset(
    frame: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split."""
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=frame[TARGET_COLUMN],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def file_sha256(path: Path) -> str:
    """Hash of a data file, recorded in the model metadata for lineage."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
