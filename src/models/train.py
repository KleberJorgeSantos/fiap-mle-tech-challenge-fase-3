"""Training of the TF-IDF + Logistic Regression triage classifier."""

import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import TARGET_COLUMN, TEXT_COLUMN

logger = logging.getLogger(__name__)


def build_pipeline(model_params: dict[str, Any], random_state: int = 42) -> Pipeline:
    """Create the untrained sklearn pipeline.

    The default ``token_pattern`` is kept on purpose: skl2onnx translates it into
    the ONNX ``Tokenizer`` regex, so custom tokenizers would break the export.

    ``lowercase=False`` because ``clean_text`` already lowercases; otherwise the
    ONNX graph gets a ``StringNormalizer`` op that requires the ``en_US.UTF-8``
    locale, absent from slim container images.
    """
    tfidf = model_params["tfidf"]
    clf = model_params["classifier"]
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=tuple(tfidf["ngram_range"]),
                    max_features=tfidf["max_features"],
                    min_df=tfidf["min_df"],
                    sublinear_tf=tfidf["sublinear_tf"],
                    lowercase=False,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=clf["C"],
                    max_iter=clf["max_iter"],
                    class_weight=clf["class_weight"],
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_model(train: pd.DataFrame, model_params: dict[str, Any]) -> Pipeline:
    """Fit the pipeline on the training split."""
    pipeline = build_pipeline(model_params)
    logger.info("Training on %d samples", len(train))
    pipeline.fit(train[TEXT_COLUMN].tolist(), train[TARGET_COLUMN].tolist())
    logger.info("Vocabulary size: %d", len(pipeline.named_steps["tfidf"].vocabulary_))
    return pipeline


def save_model(pipeline: Pipeline, path: Path) -> Path:
    """Serialize the pipeline so it loads in any environment (``joblib.load`` reads it).

    Uses the C ``pickle`` implementation on purpose. Airflow imports ``dill``, which
    registers its reducers in the pure-Python pickler at import time, and joblib copies
    that dispatch table when *it* is imported. Inside an Airflow task (dill imported
    first) ``joblib.dump`` therefore writes references to ``dill._dill`` and the model
    fails to load where dill is absent: the API image, MLflow, a laptop. The C pickler
    ignores that table.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path
