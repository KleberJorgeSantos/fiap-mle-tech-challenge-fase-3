"""Shared fixtures. Everything runs offline on a small synthetic corpus, so the
suite is fast and works in CI without downloading the real dataset."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import get_settings, load_params
from src.pipelines import steps

# Vocabulary loosely inspired by each original category of the real corpus.
_VOCAB = {
    1: ["tumor", "carcinoma", "metastasis", "chemotherapy", "neoplasm", "biopsy", "lymphoma"],
    2: ["gastric", "liver", "colon", "hepatitis", "pancreatitis", "bowel", "ulcer"],
    3: ["stroke", "seizure", "cerebral", "neuropathy", "epilepsy", "brain", "hemorrhage"],
    4: ["myocardial", "infarction", "coronary", "arrhythmia", "hypertension", "cardiac", "artery"],
    5: ["inflammation", "tissue", "wound", "infection", "fever", "syndrome", "cells"],
}
_FILLER = ["patient", "study", "results", "treatment", "clinical", "observed", "with", "the"]


def make_raw_corpus(n_per_class: int = 60, seed: int = 7) -> pd.DataFrame:
    """Raw-format corpus (``condition_label``, ``medical_abstract``)."""
    rng = np.random.default_rng(seed)
    rows = []
    for label, words in _VOCAB.items():
        for _ in range(n_per_class):
            tokens = list(rng.choice(words, size=6)) + list(rng.choice(_FILLER, size=6))
            rng.shuffle(tokens)
            rows.append({"condition_label": label, "medical_abstract": " ".join(tokens).title()})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def params() -> dict:
    """Real params.yaml with limits scaled down for the synthetic corpus."""
    values = copy.deepcopy(load_params())
    values["data"]["min_rows"] = 100
    values["model"]["tfidf"]["max_features"] = 2000
    values["model"]["tfidf"]["min_df"] = 1
    return values


@pytest.fixture(scope="session")
def raw_corpus() -> pd.DataFrame:
    return make_raw_corpus()


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory, raw_corpus, params) -> dict:
    """Run the whole pipeline (minus download) once and share the artifacts."""
    data_dir = tmp_path_factory.mktemp("data")
    model_dir = tmp_path_factory.mktemp("models")
    raw_path = data_dir / "raw.csv"
    raw_corpus.to_csv(raw_path, index=False)

    dataset_path = steps.validate_data(str(raw_path), data_dir, params)
    splits = steps.preprocess(dataset_path, data_dir, params)
    model_path = steps.train(splits["train"], data_dir / "staging", params)
    metrics = steps.evaluate_model(model_path, splits["test"], params)
    parity = steps.export_onnx(model_path, splits["test"], params)
    version = steps.publish_model(
        data_dir / "staging", model_dir, metrics, parity, str(raw_path), params
    )
    return {
        "model_dir": model_dir,
        "staging_dir": data_dir / "staging",
        "splits": splits,
        "metrics": metrics,
        "parity": parity,
        "version": version,
    }


@pytest.fixture
def model_dir(trained_artifacts) -> Path:
    return trained_artifacts["model_dir"]


@pytest.fixture
def use_settings(monkeypatch):
    """Override environment-based settings for one test."""

    def _apply(**env: str) -> None:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()

    yield _apply
    get_settings.cache_clear()
