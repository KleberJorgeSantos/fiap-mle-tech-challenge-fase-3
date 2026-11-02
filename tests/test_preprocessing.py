import pandas as pd
import pytest

from src.config import LABELS, TARGET_COLUMN, TEXT_COLUMN
from src.data.preprocessing import (
    file_sha256,
    map_labels,
    split_dataset,
    validate_dataset,
)
from src.data.text import clean_text


def test_clean_text_normalizes_case_spaces_and_accents() -> None:
    assert clean_text("  Dor   TORÁCICA\n aguda ") == "dor toracica aguda"


def test_map_labels_produces_urgency_schema(raw_corpus, params) -> None:
    frame = map_labels(raw_corpus, params)
    assert list(frame.columns) == [TEXT_COLUMN, TARGET_COLUMN]
    assert set(frame[TARGET_COLUMN]) == set(LABELS)


def test_map_labels_follows_params_mapping(params) -> None:
    raw = pd.DataFrame(
        {"condition_label": [4, 5, 1], "medical_abstract": ["heart attack", "fever", "tumor"]}
    )
    assert map_labels(raw, params)[TARGET_COLUMN].tolist() == ["urgente", "normal", "atencao"]


def test_map_labels_drops_unmapped_and_duplicates(params) -> None:
    raw = pd.DataFrame(
        {"condition_label": [9, 5, 5], "medical_abstract": ["unknown", "fever", "FEVER"]}
    )
    assert len(map_labels(raw, params)) == 1


def test_validate_dataset_accepts_valid_frame(raw_corpus, params) -> None:
    distribution = validate_dataset(map_labels(raw_corpus, params), min_rows=100)
    assert set(distribution) == set(LABELS)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"text": ["a"]}), "Missing columns"),
        (pd.DataFrame({"text": ["a"], "label": ["normal"]}), "minimum"),
        (pd.DataFrame({"text": ["a", "b", "c"], "label": ["normal", "atencao", "x"]}), "Unknown"),
        (pd.DataFrame({"text": ["a", "b"], "label": ["normal", "atencao"]}), "without samples"),
    ],
)
def test_validate_dataset_rejects_contract_violations(frame, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_dataset(frame, min_rows=2 if "Missing" not in message else 1)


def test_split_dataset_is_stratified(raw_corpus, params) -> None:
    frame = map_labels(raw_corpus, params)
    train, test = split_dataset(frame, test_size=0.2, random_state=42)
    assert len(train) + len(test) == len(frame)
    assert set(test[TARGET_COLUMN]) == set(LABELS)


def test_file_sha256_is_stable(tmp_path) -> None:
    path = tmp_path / "f.txt"
    path.write_text("abc")
    assert file_sha256(path) == file_sha256(path)
    assert len(file_sha256(path)) == 64
