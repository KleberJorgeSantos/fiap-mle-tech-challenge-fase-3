"""Conversion of the sklearn pipeline to ONNX and parity validation."""

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
from sklearn.pipeline import Pipeline

from src.models.predictor import OnnxPredictor

logger = logging.getLogger(__name__)

# sklearn's default token_pattern keeps only tokens of 2+ word chars, but skl2onnx
# emits "[a-zA-Z0-9_]+", which keeps 1-char tokens and shifts the bigrams. Texts are
# ASCII-folded by clean_text, so this ASCII regex reproduces sklearn exactly.
ONNX_TOKEN_REGEX = "[a-zA-Z0-9_][a-zA-Z0-9_]+"


class OnnxParityError(RuntimeError):
    """Raised when ONNX predictions diverge from the original sklearn model."""


def export_to_onnx(pipeline: Pipeline, output_path: Path, opset: int = 15) -> Path:
    """Serialize the full pipeline (TF-IDF + classifier) as a single ONNX graph.

    ``zipmap=False`` makes the classifier return a plain probability tensor instead
    of a list of dicts, which is faster and simpler to consume.
    """
    onnx_model = to_onnx(
        pipeline,
        initial_types=[("text", StringTensorType([None, 1]))],
        target_opset={"": opset, "ai.onnx.ml": 3},
        options={
            id(pipeline.named_steps["tfidf"]): {"tokenexp": ONNX_TOKEN_REGEX},
            id(pipeline.named_steps["clf"]): {"zipmap": False},
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(onnx_model.SerializeToString())
    logger.info("ONNX model saved to %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
    return output_path


def check_parity(
    pipeline: Pipeline, onnx_path: Path, texts: Sequence[str], min_parity: float = 0.99
) -> dict[str, float]:
    """Compare sklearn vs ONNX Runtime predictions on the same texts.

    Raises:
        OnnxParityError: If label agreement is below ``min_parity``.
    """
    texts = list(texts)
    sk_proba = pipeline.predict_proba(texts)
    onnx_proba = OnnxPredictor(onnx_path, classes=list(pipeline.classes_)).predict_proba(texts)

    agreement = float(np.mean(sk_proba.argmax(axis=1) == onnx_proba.argmax(axis=1)))
    max_abs_diff = float(np.max(np.abs(sk_proba - onnx_proba)))
    result = {"label_agreement": round(agreement, 4), "max_proba_diff": round(max_abs_diff, 6)}
    logger.info("ONNX parity: %s", result)
    if agreement < min_parity:
        raise OnnxParityError(f"label agreement {agreement:.4f} < {min_parity}")
    return result
