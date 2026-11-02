"""Inference backends behind a common interface.

* ``SklearnPredictor`` -- original model (joblib), the latency baseline.
* ``OnnxPredictor``    -- same pipeline exported to ONNX, served by ONNX Runtime.

Only ``onnxruntime`` / ``joblib`` are needed at serving time; skl2onnx stays in the
training extra.
"""

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.config import METADATA_FILE, ONNX_MODEL_FILE, SKLEARN_MODEL_FILE

logger = logging.getLogger(__name__)


class Predictor(ABC):
    """Common interface: probabilities per class for a batch of texts."""

    backend: str
    classes: list[str]

    @abstractmethod
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return an array of shape ``(n_texts, n_classes)``."""

    def predict(self, texts: Sequence[str]) -> list[str]:
        """Return the most likely class of each text."""
        proba = self.predict_proba(texts)
        return [self.classes[i] for i in proba.argmax(axis=1)]

    def warmup(self, n: int = 5) -> None:
        """Run a few dummy inferences so the first real request is not slow."""
        for _ in range(n):
            self.predict_proba(["warmup chest pain and shortness of breath"])


class SklearnPredictor(Predictor):
    """Original scikit-learn pipeline loaded with joblib."""

    backend = "sklearn"

    def __init__(self, model_path: Path) -> None:
        import joblib

        self.pipeline = joblib.load(model_path)
        self.classes = [str(c) for c in self.pipeline.classes_]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.pipeline.predict_proba(list(texts))


class OnnxPredictor(Predictor):
    """Same pipeline exported to ONNX and executed by ONNX Runtime."""

    backend = "onnx"

    def __init__(self, model_path: Path, classes: list[str]) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # A single request is tiny: extra threads only add scheduling overhead.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        # Outputs are [label, probabilities]; only the probabilities are needed.
        self.proba_name = self.session.get_outputs()[1].name
        self.classes = classes

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        batch = np.asarray(list(texts), dtype=object).reshape(-1, 1)
        return self.session.run([self.proba_name], {self.input_name: batch})[0]


def load_metadata(model_dir: Path) -> dict[str, Any]:
    """Read ``metadata.json`` written by the training pipeline."""
    return json.loads((model_dir / METADATA_FILE).read_text(encoding="utf-8"))


def load_predictor(backend: str, model_dir: Path) -> Predictor:
    """Instantiate the requested backend from the artifacts in ``model_dir``."""
    if backend == "sklearn":
        return SklearnPredictor(model_dir / SKLEARN_MODEL_FILE)
    if backend == "onnx":
        classes = load_metadata(model_dir)["classes"]
        return OnnxPredictor(model_dir / ONNX_MODEL_FILE, classes=classes)
    raise ValueError(f"Unknown backend: {backend!r}")
