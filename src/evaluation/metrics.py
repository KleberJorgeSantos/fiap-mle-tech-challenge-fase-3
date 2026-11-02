"""Classification metrics and the quality gate used by the pipeline and the DAG."""

import logging
from collections.abc import Sequence
from typing import Any

from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import LABELS

logger = logging.getLogger(__name__)


class QualityGateError(RuntimeError):
    """Raised when a trained model does not meet the minimum quality bar."""


def evaluate(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    """Compute accuracy, macro-F1, per-class F1 and urgent-class recall."""
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=LABELS, average="macro")), 4),
        "f1_per_class": {label: round(report[label]["f1-score"], 4) for label in LABELS},
        # Missing an urgent report is the costliest error in triage.
        "recall_urgente": round(report["urgente"]["recall"], 4),
    }
    logger.info("Evaluation: %s", metrics)
    return metrics


def check_quality_gate(metrics: dict[str, Any], min_macro_f1: float) -> None:
    """Block publication of models below the minimum macro-F1."""
    if metrics["macro_f1"] < min_macro_f1:
        raise QualityGateError(
            f"macro_f1={metrics['macro_f1']:.4f} below threshold {min_macro_f1:.2f}"
        )
    logger.info("Quality gate passed (macro_f1 >= %.2f)", min_macro_f1)
