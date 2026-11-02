"""Champion x challenger: should a freshly trained model replace the one in production?

The quality gate (``metrics.check_quality_gate``) is *absolute*: "is this model good
enough?". This check is *relative*: "is it at least as good as the current champion?".
A retrain that passes the gate but scores below the champion must not be published.
"""

import logging
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)


def decide_promotion(
    challenger: dict[str, Any],
    champion: dict[str, Any] | None,
    metric_names: Sequence[str],
    tolerance: float,
) -> dict[str, Any]:
    """Compare challenger and champion metrics measured on the same test split.

    The challenger wins if **no** watched metric drops more than ``tolerance`` below
    the champion's. Ties are promoted: same quality with fresher data is preferred.

    Args:
        challenger: Metrics of the new model (output of ``evaluate``).
        champion: Metrics of the model in production, or ``None`` if there is none.
        metric_names: Metrics that must not get worse (e.g. macro_f1, recall_urgente).
        tolerance: Allowed drop, absorbing noise between retrains.

    Returns:
        ``{"promote": bool, "reason": str, "deltas": {metric: challenger - champion}}``
    """
    if champion is None:
        decision = {"promote": True, "reason": "no champion in production", "deltas": {}}
        logger.info("Promotion: %s", decision["reason"])
        return decision

    deltas = {name: round(challenger[name] - champion[name], 4) for name in metric_names}
    worse = {name: delta for name, delta in deltas.items() if delta < -tolerance}
    if worse:
        detail = ", ".join(f"{name} {delta:+.4f}" for name, delta in worse.items())
        reason = f"challenger worse than champion beyond tolerance {tolerance}: {detail}"
    else:
        detail = ", ".join(f"{name} {delta:+.4f}" for name, delta in deltas.items())
        reason = f"challenger not worse than champion ({detail})"

    decision = {"promote": not worse, "reason": reason, "deltas": deltas}
    log = logger.info if decision["promote"] else logger.warning
    log("Promotion: %s -> %s", "PROMOTE" if decision["promote"] else "REJECT", reason)
    return decision
