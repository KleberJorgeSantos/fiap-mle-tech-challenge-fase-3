import json

import pytest

from src.config import METADATA_FILE
from src.evaluation.promotion import decide_promotion
from src.pipelines import steps

WATCHED = ["macro_f1", "recall_urgente"]
CHAMPION = {"macro_f1": 0.73, "recall_urgente": 0.75}


def test_first_model_is_always_promoted() -> None:
    decision = decide_promotion(CHAMPION, None, WATCHED, tolerance=0.005)
    assert decision["promote"] is True
    assert decision["reason"] == "no champion in production"


def test_equal_metrics_are_promoted() -> None:
    assert decide_promotion(CHAMPION, CHAMPION, WATCHED, tolerance=0.005)["promote"] is True


def test_small_drop_within_tolerance_is_promoted() -> None:
    challenger = {"macro_f1": 0.727, "recall_urgente": 0.75}
    decision = decide_promotion(challenger, CHAMPION, WATCHED, tolerance=0.005)
    assert decision["promote"] is True
    assert decision["deltas"]["macro_f1"] == pytest.approx(-0.003)


@pytest.mark.parametrize(
    "challenger",
    [
        {"macro_f1": 0.70, "recall_urgente": 0.75},  # worse overall
        {"macro_f1": 0.75, "recall_urgente": 0.70},  # better F1 but misses more urgent reports
    ],
)
def test_any_watched_metric_beyond_tolerance_rejects(challenger) -> None:
    decision = decide_promotion(challenger, CHAMPION, WATCHED, tolerance=0.005)
    assert decision["promote"] is False
    assert "worse than champion" in decision["reason"]


def test_compare_with_champion_reads_production_metadata(model_dir, params) -> None:
    champion = json.loads((model_dir / METADATA_FILE).read_text())
    worse = {**champion["metrics"], "macro_f1": champion["metrics"]["macro_f1"] - 0.1}

    decision = steps.compare_with_champion(worse, model_dir, params)

    assert decision["promote"] is False
    assert decision["champion_version"] == champion["model_version"]


def test_compare_without_champion_promotes(tmp_path, params) -> None:
    decision = steps.compare_with_champion(
        {"macro_f1": 0.5, "recall_urgente": 0.5}, tmp_path, params
    )
    assert decision == {**decision, "promote": True, "champion_version": None}


def test_register_challenger_keeps_production_untouched(
    trained_artifacts, model_dir, params, monkeypatch
) -> None:
    from src.tracking import mlflow_tracking

    staging_dir = trained_artifacts["staging_dir"]
    production_before = (model_dir / METADATA_FILE).read_text()
    logged = {}

    def fake_log(model_dir_arg, metadata, **kwargs):
        logged.update(kwargs, model_dir=model_dir_arg)
        return "run-123"

    monkeypatch.setattr(mlflow_tracking, "log_training_run", fake_log)
    decision = {"promote": False, "reason": "worse", "champion_version": "v-champion"}
    run_id = steps.register_challenger(
        staging_dir,
        trained_artifacts["metrics"],
        trained_artifacts["parity"],
        decision,
        None,
        params,
    )

    assert run_id == "run-123"
    assert logged["alias"] == mlflow_tracking.CHALLENGER_ALIAS
    assert logged["extra_tags"]["promotion"] == "rejected"
    assert logged["model_dir"] == staging_dir
    assert (model_dir / METADATA_FILE).read_text() == production_before
