"""Named metric replacements must refresh scores and decisions during recovery."""

import numpy as np
import pytest
from sklearn.linear_model import SGDRegressor

from football_experiment_samples import prepared
from model_selection_samples import development, plan, sample
from test_football_experiment_search import CountingFixedFactory
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.evaluation import Metric, register_metric
import xdiyo_analytics.evaluation.metrics as registry
from xdiyo_analytics.experiments import ExperimentStore, FootballExperiment
from xdiyo_analytics.experiments.recovery import execution_key
from xdiyo_analytics.reporting import PerformanceReporter
from xdiyo_analytics.selection import Candidate, MetricSelection, ModelSelection
from xdiyo_analytics.training import PartialFitBackend


NAME = "recovery_custom_metric"


def mean_prediction(y, prediction, *, scale=1.):
    return scale * float(np.asarray(prediction).mean())


def negative_prediction(y, prediction, *, scale=1.):
    return -mean_prediction(y, prediction, scale=scale)


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, "METRICS", dict(registry.METRICS))
    register_metric(NAME, mean_prediction, kind="numeric", direction="minimize")


def reporter(metric_request):
    return PerformanceReporter(type="overall", partition="score", metrics=[metric_request] if isinstance(metric_request, (str, Metric)) else metric_request)


def candidates(models):
    return [Candidate(str(value), CountingFixedFactory(value, models), config={"constant": value})
            for value in (2., 10.)]


@pytest.mark.parametrize("metric_request", [NAME, [Metric(NAME, key="custom", parameters={"scale": 2.})]])
def test_replaced_report_metric_refreshes_final_and_keeps_old_saved_score(tmp_path, metric_request):
    bundle, models = prepared(), []
    experiment = FootballExperiment("Changed report calculation", output_dir=tmp_path)
    model = candidates(models)[0]
    analysis = PostTrainingAnalysis({"custom": reporter(metric_request)})
    first = experiment.run(bundle, model=model, post_analysis=analysis)
    original = first.record["metrics"][0]["value"]
    assert original > 0 and len(models) == 2
    register_metric(NAME, negative_prediction, kind="numeric", direction="minimize", replace=True)
    changed = experiment.run(bundle, model=model, post_analysis=analysis)
    assert changed.reused and changed.record["run_id"] == first.record["run_id"]
    assert changed.record["metrics"][0]["value"] == -original
    assert len(models) == 2
    assert experiment.load(first.record["run_id"]).record["metrics"][0]["value"] == -original
    assert experiment.run(bundle, model=model, post_analysis=analysis).reused
    assert len(models) == 2

    import json
    original_tables = json.loads((first.path / "tables.json").read_text(encoding="utf-8"))
    assert original_tables[0]["table"]["data"][0]["value"] == original


@pytest.mark.parametrize("change", ["function", "direction"])
@pytest.mark.parametrize("metric_request", [NAME, Metric(NAME, key="custom", parameters={"scale": 2.})])
def test_replaced_search_metric_rescores_saved_trials_and_changes_winner(tmp_path, change, metric_request):
    bundle, models = prepared(holdout=True), []
    experiment = FootballExperiment("Changed selection definition", output_dir=tmp_path)
    search = ModelSelection(candidates(models), metrics=metric_request)
    first = experiment.run(bundle, model_selection=search, selection_plan=plan(bundle.dataset))
    assert first.selection.winner.candidate.name == "2.0" and len(models) == 5
    saved = [trial.saved_run_id for trial in first.selection.trials]
    originals = {run_id: (experiment.store.path / "runs" / run_id / "run.json").read_bytes()
                 for run_id in saved}
    register_metric(NAME, negative_prediction if change == "function" else mean_prediction,
                    kind="numeric", direction="minimize" if change == "function" else "maximize", replace=True)
    changed = experiment.run(bundle, model_selection=search, selection_plan=plan(bundle.dataset))
    assert not changed.reused
    assert changed.record["run_group"] != first.record["run_group"]
    assert changed.selection.winner.candidate.name == "10.0"
    assert [trial.saved_run_id for trial in changed.selection.trials] == saved
    assert len(models) == 6  # Fresh final evaluation; candidate predictions are rescored.
    assert all(fold.model is None for trial in changed.selection.trials for fold in trial.training.folds)
    for run_id, original in originals.items():
        assert (experiment.store.path / "runs" / run_id / "run.json").read_bytes() == original
    assert experiment.run(bundle, model_selection=search, selection_plan=plan(bundle.dataset)).reused
    assert len(models) == 6


def test_standalone_search_group_tracks_resolved_definitions(tmp_path):
    data, models = sample(), []
    store = ExperimentStore(tmp_path, "Standalone registry definition")
    search = ModelSelection(candidates(models), metrics=NAME)
    first = search.run(data, plan(data), development_positions=development(data), experiment=store, resume=True)
    register_metric(NAME, mean_prediction, kind="numeric", direction="maximize", replace=True)
    changed = search.run(data, plan(data), development_positions=development(data), experiment=store, resume=True)
    assert changed.run_group != first.run_group
    assert first.winner.candidate.name == "2.0" and changed.winner.candidate.name == "10.0"
    assert [trial.saved_run_id for trial in changed.trials] == [trial.saved_run_id for trial in first.trials]
    assert len(models) == 4


def test_changed_report_input_kind_does_not_return_stale_final(tmp_path):
    bundle, models = prepared(), []
    experiment = FootballExperiment("Changed input kind", output_dir=tmp_path)
    model = candidates(models)[0]
    analysis = PostTrainingAnalysis({"custom": reporter(NAME)})
    experiment.run(bundle, model=model, post_analysis=analysis)
    register_metric(NAME, mean_prediction, kind="probability", direction="minimize", replace=True)
    # The replacement requires probabilities that this point predictor does not emit.
    with pytest.raises(KeyError, match="predict_proba"):
        experiment.run(bundle, model=model, post_analysis=analysis)
    assert len(experiment.store.read_runs(role="final")) == 1


def test_changed_custom_evidence_metric_invalidates_retained_evidence(tmp_path):
    data, models = sample(), []
    store = ExperimentStore(tmp_path, "Retained custom evidence")
    search = ModelSelection(candidates(models), metrics=(), decision=MetricSelection(NAME),
                            evidence_reporters={"custom": reporter(NAME)})
    first = search.run(data, plan(data), development_positions=development(data), experiment=store, resume=True)
    register_metric(NAME, negative_prediction, kind="numeric", direction="minimize", replace=True)
    changed = search.run(data, plan(data), development_positions=development(data), experiment=store, resume=True)
    assert first.winner.candidate.name == "2.0" and changed.winner.candidate.name == "10.0"
    assert {t.saved_run_id for t in first.trials}.isdisjoint(t.saved_run_id for t in changed.trials)
    assert len(models) == 8


def test_unrelated_registered_metric_keeps_final_reuse(tmp_path):
    bundle, models = prepared(), []
    experiment = FootballExperiment("Only requested definitions", output_dir=tmp_path)
    model = candidates(models)[0]
    analysis = PostTrainingAnalysis({"custom": reporter(NAME)})
    first = experiment.run(bundle, model=model, post_analysis=analysis)
    register_metric("unused_recovery_metric", negative_prediction, kind="numeric", direction="maximize")
    again = experiment.run(bundle, model=model, post_analysis=analysis)
    assert again.reused and again.record["run_id"] == first.record["run_id"]
    assert len(models) == 2


@pytest.mark.parametrize("loss", [NAME, Metric(NAME)])
def test_explicit_training_loss_definition_changes_fitting_identity(loss):
    backend = PartialFitBackend(SGDRegressor(), loss=loss)
    before = execution_key(backend)
    register_metric(NAME, negative_prediction, kind="numeric", direction="minimize", replace=True)
    assert execution_key(backend) != before
