"""Exact fit membership and the existing temporal fold's information boundary."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.ratings import BayesianScoreSerializer
from xdiyo_analytics.splits import TemporalSplit
from xdiyo_analytics.training import fit_predict
from test_bayesian_training import adapter, dataset, fit_context, prediction_context, synthetic_history


@pytest.mark.parametrize("timing_fields", [("training_boundary",), ("fit_at",), ("training_boundary", "fit_at")])
@pytest.mark.parametrize("source_history", [False, True])
def test_adapter_rejects_selected_results_released_after_declared_boundary(timing_fields, source_history):
    history = synthetic_history(seasons=1, matches=3)
    data = dataset(history)
    fit_at = data.metadata.kickoff_at.iloc[2] - pd.Timedelta(days=2)
    data.metadata["released_at"] = data.metadata.kickoff_at + pd.Timedelta(hours=3)
    data.metadata.loc[1, "released_at"] = fit_at + pd.Timedelta(days=1)
    context = fit_context(data, [0, 1])
    context.fold_metadata.update({name: fit_at for name in timing_fields}, cutoffs="explicit")
    release_by_event = data.metadata.set_index("event_id").released_at
    history["released_at"] = history.event_id.map(release_by_event)
    model = adapter(history if source_history else None, available_at="released_at")
    with pytest.raises(ValueError, match="declared fold training boundary"):
        model.fit(context)
    assert not hasattr(model, "run_")  # No silent row dropping or forward shift.


def test_gap_boundary_is_stricter_than_fit_time_and_includes_release_ties():
    data = dataset(synthetic_history(seasons=1, matches=3))
    fit_at = data.metadata.kickoff_at.iloc[2]
    boundary = fit_at - pd.Timedelta(days=2)
    data.metadata["released_at"] = data.metadata.kickoff_at
    data.metadata.loc[1, "released_at"] = boundary + pd.Timedelta(hours=1)
    context = fit_context(data, [0, 1])
    context.fold_metadata.update(training_boundary=boundary, fit_at=fit_at)
    with pytest.raises(ValueError, match="declared fold training boundary"):
        adapter(available_at="released_at").fit(context)
    context.metadata.loc[1, "released_at"] = boundary
    model = adapter(available_at="released_at")
    model.fit(context)
    assert model.training_summary_["population"]["matches"] == 2
    assert model.training_summary_["cutoff"] == boundary.isoformat()
    assert pd.Timestamp(model.model_.training_cutoff) == fit_at
    prediction = prediction_context(data, [2])
    prediction.fold_metadata.update(context.fold_metadata)
    assert model.predict(prediction)["predict"].notna().all().all()


def test_kickoff_tie_is_excluded_even_when_release_meets_boundary():
    data = dataset(synthetic_history(seasons=1, matches=3))
    context = fit_context(data, [0, 1])
    context.fold_metadata["training_boundary"] = context.metadata.kickoff_at.max()
    with pytest.raises(ValueError, match="declared fold training boundary"):
        adapter().fit(context)


@pytest.mark.parametrize("timing", [
    {"training_boundary": "2020-08-14", "fit_at": "2020-08-13"},
    {"training_boundary": "invalid"}, {"fit_at": pd.NaT},
    {"fit_at": ["2020-08-13", "2020-08-14"]}, {"training_boundary": None},
    {"fit_at": np.int64(1597312800000000000)},
])
def test_invalid_declared_timing_is_rejected(timing):
    data = dataset(synthetic_history(seasons=1))
    context = fit_context(data, [0, 1])
    context.fold_metadata.update(timing)
    with pytest.raises(ValueError, match="fold_metadata"):
        adapter().fit(context)


def test_prediction_uses_declared_time_even_when_fixture_kickoff_is_later(tmp_path):
    data = dataset(synthetic_history(seasons=1, matches=3))
    data.metadata["released_at"] = data.metadata.kickoff_at
    data.metadata.loc[1, "released_at"] = pd.Timestamp("2020-08-14T10:00Z")
    # Standalone fit keeps the existing actual-availability-derived boundary.
    model = adapter(available_at="released_at")
    model.fit(fit_context(data, [0, 1]))
    prediction = prediction_context(data, [2])
    assert model.predict(prediction)["predict"].notna().all().all()
    serializer = BayesianScoreSerializer()
    serializer.save(model, tmp_path / "fitted")
    restored = serializer.load(tmp_path / "fitted")
    for candidate in (model, restored):
        prediction.fold_metadata = {"fit_at": pd.Timestamp("2020-08-13T10:00Z"), "cutoffs": "explicit"}
        with pytest.raises(ValueError, match="declared prediction fit_at"):
            candidate.predict(prediction)
        prediction.fold_metadata = {"training_boundary": pd.Timestamp("2020-08-13T10:00Z")}
        with pytest.raises(ValueError, match="declared prediction fold training boundary"):
            candidate.predict(prediction)


def test_declared_activation_survives_native_reload_and_parameter_export(tmp_path):
    data = dataset(synthetic_history(seasons=1, matches=3))
    context = fit_context(data, [0, 1])
    context.fold_metadata.update(training_boundary="2020-08-12T10:00Z", fit_at="2020-08-14T10:00Z")
    model = adapter()
    model.fit(context)
    assert pd.Timestamp(model.model_.training_cutoff) == pd.Timestamp(context.fold_metadata["fit_at"])
    serializer = BayesianScoreSerializer()
    serializer.save(model, tmp_path / "activated")
    restored = serializer.load(tmp_path / "activated")
    assert restored.model_ == model.model_
    assert restored.training_summary_["fit_timing"] == model.training_summary_["fit_timing"]
    prediction = prediction_context(data, [2])
    prediction.fold_metadata.update(context.fold_metadata)
    for name, values in model.predict(prediction).items():
        pd.testing.assert_frame_equal(restored.predict(prediction)[name], values)
    too_early = deepcopy(prediction)
    too_early.fold_metadata = {}
    too_early.metadata["kickoff_at"] = pd.Timestamp("2020-08-13T10:00Z")
    for candidate in (model, restored):
        with pytest.raises(ValueError, match="trained after"):
            candidate.predict(too_early)


def test_temporal_runner_detects_adapter_and_split_availability_mismatch():
    data = dataset(synthetic_history(seasons=1, matches=3))
    fit_at = data.metadata.kickoff_at.iloc[2] - pd.Timedelta(days=2)
    data.metadata["released_at"] = data.metadata.kickoff_at + pd.Timedelta(hours=3)
    data.metadata.loc[1, "released_at"] = fit_at + pd.Timedelta(days=1)
    cutoffs = data.metadata.kickoff_at - pd.Timedelta(days=2)
    splitter = TemporalSplit(train_size=2, unit="kickoffs")
    proxy_fold = splitter.folds(data, cutoffs=cutoffs)[0]
    assert proxy_fold.train.tolist() == [0, 1]
    with pytest.raises(ValueError, match="declared fold training boundary"):
        fit_predict(data, proxy_fold, lambda: adapter(available_at="released_at"))
    valid_fold = splitter.folds(data, cutoffs=cutoffs, available_at="released_at")[0]
    assert valid_fold.train.tolist() == [0]
    result = fit_predict(data, valid_fold, lambda: adapter(available_at="released_at"))
    assert result.model.training_summary_["population"]["matches"] == 1
    assert result.predictions["predict"].notna().all().all()
