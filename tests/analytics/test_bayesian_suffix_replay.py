"""Suffix checkpoints must match global chronological-prefix replay exactly."""
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import json

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.ratings import BayesianModel, BayesianConfig, build_bayesian_ratings
from xdiyo_analytics.ratings.bayesian_calibration import CalibrationReplay, _States, _suffix_versions
from xdiyo_analytics.ratings import bayesian_replay as replay
from test_bayesian_calibration_activity import exchange_history
from test_bayesian_calibration_fast import PARAMETERS, assert_evidence
from test_bayesian_kickoff_replay import _history
from xdiyo_analytics.ratings.bayesian_training import DEFAULT_FIT_FIELDS

spec = spec_from_file_location('refilter_reference', Path(__file__).with_name('fixtures') / 'bayesian_refilter_6f9c2fa.py')
REFERENCE = module_from_spec(spec)
spec.loader.exec_module(REFERENCE)


@pytest.mark.parametrize('parameters', PARAMETERS)
@pytest.mark.parametrize('transition', ['mirrored', 'bridge'])
@pytest.mark.parametrize('stride', [1, 128])
def test_native_publication_and_checkpoint_equal_global_reference(monkeypatch, parameters, transition, stride):
    import xdiyo_analytics.ratings.bayesian_calibration as calibration
    monkeypatch.setattr(calibration, '_suffix_versions',
        lambda history, model, versions: _suffix_versions(history, model, versions, stride=stride))
    history, movements = exchange_history(seasons=2)
    history['release'] = history.kickoff_at
    history.loc[history.event_id.eq(history.event_id.iloc[0]), 'release'] += pd.Timedelta('9d')
    model = BayesianModel(parameters=parameters, config=BayesianConfig(transition=transition, bridge_gaps=((1, 2, .2),)))
    actual = build_bayesian_ratings(history, model=model, available_at='release', team_seasons=movements)
    with monkeypatch.context() as patch:
        patch.setattr(replay, '_refilter_versions', REFERENCE._refilter_versions)
        expected = build_bayesian_ratings(history, model=model, available_at='release', team_seasons=movements)
    for name in ('snapshots', 'league_snapshots', 'predictions'):
        pd.testing.assert_frame_equal(getattr(actual, name), getattr(expected, name), check_exact=True)
    assert actual.checkpoint == expected.checkpoint
    assert actual.metadata == expected.metadata
    plan = CalibrationReplay(history, config=model.config, cutoff='2022-01-01T00:00Z', available_at='release', team_seasons=movements)
    assert_evidence(plan.evaluate(model), expected)


def test_rollback_across_checkpoint_blocks_reuses_prefix_and_is_trial_local(monkeypatch):
    start = pd.Timestamp('2020-01-01')
    rows = []
    for i in range(300):
        kick = start + pd.Timedelta(days=i)
        release = kick + pd.Timedelta(hours=3)
        if i in (12, 135, 264):
            release += pd.Timedelta(days=19)
        rows.append((i, 1, 2020, str(kick), 10, 11, i%4, i%3, str(release)))
    history = _history(rows)
    model = BayesianModel(parameters=PARAMETERS[1])
    plan = CalibrationReplay(history, config=model.config, cutoff='2021-01-01T00:00Z', available_at='released_at')
    counts = []
    original = replay.update_matches
    def count(*args, **kwargs):
        counts.append(len(args[2]))
        return original(*args, **kwargs)
    monkeypatch.setattr(replay, 'update_matches', count)
    fast = plan.evaluate(model)
    accelerated = sum(counts)
    counts.clear()
    states = _States()
    for time, schedule in plan.versions:
        canonical = replay._replay_in_order(plan.history, model=model, _plan=schedule, _predict=False, _sink=_States(history=False))
        states.publish(canonical, time)
    assert dict(fast.states.teams) == dict(states.teams)
    assert dict(fast.states.leagues) == dict(states.leagues)
    assert fast.states.usage == states.usage
    assert accelerated < sum(counts) / 10
    # Changing parameters and returning to the first trial cannot reuse its state.
    plan.evaluate(replace(model, parameters=PARAMETERS[2]))
    again = plan.evaluate(model)
    np.testing.assert_array_equal(fast.probabilities, again.probabilities)
    assert all(schedule.start == 0 for _, schedule in plan.versions)


@pytest.mark.parametrize('field', DEFAULT_FIT_FIELDS)
def test_late_release_all_nine_finite_difference_directions(monkeypatch, field):
    from xdiyo_analytics.ratings.bayesian_training import _LOG_FIELDS
    history, movements = exchange_history(seasons=2)
    history['release'] = history.kickoff_at
    history.loc[history.event_id.eq(history.event_id.iloc[0]), 'release'] += pd.Timedelta('9d')
    config = BayesianConfig()
    plan = CalibrationReplay(history, config=config, cutoff='2022-01-01T00:00Z', available_at='release', team_seasons=movements)
    center = getattr(PARAMETERS[1], field)
    center = np.log(center) if field in _LOG_FIELDS else center
    for direction in (-1e-8, 1e-8):
        value = np.exp(center+direction) if field in _LOG_FIELDS else center+direction
        model = BayesianModel(parameters=replace(PARAMETERS[1], **{field: value}), config=config)
        with monkeypatch.context() as patch:
            patch.setattr(replay, '_refilter_versions', REFERENCE._refilter_versions)
            reference = build_bayesian_ratings(history, model=model, available_at='release', team_seasons=movements)
        assert_evidence(plan.evaluate(model), reference)


def test_real_resumed_windows_match_every_published_state_and_objective(monkeypatch, tmp_path):
    from xdiyo_analytics.ratings import BayesianRatingRun
    fixture = json.loads((Path(__file__).with_name('fixtures') / 'bayesian_completion_windows_8207e29.json').read_text())
    history = pd.DataFrame(fixture['rows'])
    for name in ('kickoff_at', 'available_at', 'result_available_at'):
        history[name] = pd.to_datetime(history[name], utc=True)
    model = BayesianModel(parameters=PARAMETERS[2])
    actual = build_bayesian_ratings(history, model=model, available_at='available_at')
    with monkeypatch.context() as patch:
        patch.setattr(replay, '_refilter_versions', REFERENCE._refilter_versions)
        expected = build_bayesian_ratings(history, model=model, available_at='available_at')
    for name in ('snapshots', 'league_snapshots', 'predictions'):
        pd.testing.assert_frame_equal(getattr(actual, name), getattr(expected, name), check_exact=True)
    assert actual.checkpoint == expected.checkpoint
    assert actual.metadata == expected.metadata
    plan = CalibrationReplay(history, config=model.config, cutoff='2012-01-01T00:00Z', available_at='available_at')
    assert_evidence(plan.evaluate(model), expected)
    actual.save(tmp_path / 'real-windows')
    loaded = BayesianRatingRun.load(tmp_path / 'real-windows')
    pd.testing.assert_frame_equal(loaded.fixture_features(history), actual.fixture_features(history), check_exact=True)
