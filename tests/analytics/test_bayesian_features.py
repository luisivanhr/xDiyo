"""Bayesian feature routing, shared state and prediction-time integration."""

from copy import deepcopy

import pandas as pd
import pytest

from xdiyo_analytics.features import (
    BayesianFixture, BayesianRating, H2H, Rating, evaluate_features,
)
from xdiyo_analytics.histories import build_team_history
from test_match_score import score_data


class _RoutingRun:
    def features(self, history, *, cutoffs, side, fields):
        return pd.DataFrame({f'{side}::{field}': 1. for field in fields}, index=history.index)

    def fixture_features(self, history, *, cutoffs, fields):
        return pd.DataFrame({field: 2. for field in fields}, index=history.index)


def test_selected_outputs_share_producer_and_forward_prediction_context(monkeypatch):
    import xdiyo_analytics.ratings as ratings

    history = build_team_history(score_data())
    calls = []

    def build(frame, **options):
        assert frame is history
        calls.append(options)
        return _RoutingRun()

    monkeypatch.setattr(ratings, 'build_bayesian_ratings', build, raising=False)
    movements = pd.DataFrame({'team_id': []})
    cutoffs = history.kickoff_at - pd.Timedelta(hours=1)
    available = history.kickoff_at + pd.Timedelta(hours=2)
    starts = {'explicit': 'boundary'}
    result = evaluate_features(history, {
        'means': BayesianRating(),
        'selected': BayesianRating(side='for', fields=('attack_sd', 'attack_mean')),
        'fixture': BayesianFixture(fields=('expected_away_goals', 'expected_home_goals')),
    }, cutoffs=cutoffs, available_at=available, team_seasons=movements, season_starts=starts)
    assert len(calls) == 1
    assert calls[0]['cutoffs'] is cutoffs
    assert calls[0]['available_at'] is available
    assert calls[0]['team_seasons'] is movements
    assert calls[0]['season_starts'] is starts
    assert list(result) == ['means::both::attack_mean', 'means::both::defence_vulnerability_mean',
                            'selected::for::attack_sd', 'selected::for::attack_mean',
                            'fixture::expected_away_goals', 'fixture::expected_home_goals']


def test_named_team_and_fixture_run_requires_no_new_filter(monkeypatch):
    import xdiyo_analytics.ratings as ratings

    def forbidden(*args, **kwargs):
        pytest.fail('A saved run should not trigger a new filter')

    monkeypatch.setattr(ratings, 'build_bayesian_ratings', forbidden, raising=False)
    history = build_team_history(score_data())
    result = evaluate_features(history, {
        'team': Rating('saved', fields=('attack_mean',)),
        'fixture': BayesianFixture(name='saved', fields=('expected_home_goals',)),
    }, ratings={'saved': _RoutingRun()})
    assert list(result) == ['team', 'fixture']
    assert result['team'].eq(1.).all() and result['fixture'].eq(2.).all()
    with pytest.raises(TypeError, match='BayesianRatingRun'):
        evaluate_features(history, {'fixture': BayesianFixture(name='saved')}, ratings={'saved': object()})
    with pytest.raises(KeyError, match='saved'):
        evaluate_features(history, {'fixture': BayesianFixture(name='saved')})
    with pytest.raises(ValueError, match='not both'):
        BayesianFixture(name='saved', model=object())


@pytest.mark.parametrize('expression', [BayesianRating(), BayesianFixture()])
def test_h2h_cannot_reinterpret_goal_filter(expression):
    with pytest.raises(ValueError, match='H2H rating'):
        evaluate_features(build_team_history(score_data()), {'bad': H2H(expression)})


def test_real_filter_selection_and_future_score_perturbation():
    history = build_team_history(score_data())
    before = deepcopy(history)
    definitions = {'state': BayesianRating(), 'fixture': BayesianFixture()}
    output = evaluate_features(history, definitions)
    selected = evaluate_features(history, {
        'state': BayesianRating(fields=('attack_mean',)),
        'fixture': BayesianFixture(fields=('expected_away_goals',)),
    })
    for name in [c for c in selected if c.startswith('state::')]:
        pd.testing.assert_series_equal(selected[name], output[name])
    pd.testing.assert_series_equal(selected.fixture, output['fixture::expected_away_goals'], check_names=False)
    cutoff = history.kickoff_at.sort_values().iloc[4]
    perturbed = history.copy(deep=True)
    perturbed.loc[history.kickoff_at >= cutoff, ['goals_for', 'goals_against']] = 10
    changed = evaluate_features(perturbed, definitions)
    pd.testing.assert_frame_equal(output.loc[history.kickoff_at <= cutoff], changed.loc[history.kickoff_at <= cutoff])
    pd.testing.assert_frame_equal(history, before)


def test_named_bayesian_run_preserves_generic_none_means_all_numeric_fields():
    from xdiyo_analytics.ratings import build_bayesian_ratings
    from xdiyo_analytics.ratings.bayesian import TEAM_FIELDS

    history = build_team_history(score_data())
    run = build_bayesian_ratings(history)
    exported = evaluate_features(history, {'all': Rating('saved')}, ratings={'saved': run})
    assert len(exported.columns) == 2 * len(TEAM_FIELDS)
    assert {column.rsplit('::', 1)[-1] for column in exported} == set(TEAM_FIELDS)
    defaults = evaluate_features(history, {'all': BayesianRating()})
    assert len(defaults.columns) == 4
    pd.testing.assert_frame_equal(exported.loc[:, defaults.columns], defaults)
