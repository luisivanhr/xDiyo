"""Paired native goal targets through the existing label and assembly contracts."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from test_match_score import score_data
from xdiyo_analytics.datasets import assemble_dataset
from xdiyo_analytics.features import IsHome, evaluate_features
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.labels import MatchGoals, create_labels


def test_match_goals_keep_native_scores_order_exact_ids_and_missing_targets():
    data = score_data()
    data.matches.loc[2, 'status'] = 'inprogress'
    data.matches.loc[3, 'home_score_current'] = np.nan
    history = build_team_history(data).sample(frac=1, random_state=7)
    before = deepcopy(history)
    label = create_labels(history, {'score': MatchGoals()})['score']
    home = history.loc[history.side.eq('home')]
    assert label.unit == 'match' and label.perspective == 'home'
    assert list(label.y) == ['score::home_goals', 'score::away_goals']
    assert label.metadata.event_id.tolist() == home.event_id.tolist()
    assert label.metadata.home_id.tolist() == home.team_id.tolist()
    assert min(label.metadata.home_id) > 2**63
    expected = home[['goals_for', 'goals_against']].to_numpy(float, na_value=np.nan)
    expected[home.status.ne('finished')] = np.nan
    np.testing.assert_allclose(label.y.to_numpy(), expected, equal_nan=True)
    features = evaluate_features(history, {'venue': IsHome()}, keyed=True)
    assembled = assemble_dataset(features.sample(frac=1, random_state=8), label,
                                 layout='match', drop_missing_targets=True)
    assert len(assembled.y) == len(data.matches) - 2
    assert list(assembled.y) == ['score::home_goals', 'score::away_goals']
    pd.testing.assert_frame_equal(history, before)


@pytest.mark.parametrize('invalid', [-1., .5])
def test_match_goals_reject_invalid_counts(invalid):
    data = score_data()
    data.matches['home_score_current'] = data.matches.home_score_current.astype(float)
    data.matches.loc[0, 'home_score_current'] = invalid
    with pytest.raises(ValueError, match='nonnegative integer'):
        create_labels(build_team_history(data), {'score': MatchGoals()})


def test_match_goals_reject_contradictory_perspectives_and_unsupported_basis():
    history = build_team_history(score_data())
    history.loc[history.side.eq('away'), 'goals_against'] = 999
    with pytest.raises(ValueError, match='agreeing'):
        create_labels(history, {'score': MatchGoals()})
    with pytest.raises(ValueError, match='score_field'):
        MatchGoals(score_field='regulation')
