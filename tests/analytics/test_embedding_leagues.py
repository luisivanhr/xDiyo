"""League presentation uses recorded fixture identities, never current rosters."""
import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.reporting.embedding_view import embedding_figure
from xdiyo_analytics.reporting.leagues import embedding_leagues


def samples():
    table = pd.DataFrame(dict(team_id=[47, 64, 99], league=pd.array([18, None, 17], dtype='UInt64'),
                              team_name=['Charlton', 'Wrexham', 'Historical team'],
                              x=[1., 2., 3.], y=[4., 5., 6.], support=[1., 0., 3.], gate=[.1, 0., .3]))
    metadata = pd.DataFrame(dict(home_id=[47, 99], away_id=[64, 47],
                                  competition_id=[18, 17], source_league=['Championship', 'Premier_League']))
    return table, metadata


def test_names_and_cold_membership_are_display_only():
    table, metadata = samples()
    before = table.copy(deep=True)
    result = embedding_leagues(table, metadata)
    assert result.league_name.tolist() == ['Championship', 'Championship', 'Premier League']
    assert result.league_label_source.tolist() == ['model_membership', 'evaluation_fixture', 'model_membership']
    assert pd.isna(result.league.iloc[1])
    pd.testing.assert_frame_equal(table, before)
    pd.testing.assert_frame_equal(result[table.columns], before)
    figure = embedding_figure(result, [None] * 3, 'Frozen')
    assert [trace.name for trace in figure.data] == ['Championship', 'Premier League']
    assert list(figure.data[0].ids) == ['47', '64']
    np.testing.assert_array_equal(figure.data[0].x, [1., 2.])
    np.testing.assert_array_equal(figure.data[0].y, [4., 5.])


@pytest.mark.parametrize('ambiguous', [False, True])
def test_missing_or_conflicting_fixture_membership_is_not_guessed(ambiguous):
    table, metadata = samples()
    metadata = metadata.iloc[[1]].copy()
    if ambiguous:
        metadata = pd.concat([metadata.assign(away_id=64), metadata.assign(away_id=64, competition_id=18)])
    with pytest.raises(ValueError, match='Team 64: cannot resolve display league'):
        embedding_leagues(table, metadata)


def test_large_competition_ids_are_not_rounded_and_other_folds_are_not_consulted():
    large = 2**53 + 1
    table = pd.DataFrame({'team_id': [47], 'league': pd.array([large], dtype='UInt64')})
    metadata = pd.DataFrame({'competition_id': [large], 'league_name': ['Recorded name']})
    result = embedding_leagues(table, metadata)
    assert result.display_league.tolist() == [str(large)]
    assert result.league_name.tolist() == ['Recorded name']
