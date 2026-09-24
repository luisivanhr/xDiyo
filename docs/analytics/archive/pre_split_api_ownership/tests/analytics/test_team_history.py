"""Two-sided histories preserve observed identities, times and missing data."""

from copy import deepcopy
from datetime import datetime, timezone
from urllib.parse import unquote

import pandas as pd
import pyarrow as pa
import pytest

from xdiyo_analytics.data import SeasonData
from xdiyo_analytics.histories import build_team_history


EVENT, HOME, AWAY = 2**53 + 1, 2**63 + 9, 2**63 + 11
EARLY = datetime(2025, 1, 1, 12, 34, 56, tzinfo=timezone.utc)
LATE = datetime(2025, 1, 2, 21, 47, 31, 125000, tzinfo=timezone.utc)


def _frame(columns):
    return pa.table(columns).to_pandas(types_mapper=pd.ArrowDtype)


@pytest.fixture
def data():
    matches = _frame({
        'event_id': [EVENT, 17, EVENT, EVENT, 8],
        'source_league': ['Alpha', 'Alpha', 'Beta', 'Alpha', 'Gamma'],
        'source_season': ['24_25', '24_25', '24_25', '23_24', '24_25'],
        'home_id': pa.array([HOME, 42, HOME, 99, 7], type=pa.uint64()),
        'away_id': pa.array([AWAY, 50, AWAY, 42, 9], type=pa.uint64()),
        'home_name': ['H one', 'H draw', 'H other', 'H prior', 'H unknown'],
        'away_name': ['A one', 'A draw', 'A other', 'A prior', 'A unknown'],
        'kickoff_utc': [LATE.timestamp(), EARLY.timestamp(), EARLY.timestamp(), None, None],
        'competition_id': [101, 101, 102, 101, 103], 'season_id': [11, 11, 12, 10, 13],
        'season_year': [2024, 2024, 2024, 2023, 2024], 'round': [1, 2, 1, 9, 1],
        'status': ['finished', 'finished', 'inprogress', None, 'postponed'],
        'home_score_current': pa.array([3, 1, 2, None, 0], type=pa.int64()),
        'away_score_current': pa.array([1, 1, 1, 2, None], type=pa.int64()),
        'home_score_normaltime': [0, 0, 0, 0, 0], 'away_score_normaltime': [0, 0, 0, 0, 0],
        'home_score_penalties': pa.array([None, 5, None, None, None], type=pa.int64()),
        'away_score_penalties': pa.array([None, 4, None, None, None], type=pa.int64()),
    })
    entries = [
        (0, 'home', 'ALL', 'Match overview', 'totalShotsOnGoal', 5, 10, '5/10'),
        (0, 'away', 'ALL', 'Match overview', 'totalShotsOnGoal', 7, 20, '7/20'),
        (0, 'home', 'ALL', 'Shots', 'totalShotsOnGoal', 6, 30, '6'),
        (0, 'home', '1ST', 'Match overview', 'totalShotsOnGoal', 2, 5, '2'),
        (0, 'away', '1ST', 'Match overview', 'totalShotsOnGoal', None, None, 'not supplied'),
        (2, 'home', 'ALL', 'Match overview', 'totalShotsOnGoal', 100, 1000, '100'),
        (2, 'away', 'ALL', 'Match overview', 'totalShotsOnGoal', 101, 1000, '101'),
        (3, 'away', 'ALL', 'Match overview', 'totalShotsOnGoal', 200, None, '200'),
    ]
    statistics = _frame({
        **{name: [matches.iloc[row][name] for row, *_ in entries]
           for name in ['source_league', 'source_season', 'event_id']},
        'side': [entry[1] for entry in entries],
        'team_id': pa.array([matches.iloc[entry[0]][entry[1] + '_id'] for entry in entries], type=pa.uint64()),
        **{name: [entry[i] for entry in entries]
           for i, name in enumerate(['period', 'group_name', 'key', 'value', 'total', 'display'], start=2)},
    })
    positions = [(0, 'home', 3), (0, 'away', 8), (2, 'home', None), (2, 'away', 2), (3, 'away', 4)]
    pregame = _frame({
        **{name: [matches.iloc[row][name] for row, _, _ in positions]
           for name in ['source_league', 'source_season', 'event_id']},
        'side': [side for _, side, _ in positions],
        'team_id': pa.array([matches.iloc[row][side + '_id'] for row, side, _ in positions], type=pa.uint64()),
        'position': pa.array([position for _, _, position in positions], type=pa.int64()),
    })
    matches.index = pd.Index([13, 2, 20, 1, 0], name='original_row')
    statistics.index = pd.Index([10, 3, 3, 8, 0, 51, 6, 18], name='original_row')
    for frame in [matches, statistics, pregame]:
        frame.attrs = {'source': {'version': 'v1'}}
    return SeasonData({'matches': matches, 'statistics': statistics, 'pregame': pregame},
                      {'sources': [{'version': 'v1'}]})


def _row(history, event=EVENT, league='Alpha', season='24_25', side='home'):
    rows = history.loc[history.event_id.eq(event) & history.source_league.eq(league)
                       & history.source_season.eq(season) & history.side.eq(side)]
    assert len(rows) == 1
    return rows.iloc[0]


def _column(history, role='team', period='ALL', group='Match overview', key='totalShotsOnGoal', field='value'):
    expected = {'role': role, 'period': period, 'group_name': group, 'key': key, 'field': field}
    return next(name for name, labels in history.attrs['stat_columns'].items() if labels == expected)


def test_perspective_reversal_scores_names_positions_and_group_identity(data):
    history = build_team_history(data)
    assert len(history) == 2 * len(data.matches)
    home, away = _row(history), _row(history, side='away')
    assert (home.team_id, home.opponent_id) == (HOME, AWAY)
    assert (away.team_id, away.opponent_id) == (AWAY, HOME)
    assert (home.team_name, home.opponent_name, away.team_name) == ('H one', 'A one', 'A one')
    assert (home.goals_for, home.goals_against, home.result) == (3, 1, 'W')
    assert (away.goals_for, away.goals_against, away.result) == (1, 3, 'L')
    assert (home.team_position, home.opponent_position, away.team_position, away.opponent_position) == (3, 8, 8, 3)
    assert (home[_column(history)], home[_column(history, role='opponent')]) == (5, 7)
    assert (away[_column(history)], away[_column(history, role='opponent')]) == (7, 5)
    assert home[_column(history, group='Shots')] == 6
    assert home[_column(history, period='1ST')] == 2
    assert pd.isna(away[_column(history, group='Shots')])
    assert _row(history, event=17).result == 'D'  # Penalties 5:4 do not break the 1:1 current-score draw.


@pytest.mark.parametrize('unrepresentable', [1e30, -1e30, float('inf'), float('-inf')])
def test_full_utc_times_sort_stably_with_missing_and_unrepresentable_dates_last(data, unrepresentable):
    data.matches.loc[0, 'kickoff_utc'] = unrepresentable
    history = build_team_history(data)
    assert str(history.kickoff_at.dt.tz) == 'UTC'
    assert _row(history).kickoff_at == pd.Timestamp(LATE)
    assert _row(history, event=17).kickoff_at == pd.Timestamp(EARLY)
    assert history.kickoff_at.isna().sum() == 4
    assert history.kickoff_at.iloc[-4:].isna().all()
    assert list(history[['source_league', 'source_season', 'event_id', 'side']].itertuples(index=False, name=None)) == [
        ('Alpha', '24_25', 17, 'away'), ('Alpha', '24_25', 17, 'home'),
        ('Beta', '24_25', EVENT, 'away'), ('Beta', '24_25', EVENT, 'home'),
        ('Alpha', '24_25', EVENT, 'away'), ('Alpha', '24_25', EVENT, 'home'),
        ('Alpha', '23_24', EVENT, 'away'), ('Alpha', '23_24', EVENT, 'home'),
        ('Gamma', '24_25', 8, 'away'), ('Gamma', '24_25', 8, 'home'),
    ]


def test_partition_identity_large_ids_and_missing_side_or_entire_observations(data):
    history = build_team_history(data)
    assert _row(history, league='Beta')[_column(history)] == 100
    assert _row(history, season='23_24', side='away')[_column(history)] == 200
    assert _row(history).team_id == HOME and _row(history).event_id == EVENT
    assert history.team_id.dtype == data.matches.home_id.dtype
    assert pd.isna(_row(history, season='23_24')[_column(history)])
    assert _row(history, season='23_24')[_column(history, role='opponent')] == 200
    assert pd.isna(_row(history, league='Beta').team_position)
    assert _row(history, league='Beta').opponent_position == 2
    assert pd.isna(_row(history, season='23_24').team_position)
    assert _row(history, season='23_24').opponent_position == 4
    absent = history.loc[history.event_id.eq(17)]
    assert absent[list(history.attrs['stat_columns'])].isna().all().all()
    assert absent[['team_position', 'opponent_position']].isna().all().all()


def test_optional_stat_fields_keep_source_text_totals_and_explicit_nulls(data):
    history = build_team_history(data, stat_fields=['value', 'total', 'display', 'value'])
    assert len(history.attrs['stat_columns']) == 18
    assert _row(history)[_column(history, field='total')] == 10
    assert _row(history)[_column(history, role='opponent', field='display')] == '7/20'
    assert _row(history, side='away')[_column(history, period='1ST', field='display')] == 'not supplied'
    assert pd.isna(_row(history, side='away')[_column(history, period='1ST', field='value')])
    display_only = build_team_history(data, stat_fields='display')
    assert {info['field'] for info in display_only.attrs['stat_columns'].values()} == {'display'}
    assert {info['field'] for info in build_team_history(data).attrs['stat_columns'].values()} == {'value'}


def test_arbitrary_labels_and_separator_escaping_are_reversible_and_collision_free(data):
    statistics = data.statistics.iloc[:3].copy()
    statistics['period'] = ['EX::TRA', 'EX::TRA', 'EX::TRA']
    statistics['group_name'] = ['G::x', 'G', 'G::x']
    statistics['key'] = ['k% p', 'x::k% p', 'k%3A p']
    history = build_team_history(SeasonData({'matches': data.matches, 'statistics': statistics}, data.provenance))
    assert 'team::EX%3A%3ATRA::G%3A%3Ax::k%25 p::value' in history
    assert len(history.attrs['stat_columns']) == 6 and history.columns.is_unique
    for name, labels in history.attrs['stat_columns'].items():
        assert tuple(unquote(piece) for piece in name.split('::')) == (
            labels['role'], labels['period'], labels['group_name'], labels['key'], labels['field'])


@pytest.mark.parametrize('status', ['inprogress', 'postponed', 'unknown', None])
def test_nonfinished_or_unknown_status_keeps_goals_but_has_no_result(data, status):
    matches = data.matches.iloc[:1].copy()
    matches['status'] = pd.Series([status], index=matches.index, dtype='string[pyarrow]')
    history = build_team_history(SeasonData({'matches': matches}, {}))
    assert len(history) == 2 and history.result.isna().all()
    assert sorted(history.goals_for.tolist()) == [1, 3]


def test_missing_scores_or_status_never_invents_result(data):
    matches = data.matches.iloc[:1].copy()
    matches['home_score_current'] = pd.NA
    history = build_team_history(SeasonData({'matches': matches}, {}))
    assert history.result.isna().all() and history.goals_for.isna().sum() == 1
    minimal = data.matches.drop(columns=['status', 'home_score_current', 'away_score_current'])
    history = build_team_history(SeasonData({'matches': minimal}, {}))
    assert len(history) == 10 and history.result.isna().all()
    assert history[['goals_for', 'goals_against']].isna().all().all()
    assert not history.attrs['stat_columns'] and 'team_position' not in history


def test_source_tables_attributes_and_provenance_are_unchanged(data):
    originals = {name: frame.copy(deep=True) for name, frame in data.tables.items()}
    provenance = deepcopy(data.provenance)
    history = build_team_history(data, stat_fields=['display', 'value', 'total'])
    for name, original in originals.items():
        pd.testing.assert_frame_equal(data[name], original, check_exact=True)
        assert data[name].attrs == original.attrs
    assert data.provenance == provenance and history.attrs['source'] == provenance


@pytest.mark.parametrize('table', ['matches', 'statistics', 'pregame'])
def test_duplicate_observations_raise_instead_of_aggregating(data, table):
    frames = dict(data.tables)
    frames[table] = pd.concat([frames[table], frames[table].iloc[:1]])
    with pytest.raises(ValueError):
        build_team_history(SeasonData(frames, data.provenance))


@pytest.mark.parametrize('table', ['statistics', 'pregame'])
def test_orphan_partition_or_side_is_rejected(data, table):
    frames = dict(data.tables)
    frames[table] = frames[table].copy()
    frames[table]['source_season'] = '99_00'
    with pytest.raises(ValueError, match='absent from matches'):
        build_team_history(SeasonData(frames, data.provenance))
    frames[table] = data[table].copy()
    frames[table]['side'] = 'neutral'
    with pytest.raises(ValueError, match='absent from matches'):
        build_team_history(SeasonData(frames, data.provenance))


@pytest.mark.parametrize('table', ['statistics', 'pregame'])
def test_supplied_team_disagreement_is_rejected(data, table):
    frames = dict(data.tables)
    frames[table] = frames[table].copy()
    frames[table]['team_id'] = 123
    with pytest.raises(ValueError):
        build_team_history(SeasonData(frames, data.provenance))


def test_missing_required_fields_or_null_identities_fail_clearly(data):
    with pytest.raises(KeyError, match='Load matches'):
        build_team_history(SeasonData({}, {}))
    with pytest.raises(KeyError, match='kickoff_utc'):
        build_team_history(SeasonData({'matches': data.matches.drop(columns='kickoff_utc')}, {}))
    frames = dict(data.tables)
    frames['statistics'] = data.statistics.drop(columns='total')
    with pytest.raises(KeyError, match='total'):
        build_team_history(SeasonData(frames, {}), stat_fields='total')
    frames['statistics'] = data.statistics.copy()
    frames['statistics']['period'] = pd.NA
    with pytest.raises(ValueError, match='must be nonmissing'):
        build_team_history(SeasonData(frames, {}))
    matches = data.matches.copy()
    matches['event_id'] = pd.NA
    with pytest.raises(ValueError, match='unique, nonmissing'):
        build_team_history(SeasonData({'matches': matches}, {}))


@pytest.mark.parametrize('stat_fields', [[], ['ratio']])
def test_invalid_stat_fields_are_rejected(data, stat_fields):
    with pytest.raises(ValueError, match='stat_fields'):
        build_team_history(data, stat_fields=stat_fields)


@pytest.mark.parametrize('with_optional_tables', [False, True])
def test_empty_matches_return_an_empty_history(data, with_optional_tables):
    frames = {name: frame.iloc[:0] for name, frame in data.tables.items()
              if with_optional_tables or name == 'matches'}
    history = build_team_history(SeasonData(frames, {}))
    assert history.empty and history.index.equals(pd.RangeIndex(0))
    assert str(history.kickoff_at.dt.tz) == 'UTC'
    assert {'team_id', 'opponent_id', 'side', 'result', 'goals_for', 'goals_against'} <= set(history.columns)
    assert history.attrs['stat_columns'] == {}
