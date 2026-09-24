"""Selection keeps observed identities, values and periods in separate tables."""

from copy import deepcopy
import math

import pandas as pd
import pyarrow as pa
import pytest

from xdiyo_analytics.data import SeasonData, STAT_CATEGORIES, list_stat_bundles, select_stats


EVENT = 2**53 + 1
FIELDS = ['period', 'group_name', 'key']


def _frame(columns):
    return pa.table(columns).to_pandas(types_mapper=pd.ArrowDtype)


@pytest.fixture
def data():
    identities = [
        ('ALL', 'Match overview', 'cornerKicks'),
        ('1ST', 'Match overview', 'cornerKicks'),
        ('2ND', 'Match overview', 'cornerKicks'),
        ('EXTRA', 'Match overview', 'cornerKicks'),
        ('ALL', 'Defending', 'interceptionWon'),
        ('1ST', 'Defending', 'interceptionWon'),
        ('2ND', 'Defending', 'interceptionWon'),
        ('EXTRA', 'Defending', 'interceptionWon'),
        ('ALL', 'Novel provider group', 'newMeasure'),
        ('1ST', 'Discipline', 'yellowCards'),
        ('ALL', 'Shots', 'totalShotsOnGoal'),
        ('ALL', 'Match overview', 'totalShotsOnGoal'),
        ('1ST', 'Attack', 'offsides'),  # Observed in a half without a totals row.
        ('ALL', 'Defending', 'totalClearance'),  # Observed totals without halves.
        ('ALL', 'Match overview', 'cornerKicks'),
    ]
    statistics = _frame({
        'event_id': [EVENT] * 13 + [7, 7],
        **{field: [triple[i] for triple in identities] for i, field in enumerate(FIELDS)},
        'side': ['home'] * 4 + ['away'] * 6 + ['home', 'home', 'away', 'home', 'away'],
        'team_id': pa.array([EVENT + 2] * 13 + [None, 42], type=pa.int64()),
        'value': pa.array([100.0, 2, 3, None, 40, 4, 5, 6, float('nan'), 0, 1, 1, 1, 9, 0],
                          type=pa.float64(), from_pandas=False),
        'external_id': pa.array([2**63 + i for i in range(15)], type=pa.uint64()),
        'extra_flag': pa.array([False, None, True] * 5, type=pa.bool_()),
        'source_league': ['Alpha'] * 13 + ['Beta'] * 2,
        'source_season': ['24_25'] * 13 + ['22_23'] * 2,
    })
    statistics.index = pd.Index([91, 12, 70, 5, 91, 48, 3, 84, 22, 9, 35, 62, 11, 7, 4], name='stored_row')
    statistics.attrs = {'source': {'version': 'v1'}}
    matches = _frame({'event_id': [EVENT, 7], 'optional_flag': [None, False]})
    pregame = _frame({
        'event_id': [EVENT, 7], 'side': ['home', 'away'],
        'position': pa.array([None, 3], type=pa.int64()), 'team_id': [EVENT + 2, 42],
        'observed_at': [1700000000.0, 1700000010.0],
        'source_league': ['Alpha', 'Beta'], 'source_season': ['24_25', '22_23'],
        'value_json': ['provider context', None],
    })
    pregame.index = pd.Index([19, 2], name='stored_row')
    return SeasonData(
        {'matches': matches, 'statistics': statistics, 'pregame': pregame,
         'shots': _frame({'shot_id': [8, 9]})},
        {'sources': [{'season': 'Alpha_24_25', 'version': 'v1'}]},
    )


@pytest.mark.parametrize(('bundle', 'positions'), [
    ('all_all', list(range(15))),
    ('all_totals', [0, 4, 8, 10, 11, 13, 14]),
    ('all_first_half', [1, 5, 9, 12]),
    ('all_second_half', [2, 6]),
    ('attack_all', [0, 1, 2, 3, 10, 11, 12, 14]),
    ('attack_totals', [0, 10, 11, 14]),
    ('attack_first_half', [1, 12]),
    ('attack_second_half', [2]),
    ('defense_all', [4, 5, 6, 7, 13]),
    ('defense_totals', [4, 13]),
    ('defense_first_half', [5]),
    ('defense_second_half', [6]),
])
def test_category_period_filters_keep_observations_without_derivation(data, bundle, positions):
    selected = select_stats(data, bundles=bundle)
    pd.testing.assert_frame_equal(selected.statistics, data.statistics.iloc[positions], check_exact=True)
    pd.testing.assert_frame_equal(selected.matches, data.matches, check_exact=True)
    assert selected.shots is None and selected.pregame is None
    category = bundle.split('_', 1)[0]
    all_periods = select_stats(data, bundles=f'{category}_all')
    reapplied = select_stats(all_periods, bundles=bundle)
    pd.testing.assert_frame_equal(reapplied.statistics, selected.statistics, check_exact=True)
    # Stored totals are 100 and 40; their halves deliberately do not sum to them.


def test_all_aliases_share_the_same_period_selections(data):
    for alias, canonical in {
        'all_stats': 'all_all', 'all_totals_only': 'all_totals',
        'all_first_period_only': 'all_first_half', 'all_second_period_only': 'all_second_half',
    }.items():
        pd.testing.assert_frame_equal(select_stats(data, bundles=alias).statistics,
                                      select_stats(data, bundles=canonical).statistics)


def test_composable_union_retains_group_identity_without_duplicate_selection(data):
    selected = select_stats(
        data, bundles=['standings', 'attack_totals', 'defense_first_half', 'attack_totals'],
        stats=[('1ST', 'Discipline', 'yellowCards'), ('ALL', 'Match overview', 'cornerKicks')],
    )
    expected = data.statistics.iloc[[0, 5, 9, 10, 11, 14]]
    pd.testing.assert_frame_equal(selected.statistics, expected, check_exact=True)
    assert list(selected.tables) == ['matches', 'statistics', 'pregame']
    assert selected.statistics.loc[selected.statistics.key.eq('totalShotsOnGoal'), 'group_name'].tolist() == ['Shots', 'Match overview']
    assert selected.provenance['selection']['statistics'] == tuple(expected[FIELDS].drop_duplicates().itertuples(index=False, name=None))
    assert selected.provenance['selection']['context'] == ('standings',)


def test_exact_arbitrary_statistic_keeps_only_its_named_group(data):
    selected = select_stats(data, stats=[('ALL', 'Shots', 'totalShotsOnGoal'),
                                        ('ALL', 'Novel provider group', 'newMeasure'),
                                        ('ALL', 'Shots', 'totalShotsOnGoal')])
    pd.testing.assert_frame_equal(selected.statistics, data.statistics.iloc[[8, 10]], check_exact=True)


def test_all_selection_preserves_nulls_valid_nan_large_ids_order_and_inputs(data):
    snapshots = {name: frame.copy(deep=True) for name, frame in data.tables.items()}
    provenance_before = deepcopy(data.provenance)
    selected = select_stats(data, bundles=['all_stats', 'standings'])
    pd.testing.assert_frame_equal(selected.statistics, snapshots['statistics'], check_exact=True)
    values = selected.statistics['value'].array.__arrow_array__()
    assert values.null_count == 1 and not values[3].is_valid
    assert values[8].is_valid and math.isnan(values[8].as_py())
    assert selected.statistics['external_id'].iloc[0] == 2**63
    assert len(selected.pregame) == 2  # Missing event/side observations are not filled.
    for name, frame in data.tables.items():
        pd.testing.assert_frame_equal(frame, snapshots[name], check_exact=True)
        assert frame.attrs == snapshots[name].attrs
    assert data.provenance == provenance_before


def test_per_call_categories_replace_or_extend_without_changing_defaults(data):
    before = deepcopy(STAT_CATEGORIES)
    categories = {'attack': [('Discipline', 'yellowCards')],
                  'custom': [('Novel provider group', 'newMeasure')]}
    chosen = select_stats(data, bundles=['attack_all', 'custom_totals'], categories=categories)
    pd.testing.assert_frame_equal(chosen.statistics, data.statistics.iloc[[8, 9]], check_exact=True)
    assert STAT_CATEGORIES == before
    pd.testing.assert_frame_equal(select_stats(data, bundles='all_stats', categories=categories).statistics, data.statistics)


def test_default_categories_can_be_edited_independently_of_period(data, monkeypatch):
    monkeypatch.setitem(STAT_CATEGORIES, 'attack', (('Discipline', 'yellowCards'),))
    pd.testing.assert_frame_equal(select_stats(data, bundles='attack_first_half').statistics, data.statistics.iloc[[9]])
    assert select_stats(data, bundles='attack_totals').statistics.empty


def test_bundle_discovery_includes_custom_categories_without_mutating_defaults():
    before = deepcopy(STAT_CATEGORIES)
    bundles = list_stat_bundles(categories={'custom': [('Discipline', 'yellowCards')]})
    expected = {f'{category}_{period}' for category in ['all', 'attack', 'defense', 'custom']
                for period in ['all', 'totals', 'first_half', 'second_half']}
    expected |= {'all_stats', 'all_totals_only', 'all_first_period_only', 'all_second_period_only', 'standings'}
    assert set(bundles) == expected and all(isinstance(text, str) and text for text in bundles.values())
    assert 'all available periods' in bundles['custom_all'] and 'period=1ST' in bundles['custom_first_half']
    assert STAT_CATEGORIES == before


def test_standings_needs_no_statistics_or_source_key(data):
    positions = data.pregame
    expected_columns = ['event_id', 'side', 'position', 'team_id', 'observed_at', 'source_league', 'source_season']
    minimal = SeasonData({'pregame': positions}, {'season': 'example'})
    result = select_stats(minimal, bundles='standings')
    assert list(result.tables) == ['pregame']
    pd.testing.assert_frame_equal(result.pregame, positions[expected_columns], check_exact=True)
    irrelevant = positions.assign(source_key=['unrelated', None])
    pd.testing.assert_frame_equal(select_stats(SeasonData({'pregame': irrelevant}, {}), bundles='standings').pregame,
                                  result.pregame, check_exact=True)


def test_statistics_selection_does_not_require_matches(data):
    result = select_stats(SeasonData({'statistics': data.statistics}, {}), bundles='attack_totals')
    assert list(result.tables) == ['statistics'] and result.matches is None
    pd.testing.assert_frame_equal(result.statistics, data.statistics.iloc[[0, 10, 11, 14]])


@pytest.mark.parametrize('members', [[], [('Unavailable group', 'unavailableKey')]])
def test_empty_or_unavailable_category_members_return_empty_observed_schema(data, members):
    selected = select_stats(data, bundles='custom_all', categories={'custom': members})
    pd.testing.assert_frame_equal(selected.statistics, data.statistics.iloc[:0])
    pd.testing.assert_frame_equal(selected.matches, data.matches)


def test_empty_statistics_table_retains_its_schema(data):
    frame = data.statistics.iloc[:0]
    selected = select_stats(SeasonData({'statistics': frame}, {}), bundles='all_stats')
    pd.testing.assert_frame_equal(selected.statistics, frame)


@pytest.mark.parametrize(('table', 'bundle', 'message'), [
    ('statistics', 'all_stats', 'Load the statistics table'),
    ('pregame', 'standings', 'Load the pregame table'),
])
def test_unavailable_requested_tables_raise(data, table, bundle, message):
    subset = SeasonData({name: frame for name, frame in data.tables.items() if name != table}, data.provenance)
    with pytest.raises(KeyError, match=message):
        select_stats(subset, bundles=bundle)


def test_unknown_bundle_and_absent_explicit_statistic_raise(data):
    with pytest.raises(KeyError, match='Unknown bundles'):
        select_stats(data, bundles=['attack_totals', 'unknown_bundle'])
    with pytest.raises(KeyError, match='Statistics absent from the input'):
        select_stats(data, bundles='all_stats', stats=[('ALL', 'Attack', 'offsides')])
    with pytest.raises(KeyError, match='standings columns'):
        select_stats(SeasonData({'pregame': data.pregame.drop(columns='position')}, {}), bundles='standings')


def test_empty_selection_and_malformed_exact_identity_raise(data):
    with pytest.raises(ValueError, match='Choose at least one'):
        select_stats(data)
    with pytest.raises(ValueError, match='triple'):
        select_stats(data, stats=[('ALL', 'cornerKicks')])
