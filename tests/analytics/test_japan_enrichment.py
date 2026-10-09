import json

import pandas as pd
import pytest

from examples.backfill_statistics import statistics_scope
from examples.prepare_japan_20261009 import derive_japan
from xdiyo_analytics.data import load_season, load_seasons


@pytest.mark.parametrize('season,expected', [
    ('2019', ('J1', '19_19')), ('2026', ('J1_Transition', '26_26')),
    ('2026/2027', ('J1', '26_27')), ('2026/27', ('J1', '26_27')),
    ('2026/28', ('J1', None)),
])
def test_calendar_workbook_identity(season, expected):
    assert statistics_scope(dict(country='Japan', league='J1 League', season=season)) == expected
    assert statistics_scope(dict(country='England', league='Premier League', season='2024/2025')) == ('Premier_League', '24_25')


def calendar_export(export, end=2024):
    stem = f'Premier_League_24_{end % 100:02}'
    export.change_manifest(lambda m: m['scope'].update(season_end=end))
    pub = json.loads(export.publication_path.read_bytes())
    pub['season_file'] = stem + '.parquet'
    export.publication_path.unlink()
    (export.root / (stem + '.manifest.json')).write_text(json.dumps(pub))
    return stem


def test_calendar_loader_discovery_and_pinned_replay(export, tmp_path):
    expected = load_season(export.root, 'Premier_League_24_25', verify_hashes=True).matches
    stem = calendar_export(export)
    record = tmp_path / 'pinned.json'
    first = load_season(export.root, stem, record_path=record, verify_hashes=True)
    again = load_season(export.root, stem, record_path=record, verify_hashes=True)
    pd.testing.assert_frame_equal(first.matches, expected)
    pd.testing.assert_frame_equal(again.matches, expected)
    combined = load_seasons(export.root, ['24_24'], leagues=['Premier_League'], verify_hashes=True)
    assert combined.matches.source_season.tolist() == ['24_24', '24_24']


@pytest.mark.parametrize('end', [2023, 2026])
def test_invalid_season_ranges_still_rejected(export, end):
    stem = calendar_export(export, end)
    with pytest.raises(ValueError, match='Manifest competition/season'):
        load_season(export.root, stem)
    with pytest.raises(ValueError, match='Choose seasons'):
        load_seasons(export.root, [stem[-5:]])


def japan_rosters():
    return [dict(stem=stem, league=league, year=year, competition_id=competition,
                 season_id=season, teams=teams) for stem, league, year, competition, season, teams in [
        ('J1_25_25', 'J1', 2025, 196, 1, {'1': 'Retained', '2': 'Relegated'}),
        ('J2_25_25', 'J2', 2025, 402, 2, {'3': 'Promoted', '4': 'Lower'}),
        ('J1_26_27', 'J1', 2026, 196, 3, {'1': 'Retained', '3': 'Promoted'}),
        ('J2_26_27', 'J2', 2026, 402, 4, {'2': 'Relegated', '4': 'Lower'}),
        ('J1_Transition_26_26', 'J1_Transition', 2026, 196, 5, {'1': 'Retained', '3': 'Promoted'}),
    ]]


def initial_boundaries():
    return [dict(stem=stem, promoted_ids=[], relegated_ids=[], other_entry_ids=[],
                 complete_review=True, sources=['https://example.org/review'])
            for stem in ['J1_25_25', 'J2_25_25']]


def test_transition_keeps_actual_prior_division_and_separate_identity():
    flags = derive_japan(japan_rosters(), initial_boundaries())
    indexed = flags.set_index(['season_stem', 'team_id'])
    for stem in ['J1_26_27', 'J1_Transition_26_26']:
        assert indexed.loc[(stem, 3), 'movement'] == 'promoted'
        assert indexed.loc[(stem, 3), 'previous_competition_id'] == 402
        assert indexed.loc[(stem, 3), 'previous_season_id'] == 2
        assert indexed.loc[(stem, 1), 'movement'] == 'retained'
    assert indexed.loc[('J2_26_27', 2), 'movement'] == 'relegated'
    assert not flags.duplicated(['competition_id', 'season_id', 'team_id']).any()


def test_transition_population_change_requires_review():
    rosters = japan_rosters()
    rosters[-1]['teams'] = {'1': 'Retained', '9': 'Unexpected'}
    with pytest.raises(ValueError, match='membership differ'):
        derive_japan(rosters, initial_boundaries())
