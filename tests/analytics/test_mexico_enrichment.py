import json
import shutil

import pandas as pd
import pytest

from examples.backfill_statistics import statistics_scope, match_fixtures
from examples.prepare_mexico_20261010 import derive_mexico
from xdiyo_analytics.data import load_season
from xdiyo_analytics.data.loading import _coalesce_shared_movements


@pytest.mark.parametrize('date,season,expected', [
    ('2020-07-30', '2020/2021', ('Liga_MX_Apertura', '20_21')),
    ('2021-01-10', '2020/2021', ('Liga_MX_Clausura', '20_21')),
    ('2026-08-10', '2026/2027', ('Liga_MX_Apertura', '26_26')),
    ('2027-02-10', '2026/2027', ('Liga_MX_Clausura', '26_27')),
    ('2024-08-10', '2020/2021', (None, None)),
])
def test_workbook_short_tournament_scope(date, season, expected):
    serial = (pd.Timestamp(date) - pd.Timestamp('1899-12-30')).total_seconds() / 86400
    assert statistics_scope(dict(country='Mexico', league='Liga MX', season=season,
                                 start_datetime=str(serial))) == expected


def shared_export(export):
    stem = 'Premier_League_24_25'
    destination = export.root / '_tables' / 'cohorts' / stem / 'competition=17/season=61627/versions/v1'
    shutil.copytree(export.version, destination)
    pub = json.loads(export.publication_path.read_bytes())
    pub.update(manifest=(destination / 'manifest.json').relative_to(export.root).as_posix(),
               reconciliation={'archive_cohort': dict(shared_provider_identity=True,
                   stem=stem, league='Premier_League', competition_id=17, season_id=61627)})
    export.publication_path.write_text(json.dumps(pub))
    return pub


def test_shared_cohort_loader_and_pins(export, tmp_path):
    expected = load_season(export.root, 'Premier_League_24_25').matches
    shared_export(export)
    record = tmp_path / 'pin.json'
    for _ in range(2):
        loaded = load_season(export.root, 'Premier_League_24_25', record_path=record, verify_hashes=True)
        pd.testing.assert_frame_equal(loaded.matches, expected)
        assert loaded.provenance['shared_provider_identity'] is True


@pytest.mark.parametrize('key,value', [('stem', 'Other_24_25'), ('league', 'Other'),
    ('competition_id', 99), ('season_id', 99), ('shared_provider_identity', False)])
def test_shared_cohort_rejects_wrong_scope(export, key, value):
    pub = shared_export(export)
    pub['reconciliation']['archive_cohort'][key] = value
    export.publication_path.write_text(json.dumps(pub))
    with pytest.raises(ValueError, match='cohort|location'):
        load_season(export.root, 'Premier_League_24_25')


def movement_copies():
    return pd.DataFrame([dict(competition_id=1, season_id=2, team_id=3,
        movement='retained', got_promoted=False, got_demoted=False, previous_season_id=None,
        previous_competition_id=None, evidence='review', team_name='Team',
        source_league=league, source_season='10_11', season_stem=league+'_10_11')
        for league in ['Opening', 'Closing']])


def test_only_explicit_identical_shared_movement_copies_coalesce():
    frame = movement_copies()
    sources = [dict(season=s, shared_provider_identity=True) for s in frame.season_stem]
    assert len(_coalesce_shared_movements(frame, sources)) == 1
    assert len(_coalesce_shared_movements(frame, [])) == 2
    for field, value in [('movement', 'promoted'), ('previous_season_id', 1), ('evidence', 'different')]:
        changed = frame.copy()
        changed.loc[1, field] = value
        with pytest.raises(ValueError, match='Conflicting'):
            _coalesce_shared_movements(changed, sources)


def test_mexico_annual_flags_shared_identity_and_admin_entry():
    rosters = [dict(stem=f'{league}_{year % 100:02}_{(year+1) % 100:02}', league=league,
                   year=year, competition_id=comp, season_id=season, teams=teams)
        for league, year, comp, season, teams in [
            ('Liga_MX_Apertura', 2010, 1, 10, {'1':'A','2':'B'}),
            ('Liga_MX_Clausura', 2010, 1, 10, {'1':'A','2':'B'}),
            ('Liga_MX_Apertura', 2011, 1, 11, {'1':'A','3':'C'}),
            ('Liga_MX_Clausura', 2011, 2, 11, {'1':'A','3':'C'}),
        ]]
    boundaries = [dict(stem='Liga_MX_Apertura_10_11', promoted_ids=[2], complete_review=True,
                       sources=['https://example.org/initial']),
                  dict(stem='Liga_MX_Apertura_11_12', other_entry_ids=[3],
                       sources=['https://example.org/franchise'])]
    flags = derive_mexico(rosters, boundaries)
    assert flags.loc[flags.team_id.eq(2), 'got_promoted'].all()
    assert flags.loc[flags.team_id.eq(3), 'movement'].eq('other_entry').all()
    closing = flags[flags.season_stem.eq('Liga_MX_Clausura_11_12')]
    assert closing.previous_competition_id.isna().all()
    assert flags.loc[flags.season_stem.eq('Liga_MX_Apertura_11_12') & flags.team_id.eq(1),
                     'previous_season_id'].iloc[0] == 10
    rosters[-1]['teams'] = {'1':'A','9':'Unexpected'}
    with pytest.raises(ValueError, match='Unreviewed Clausura entrant'):
        derive_mexico(rosters, boundaries)


def test_exact_reviewed_cross_midnight_exception():
    date = pd.Timestamp('2026-08-02 01:00')
    row = dict(source_league='Liga_MX_Apertura', source_season='26_26',
               home_team='Home', away_team='Away',
               start_datetime=str((date-pd.Timestamp('1899-12-30')).total_seconds()/86400))
    native = dict(source_league=row['source_league'], source_season=row['source_season'],
                  home_name='Home', away_name='Away', home_id=1, away_id=2,
                  event_id=3, kickoff_utc=pd.Timestamp('2026-08-01 23:00',tz='UTC').timestamp(), stem='test')
    override = dict(vendor_match_id='vendor', event_id=3, source_league=row['source_league'],
                    source_season=row['source_season'], vendor_scheduled_at=str(date),
                    native_kickoff_at='2026-08-01T23:00:00+00:00')
    assert not match_fixtures({'vendor':[row]}, [native], {}, [])[0]
    assert match_fixtures({'vendor':[row]}, [native], {}, [override])[0]['vendor']==native
    override['native_kickoff_at']='2026-08-01T22:00:00+00:00'
    assert not match_fixtures({'vendor':[row]}, [native], {}, [override])[0]
