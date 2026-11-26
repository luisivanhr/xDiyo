from datetime import date, datetime, timezone
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from xdiyo_analytics.data.scorelines import (
    build, historical_season, materialize, native_score, read_external,
)

UTC=timezone.utc
DAY=date(2024,1,3)


def native(**changes):
    record=dict(fixture_key='sofascore:123',league='Premier_League',season='2023/2024',
                competition_id=17,season_id=42,league_name='Premier League',season_name='Premier League 23/24',
                home_id=1,away_id=2,home_team='Alpha',away_team='Beta',home_goals=1,away_goals=0,
                source='SofaScore',source_fixture_id='123',source_manifest='Premier_League_23_24',
                source_reported_date=DAY,kickoff_utc=datetime(2024,1,3,22,tzinfo=UTC),
                result_available_at=None,team_identity='sofascore_native')
    return dict(record,**changes)


def historical(key='archive:1',**changes):
    record=dict(fixture_key=key,league='Premier_League',season='2023/2024',home_team='Alpha',away_team='Beta',
                home_goals=1,away_goals=0,source='External',source_fixture_id='1',
                source_home_key='alpha',source_away_key='beta',source_reported_date=DAY,kickoff_utc=None,
                date_precision='date_only',score_basis='historical_coverage_checked')
    return dict(record,**changes),None


DIRECTORY={'teams':[
    dict(country='England',team_id=i,canonical_name=name,
         names=[dict(name=name,start='2020-01-01',end='2030-01-01')])
    for i,name in [(1,'Alpha'),(2,'Beta')]
]}


def test_native_wins_only_identical_confirmed_overlap():
    ext=[historical(),historical('archive:2',away_goals=3),historical('archive:3',home_team='Alfa')]
    rows,links,held=materialize([native()],[],ext,DIRECTORY)
    assert [r['fixture_key'] for r in rows]==['sofascore:123']
    assert [r['source_key'] for r in links]==['archive:1']
    assert {r['reason'] for r in held}=={'native_overlap_score_conflict','native_overlap_needs_review'}
    assert rows[0]['replay_eligible'] is False


def test_native_excluded_result_is_not_backfilled_by_an_archive():
    rows,links,held=materialize([native(home_goals=None,away_goals=None)],[],[historical()],DIRECTORY)
    assert not rows and not links
    assert held[0]['reason']=='native_overlap_needs_review'


def test_new_season_uses_reviewed_teams_without_inventing_a_season_id():
    rows,links,held=materialize([native()],[],[historical(season='2022/2023',home_team='Alfa')],DIRECTORY)
    assert not links and not held
    ext=next(r for r in rows if r['source']=='External')
    assert ext['season_id'] is None and ext['competition_id']==17
    assert ext['home_id'] is None and ext['home_team_key']=='External:Premier_League:alpha'
    assert ext['away_id']==2 and ext['kickoff_utc'] is None and not ext['replay_eligible']


def test_reviewed_completion_uses_utc_day_guard_without_fabricating_kickoff():
    review=dict(fixture_key='archive:1',completion_date='2024-01-05',timezone='America/Lima',
                regulation_verified=True,evidence='match report, replay completed on Jan 5')
    rows,_,_=materialize([native()],[],[historical(season='2022/2023')],DIRECTORY,[review])
    ext=next(r for r in rows if r['source']=='External')
    assert ext['kickoff_utc'] is None and ext['replay_eligible']
    assert ext['replay_available_at']==datetime(2024,1,7,tzinfo=UTC)
    with pytest.raises(ValueError,match='predates'):
        materialize([native()],[],[historical(season='2022/2023')],DIRECTORY,
                    [dict(review,completion_date='2024-01-01')])


def test_explicit_result_availability_waits_past_completion_day():
    r=native(result_available_at=datetime(2024,1,4,9,tzinfo=UTC))
    rows,_,_=materialize([r],[],[],DIRECTORY)
    assert rows[0]['replay_available_at']==datetime(2024,1,5,tzinfo=UTC)


@pytest.mark.parametrize('changes',[dict(is_awarded=True),dict(is_awarded=None),
                     dict(status='notstarted'),dict(home_score_extra1=0),dict(status_code=120)])
def test_nonregulation_and_admin_results_are_excluded(changes):
    r=dict(status='finished',is_awarded=False,home_score_current=1,away_score_current=0)
    assert native_score(r)==(1,0)
    assert native_score(dict(r,**changes)) is None


def test_calendar_leagues_and_isdb_revision():
    assert historical_season('J1','16-17',date(2016,2,1))=='2016/2016'
    assert historical_season('Premier_League','16-17',date(2017,2,1))=='2016/2017'


def test_cli_build_is_readonly_versioned_and_reproducible(export,tmp_path):
    from xdiyo_analytics.data.scorelines import main
    table=export.table('matches')
    for key, values in dict(home_name=['Alpha','Gamma'],away_name=['Beta','Delta'],
                            is_awarded=[False,True],status=['finished','finished'],
                            home_score_current=[1.,1.],away_score_current=[0.,1.]).items():
        table=table.append_column(key,pa.array(values))
    export.replace('matches',table)
    before=export.snapshot()
    txt=tmp_path/'results.txt'
    txt.write_text('Sea\tLge\tDate\tHT\tAT\tHS\tAS\tGD\tWDL\n20-21\tENG1\t03/02/2021\tAlpha\tBeta\t1\t0\t1\tW\n')
    arc=tmp_path/'archive.tsv'
    arc.write_text('league\tseason\thome_id\taway_id\thome_team\taway_team\thome_goals\taway_goals\tsource\tsource_fixture_id\tsource_reported_date\tkickoff_utc\tsource_url\n')
    args=['--sofascore-root',str(export.root),'--isdb',str(txt),'--archive',str(arc)]
    i=tmp_path/'one';j=tmp_path/'two'
    main([*args,'--output',str(i)]);main([*args,'--output',str(j)])
    assert export.snapshot()==before
    assert {(m['source'],m['event_id']) for m in pq.read_table(i/'scorelines.parquet').to_pylist()}=={
        ('SofaScore',2**53+1),('ISDBv2',None)}
    assert pq.read_table(i/'scorelines_replay.parquet').num_rows==0
    assert json.loads((i/'manifest.json').read_bytes())==json.loads((j/'manifest.json').read_bytes())
    with pytest.raises(ValueError,match='new output'):
        build(export.root,txt,arc,i)
