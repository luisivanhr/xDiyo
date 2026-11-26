from datetime import date, datetime, timedelta, timezone
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
    assert rows[0]['replay_eligible'] is True


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


def test_publication_timestamp_does_not_override_prior_day_policy():
    r=native(result_available_at=datetime(2024,1,4,9,tzinfo=UTC))
    rows,_,_=materialize([r],[],[],DIRECTORY)
    assert rows[0]['replay_available_at']==datetime(2024,1,4,tzinfo=UTC)
    assert rows[0]['replay_basis']=='source_reported_prior_day'


def test_known_resumption_without_completion_evidence_is_held_from_replay():
    rows,_,_=materialize([native(completion_needs_review=True)],[],[],DIRECTORY)
    assert not rows[0]['replay_eligible']
    assert rows[0]['replay_basis']=='completion_day_needs_review'


def test_date_only_result_with_reviewed_identity_can_supply_prior_day_history():
    rows,_,_=materialize([native()],[],[historical(season='2022/2023')],DIRECTORY)
    ext=next(r for r in rows if r['source']=='External')
    assert ext['replay_eligible'] and ext['kickoff_utc'] is None
    assert ext['replay_available_at']==datetime(2024,1,4,tzinfo=UTC)


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
    replay=pq.read_table(i/'scorelines_replay.parquet').to_pylist()
    assert len(replay)==2
    assert replay[0]['home_id']==replay[1]['home_id']==2**53+3
    assert replay[0]['kickoff_utc'] is None
    assert (i/'team_crosswalk.json').is_file()
    assert json.loads((i/'manifest.json').read_bytes())==json.loads((j/'manifest.json').read_bytes())
    with pytest.raises(ValueError,match='new output'):
        build(export.root,txt,arc,i)


def test_crosswalk_uses_two_independent_ordered_fixtures_for_alias():
    from xdiyo_analytics.data.scoreline_identities import compile_crosswalk

    second=DAY.replace(day=10)
    natives=[native(),native(fixture_key='sofascore:124',source_reported_date=second+timedelta(days=1),
                            away_id=3,away_team='Gamma')]
    ext=[historical(home_team='Alias Alpha'),historical('archive:2',home_team='Alias Alpha',away_team='Gamma',
                        source_away_key='gamma',source_reported_date=second)]
    decisions,proof=compile_crosswalk(natives,ext,[],lambda *_:None)
    k=('External','Premier_League','2023/2024','alpha')
    assert decisions[k]==1
    record=next(r for r in proof if r['source_team_key']=='alpha')
    assert record['decision']=='repeated_ordered_fixture'
    assert record['evidence_fixture_keys']==['sofascore:123','sofascore:124']

    # A repeat against the same opponent doesn't independently identify a club.
    natives[1].update(away_id=2,away_team='Beta')
    ext[1][0].update(away_team='Beta',source_away_key='beta')
    decisions,_=compile_crosswalk(natives,ext,[],lambda *_:None)
    assert k not in decisions


def test_crosswalk_preserves_reserve_name_and_holds_reused_names():
    from xdiyo_analytics.data.scoreline_identities import compile_crosswalk

    old=native(fixture_key='sofascore:1',home_id=77,home_team='Alpha',season='2020/2021')
    reserve=native(fixture_key='sofascore:2',home_id=9,home_team='Alpha B')
    ext=[historical(),historical('archive:2',home_team='Alpha B',source_home_key='alpha_b',
                                 source_reported_date=DAY.replace(day=12))]
    # Put the external names out of fixture support so this checks fallback.
    ext[0][0]['source_reported_date']=DAY.replace(day=11)
    decisions,_=compile_crosswalk([native(),old,reserve],ext,[],lambda *_:None)
    assert ('External','Premier_League','2023/2024','alpha') not in decisions
    assert decisions['External','Premier_League','2023/2024','alpha_b']==9


def test_crosswalk_holds_exact_name_when_ordered_fixture_contradicts_it():
    from xdiyo_analytics.data.scoreline_identities import compile_crosswalk

    exact=native(fixture_key='sofascore:200',source_reported_date=DAY.replace(day=10))
    other=native(home_id=12,home_team='Other Alpha')
    decisions,proof=compile_crosswalk([exact,other],[historical()],[],lambda *_:None)
    assert ('External','Premier_League','2023/2024','alpha') not in decisions
    assert next(r for r in proof if r['source_team_key']=='alpha')['alternative_ids']==[1,12]


def test_compiler_quarantines_two_source_keys_that_play_against_each_other():
    from xdiyo_analytics.data.scoreline_identities import compile_crosswalk

    match=historical(home_team='Alpha',away_team='Alpha alias',source_away_key='alias',
                     source_reported_date=DAY.replace(day=20))
    decisions,proof=compile_crosswalk([native()], [match], [], lambda *args:1)
    assert not decisions
    assert {r['decision'] for r in proof}=={'opponents_collapse'}


def test_slug_pass_crosses_divisions_but_not_borders_or_reserve_qualifiers():
    from xdiyo_analytics.data.scoreline_identities import compile_crosswalk, slug_key

    assert slug_key('São Paulo B')=='sao-paulo-b'
    assert slug_key('Łódź II')=='lodz-ii'
    top=native(home_team='Current name',home_source_slug='alpha',away_source_slug='beta')
    lower=native(fixture_key='sofascore:124',league='Championship',home_id=9,home_team='Étoile',home_source_slug='etoile')
    abroad=native(fixture_key='sofascore:125',league='Ligue_1',home_id=11,home_team='Étoile',home_source_slug='etoile')
    countries={'Premier_League':'England','Championship':'England','Ligue_1':'France'}
    ext=[historical(season='2015/2016',home_team='Étoile'),historical('archive:2',season='2015/2016',
                           home_team='Étoile B',source_home_key='reserve')]
    idx=[]
    decisions,_=compile_crosswalk([top,lower,abroad],ext,[],lambda *_:None,countries=countries,slug_index=idx)
    assert decisions['External','Premier_League','2015/2016','alpha']==9
    assert ('External','Premier_League','2015/2016','reserve') not in decisions
    assert {r['team_id'] for r in idx if r['slug']=='etoile'}=={9,11}


def test_reviewed_source_team_is_bounded_and_does_not_override_conflict():
    from xdiyo_analytics.data.scoreline_identities import compile_crosswalk

    review=dict(source='External',league='Premier_League',season='2022/2023',
                source_team_key='old',source_name='Former Alpha',sofascore_team_id=1,
                evidence_urls=['https://league.example/old','https://provider.example/team/1'])
    candidate=historical(season='2022/2023',source_home_key='old',home_team='Former Alpha')
    nearby=historical('archive:2',season='2021/2022',source_home_key='old',home_team='Former Alpha')
    args=([native()], [candidate,nearby], [])
    resolver=lambda *_:None
    decisions,proof=compile_crosswalk(*args,resolver,reviewed_source_teams=[review])
    key=('External','Premier_League','2022/2023','old')
    assert decisions[key]==1 and ('External','Premier_League','2021/2022','old') not in decisions
    assert any(x['decision']=='reviewed_source_team' for x in proof)
    conflict,_=compile_crosswalk(*args,lambda league,label,day:2 if label=='Former Alpha' else None,
                                 reviewed_source_teams=[review])
    assert key not in conflict


@pytest.mark.parametrize(('overrides','match'),[
    ({'source_name':'Different'},'name disagrees'),
    ({'season':'2021/2022'},'not found'),
    ({'sofascore_team_id':99},'evidence or target'),
    ({'evidence_urls':['https://league.example/a','https://league.example/b']},'independent'),
])
def test_reviewed_source_team_rejects_stale_or_unscoped_input(overrides,match):
    from xdiyo_analytics.data.scoreline_identities import compile_crosswalk

    review=dict(source='External',league='Premier_League',season='2022/2023',
                source_team_key='old',source_name='Former Alpha',sofascore_team_id=1,
                evidence_urls=['https://league.example/old','https://provider.example/team/1'])
    review.update(overrides)
    candidate=historical(season='2022/2023',source_home_key='old',home_team='Former Alpha')
    with pytest.raises(ValueError,match=match):
        compile_crosswalk([native()],[candidate],[],lambda *_:None,reviewed_source_teams=[review])


def test_mens_porto_b_slug_uses_pinned_mens_competitions_only():
    from xdiyo_analytics.data.scoreline_identities import compile_crosswalk
    men=native(league='Liga_Portugal_2',home_team='FC Porto B',home_id=74469,
               home_source_slug='fc-porto-b')
    women=native(fixture_key='sofascore:456',league='Portugal_Feminina',
                 home_team='FC Porto B',home_id=1164689,home_source_slug='fc-porto-b')
    archive=historical(league='Liga_Portugal_2',season='2013/2014',home_team='FC Porto B',
                       source_home_key='old-porto-b')
    slug_index=[]
    decisions,_=compile_crosswalk([men,women],[archive],[],lambda *_:None,
                                  countries={'Liga_Portugal_2':'Portugal'},slug_index=slug_index)
    assert decisions[('External','Liga_Portugal_2','2013/2014','old-porto-b')]==74469
    assert [row['team_id'] for row in slug_index if row['slug']=='fc-porto-b']==[74469]
