"""Independent reconstruction oracle and native integration regressions."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.features import DerivedStandings, NormalizedStanding, evaluate_features
from xdiyo_analytics.ratings import build_standings, StandingsRun
from xdiyo_analytics.ratings.standings import competition_ranks, TOTALS


def population(n=8, seasons=(1,), base=0):
    rows, members = [], []
    for season in seasons:
        teams = [base+i+1 for i in range(n)]
        members.extend(dict(competition_id=17, season_id=season, team_id=t) for t in teams)
        wheel = teams[:]
        rounds = []
        for r in range(n-1):
            rounds.append(list(zip(wheel[:n//2], reversed(wheel[n//2:]))))
            wheel = [wheel[0], wheel[-1], *wheel[1:-1]]
        rounds += [[(a,h) for h,a in pairs] for pairs in rounds]
        for r,pairs in enumerate(rounds,1):
            for j,(h,a) in enumerate(pairs):
                event = season*10000+r*100+j
                kick = pd.Timestamp(f'{2019+season}-01-01', tz='UTC')+pd.Timedelta(days=r*3+j/10)
                gf,ga = (r*3+j)%5,(r+j*2)%4
                for side,t,o,x,y in [('home',h,a,gf,ga),('away',a,h,ga,gf)]:
                    rows.append(dict(event_id=event,competition_id=17,season_id=season,
                        team_id=t,opponent_id=o,side=side,round=r,kickoff_at=kick,
                        release=kick+pd.Timedelta(hours=3),status='finished',goals_for=x,goals_against=y,
                        team_position=2,opponent_position=3,result='W' if x>y else 'D' if x==y else 'L'))
    return pd.DataFrame(rows),pd.DataFrame(members)


def oracle(h, members, query, q, w):
    """Slow from-scratch logic, independent of producer canonicalization/replay."""
    group = h.loc[(h.competition_id==query.competition_id)&(h.season_id==query.season_id)]
    first = group.loc[group['round']==1]
    good = group.status.eq('finished') & group.goals_for.notna() & group.goals_against.notna()
    permitted = good & group.kickoff_at.lt(q) & group.release.le(q) & group.event_id.ne(query.event_id)
    gate = first.event_id.nunique() == group.loc[permitted & group['round'].eq(1)].event_id.nunique()
    teams=members.loc[(members.competition_id==query.competition_id)&(members.season_id==query.season_id),'team_id'].tolist()
    totals={}
    for t in teams:
        played=group.loc[permitted & group.team_id.eq(t)].sort_values(['kickoff_at','event_id'])
        if w:played=played.tail(w)
        games=list(played[['goals_for','goals_against']].itertuples(index=False,name=None)) if gate else []
        wins=sum(a>b for a,b in games); draws=sum(a==b for a,b in games)
        gf=sum(int(a) for a,b in games);ga=sum(int(b) for a,b in games)
        totals[t]=[len(games),wins,draws,len(games)-wins-draws,3*wins+draws,gf,ga,gf-ga]
    keys={t:(v[4],v[7],v[5],-v[6]) for t,v in totals.items()}
    ranks={t:1+sum(k>keys[t] for k in keys.values()) for t in teams}
    return totals,ranks,gate


@pytest.mark.parametrize('windows',[(3,6),(1,5,12,30),(),(7,)])
@pytest.mark.parametrize('delayed',[False,True])
def test_replay_matches_independent_oracle(windows,delayed):
    h,m=population()
    if delayed:
        h.loc[h['round'].eq(1)&h.event_id.eq(h.event_id.iloc[0]),'release']+=pd.Timedelta(days=12)
        h.loc[h['round'].eq(4),'release']+=pd.Timedelta(days=8)
    before=h.copy(deep=True)
    run=build_standings(h,team_seasons=m,normalized_ranks=windows,available_at='release')
    for row in run.snapshots.itertuples(index=False):
        query=h.loc[(h.season_id==row.season_id)&(h.kickoff_at>=row.recorded_at)].iloc[0]
        sums,ranks,gate=oracle(h,m,query,row.recorded_at,row.window)
        assert row.rank==ranks[row.team_id]
        assert [getattr(row,c) for c in TOTALS]==sums[row.team_id]
        assert row.gate_open==gate
    pd.testing.assert_frame_equal(h,before)
    assert run.metadata['table_computations']==h['round'].nunique()
    if delayed:assert run.metadata['late_team_insertions']>0


def test_points_ties_and_redundant_goals_against():
    totals=[[0,0,0,0,3,0,1000,-1000],[0,0,0,0,2,10000,0,10000],
            [0,0,0,0,3,0,1000,-1000]]
    assert competition_ranks(totals)==[1,3,1]
    totals=[[0,0,0,0,4,3,1,2],[0,0,0,0,4,4,2,2],[0,0,0,0,4,4,1,3]]
    assert competition_ranks(totals)==[3,2,1]
    assert [t[6] for t in totals]==[t[5]-t[7] for t in totals]


def test_cold_start_gate_delayed_first_round_and_counts():
    h,m=population(4)
    first=h.event_id.iloc[0]
    h.loc[h.event_id.eq(first),'release']=h.kickoff_at.max()+pd.Timedelta(days=1)
    run=build_standings(h,team_seasons=m,normalized_ranks=[1,3,6],available_at='release')
    assert run.snapshots['rank'].eq(1).all()
    assert not run.snapshots.gate_open.any()
    assert run.snapshots.effective_count.max()>0
    raw=run.features(h,DerivedStandings([],True,True))
    assert raw.eq(.25).all().all()
    assert run.features(h,DerivedStandings([3,6])).eq(1).all().all()
    # Rank remains initial even when later-round matches are available.
    assert any(p['contributing_events'] for p in run.publications)


def test_missing_valid_zero_and_arithmetic():
    h,m=population(4)
    h[['goals_for','goals_against']]=0
    missing=h.event_id.iloc[-1]
    h.loc[h.event_id.eq(missing),['goals_for','goals_against']]=np.nan
    r=build_standings(h,team_seasons=m,available_at='release')
    s=r.snapshots
    assert s['rank'].eq(1).all()
    assert s.points.eq(s.played).all()
    assert s.gf.eq(0).all()
    assert any(x['event']==missing for x in r.metadata['skipped'])


@pytest.mark.parametrize('args',[{'normalized_ranks':[True]}, {'normalized_ranks':[0]},
    {'normalized_ranks':[-1]}, {'normalized_ranks':[1.5]}, {'normalized_ranks':[3,3]},
    {'normalized_ranks':3}, {'normalized_ranks':[]}, {'normalized_ranks':None}, {'side':'home'}])
def test_invalid_configuration(args):
    with pytest.raises((ValueError,TypeError)):DerivedStandings(**args)


def test_selection_shared_cache_naming_order_keyed_and_normalization(monkeypatch):
    import xdiyo_analytics.ratings.standings as module
    h,m=population(4)
    calls=[];original=module.build_standings
    def counted(*args,**kw):calls.append(kw);return original(*args,**kw)
    monkeypatch.setattr(module,'build_standings',counted)
    definitions={'form':DerivedStandings([3,6]),'with_raw':DerivedStandings([3,6],True),
                 'norm':DerivedStandings([3,6],True,True),'provider':NormalizedStanding()}
    f=evaluate_features(h,definitions,team_seasons=m,available_at='release',keyed=True)
    assert len(calls)==1
    assert f.shape[1]==9
    assert list(f.columns[:2])==['form::team::form_rank_3','form::team::form_rank_6']
    np.testing.assert_array_equal(f['form::team::form_rank_3'],f['norm::team::form_rank_3'])
    np.testing.assert_array_equal(f['with_raw::team::season_position']/4,f['norm::team::season_position_over_n'])
    baseline=evaluate_features(h,{'provider':NormalizedStanding()},keyed=True)
    pd.testing.assert_series_equal(f['provider'],baseline['provider'])
    assert f.attrs['derived_standings']['form'][0]['source']['source_sha256']==f.attrs['derived_standings']['norm'][0]['source']['source_sha256']


def test_roundtrip_immutable_and_queries_permutation(tmp_path):
    h,m=population(4,base=2**53+10)
    h.index=[1]*len(h)
    run=build_standings(h,team_seasons=m,normalized_ranks=[3,6],available_at='release')
    e=DerivedStandings([6,3],True,True,side='both')
    expected=run.features(h,e)
    run.save(tmp_path)
    loaded=StandingsRun.load(tmp_path)
    pd.testing.assert_frame_equal(loaded.features(h,e),expected)
    with pytest.raises(FileExistsError):run.save(tmp_path)
    out=run.snapshots;out['rank']=99
    assert run.snapshots['rank'].max()<99
    shuffled=h.sample(frac=1,random_state=2)
    rebuilt=build_standings(shuffled,team_seasons=m,normalized_ranks=[3,6],available_at='release')
    pd.testing.assert_frame_equal(rebuilt.features(h,e),expected)
    pd.testing.assert_frame_equal(loaded.features(h.iloc[::-1],e),expected.iloc[::-1])
    assert run.metadata['source_sha256']==rebuilt.metadata['source_sha256']


def test_late_result_future_mutation_and_cutoff_guards():
    h,m=population()
    q=h.kickoff_at.iloc[40]
    query=h.loc[h.kickoff_at.ge(q)].copy()
    run=build_standings(h,team_seasons=m,queries=query,cutoffs=q,available_at='release')
    changed=h.copy()
    changed.loc[changed.release.gt(q),'goals_for']+=40
    changed.loc[changed.release.gt(q),'goals_against']+=40
    other=build_standings(changed,team_seasons=m,queries=query,cutoffs=q,available_at='release')
    pd.testing.assert_frame_equal(run.snapshots,other.snapshots)
    assert run.metadata['table_computations']==1
    with pytest.raises(ValueError,match='cutoff'):
        build_standings(h,team_seasons=m,cutoffs=h.kickoff_at+pd.Timedelta(days=1))
    with pytest.raises(ValueError,match='exact query cutoff'):
        run.features(query,DerivedStandings(),cutoffs=q-pd.Timedelta(seconds=1))


def test_season_reset_and_competition_isolation():
    h,m=population(4,seasons=(1,2))
    extra,em=population(6,seasons=(3,))
    extra.competition_id=99;em.competition_id=99
    h=pd.concat([h,extra]);m=pd.concat([m,em])
    r=build_standings(h,team_seasons=m,normalized_ranks=[30],available_at='release')
    for _,s in r.snapshots.groupby(['competition_id','season_id']):
        cold=s.loc[s.recorded_at.eq(s.recorded_at.min())]
        assert cold['rank'].eq(1).all() and cold.played.eq(0).all()
    assert set(r.snapshots.team_count)=={4,6}


@pytest.mark.parametrize('issue',['membership','schedule','round','singleton','stage','paired','floatid'])
def test_incomplete_conflicting_inputs_fail(issue):
    h,m=population(4)
    if issue=='membership':m=m.iloc[:-1]
    if issue=='schedule':h=h.loc[h.event_id.ne(h.event_id.iloc[0])]
    if issue=='round':h.loc[h['round'].eq(1),'round']=99
    if issue=='singleton':m=m.iloc[:1]
    if issue=='stage':h['stage_json']='{"name":"Playoff"}'
    if issue=='paired':h.loc[0,'goals_for']=90
    if issue=='floatid':h['team_id']=float(2**53)
    with pytest.raises(ValueError):build_standings(h,team_seasons=m)


def test_awarded_and_revised_scores_require_known_ruling():
    h,m=population(4)
    h['is_awarded']=False;h.loc[h.event_id.eq(h.event_id.iloc[0]),'is_awarded']=True
    with pytest.raises(ValueError,match='approved'):build_standings(h,team_seasons=m)
    h['standings_goals_for']=h.goals_for;h['standings_goals_against']=h.goals_against
    h['standings_result_available_at']=h.release
    r=build_standings(h,team_seasons=m,available_at='release')
    assert r.snapshots.gate_open.any()


def test_idle_team_reranking_and_independent_windows():
    h,m=population()
    r=build_standings(h,team_seasons=m,normalized_ranks=[1,3,6],cutoffs=h.kickoff_at,available_at='release')
    s=r.snapshots
    changed=False
    for _,g in s[s.window.eq(0)].groupby('team_id'):
        g=g.sort_values('recorded_at')
        changed |= bool((g.played.diff().eq(0)&g['rank'].diff().ne(0)&g.gate_open).any())
    assert changed
    wide=s.pivot(index=['team_id','recorded_at'],columns='window',values='rank')
    assert (wide[0]!=wide[3]).any() and (wide[3]!=wide[6]).any()
    # Last-w counts grow and then cap, not padded with synthetic matches.
    assert set(s.loc[s.window.eq(6),'effective_count'])==set(range(7))


def test_catalog_json_python_and_fixture_assembly():
    from xdiyo_analytics.ui.recipe import catalog_for_ui,default_recipe,export_python
    from xdiyo_analytics.labels import create_labels,Outcome
    from xdiyo_analytics.datasets import assemble_dataset
    cat=catalog_for_ui();e=DerivedStandings([3,6])
    assert cat.build(json.loads(json.dumps(cat.encode(e))))==e
    recipe=default_recipe();recipe['features']={'form':cat.encode(e)}
    text=export_python(recipe)
    compile(text,'recipe.py','exec')
    assert 'features.DerivedStandings' in text
    h,m=population(4)
    f=evaluate_features(h,{'form':e},team_seasons=m,available_at='release',keyed=True)
    labels=create_labels(h,{'outcome':Outcome(perspective='home')})
    d=assemble_dataset(f,labels['outcome'],layout='match')
    assert d.X.shape[1]==4
    assert all('form_rank' in c for c in d.X)


def test_form_table_is_not_average_of_previous_cumulative_positions():
    h,m=population()
    run=build_standings(h,team_seasons=m,normalized_ranks=[3,6],available_at='release')
    s=run.snapshots
    cumulative=s.loc[s.window.eq(0)].pivot(index='recorded_at',columns='team_id',values='rank')
    form=s.loc[s.window.eq(3)].pivot(index='recorded_at',columns='team_id',values='rank')
    average=cumulative.rolling(3,min_periods=1).mean()
    assert (form.ne(average)).any().any()


def test_simultaneous_atomic_boundary_and_reordered_round_dates():
    h,m=population(4)
    h['kickoff_at']=h.groupby('round').kickoff_at.transform('min')
    h['release']=h.kickoff_at+pd.Timedelta(hours=3)
    q=h.loc[h['round'].eq(1),'release'].iloc[0]
    query=h.loc[h['round'].eq(2)]
    run=build_standings(h,team_seasons=m,queries=query,cutoffs=q,available_at='release')
    assert run.snapshots.gate_open.all()
    assert run.snapshots.played.eq(1).all()
    assert len(run.publications)==1 and len(run.publications[0]['contributing_events'])==2
    before=build_standings(h,team_seasons=m,queries=query,cutoffs=q-pd.Timedelta(1,'ns'),available_at='release')
    assert not before.snapshots.gate_open.any()
    # Opening-round identity is the original assigned round, not date order.
    h.loc[h['round'].eq(1),'kickoff_at']+=pd.Timedelta(days=9)
    h.loc[h['round'].eq(1),'release']+=pd.Timedelta(days=9)
    moved=build_standings(h,team_seasons=m,available_at='release')
    earliest=moved.snapshots.recorded_at.min()
    assert not moved.snapshots.loc[moved.snapshots.recorded_at.eq(earliest),'gate_open'].any()


def test_explicit_independent_phase_membership_and_scope():
    h,m=population(4)
    second=h.copy();second['event_id']+=100000;second.kickoff_at+=pd.Timedelta(days=100);second.release+=pd.Timedelta(days=100)
    h['standings_table_id']='opening';second['standings_table_id']='closing'
    h=pd.concat([h,second],ignore_index=True);h['standings_rule']='independent_3_1_0';h['stage_json']='{"name":"Apertura/Clausura"}'
    m=pd.concat([m.assign(standings_table_id='opening'),m.assign(standings_table_id='closing')])
    r=build_standings(h,team_seasons=m,available_at='release')
    for _,g in r.snapshots.groupby('standings_table_id'):
        assert g.loc[g.recorded_at.eq(g.recorded_at.min()),'played'].eq(0).all()
    h['standings_rule']='carry_half_points'
    with pytest.raises(ValueError,match='carry/halving'):build_standings(h,team_seasons=m)


def test_full_source_subset_queries_and_current_target_exclusion():
    h,m=population(4)
    query=h.iloc[-4:].copy()
    full=evaluate_features(h,{'r':DerivedStandings([3,6])},team_seasons=m,available_at='release')
    selected=evaluate_features(query,{'r':DerivedStandings([3,6])},team_seasons=m,
        standings_history=h,standings_available_at='release')
    np.testing.assert_array_equal(full.iloc[-4:].to_numpy(),selected.to_numpy())
    with pytest.raises(ValueError,match='Incomplete schedule'):
        evaluate_features(query,{'r':DerivedStandings([3,6])},team_seasons=m)
    q=h.iloc[-2:]
    a=build_standings(h,team_seasons=m,queries=q,cutoffs=q.kickoff_at,available_at='release')
    for pub in a.publications:
        assert not set(q.event_id)&set(pub['contributing_events'])


def test_roundtrip_python_recipe_prepares_without_fit(tmp_path):
    import hashlib
    from xdiyo_analytics.ui.recipe import export_python,prepare_recipe
    from examples.derived_standings import recipe
    h,m=population(4)
    m=m.assign(movement='other_entry',got_promoted=False,got_demoted=False)
    home=h.loc[h.side.eq('home')]
    matches=home[['event_id','competition_id','season_id','round','status']].copy()
    matches['home_id']=home.team_id;matches['away_id']=home.opponent_id
    matches['kickoff_utc']=home.kickoff_at.astype('int64')//10**9
    # pandas datetime may use microsecond storage; timestamp() is unit explicit.
    matches['kickoff_utc']=[t.timestamp() for t in home.kickoff_at]
    matches['home_score_current']=home.goals_for;matches['away_score_current']=home.goals_against
    version=tmp_path/'_tables'/'competition=17'/'season=1'/'versions'/'v1';version.mkdir(parents=True)
    manifest=dict(schema_version='2',parser_version='2',version='v1',scope=dict(league_id=17,league_name='Premier_League',season_id=1,season_start=2022,season_end=2023),tables={})
    for name,frame in dict(matches=matches,team_seasons=m).items():
        path=version/(name+'.parquet');frame.to_parquet(path,index=False,compression='zstd')
        manifest['tables'][name]=dict(file=path.name,rows=len(frame),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (version/'manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'Premier_League_22_23.manifest.json').write_text(json.dumps(dict(schema_version='2',complete=True,matches=len(matches),season_file='Premier_League_22_23.parquet',manifest='_tables/competition=17/season=1/versions/v1/manifest.json')))
    r=recipe(tmp_path);r['split']['params'].update(train_size=2,test_size=1)
    prepared=prepare_recipe(r)
    assert prepared.dataset.X.shape==(12,6)
    namespace={}
    # Execute only exported preparation, explicitly never its run_recipe line.
    exec(export_python(r).split('result = run_recipe')[0],namespace)
    pd.testing.assert_frame_equal(namespace['prepared'].dataset.X,prepared.dataset.X)

    # Saved-state recipe lookup needs neither membership reloading nor replay.
    from xdiyo_analytics.data import load_seasons
    from xdiyo_analytics.histories import build_team_history
    native=load_seasons(tmp_path,['22_23'],leagues=['Premier_League'],tables=['matches','team_seasons'])
    run=build_standings(build_team_history(native),team_seasons=native['team_seasons'],normalized_ranks=[3,6])
    saved=tmp_path/'saved';run.save(saved)
    r['feature_options']['standings_run']={'component':'input.StandingsRun','params':{'directory':str(saved)}}
    manifest['tables'].pop('team_seasons')
    (version/'manifest.json').write_text(json.dumps(manifest))
    restored=prepare_recipe(r)
    np.testing.assert_array_equal(restored.dataset.X.to_numpy(),prepared.dataset.X.to_numpy())


def test_conflicting_paired_cutoffs_are_rejected():
    h,m=population(4)
    cutoffs=h.kickoff_at.copy()
    cutoffs.iloc[0]-=pd.Timedelta(seconds=1)
    with pytest.raises(ValueError,match='identical standings prediction cutoffs'):
        build_standings(h,team_seasons=m,cutoffs=cutoffs)
