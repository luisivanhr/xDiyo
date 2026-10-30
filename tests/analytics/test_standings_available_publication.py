"""Publication repair: no initial-round barrier, exact causal and legacy contracts."""
import json
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.features import DerivedStandings, evaluate_features
from xdiyo_analytics.ratings import build_standings, StandingsRun
from xdiyo_analytics.ratings.standings import POLICY, LEGACY_POLICY, PUBLICATION_RULE, TOTALS
from test_derived_standings import population


@pytest.mark.parametrize('never',[False,True])
@pytest.mark.parametrize('venue',['all','home','away'])
def test_opening_delay_does_not_gate_any_window(never,venue):
    h,m=population(4)
    delayed=h.event_id.iloc[0]
    mask=h.event_id.eq(delayed)
    h.loc[mask,'release']=pd.NaT if never else h.release.max()+pd.Timedelta(days=120)
    run=build_standings(h,team_seasons=m,normalized_ranks=[1,2,7,99],available_at='release',venues=[venue])
    assert run.metadata['policy']==POLICY
    assert 'gate_open' not in run.snapshots
    for w,g in run.snapshots.groupby('window'):
        assert g.played.max()>0 and g['rank'].max()>1
    assert all(delayed not in p['contributing_events'] for p in run.publications)
    assert all(p['publication_rule']==PUBLICATION_RULE for p in run.publications)


def test_initial_normalization_idle_teams_and_exact_ties():
    h,m=population(4);event=h.event_id.iloc[0]
    home=h.loc[h.event_id.eq(event)&h.side.eq('home'),'team_id'].iloc[0]
    away=h.loc[h.event_id.eq(event)&h.side.eq('away'),'team_id'].iloc[0]
    h.loc[h.event_id.eq(event)&h.side.eq('home'),['goals_for','goals_against']]=[2,0]
    h.loc[h.event_id.eq(event)&h.side.eq('away'),['goals_for','goals_against']]=[0,2]
    h['release']=pd.Series(pd.NaT,index=h.index,dtype='datetime64[ns, UTC]')
    q=h.kickoff_at.min()+pd.Timedelta(hours=4)
    h.loc[h.event_id.eq(event),'release']=q
    query=h.loc[h.kickoff_at.ge(q)]
    def at(t):return build_standings(h,team_seasons=m,queries=query,cutoffs=t,available_at='release',normalized_ranks=[1,7])
    before=at(q-pd.Timedelta(1,'ns'));after=at(q)
    s=after.snapshots.query('window==0').set_index('team_id')
    assert s.loc[home,'rank']==1 and s.loc[away,'rank']==4
    assert s.loc[~s.index.isin([home,away]),'rank'].eq(2).all()
    assert s.played.sum()==2
    assert before.snapshots['rank'].eq(1).all() and before.snapshots.played.eq(0).all()
    for expr,value in [(DerivedStandings([7]),1),(DerivedStandings([],True),1),(DerivedStandings([],True,True),.25)]:
        assert before.features(query,expr,cutoffs=q-pd.Timedelta(1,'ns')).eq(value).all().all()
    assert after.snapshots.query('window==7').played.max()==1


def test_unknown_release_is_not_kickoff_and_resumed_bounds():
    h,m=population(4);q=h.kickoff_at.iloc[-1];query=h.loc[h.kickoff_at.eq(q)]
    unknown=build_standings(h,team_seasons=m,queries=query,cutoffs=q)
    assert unknown.snapshots.played.eq(0).all()
    h['result_available_at']=h.release
    event=h.event_id.iloc[0];h.loc[h.event_id.eq(event),'result_available_at']=q
    def at(t):return build_standings(h,team_seasons=m,queries=query,cutoffs=t,available_at='release')
    early=at(q-pd.Timedelta(1,'ns'));late=at(q)
    assert event not in early.publications[0]['contributing_events']
    assert event in late.publications[0]['contributing_events']
    assert query.event_id.iloc[0] not in late.publications[0]['contributing_events']
    known=build_standings(h,team_seasons=m,queries=query,cutoffs=q)
    pd.testing.assert_frame_equal(known.snapshots,late.snapshots)


@pytest.mark.parametrize('side',['for','against','both'])
@pytest.mark.parametrize('venue',['all','home','away','same'])
def test_equal_cutoff_permutation_truncated_results_and_future_mutation(side,venue):
    h,m=population(4,seasons=(1,2));q=pd.Timestamp('2020-01-10',tz='UTC')
    query=h.loc[h.season_id.eq(1)&h.kickoff_at.ge(q)]
    expr=DerivedStandings([1,4,99],True,side=side,venue=venue)
    def evaluate(source,queries=query):return evaluate_features(queries,{'r':expr},standings_history=source,team_seasons=m,standings_available_at='release',cutoffs=q)
    expected=evaluate(h)
    shuffled=evaluate(h.sample(frac=1,random_state=23),query.iloc[::-1]).loc[query.index]
    np.testing.assert_array_equal(expected,shuffled)
    future=h.release.gt(q)
    changed=h.copy();changed.loc[future,['goals_for','goals_against']]+=10
    np.testing.assert_array_equal(expected,evaluate(changed))
    # Keep verified fixture membership; truncate only unavailable outcomes.
    changed.loc[future,['goals_for','goals_against']]=np.nan
    np.testing.assert_array_equal(expected,evaluate(changed))
    assert expected.attrs['derived_standings']['r'][0]['source']['policy']==POLICY


def test_legacy_inspection_only_and_new_save_reload(tmp_path):
    h,m=population(4);run=build_standings(h,team_seasons=m,available_at='release')
    old=run.snapshots.copy();old['gate_open']=False
    old.loc[:,list(TOTALS)]=0;old['rank']=1
    pubs=[dict(p,gate_open=False,pending_first_round=[1]) for p in run.publications]
    for p in pubs:p.pop('publication_rule');p.pop('eligible_result_count')
    metadata=run.metadata;metadata['policy']=LEGACY_POLICY;metadata.pop('publication_rule')
    legacy=StandingsRun(old,pubs,metadata,run._fixtures)
    legacy.save(tmp_path/'old');loaded=StandingsRun.load(tmp_path/'old')
    pd.testing.assert_frame_equal(old,loaded.snapshots)
    with pytest.raises(ValueError,match='rebuild'):loaded.features(h,DerivedStandings())
    with pytest.raises(ValueError,match='rebuild'):evaluate_features(h,{'rank':DerivedStandings()},standings_run=loaded)
    loaded.save(tmp_path/'copy')
    assert json.loads((tmp_path/'copy'/'standings.json').read_text())['metadata']['policy']==LEGACY_POLICY
    run.save(tmp_path/'new');new=StandingsRun.load(tmp_path/'new')
    pd.testing.assert_frame_equal(run.snapshots,new.snapshots)
    pd.testing.assert_frame_equal(run.features(h,DerivedStandings()),new.features(h,DerivedStandings()))
    body=json.loads((tmp_path/'old'/'standings.json').read_text());body['version']=3;body['metadata']['policy']=POLICY
    (tmp_path/'old'/'standings.json').write_text(json.dumps(body))
    with pytest.raises(ValueError,match='schema'):StandingsRun.load(tmp_path/'old')


def test_policy_identity_and_native_cache_inputs(monkeypatch):
    import xdiyo_analytics.ratings.standings as module
    from xdiyo_analytics.experiments.recovery import training_source_files
    from pathlib import Path
    h,m=population(4)
    current=build_standings(h,team_seasons=m,available_at='release')
    monkeypatch.setattr(module,'POLICY',LEGACY_POLICY)
    previous_identity=build_standings(h,team_seasons=m,available_at='release')
    assert current.metadata['source_sha256']!=previous_identity.metadata['source_sha256']
    assert current.metadata['state_sha256']!=previous_identity.metadata['state_sha256']
    root=Path(module.__file__).resolve().parents[1]
    assert Path(module.__file__).resolve() in training_source_files(root)


def test_default_approved_release_but_explicit_unknown_stays_unknown():
    h,m=population(4)
    h['standings_goals_for']=h.goals_for;h['standings_goals_against']=h.goals_against
    h['standings_result_available_at']=h.release
    known=build_standings(h,team_seasons=m)
    assert known.snapshots.played.max()>0
    unknown=pd.Series(pd.NaT,index=h.index,dtype='datetime64[ns, UTC]')
    blocked=build_standings(h,team_seasons=m,available_at=unknown)
    assert blocked.snapshots.played.eq(0).all()


@pytest.mark.parametrize('empty',['unfinished','missing_scores','unknown_release'])
def test_season_with_no_eligible_results(empty):
    h,m=population(4)
    if empty=='unfinished':h['status']='notstarted'
    if empty=='missing_scores':h[['goals_for','goals_against']]=np.nan
    if empty=='unknown_release':h['release']=pd.Series(pd.NaT,index=h.index,dtype='datetime64[ns, UTC]')
    run=build_standings(h,team_seasons=m,available_at='release',normalized_ranks=[1,99],venues=['all','home','away'])
    assert run.snapshots[list(TOTALS)].eq(0).all().all()
    assert run.snapshots['rank'].eq(1).all()
    assert all(p['eligible_result_count']==0 for p in run.publications)
