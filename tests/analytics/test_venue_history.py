"""Fixed historical venue is independent of target side and for/against role."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.features import (MatchScore, RollingMean, RollingStd,
    RollingSkewness, RollingZScore, EMA, Lag, H2H, ForAgainst,
    RollingWeightedMean, WarmStart, SeededEMA, Hard, DerivedStandings, evaluate_features)
from xdiyo_analytics.ratings import build_standings, StandingsRun
from test_derived_standings import population, TOTALS
from transition_samples import A, B, C, STAT, OWN, history_from_games


def paper_history():
    # Liverpool's six league games preceding Manchester City on 1 April 2023.
    games = [('2023-02-13','home',2,0), ('2023-02-18','away',2,0),
             ('2023-02-25','away',0,0), ('2023-03-01','home',2,0),
             ('2023-03-05','home',7,0), ('2023-03-11','away',0,1),
             ('2023-04-01','away',1,4)]
    return pd.DataFrame([dict(event_id=i,team_id=1,opponent_id=2,
        competition_id=17,season_id=1,status='finished',side=side,
        kickoff_at=pd.Timestamp(day,tz='UTC'),goals_for=gf,goals_against=ga)
        for i,(day,side,gf,ga) in enumerate(games)])


def test_paper_example_independent_venue_and_goal_perspective():
    h=paper_history()
    defs={f'{side}_{venue}':RollingMean(MatchScore(side=side),3,venue=venue)
          for side in ('for','against') for venue in ('all','home','away','same')}
    defs['total6']=RollingMean(MatchScore(),6)
    f=evaluate_features(h,defs)
    expected={'for_all':3,'for_home':11/3,'for_away':2/3,'for_same':2/3,
              'against_all':1/3,'against_home':0,'against_away':1/3,'against_same':1/3,
              'total6':13/6}
    assert f.iloc[-1].to_dict()==pytest.approx(expected)
    h.loc[h.index[-1],'side']='home'
    home=evaluate_features(h,defs).iloc[-1]
    for name in expected:
        if name.endswith('_same'): continue
        assert home[name]==f.iloc[-1][name]
    assert home.for_same==pytest.approx(11/3)


@pytest.mark.parametrize('venue',['home','away'])
@pytest.mark.parametrize('op', [Lag(MatchScore(),2), RollingMean(MatchScore(),3),
    RollingStd(MatchScore(),3), RollingSkewness(MatchScore(),3),
    RollingZScore(MatchScore(),3), EMA(MatchScore(),3)])
def test_fixed_venue_matches_independent_prefiltered_history(op,venue):
    h=paper_history()
    definitions={'v':replace(op,venue=venue)}
    got=evaluate_features(h,definitions)
    # Independent oracle: physically omit other venues before computing defaults.
    for i in range(len(h)):
        subset=h.loc[(h.index<i)&h.side.eq(venue)]
        subset=pd.concat([subset,h.iloc[[i]]])
        expected=evaluate_features(subset,{'v':replace(op,venue='all')}).iloc[-1]
        np.testing.assert_allclose(got.iloc[i],expected,equal_nan=True)


def test_missing_window_slots_release_and_future_safety():
    h=paper_history();h.loc[4,'goals_for']=np.nan
    d={'v':RollingMean(MatchScore(),2,venue='home'),
       'strict':RollingMean(MatchScore(),2,min_periods=2,venue='home')}
    got=evaluate_features(h,d)
    assert got.v.iloc[-1]==2 and np.isnan(got.strict.iloc[-1])
    # First home observation cannot fill the missing last-two-home slot.
    h.loc[0,'goals_for']=100
    assert evaluate_features(h,d).v.iloc[-1]==2
    release=h.kickoff_at+pd.Timedelta(hours=3)
    release.iloc[3]=h.kickoff_at.iloc[-1]+pd.Timedelta(days=1)
    assert evaluate_features(h,d,available_at=release).v.iloc[-1]==100
    # Changing target/future values has no effect on earlier predictions.
    original=evaluate_features(h,d)
    h.loc[5:,'goals_for']=999
    pd.testing.assert_frame_equal(evaluate_features(h,d).iloc[:6],original.iloc[:6])


def test_stat_opponent_view_h2h_and_weighted():
    h=history_from_games([dict(home=A,away=B,values=(1,10)),
        dict(home=B,away=A,values=(2,20)),dict(home=A,away=C,values=(3,30)),
        dict(home=A,away=B,values=(4,40)),dict(home=B,away=A,values=(5,50))])
    d={'home_conceded':RollingMean(ForAgainst(STAT,'against'),2,venue='home'),
       'h2h':H2H(RollingMean(ForAgainst(STAT,'against'),2,venue='home')),
       'weighted':RollingWeightedMean(STAT,STAT,window=2,venue='home')}
    f=evaluate_features(h,d)
    assert f.home_conceded.iloc[-1]==35
    assert f.h2h.iloc[-1]==25
    assert f.weighted.iloc[-1]==pytest.approx((3*3+4*4)/7)


@pytest.mark.parametrize('mode',['uniform','legacy'])
@pytest.mark.parametrize('venue,expected',[('home',4),('away',20)])
def test_seeded_prior_filters_before_boundary_window(venue,expected,mode):
    h=history_from_games([dict(day=0,home=A,away=B,values=(2,8)),
        dict(day=2,home=B,away=A,values=(99,20)),
        dict(day=4,home=A,away=B,values=(6,12)),
        dict(day=10,season=2,round=1,home=B,away=A,values=(100,200))])
    op=WarmStart(RollingMean(STAT,2,venue=venue),SeededEMA(mode=mode,alpha=.5,handoff=Hard(10)))
    result=evaluate_features(h,{'v':op})
    assert result.v.iloc[-1]==expected


def test_venue_requires_valid_side_and_preserves_population_restriction():
    from xdiyo_analytics.features import League
    h=paper_history()
    for invalid in (h.drop(columns='side'),h.assign(side='neutral')):
        with pytest.raises(ValueError,match='side metadata'):
            evaluate_features(invalid,{'v':RollingMean(MatchScore(),venue='home')})
    with pytest.raises(ValueError,match='venue'):
        DerivedStandings(venue='neutral')
    h=history_from_games([{},{}])
    with pytest.raises(ValueError,match='team-history'):
        evaluate_features(h,{'v':RollingMean(League(STAT),venue='away')})


def test_venue_rank_future_mutation_and_shared_build(monkeypatch):
    import xdiyo_analytics.ratings.standings as module
    h,m=population(4)
    q=h.loc[h['round'].eq(4),'kickoff_at'].min()
    query=h.loc[h.kickoff_at.ge(q)]
    a=build_standings(h,team_seasons=m,queries=query,cutoffs=q,
        available_at='release',venues=['home','away'])
    changed=h.copy()
    changed.loc[changed.release.gt(q),['goals_for','goals_against']]+=100
    b=build_standings(changed,team_seasons=m,queries=query,cutoffs=q,
        available_at='release',venues=['home','away'])
    pd.testing.assert_frame_equal(a.snapshots,b.snapshots)
    calls=[];native=module.build_standings
    def counted(*args,**kwargs):
        calls.append(kwargs)
        return native(*args,**kwargs)
    monkeypatch.setattr(module,'build_standings',counted)
    defs={v:DerivedStandings([3],venue=v) for v in ('all','home','away','same')}
    evaluate_features(h,defs,team_seasons=m)
    assert len(calls)==1 and calls[0]['venues']==['all','home','away']


@pytest.mark.parametrize('venue',['home','away'])
@pytest.mark.parametrize('delayed',[False,True])
def test_venue_standings_match_independent_reconstruction(venue,delayed):
    h,m=population(4,seasons=(1,2))
    if delayed:
        h.loc[h['round'].eq(2),'release']+=pd.Timedelta(days=8)
    run=build_standings(h,team_seasons=m,normalized_ranks=[1,3],venues=[venue],available_at='release')
    for row in run.snapshots.itertuples(index=False):
        query=h.loc[(h.season_id==row.season_id)&(h.kickoff_at>=row.recorded_at)].iloc[0]
        # Mask contributions at the other venue, retaining first-round gate from
        # the complete population. Build totals/ranks without production replay.
        group=h.loc[h.season_id.eq(row.season_id)]
        allowed=group.status.eq('finished')&group.kickoff_at.lt(row.recorded_at)&group.release.le(row.recorded_at)
        gate=set(group.loc[group['round'].eq(1),'event_id'])<=set(group.loc[allowed,'event_id'])
        totals={}
        for team in m.loc[m.season_id.eq(row.season_id),'team_id']:
            games=group.loc[allowed&group.side.eq(venue)&group.team_id.eq(team)].sort_values(['kickoff_at','event_id'])
            if row.window: games=games.tail(row.window)
            count=len(games)
            if not gate: games=games.iloc[:0]
            gf,ga=games.goals_for,games.goals_against
            wins=int((gf>ga).sum());draws=int((gf==ga).sum())
            totals[team]=[len(games),wins,draws,len(games)-wins-draws,3*wins+draws,int(gf.sum()),int(ga.sum()),int((gf-ga).sum())]
            if team==row.team_id: assert row.effective_count==count
        keys={t:(v[4],v[7],v[5],-v[6]) for t,v in totals.items()}
        rank=1+sum(k>keys[row.team_id] for k in keys.values())
        assert row.rank==rank and row.gate_open==gate
        assert [getattr(row,c) for c in TOTALS]==totals[row.team_id]


def test_rank_roles_same_venue_persistence_and_cache_identity(tmp_path):
    h,m=population(4)
    run=build_standings(h,team_seasons=m,normalized_ranks=[3],venues=['all','home','away'],available_at='release')
    plain=build_standings(h,team_seasons=m,normalized_ranks=[3],available_at='release')
    expr=DerivedStandings([3])
    pd.testing.assert_frame_equal(run.features(h,expr),plain.features(h,expr))
    assert run.metadata['state_sha256']!=plain.metadata['state_sha256']
    outputs={v:run.features(h,replace(expr,side='both',venue=v)) for v in ('home','away','same')}
    for i,row in h.iterrows():
        other='away' if row.side=='home' else 'home'
        assert outputs['same'].iloc[i,0]==outputs[row.side].iloc[i,0]
        assert outputs['same'].iloc[i,1]==outputs[other].iloc[i,1]
    run.save(tmp_path/'extended');plain.save(tmp_path/'legacy')
    for venue in ('all','home','away','same'):
        e=replace(expr,venue=venue,side='both')
        pd.testing.assert_frame_equal(StandingsRun.load(tmp_path/'extended').features(h,e),run.features(h,e))
    pd.testing.assert_frame_equal(StandingsRun.load(tmp_path/'legacy').features(h,expr),plain.features(h,expr))
    with pytest.raises(ValueError,match='venues are absent'): plain.features(h,replace(expr,venue='home'))


def test_native_recipe_inventory_and_eighteen_fixture_predictors():
    from examples.venue_history import definitions,recipe
    from xdiyo_analytics.ui.recipe import catalog_for_ui,export_python
    from xdiyo_analytics.ui.inventory import inventory
    from xdiyo_analytics.labels import create_labels,Outcome
    from xdiyo_analytics.datasets import assemble_dataset
    cat=catalog_for_ui()
    for expr in definitions().values():
        assert cat.build(json.loads(json.dumps(cat.encode(expr))))==expr
    compile(export_python(recipe('unused')),'recipe.py','exec')
    for component in ('RollingMean','RollingStd','RollingSkewness','RollingZScore','EMA','Lag','RollingWeightedMean','DerivedStandings'):
        field=next(f for f in inventory()['components']['features.'+component]['fields'] if f['name']=='venue')
        assert [c['value'] for c in field['choices']]==['all','same','home','away']
        assert not any(k.startswith('visible_when') for k in field)
    h,m=population(4)
    f=evaluate_features(h,definitions(),team_seasons=m,available_at='release',keyed=True)
    assert f.shape[1]==9
    labels=create_labels(h,{'outcome':Outcome(perspective='home')})
    data=assemble_dataset(f,labels['outcome'],layout='match')
    assert data.X.shape[1]==18
