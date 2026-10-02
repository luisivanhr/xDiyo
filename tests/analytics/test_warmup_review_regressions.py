"""Permanent R1-R5 regressions from Ayre's 4fba768 review, plus nearby cases."""
import pandas as pd
import pytest

from transition_samples import A, D, OWN, STAT, history_from_games, mover_games, movement
from test_explicit_warmup import cohort_history, policy, run, toy
from xdiyo_analytics.features import RollingMean, RollingStd, TeamMovement, evaluate_features
from xdiyo_analytics.features.seeded import moment_update


@pytest.mark.parametrize('status',['other_entry','promoted','relegated'])
def test_unrepresented_entrant_evidence_controls_donors_identity_and_audit(status):
    h=cohort_history(); p=policy(mode='w_league_prior',bottom=-1)
    baseline=run(h,p=p,team_seasons=movement())
    records=movement()+[dict(competition_id=10,season_id=2,team_id=D,movement=status)]
    actual=run(h,p=p,team_seasons=records)
    assert actual.value.iloc[8]==2
    assert actual.attrs['warm_start_input_hash']!=baseline.attrs['warm_start_input_hash']
    assert any(r['team_id']==D and r['season_id']==2 and r['movement']==status
               for r in actual.attrs['movement_evidence'])
    seed=next(a for a in actual.attrs['warm_start_audit']['value'] if a['team_id']==A and a['season_id']==2)
    assert D not in [d['team_id'] for d in seed['donors']]
    shuffled=run(h.sample(frac=1,random_state=12),p=p,team_seasons=list(reversed(records)),keyed=True)
    assert shuffled.loc[shuffled.index.get_level_values('event_id')==h.event_id.iloc[8]].value.iloc[0]==2


def test_own_window_excludes_newer_origin_and_preserves_proven_older_history():
    games=mover_games()
    games.insert(4,dict(day=5,competition=20,season=12,label='24_25',home=A,away=D,values=(100,0)))
    games.insert(0,dict(day=-10,competition=20,season=10,label='22_23',home=A,away=D,values=(12,0)))
    h=history_from_games(games)
    row=h.index[(h.team_id.astype(object)==A)&(h.season_id==2)][0]
    assert run(h,team_seasons=movement()).value.iloc[row]==3
    assert run(h,RollingMean(STAT,3),team_seasons=movement()).value.iloc[row]==6
    assert run(h,RollingMean(STAT,3),team_seasons=movement(),
               group_by=('team_id','competition_id','season_id')).value.iloc[row]==3


@pytest.mark.parametrize('explicit_cutoff',[False,True])
def test_undated_targets_remain_missing_without_blocking_dated_predictions(explicit_cutoff):
    h=toy(); times=h.kickoff_at.copy()
    h.loc[10:13,'kickoff_at']=pd.NaT; h.loc[10:13,'status']='notstarted'
    kwargs={'cutoffs':times} if explicit_cutoff else {}
    actual=run(h,**kwargs)
    assert actual.value.iloc[8]==15 and actual.value.iloc[10:].isna().all()
    ordinary=evaluate_features(h,{'value':RollingMean(STAT,2)},**kwargs)
    assert ordinary.value.iloc[10:].isna().all()


@pytest.mark.parametrize('flags',[(False,False),(False,None),(None,False),(True,None),(None,True),(None,None)])
@pytest.mark.parametrize('supplied',[False,True])
def test_partial_flags_remain_independent_through_warmup_and_recovery(flags,supplied):
    from xdiyo_analytics.features import WarmStart
    from xdiyo_analytics.experiments.recovery import pack,unpack
    h=toy(); kwargs={}
    if supplied:
        kwargs['team_seasons']=[dict(competition_id=c,season_id=s,team_id=t,
            got_promoted=flags[0],got_demoted=flags[1])
            for c,s,t in h[['competition_id','season_id','team_id']].drop_duplicates().itertuples(index=False,name=None)]
    else:
        h['team_got_promoted']=flags[0];h['team_got_demoted']=flags[1]
    # Evaluate warm-up first so flags also exercise the shared resolved context.
    features={'warm':WarmStart(RollingMean(STAT,2),policy()),
              'was_promoted':TeamMovement(), 'was_relegated':TeamMovement('relegated')}
    actual=unpack(pack(evaluate_features(h,features,**kwargs)))
    for col,expected in zip(['was_promoted','was_relegated'],flags):
        assert actual[col].isna().all() if expected is None else actual[col].eq(float(expected)).all()
    if True not in flags:
        assert all(r['movement']=='unknown' for r in actual.attrs['movement_evidence'])


def test_status_only_flags_and_explicit_missing_are_distinct():
    h=toy()
    record=dict(competition_id=10,season_id=2,team_id=A,movement='promoted')
    features={'was_promoted':TeamMovement(),'was_relegated':TeamMovement('relegated')}
    a=evaluate_features(h,features,team_seasons=[record])
    assert a.iloc[8].tolist()==[1,0]
    b=evaluate_features(h,features,team_seasons=[dict(record,got_demoted=None)])
    assert b.iloc[8,0]==1 and pd.isna(b.iloc[8,1])
    for invalid in (dict(record,got_promoted=False),dict(record,movement='unknown',got_promoted=True,got_demoted=True)):
        with pytest.raises(ValueError): evaluate_features(h,features,team_seasons=[invalid])


def test_alpha_one_assigns_exact_finite_value_and_matching_moments():
    assert moment_update(1e16,0,.5,1,1)==(1.,0.,1.)
    assert moment_update(-1e308,0,.5,1e308,1)==(1e308,0.,1.)
    h=toy();h.loc[[4,6],OWN]=1e16;h.loc[8,OWN]=1
    assert run(h,p=policy(alpha=1)).value.iloc[10]==1
    assert run(h,RollingStd(STAT,2,ddof=0),policy(alpha=1)).value.iloc[10]==0
    assert pd.isna(run(h,RollingStd(STAT,2,ddof=1),policy(alpha=1,variance_estimator='weighted_sample')).value.iloc[10])
