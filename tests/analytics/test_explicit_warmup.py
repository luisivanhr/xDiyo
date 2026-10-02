"""Hand-calculated boundary, cohort and dispersion acceptance cases (no models)."""
from dataclasses import replace
import math
import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.features import (SeededEMA, WarmStart, Hard, LinearFade,
    ObservationCount, RollingMean, RollingStd, RollingZScore, TeamMovement,
    Stat, ForAgainst, Lag, EMA, League, LeaveOneOut, H2H, Difference, MatchScore,
    evaluate_features)
from xdiyo_analytics.features.seeded import moment_update
from transition_samples import A,B,C,D,STAT,OWN,START,history_from_games,warm_games,mover_games,movement

def policy(**kw):
    return SeededEMA(**(dict(mode='uniform', alpha=.5, handoff=Hard(10)) | kw))

def run(h, op=None, p=None, **kw):
    return evaluate_features(h, {'value':WarmStart(op or RollingMean(STAT,2),p or policy())}, **kw)

def toy():
    return history_from_games([dict(day=i, values=(v,2*v), season=1) for i,v in enumerate([0,0,12,18])]
        +[dict(day=10+i*2,season=2,round=i+1,values=(v,2*v)) for i,v in enumerate([14,8,30])])

def test_boundary_uses_final_window_and_freezes_availability():
    h=toy(); r=run(h)
    assert r.value.iloc[8]==15  # Neither full season mean 7.5 nor last prematch mean 6.
    assert r.value.iloc[10]==14.5
    late=h.available_at.copy(); late.iloc[6:8]=START+pd.Timedelta(days=11)
    r=run(h,available_at=late)
    assert r.value.iloc[8]==6 and r.value.iloc[10]==10
    audit=[a for a in r.attrs['warm_start_audit']['value'] if a['season_id']==2 and a['team_id']==A][0]
    assert audit['seed_mean']==6 and len(audit['own_window']['events'])==2

def test_recursion_and_missing_skip():
    h=toy(); h.loc[[4,6],OWN]=10
    assert run(h).value.iloc[[8,10,12]].tolist()==[10,12,10]
    h.loc[8,OWN]=np.nan
    assert run(h).value.iloc[[8,10,12]].tolist()==[10,10,9]

def cohort_history():
    games=mover_games()
    games[0]['values']=(20,0); games[1]['values']=(20,0)
    games[2]['values']=(1,5); games[3]['values']=(3,11)
    games[4]['values']=(10,0)
    return history_from_games(games)

@pytest.mark.parametrize('move,count,mean,var',[('promoted',1,8,9),('relegated',1,2,1),('promoted',-1,5,5),('relegated',-1,5,5),('promoted',20,5,5)])
def test_cohort_equal_team_mean_within_team_variance(move,count,mean,var):
    h=cohort_history(); records=movement(); records[0]['movement']=move
    p=policy(mode='w_league_prior',top=count,bottom=count)
    assert run(h,p=p,team_seasons=records).value.iloc[8]==mean
    assert run(h,RollingStd(STAT,2,ddof=0),p,team_seasons=records).value.iloc[8]==pytest.approx(math.sqrt(var))
    # Own value 20 is replaced, never mixed into cohort mean.
    assert run(h,p=p,team_seasons=records).value.iloc[10]==(mean+10)/2

def test_all_donors_need_no_rank_and_missing_donors_fallback():
    h=cohort_history(); h['team_position']=np.nan
    assert run(h,p=policy(mode='w_league_prior',bottom=-1),team_seasons=movement()).value.iloc[8]==5
    assert run(h,p=policy(mode='w_league_prior',bottom=2),team_seasons=movement()).value.iloc[8]==20
    h.loc[h.team_id.astype(object)==A,OWN]=np.nan
    assert pd.isna(run(h,p=policy(mode='w_league_prior'),team_seasons=movement()).value.iloc[8])

def test_missing_origin_does_not_manufacture_seed_or_reseed_later():
    h=history_from_games(warm_games()[2:]); p=policy()
    r=run(h,p=p); ordinary=evaluate_features(h,{'value':RollingMean(STAT,2)})
    np.testing.assert_allclose(r,ordinary,equal_nan=True)
    h=cohort_history(); records=movement(); records[0].update(previous_competition_id=None,previous_season_id=None)
    assert pd.isna(run(h,team_seasons=records).value.iloc[8])
    assert run(h,p=policy(mode='w_league_prior',bottom=-1),team_seasons=records).value.iloc[8]==5

@pytest.mark.parametrize('count',[0,-2,True,1.5])
def test_counts_reject_invalid(count):
    with pytest.raises(ValueError): policy(bottom=count)
    with pytest.raises(ValueError): policy(top=count)

def test_population_and_corrected_weighted_moments():
    assert moment_update(4,9,.5,10,.25)==pytest.approx((5.5,13.5,.34375))
    h=toy(); h.loc[[4,6],OWN]=[1,7]  # mean 4, population variance 9, Q=.5
    p=policy(alpha=.25)
    h.loc[8,OWN]=10
    assert run(h,RollingStd(STAT,2,ddof=0),p).value.iloc[10]==pytest.approx(math.sqrt(13.5))
    p=replace(p,variance_estimator='weighted_sample')
    assert run(h,RollingStd(STAT,2,ddof=1),p).value.iloc[8]==pytest.approx(math.sqrt(18))
    assert run(h,RollingStd(STAT,2,ddof=1),p).value.iloc[10]==pytest.approx(math.sqrt(13.5/(1-.34375)))
    assert pd.isna(run(h,RollingStd(STAT,2,ddof=1),replace(p,alpha=1)).value.iloc[10])
    with pytest.raises(ValueError,match='ddof'): run(h,RollingStd(STAT,2,ddof=2),p)
    with pytest.raises(ValueError,match='prior_strength'):
        run(h,RollingStd(STAT,2),replace(p,mode='w_league_prior'))

def test_sample_cohort_requires_and_uses_explicit_strength():
    h=cohort_history(); p=policy(mode='w_league_prior',bottom=-1,variance_estimator='weighted_sample',prior_strength=4)
    r=run(h,RollingStd(STAT,2),p,team_seasons=movement())
    assert r.value.iloc[8]==pytest.approx(math.sqrt(10)) # average sample variances 2 and 18
    # mean 5, C=10*(1-1/4), x=10, alpha=.5 -> C=10, Q=.3125
    assert r.value.iloc[10]==pytest.approx(math.sqrt(10/.6875))

def test_fade_interpolates_estimates_and_zero_restores_exact_ordinary():
    h=history_from_games(warm_games()); p=policy(handoff=LinearFade())
    std=run(h,RollingStd(STAT,2,ddof=0),p).value
    # Seed mean3 C1; x10->6.5 C12.75; x14->10.25 C20.4375
    assert std.iloc[8]==pytest.approx(math.sqrt(.5*20.4375+.5*4))
    assert std.iloc[10]==2
    h.loc[[6,8],OWN]=np.nan
    assert pd.isna(run(h,RollingStd(STAT,2,ddof=0),p).value.iloc[10])

def test_zscore_original_reference_and_zero_spread():
    h=toy(); p=policy()
    assert run(h,RollingZScore(STAT,2,ddof=0),p).value.iloc[8]==1
    h.loc[[4,6],OWN]=10
    assert pd.isna(run(h,RollingZScore(STAT,2,ddof=0),p).value.iloc[8])

@pytest.mark.parametrize('op',[Lag(STAT),EMA(STAT,span=3),RollingMean(League(STAT),2),RollingMean(LeaveOneOut(League(STAT)),2)])
def test_incompatible_adapters_fail_explicitly(op):
    with pytest.raises((TypeError,ValueError)): run(toy(),op)

def test_shuffle_large_ids_and_ambiguous_kickoff():
    h=toy(); r=run(h,keyed=True).sort_index()
    sh=h.sample(frac=1,random_state=8); actual=run(sh,keyed=True).sort_index()
    np.testing.assert_allclose(actual,r,equal_nan=True)
    h.loc[2:3,'kickoff_at']=h.kickoff_at.iloc[0]
    with pytest.raises(ValueError,match='unambiguous'): run(h)

def test_movement_override_missing_admin_and_validation():
    h=toy(); mask=h.season_id==2
    h['team_season_entry']=None; h['team_got_promoted']=None; h['team_got_demoted']=None
    h.loc[mask,['team_season_entry','team_got_promoted','team_got_demoted']]=['other_entry',False,False]
    feats={'was_promoted':TeamMovement('promoted'),'was_relegated':TeamMovement('relegated')}
    r=evaluate_features(h,feats)
    assert r.iloc[:8].isna().all().all() and (r.iloc[8:]==0).all().all()
    override=[dict(competition_id=10,season_id=2,team_id=A,movement='promoted')]
    r=evaluate_features(h,feats,team_seasons=override)
    assert r.was_promoted.iloc[8]==1 and r.was_promoted.iloc[9]==0
    assert r.attrs['movement_evidence'][-1]['movement']=='other_entry'
    with pytest.raises(ValueError): evaluate_features(h,feats,team_seasons=override*2)
    with pytest.raises(ValueError): evaluate_features(h,feats,team_seasons=[dict(override[0],got_promoted=False)])
    with pytest.raises(ValueError): run(h,team_seasons=[dict(override[0],previous_competition_id=20)])

def test_config_recovery_and_input_identity_changes():
    from xdiyo_analytics.experiments.recovery import pack,unpack,signature
    from xdiyo_analytics.ui.catalog import default_catalog
    p=policy(mode='w_league_prior',bottom=-1)
    assert unpack(pack(p))==p
    catalog=default_catalog(); expr=WarmStart(RollingMean(STAT,2),p)
    assert catalog.build(catalog.encode(expr))==expr
    assert catalog.build(catalog.encode(TeamMovement()))==TeamMovement()
    assert signature(p)!=signature(replace(p,bottom=2))
    h=toy(); before=run(h).attrs['warm_start_input_hash']
    h.loc[0,OWN]=50
    assert run(h).attrs['warm_start_input_hash']!=before

@pytest.mark.parametrize('operator',[RollingMean,RollingStd,RollingZScore])
@pytest.mark.parametrize('side',['for','against','both'])
def test_generic_periods_fields_perspectives(operator,side):
    h=toy(); meta=h.attrs['stat_columns']; renamed={}
    for col,spec in list(meta.items()):
        name=col.replace('cornerKicks','arbitraryNumeric').replace('::value','::total')
        renamed[name]={**spec,'key':'arbitraryNumeric','field':'total'}
        h[name]=h[col]
    h.attrs['stat_columns']=renamed
    source=ForAgainst(Stat(None,'Match overview','arbitraryNumeric',field='total'),side)
    options={} if operator is RollingMean else dict(ddof=0)
    r=run(h,operator(source,2,**options))
    width=4 if side=='both' else 2
    assert r.shape==(len(h),width)
    expected=(15 if operator is RollingMean else 3 if operator is RollingStd else 1)
    if side=='against' and operator is not RollingZScore: expected*=2
    assert r.iloc[8,0]==expected
    if operator is RollingZScore: assert pd.isna(r.iloc[8,1]) # constant half-period

def test_derived_observations_nested_lags_and_warmed_leaves():
    h=toy()
    derived=Difference(STAT,ForAgainst(STAT,'against'))
    assert run(h,RollingMean(derived,2)).value.iloc[8]==-15
    assert run(h,RollingStd(derived,2,ddof=0)).value.iloc[8]==3
    # Historical lag observations at days 2,3 are 0,12; no unwrapping to 12,18.
    assert run(h,RollingMean(Lag(STAT),2)).value.iloc[8]==6
    from xdiyo_analytics.features import Constant
    r=evaluate_features(h,{'d':Difference(WarmStart(RollingMean(STAT,2),policy()),Constant(2))})
    assert r.d.iloc[8]==13 and r.attrs['warm_start_audit']['d']
    with pytest.raises(ValueError,match='Observed'):
        evaluate_features(h,{'bad':Difference(STAT,Constant(1))})

def test_no_refill_minimums_cutoffs_future_mutations_and_postponement():
    h=toy(); h.loc[6,OWN]=np.nan
    assert run(h).value.iloc[8]==12 # Not (0+12)/2; missing counts toward window.
    assert pd.isna(run(h,RollingMean(STAT,2,min_periods=2)).value.iloc[8])
    h=toy(); r=run(h)
    changed=h.copy(); changed.loc[10:,OWN]=999; changed.loc[8:,'team_position']=999
    assert run(changed).value.iloc[8:11].tolist()==r.value.iloc[8:11].tolist()
    cutoffs=h.kickoff_at.copy(); cutoffs.iloc[10:12]=h.kickoff_at.iloc[8]
    assert run(h,cutoffs=cutoffs).value.iloc[10]==15
    # An extra unplayed fixture holds round 1 open. Hard(1) keeps the EMA.
    games=warm_games()+[dict(day=20,season=2,round=1,home=C,away=D,status='notstarted')]
    h=history_from_games(games)
    assert run(h,p=policy(handoff=Hard(1))).value.iloc[6]==6.5

def test_h2h_custom_groups_and_season_scope():
    h=toy(); h.loc[6:7,'stage']='playoff'
    assert run(h,group_by=('team_id','competition_id','stage')).value.iloc[8]==6
    assert evaluate_features(h,{'x':H2H(WarmStart(RollingMean(STAT,2),policy()))}).x.iloc[8]==15
    # Extra custom grouping is retained for updates, too.
    h.loc[8:9,'stage']='playoff'
    assert run(h,group_by=('team_id','competition_id','stage')).value.iloc[10]==6
    scoped=run(h,p=policy(handoff=Hard(0)),group_by=('team_id','competition_id','season_id'))
    assert pd.isna(scoped.value.iloc[8])

@pytest.mark.parametrize('layout',['match','team_match'])
def test_flags_large_identity_keyed_assembly(layout):
    from xdiyo_analytics.datasets import assemble_dataset
    from xdiyo_analytics.labels import create_labels,MatchTotal,TeamValue
    h=toy(); feats={'was_promoted':TeamMovement(),'was_relegated':TeamMovement('relegated')}
    records=[dict(competition_id=10,season_id=2,team_id=A,movement='promoted'),
             dict(competition_id=10,season_id=2,team_id=B,movement='relegated')]
    f=evaluate_features(h,feats,team_seasons=records,keyed=True).sample(frac=1,random_state=3)
    label=create_labels(h,{'y':(MatchTotal if layout=='match' else TeamValue)(STAT)})['y']
    data=assemble_dataset(f,label,layout=layout)
    assert len(data.X)==(7 if layout=='match' else 14)
    if layout=='match':
        assert data.X.iloc[4].tolist()==[1,0,0,1]
    else: assert data.X.iloc[8:10].values.tolist()==[[1,0],[0,1]]
    assert data.X.iloc[0].isna().all()
    assert 'movement_evidence' in data.definitions

@pytest.mark.parametrize('normalization',['mass','density','count'])
@pytest.mark.parametrize('operator',[RollingMean,RollingStd,RollingZScore])
@pytest.mark.parametrize('method',['grid','gaussian'])
def test_spatial_moments_regions_metadata_and_rotation(normalization,operator,method):
    from xdiyo_analytics.features import Heatmap,RegionMass
    h=toy(); maps=[]
    for row in h.itertuples():
        weight=float(h.loc[row.Index,OWN])
        maps.append(dict(event_id=row.event_id,team_id=row.team_id,source_season=row.source_season,
                         x=0.,y=0.,kind='touch',weight=weight+1))
        maps.append(dict(event_id=row.event_id,team_id=row.team_id,source_season=row.source_season,
                         x=100.,y=100.,kind='touch',weight=1.))
    maps=pd.DataFrame(maps,dtype=object)
    maps['weight']=maps.weight.astype(float)
    src=Heatmap(grid_size=2,normalization=normalization,use_weights=True,method=method)
    options={} if operator is RollingMean else dict(ddof=0)
    r=run(h,operator(src,2,**options),heatmaps=maps)
    assert r.attrs['spatial_features']
    # Home focal lower-left cell; away displayed upper-right after final rotation.
    assert r.iloc[8,0]>0 and r.iloc[9,3]>0
    assert r.iloc[8,1]==0 if operator is not RollingZScore else pd.isna(r.iloc[8,1])
    region=run(h,operator(RegionMass(src,'own_half'),2,**options),heatmaps=maps)
    assert region.value.iloc[8]==pytest.approx(r.iloc[8,0]*(2500 if normalization=='density' and operator is not RollingZScore else 1))
    # Same-venue history is filtered before the two-match window.
    h.loc[6,'side']='away'; h.loc[7,'side']='home'
    same=run(h,RollingMean(src,2,venue='same'),heatmaps=maps)
    assert same.iloc[8,0]!=run(h,RollingMean(src,2),heatmaps=maps).iloc[8,0]
    maps.loc[4,'weight']+=10
    assert r.attrs['warm_start_input_hash']!=run(h,operator(src,2,**options),heatmaps=maps).attrs['warm_start_input_hash']

def test_explicit_z_reference_large_offsets_and_legacy_recipe():
    from xdiyo_analytics.features import Constant
    h=toy(); op=RollingZScore(STAT,2,ddof=0,reference=Constant(21))
    assert run(h,op).value.iloc[8]==2
    h.loc[[4,6],OWN]=[1e12-1,1e12+1]
    assert run(h,RollingStd(STAT,2,ddof=0)).value.iloc[8]==1
    from xdiyo_analytics.ui.catalog import default_catalog
    old={'component':'features.SeededEMA','params':{'alpha':.5}}
    assert default_catalog().build(old)==SeededEMA(.5)

def test_cohort_ties_unequal_counts_and_eligibility_before_selection():
    h=cohort_history(); h['team_position']=1
    r=run(h,p=policy(mode='w_league_prior',bottom=1),team_seasons=movement())
    assert r.value.iloc[8]==2 # identity B precedes D, regardless of source magnitude
    h.loc[4,OWN]=np.nan # B has one valid sample (3), D has two (5,11 -> 8)
    r=run(h,p=policy(mode='w_league_prior',bottom=-1),team_seasons=movement())
    assert r.value.iloc[8]==5.5 # Not the pooled (3+5+11)/3
    r=run(h,RollingStd(STAT,2,ddof=1),policy(mode='w_league_prior',bottom=1,
          variance_estimator='weighted_sample',prior_strength=4),team_seasons=movement())
    assert r.value.iloc[8]==pytest.approx(math.sqrt(18)) # B is ineligible before tie/rank

def test_pure_replacement_and_current_entrant_exclusion():
    h=cohort_history(); h.loc[[4,6],OWN]=[3,5] # donor B mean4; D mean8
    p=policy(mode='w_league_prior',bottom=-1)
    assert run(h,p=p,team_seasons=movement()).value.iloc[[8,10]].tolist()==[6,8]
    # B appears this season with explicitly administrative status: exclude it.
    records=movement()+[dict(competition_id=10,season_id=2,team_id=B,movement='other_entry')]
    assert run(h,p=p,team_seasons=records).value.iloc[8]==8

def test_uniform_mover_ignores_legacy_weight_and_stale_destination_is_only_ordinary():
    games=[dict(day=-10,competition=10,season=9,label='22_23',home=A,away=B,values=(100,0))]+mover_games()
    h=history_from_games(games); p=policy(handoff=Hard(1))
    # Source league own state = mean(2,4)=3. Older destination visit is not its seed.
    assert run(h,p=p,team_seasons=movement()).value.iloc[10]==3
    assert run(h,p=replace(p,league_weight=1),team_seasons=movement()).value.iloc[10]==3
    assert run(h,p=p,team_seasons=movement()).value.iloc[12]==75 # ordinary (100+50)/2 after handoff
    # Larger own window may extend into still older seasons of the origin league.
    origin=[dict(day=-20,competition=20,season=8,label='22_23',home=A,away=C,values=(12,0))]+mover_games()
    h=history_from_games(origin)
    assert run(h,RollingMean(STAT,3),team_seasons=movement()).value.iloc[10]==6

def test_cohort_flags_from_history_match_explicit_evidence():
    h=cohort_history()
    h['team_season_entry']='unknown';h['team_got_promoted']=None;h['team_got_demoted']=None
    mask=(h.team_id.astype(object)==A)&(h.season_id==2)
    h.loc[mask,['team_season_entry','team_got_promoted','team_got_demoted']]=['promoted',True,False]
    p=policy(mode='w_league_prior',bottom=-1)
    assert run(h,p=p).value.iloc[8]==5
    assert evaluate_features(h,{'flag':TeamMovement()}).flag.iloc[8]==1
    h.loc[8,'team_got_demoted']=True
    with pytest.raises(ValueError): run(h,p=p)
    with pytest.raises(ValueError): evaluate_features(h,{'flag':TeamMovement()})

def test_changed_boundary_rank_or_movement_changes_identity_and_missing_n():
    h=cohort_history(); records=movement(); p=policy(mode='w_league_prior',bottom=1)
    first=run(h,p=p,team_seasons=records).attrs['warm_start_input_hash']
    changed=h.copy(); changed['team_position']=3-changed.team_position
    assert run(changed,p=p,team_seasons=records).attrs['warm_start_input_hash']!=first
    records[0]['movement']='relegated'
    assert run(h,p=p,team_seasons=records).attrs['warm_start_input_hash']!=first
    assert run(h,p=p,team_seasons=movement(),season_starts={(10,2):START+pd.Timedelta(days=9)}).attrs['warm_start_input_hash']!=first
    h=toy(); h.loc[6,OWN]=np.nan
    assert pd.isna(run(h,RollingStd(STAT,2),policy(variance_estimator='weighted_sample')).value.iloc[8])
    # An empty/insufficient boundary sample stays ordinary after entry.
    assert run(h,RollingStd(STAT,2),policy(variance_estimator='weighted_sample')).value.iloc[12]==pytest.approx(math.sqrt(18))

def test_h2h_wrapper_order_and_mismatched_predecessor_rejection():
    h=toy();h.loc[6,'opponent_id']=C
    a=evaluate_features(h,{'a':H2H(WarmStart(RollingMean(STAT,2),policy())),
                          'b':WarmStart(H2H(RollingMean(STAT,2)),policy())})
    assert a.a.iloc[8]==a.b.iloc[8]==6
    invalid=[dict(competition_id=10,season_id=2,team_id=A,movement='promoted',
                  previous_competition_id=10,previous_season_id=1)]
    with pytest.raises(ValueError,match='Predecessor'):
        evaluate_features(h,{'flag':TeamMovement()},team_seasons=invalid)

def test_ratio_and_sum_reduce_each_observation_before_moments():
    from xdiyo_analytics.features import Ratio,Sum,Constant
    h=toy()
    source=Sum(STAT,ForAgainst(STAT,'against'))
    assert run(h,RollingStd(source,2,ddof=0)).value.iloc[8]==9
    source=Ratio(STAT,Sum(STAT,Constant(1)))
    assert run(h,RollingMean(source,2)).value.iloc[8]==pytest.approx((12/13+18/19)/2)
