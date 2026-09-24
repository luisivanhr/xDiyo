"""Observed label arithmetic, identity, layout and validation contracts."""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.features import Stat, Lag, evaluate_features
from xdiyo_analytics.labels import LabelExpr, LabelData, TeamValue, MatchTotal, Outcome, Above, BetOption, create_labels
from label_samples import label_history, STAT, ALL_PERIODS, OWN, OTHER, TEAM

def values(actual, expected):
    np.testing.assert_allclose(np.asarray(actual,dtype=float),expected,equal_nan=True,rtol=0,atol=0)

@pytest.mark.parametrize('side,expected',[
    ('for',[3,8,5,5,11,2,np.nan,np.nan,np.nan,np.nan,np.nan,3]),
    ('against',[8,3,5,5,2,11,np.nan,np.nan,np.nan,np.nan,3,np.nan]),
])
def test_team_values_exact_observations_missing_and_layout(side,expected):
    h=label_history(); node=TeamValue(STAT,side)
    result=create_labels(h,{'count':node})['count']
    assert isinstance(result,LabelData) and isinstance(node,LabelExpr)
    values(result.y['count'],expected)
    pd.testing.assert_index_equal(result.y.index,h.index)
    pd.testing.assert_index_equal(result.metadata.index,h.index)
    assert result.unit=='team_match' and result.perspective=='team'
    assert result.identity_columns==('source_league','source_season','competition_id','season_id','event_id','team_id')
    assert result.definition==node and result.settlement is None
    for column in result.identity_columns:
        pd.testing.assert_series_equal(result.metadata[column],h[column])

def test_both_periods_and_stat_field_group_selection():
    h=label_history()
    labels=create_labels(h,{'both':TeamValue(ALL_PERIODS,'both'), 'total_field':MatchTotal(Stat('ALL','Match overview','cornerKicks',field='total')), 'other_group':TeamValue(Stat('ALL','Other group','cornerKicks'))})
    both=labels['both']
    assert both.y.shape==(12,4)
    assert list(both.y)==['both::'+name for name in [OWN,'team::1ST::Match overview::cornerKicks::value',OTHER,'opponent::1ST::Match overview::cornerKicks::value']]
    values(both.y.iloc[0],[3,1,8,2])
    values(both.y.iloc[10],[np.nan,1,3,2])
    values(labels['total_field'].y.iloc[:,0],[300,300,300,np.nan,np.nan,300])
    values(labels['other_group'].y.iloc[:6,0],[99]*6)

def test_match_total_requires_two_finite_values_and_keeps_home_order():
    h=label_history().iloc[[3,0,11,8,4,1,6,9,2,7,10,5]].copy()
    result=create_labels(h,{'total':MatchTotal(STAT)})['total']
    home=h.loc[h.side.eq('home')]
    values(result.y['total'],[11,np.nan,13,np.nan,10,np.nan])
    pd.testing.assert_index_equal(result.y.index,home.index)
    assert result.unit=='match' and result.perspective=='total'
    assert result.identity_columns==('source_league','source_season','competition_id','season_id','event_id')
    assert not {'team_id','opponent_id','side'} & set(result.metadata)
    assert result.metadata.home_id.tolist()==[TEAM]*6 and result.metadata.away_id.tolist()==[TEAM+1]*6
    assert result.metadata.home_id.dtype==home.team_id.dtype
    assert result.metadata.event_id.tolist()==home.event_id.tolist()

@pytest.mark.parametrize('perspective,expected',[
    ('team',[1,-1,0,0,-1,1,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]),
    ('home',[1,0,-1,np.nan,np.nan,np.nan]),
    ('away',[-1,0,1,np.nan,np.nan,np.nan]),
])
def test_wdl_outcome_perspectives(perspective,expected):
    h=label_history(); result=create_labels(h,{'outcome':Outcome(perspective=perspective)})['outcome']
    values(result.y['outcome'],expected)
    assert result.perspective==perspective
    assert result.unit==('team_match' if perspective=='team' else 'match')

@pytest.mark.parametrize('perspective', ['team','home','away'])
@pytest.mark.parametrize('higher', [True,False])
def test_stat_outcome_sign_without_overflow(perspective,higher):
    h=label_history(); result=create_labels(h,{'outcome':Outcome(STAT,perspective,higher)})['outcome']
    expected=np.array([-1,1,0,0,1,-1,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
    if perspective=='home':expected=expected[::2]
    if perspective=='away':expected=expected[1::2]
    if not higher:expected=-expected
    values(result.y['outcome'],expected)

def test_away_outcomes_align_by_match_identity_after_shuffling():
    h=label_history().iloc[[3,0,11,8,4,1,6,9,2,7,10,5]].copy()
    labels=create_labels(h,{'away':Outcome(perspective='away'),'home':Outcome(perspective='home')})
    values(labels['away'].y['away'],[-1,np.nan,1,np.nan,0,np.nan])
    values(labels['home'].y['home'],[1,np.nan,-1,np.nan,0,np.nan])
    pd.testing.assert_frame_equal(labels['away'].metadata,labels['home'].metadata)

@pytest.mark.parametrize('node,width,unit',[(TeamValue(ALL_PERIODS),2,'team_match'),(MatchTotal(ALL_PERIODS),2,'match'),(Outcome(ALL_PERIODS,'away'),2,'match'),(Above(TeamValue(ALL_PERIODS,'both'),4),4,'team_match')])
def test_period_expansion_and_shape_propagation(node,width,unit):
    result=create_labels(label_history(),{'y':node})['y']
    assert result.y.shape==(12 if unit=='team_match' else 6,width)
    assert result.unit==unit and all(c.startswith('y::') for c in result.y)

@pytest.mark.parametrize('node,expected',[(Above(TeamValue(STAT),5),[0,1,0,0,1,0,np.nan,np.nan,np.nan,np.nan,np.nan,0]),(Above(MatchTotal(STAT),11),[0,0,1,np.nan,np.nan,np.nan]),(Above(Outcome(perspective='away'),0),[0,0,1,np.nan,np.nan,np.nan])])
def test_strict_above_and_missing_propagation(node,expected):
    result=create_labels(label_history(),{'y':node})['y']; values(result.y.iloc[:,0],expected)

def test_partition_collisions_do_not_pair_same_event_in_different_partition():
    first=label_history().iloc[:2].copy(); other=first.copy()
    other['source_league']='Other'; other['source_season']='25_26'
    other['competition_id']=pd.array([2**53+77]*2,dtype='uint64[pyarrow]')
    other['season_id']=pd.array([2**53+99]*2,dtype='uint64[pyarrow]')
    other[OWN]=[20.,40.]; other[OTHER]=[40.,20.]
    h=pd.concat([other.iloc[[1]],first.iloc[[0]],other.iloc[[0]],first.iloc[[1]]]);h.attrs=deepcopy(first.attrs)
    result=create_labels(h,{'total':MatchTotal(STAT)})['total']
    values(result.y['total'],[11,60])
    assert result.metadata.source_league.tolist()==['Synthetic','Other']
    assert result.metadata.event_id.tolist()==[2**63+101]*2

def test_sources_cached_with_independent_outputs_and_immutable_history():
    h=label_history(); before=h.copy(deep=True); attrs=deepcopy(h.attrs)
    child=TeamValue(STAT); parent=Above(child,5)
    labels=create_labels(h,{'above':parent,'raw':child,'same':child,'bet':BetOption(parent)})
    values(labels['raw'].y.iloc[:6,0],[3,8,5,5,11,2])
    labels['raw'].y.iloc[0,0]=999
    labels['raw'].metadata.iloc[0,0]='changed'
    assert labels['same'].y.iloc[0,0]==3 and labels['above'].y.iloc[0,0]==0
    pd.testing.assert_frame_equal(h,before);assert h.attrs==attrs
    assert child==TeamValue(STAT) and parent==Above(child,5)

def test_null_status_and_bad_outcome_are_missing_without_row_loss():
    h=label_history();h.iloc[:2,h.columns.get_loc('status')]=pd.NA;h.iloc[2:4,h.columns.get_loc('result')]='?'
    labels=create_labels(h,{'raw':TeamValue(STAT),'outcome':Outcome()})
    assert labels['raw'].y.iloc[:2].isna().all().all()
    assert labels['outcome'].y.iloc[:4].isna().all().all()
    assert len(labels['raw'].y)==len(h)

@pytest.mark.parametrize('bad',[np.inf,-np.inf,np.nan,pd.NA])
def test_missing_one_side_invalidates_total_and_stat_outcome(bad):
    h=label_history().iloc[:2].copy();h[OWN]=[bad,8.];h[OTHER]=[8.,bad]
    labels=create_labels(h,{'raw':TeamValue(STAT),'total':MatchTotal(STAT),'outcome':Outcome(STAT)})
    assert pd.isna(labels['raw'].y.iloc[0,0]) and labels['raw'].y.iloc[1,0]==8
    assert labels['total'].y.isna().all().all() and labels['outcome'].y.isna().all().all()

def test_total_overflow_is_missing_but_comparison_avoids_subtraction_overflow():
    h=label_history().iloc[:2].copy();h[OWN]=[1.7e308,1.6e308];h[OTHER]=[1.6e308,1.7e308]
    labels=create_labels(h,{'total':MatchTotal(STAT),'outcome':Outcome(STAT)})
    assert labels['total'].y.isna().all().all();values(labels['outcome'].y.iloc[:,0],[1,-1])
    h[OTHER]=[-1.7e308,1.7e308];h[OWN]=[1.7e308,-1.7e308]
    values(create_labels(h,{'outcome':Outcome(STAT)})['outcome'].y.iloc[:,0],[1,-1])

@pytest.mark.parametrize('change,match',[
    ('side','sides'),('missing_side','both team rows'),('duplicate_side','duplicate match sides'),
    ('same_team','opposite teams'),('opponent_mismatch','opposite teams'),('status_mismatch','agree on match status'),
    ('missing_id','nonmissing'),
])
def test_rejects_invalid_paired_population(change,match):
    h=label_history()
    if change=='side':h.iloc[0,h.columns.get_loc('side')]='neutral'
    elif change=='missing_side':h=h.iloc[1:]
    elif change=='duplicate_side':h=pd.concat([h,h.iloc[[0]]])
    elif change=='same_team':h.iloc[0,h.columns.get_loc('team_id')]=TEAM+1
    elif change=='opponent_mismatch':h.iloc[1,h.columns.get_loc('opponent_id')]=TEAM+7
    elif change=='status_mismatch':h.iloc[1,h.columns.get_loc('status')]='notstarted'
    elif change=='missing_id':h.iloc[0,h.columns.get_loc('event_id')]=pd.NA
    with pytest.raises(ValueError,match=match):create_labels(h,{'y':Outcome()})

@pytest.mark.parametrize('column',['event_id','team_id','opponent_id','side','status'])
def test_required_columns(column):
    with pytest.raises(KeyError,match='missing columns'):create_labels(label_history().drop(columns=column),{'y':Outcome()})

@pytest.mark.parametrize('definition,exception',[
    (STAT,TypeError),(Lag(STAT),TypeError),(LabelExpr(),TypeError),(TeamValue(Lag(STAT)),TypeError),
    (TeamValue(STAT,'home'),ValueError),(Outcome(perspective='total'),ValueError),(Outcome(higher_is_better=False),ValueError),
    (Outcome(higher_is_better=1),TypeError),(Above(TeamValue(STAT),True),ValueError),(Above(TeamValue(STAT),np.inf),ValueError),
    (Above(TeamValue(STAT),np.nan),ValueError),(Above(BetOption(Above(TeamValue(STAT),2)),.5),TypeError),
    (TeamValue(Stat('ALL','Absent','cornerKicks')),KeyError),
])
def test_invalid_sources_and_parameters(definition,exception):
    with pytest.raises(exception):create_labels(label_history(),{'y':definition})

@pytest.mark.parametrize('mapping',[{}, {'':Outcome()},{1:Outcome()}])
def test_invalid_label_names(mapping):
    with pytest.raises(ValueError):create_labels(label_history(),mapping)

def test_feature_and_label_namespaces_stay_separate():
    with pytest.raises(TypeError):evaluate_features(label_history(),{'y':Outcome()})

def test_opponent_stat_metadata_must_be_unique():
    h=label_history();del h.attrs['stat_columns'][OTHER]
    with pytest.raises(ValueError,match='matching opponent'):create_labels(h,{'y':TeamValue(STAT)})
    h=label_history();h['extra']=h[OTHER];h.attrs['stat_columns']['extra']=dict(h.attrs['stat_columns'][OTHER])
    with pytest.raises(ValueError,match='matching opponent'):create_labels(h,{'y':MatchTotal(STAT)})

def test_empty_typed_history_preserves_empty_layouts():
    h=label_history().iloc[:0]
    results=create_labels(h,{'team':TeamValue(STAT),'match':MatchTotal(STAT),'away':Outcome(perspective='away')})
    assert all(r.y.empty and r.metadata.empty for r in results.values())

@pytest.mark.parametrize('half_dtype',['object','Float64','float64[pyarrow]'])
def test_nullable_and_all_missing_object_columns_across_periods(half_dtype):
    h=label_history();before=h.copy(deep=True)
    h[OWN]=pd.Series([pd.NA]*len(h),index=h.index,dtype=object)
    h[OTHER]=pd.Series([pd.NA]*len(h),index=h.index,dtype=object)
    half_own='team::1ST::Match overview::cornerKicks::value'
    half_other='opponent::1ST::Match overview::cornerKicks::value'
    h[half_own]=pd.array([pd.NA,2]+[1,2]*5,dtype=half_dtype)
    h[half_other]=pd.array([2,pd.NA]+[2,1]*5,dtype=half_dtype)
    original=h.copy(deep=True)
    result=create_labels(h,{'team':TeamValue(ALL_PERIODS),'total':MatchTotal(ALL_PERIODS),'outcome':Outcome(ALL_PERIODS)})
    assert result['team'].y.iloc[:,0].isna().all()
    assert result['total'].y.iloc[:,0].isna().all()
    assert result['outcome'].y.iloc[:,0].isna().all()
    values(result['team'].y.iloc[:4,1],[np.nan,2,1,2])
    values(result['total'].y.iloc[:3,1],[np.nan,3,3])
    values(result['outcome'].y.iloc[:4,1],[np.nan,np.nan,-1,1])
    pd.testing.assert_frame_equal(h,original)
