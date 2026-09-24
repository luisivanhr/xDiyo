"""Independent layout, identity, filtering and source-preservation checks."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.labels import TeamValue,MatchTotal,Outcome,Above,BetOption,create_labels
from xdiyo_analytics.datasets import assemble_dataset,ModelDataset
from label_samples import STAT,ALL_PERIODS,TEAM
from dataset_samples import inputs,reorder_label,KEYS

@pytest.mark.parametrize('explicit',[False,True])
@pytest.mark.parametrize('layout,perspective',[('team_match','team'),('match','home'),('match','away'),('match','total')])
def test_shuffled_feature_and_label_rows_align_by_exact_ids(explicit,layout,perspective):
    h,f=inputs();definition=TeamValue(STAT) if layout=='team_match' else MatchTotal(STAT) if perspective=='total' else Outcome(perspective=perspective)
    label=create_labels(h,{'target':definition})['target']
    order=[3,0,11,8,4,1,6,9,2,7,10,5] if layout=='team_match' else [3,0,5,2,1,4]
    label=reorder_label(label,order)
    shuffled=f.iloc[[7,0,9,2,11,4,1,8,3,10,5,6]].copy()
    if explicit:shuffled=shuffled.reset_index();shuffled.index=[4]*len(shuffled)
    result=assemble_dataset(shuffled,label,layout=layout)
    assert isinstance(result,ModelDataset)
    if layout=='team_match':
        assert list(result.X)==['form','conceded']
        assert result.X.form.tolist()==[100+i for i in order]
        assert result.X.conceded.tolist()==[200+i for i in order]
    else:
        assert list(result.X)==['home::form','home::conceded','away::form','away::conceded']
        assert result.X['home::form'].tolist()==[100+2*i for i in order]
        assert result.X['away::form'].tolist()==[101+2*i for i in order]
        assert result.X['home::conceded'].tolist()==[200+2*i for i in order]
        assert result.X['away::conceded'].tolist()==[201+2*i for i in order]
    pd.testing.assert_frame_equal(result.y,label.y.reset_index(drop=True))
    for col in label.metadata:
        pd.testing.assert_series_equal(result.metadata[col],label.metadata[col].reset_index(drop=True))
    assert result.target_perspective==perspective and result.layout==layout
    assert result.identity_columns==label.identity_columns
    assert all(frame.index.equals(pd.RangeIndex(len(result.y))) for frame in [result.X,result.y,result.metadata])
    assert result.metadata.event_id.dtype==h.event_id.dtype
    assert result.X.iloc[:,0].dtype==f.form.dtype
    assert result.X.iloc[:,1].dtype==f.conceded.dtype
    assert result.definitions['features']==f.attrs['features']
    assert result.definitions['label']==label.definition

@pytest.mark.parametrize('layout',['match','team_match'])
@pytest.mark.parametrize('selector',['string','sequence','default'])
def test_feature_target_selection_order_and_period_expansion(layout,selector):
    h,f=inputs();node=TeamValue(ALL_PERIODS,'both') if layout=='team_match' else MatchTotal(ALL_PERIODS)
    label=create_labels(h,{'target':node})['target']; names=list(label.y)
    features='conceded' if selector=='string' else ['conceded','form'] if selector=='sequence' else None
    targets=names[-1] if selector=='string' else list(reversed(names)) if selector=='sequence' else None
    result=assemble_dataset(f,label,layout=layout,feature_columns=features,target_columns=targets)
    chosen=['conceded'] if selector=='string' else ['conceded','form'] if selector=='sequence' else ['form','conceded']
    assert list(result.X)==chosen if layout=='team_match' else list(result.X)==[f'{s}::{c}' for s in ['home','away'] for c in chosen]
    expected=[names[-1]] if selector=='string' else list(reversed(names)) if selector=='sequence' else names
    pd.testing.assert_frame_equal(result.y,label.y[expected].reset_index(drop=True))

@pytest.mark.parametrize('layout',['match','team_match'])
def test_extra_feature_records_are_allowed_missing_values_pass_unchanged(layout):
    h,f=inputs();node=TeamValue(STAT) if layout=='team_match' else MatchTotal(STAT)
    label=create_labels(h.iloc[:4],{'target':node})['target']
    f.iloc[0,0]=pd.NA
    result=assemble_dataset(f,label,layout=layout)
    assert pd.isna(result.X.iloc[0,0]) and len(result.X)==len(label.y)
    assert len(result.X)==(4 if layout=='team_match' else 2)

@pytest.mark.parametrize('layout',['match','team_match'])
@pytest.mark.parametrize('drop',[False,True])
def test_whole_match_target_missing_policy(layout,drop):
    h,f=inputs();label=create_labels(h,{'target':TeamValue(STAT) if layout=='team_match' else MatchTotal(STAT)})['target']
    result=assemble_dataset(f,label,layout=layout,drop_missing_targets=drop)
    expected=6 if layout=='team_match' else 3
    assert len(result.y)==(expected if drop else len(label.y))
    assert result.metadata.event_id.nunique()==(3 if drop else 6)
    if layout=='team_match':assert result.groups.value_counts().eq(2).all()
    if not drop:assert result.y.isna().any().any()

def test_nonselected_missing_targets_do_not_drop_a_pair():
    h,f=inputs();label=create_labels(h,{'target':TeamValue(ALL_PERIODS)})['target']
    half=[c for c in label.y if '1ST' in c][0]
    result=assemble_dataset(f,label,layout='team_match',target_columns=half,drop_missing_targets=True)
    assert len(result.y)==8 and result.metadata.event_id.nunique()==4
    assert result.metadata.event_id.iloc[-1]==2**63+106
    assert result.groups.value_counts().eq(2).all()

@pytest.mark.parametrize('layout',['match','team_match'])
def test_one_missing_selected_target_drops_entire_match_with_all_columns(layout):
    h,f=inputs();label=create_labels(h.iloc[:6],{'target':TeamValue(ALL_PERIODS) if layout=='team_match' else MatchTotal(ALL_PERIODS)})['target']
    label.y.iloc[0,1]=np.nan
    result=assemble_dataset(f,label,layout=layout,drop_missing_targets=True)
    assert set(result.metadata.event_id)=={2**63+102,2**63+103}
    assert len(result.y)==(4 if layout=='team_match' else 2)

@pytest.mark.parametrize('mapped',[False,True])
def test_settlement_selected_columns_stay_in_metadata_and_finite_encodings_retained(mapped):
    h,f=inputs();node=BetOption(MatchTotal(ALL_PERIODS),'over',line=11,push_value=.5 if mapped else None,void_value=0 if mapped else None,void_statuses=['cancelled'])
    label=create_labels(h,{'target':node})['target']; selected=list(label.y)[0]
    result=assemble_dataset(f,label,layout='match',target_columns=selected,drop_missing_targets=True)
    assert not any(str(c).startswith('settlement::') for c in result.X)
    assert [c for c in result.metadata if str(c).startswith('settlement::')]==['settlement::'+selected]
    assert len(result.y)==(4 if mapped else 2)
    assert set(result.metadata['settlement::'+selected])==({'win','loss','push','void'} if mapped else {'win','loss'})

def test_manually_supplied_inf_target_is_not_reencoded_or_dropped():
    h,f=inputs();label=create_labels(h.iloc[:2],{'target':TeamValue(STAT)})['target'];label.y.iloc[0,0]=np.inf
    result=assemble_dataset(f,label,layout='team_match',drop_missing_targets=True)
    assert len(result.y)==2 and np.isposinf(result.y.iloc[0,0])

def test_groups_use_all_partition_fields_and_keep_huge_ids_exact():
    h,f=inputs();h=h.iloc[:2].copy();other=h.copy();other['source_league']='Second';other['source_season']='25_26'
    other['competition_id']=pd.array([2**63+123]*2,dtype='uint64[pyarrow]')
    h=pd.concat([h,other]);h.attrs=deepcopy(other.attrs)
    frame=pd.DataFrame({'x':[10,11,20,21]},index=pd.MultiIndex.from_frame(h[KEYS]))
    label=create_labels(h,{'target':TeamValue(STAT)})['target']
    result=assemble_dataset(frame.iloc[::-1],label,layout='team_match')
    assert result.X.x.tolist()==[10,11,20,21]
    assert result.groups.tolist()[0]==result.groups.tolist()[1]
    assert result.groups.tolist()[2]==result.groups.tolist()[3]
    assert result.groups.iloc[0]!=result.groups.iloc[2]
    assert result.groups.iloc[2][2]==2**63+123
    assert result.groups.name=='match_group' and result.groups.dtype==object
    assert all(isinstance(key,tuple) for key in result.groups)

@pytest.mark.parametrize('layout',['match','team_match'])
def test_empty_population_is_a_valid_empty_dataset(layout):
    h,f=inputs();label=create_labels(h.iloc[:0],{'target':TeamValue(STAT) if layout=='team_match' else MatchTotal(STAT)})['target']
    result=assemble_dataset(f.iloc[:0],label,layout=layout,drop_missing_targets=True)
    assert all(frame.empty and isinstance(frame.index,pd.RangeIndex) for frame in [result.X,result.y,result.metadata])
    assert result.groups.empty and list(result.y)==['target']

def test_no_input_mutation_or_shared_metadata_definitions():
    h,f=inputs();label=create_labels(h,{'target':BetOption(Above(TeamValue(STAT),5))})['target']
    f_before=f.copy(deep=True);attrs=deepcopy(f.attrs);label_before=deepcopy(label)
    result=assemble_dataset(f,label,layout='team_match')
    result.X.iloc[0,0]=999;result.y.iloc[0,0]=999;result.metadata.iloc[0,0]='changed';result.definitions['features']['form']='changed'
    pd.testing.assert_frame_equal(f,f_before);assert f.attrs==attrs
    for name in ['y','metadata','settlement']:pd.testing.assert_frame_equal(getattr(label,name),getattr(label_before,name))

@pytest.mark.parametrize('layout',['match','team_match'])
def test_reordered_index_levels_and_selected_label_subset(layout):
    h,f=inputs();f.index=f.index.reorder_levels(list(reversed(KEYS)))
    label=create_labels(h.iloc[4:6],{'target':TeamValue(STAT) if layout=='team_match' else MatchTotal(STAT)})['target']
    result=assemble_dataset(f.iloc[::-1],label,layout=layout)
    assert result.X.iloc[:,0].tolist()==([104,105] if layout=='team_match' else [104])

def test_explicit_identifier_columns_cannot_be_selected_as_features():
    h,f=inputs();label=create_labels(h,{'target':MatchTotal(STAT)})['target']
    with pytest.raises(KeyError):assemble_dataset(f.reset_index(),label,layout='match',feature_columns='team_id')

def test_all_missing_targets_produce_empty_pair_preserving_output():
    h,f=inputs();label=create_labels(h.iloc[6:10],{'target':TeamValue(STAT)})['target']
    result=assemble_dataset(f,label,layout='team_match',drop_missing_targets=True)
    assert result.X.empty and result.y.empty and result.metadata.empty and result.groups.empty

@pytest.mark.parametrize('layout',['match','team_match'])
@pytest.mark.parametrize('change',['missing','duplicate','null_key','bad_side','wrong_team','side_duplicate'])
def test_bad_feature_identity_records_raise(layout,change):
    h,f=inputs();label=create_labels(h,{'target':TeamValue(STAT) if layout=='team_match' else MatchTotal(STAT)})['target'];frame=f.reset_index()
    if change=='missing':frame=frame.iloc[1:]
    elif change=='duplicate':frame=pd.concat([frame,frame.iloc[[0]]])
    elif change=='null_key':frame.iloc[0,frame.columns.get_loc('event_id')]=pd.NA
    elif change=='bad_side':frame.iloc[0,frame.columns.get_loc('side')]='neutral'
    elif change=='wrong_team':frame.iloc[0,frame.columns.get_loc('team_id')]=TEAM+77
    elif change=='side_duplicate':frame.iloc[1,frame.columns.get_loc('side')]='home'
    with pytest.raises(ValueError):assemble_dataset(frame,label,layout=layout)

@pytest.mark.parametrize('change',['unkeyed','missing_key','duplicate_level_names'])
def test_explicit_feature_keys_are_required(change):
    h,f=inputs();label=create_labels(h,{'target':MatchTotal(STAT)})['target']
    if change=='unkeyed':f=f.reset_index(drop=True)
    elif change=='missing_key':f=f.reset_index().drop(columns='source_league')
    else:f.index=f.index.set_names([*f.index.names[:-1],'team_id'])
    with pytest.raises(ValueError):assemble_dataset(f,label,layout='match')

@pytest.mark.parametrize('change',['missing_pair','wrong_opponent','self_match','side_mismatch','missing_opponent','missing_side','duplicate_label','null_identity','bad_identity_tuple','missing_metadata_key'])
def test_invalid_team_label_population_raises(change):
    h,f=inputs();label=create_labels(h,{'target':TeamValue(STAT)})['target']
    if change=='missing_pair':label=reorder_label(label,list(range(1,12)))
    elif change=='wrong_opponent':label.metadata.iloc[1,label.metadata.columns.get_loc('opponent_id')]=TEAM+88
    elif change=='self_match':label.metadata.iloc[0,label.metadata.columns.get_loc('opponent_id')]=TEAM
    elif change=='side_mismatch':label.metadata.iloc[0,label.metadata.columns.get_loc('side')]='away'
    elif change=='missing_opponent':label.metadata=label.metadata.drop(columns='opponent_id')
    elif change=='missing_side':label.metadata=label.metadata.drop(columns='side')
    elif change=='duplicate_label':label=reorder_label(label,[0,0,*range(2,12)])
    elif change=='null_identity':label.metadata.iloc[0,label.metadata.columns.get_loc('event_id')]=pd.NA
    elif change=='bad_identity_tuple':label.identity_columns=('event_id','event_id','team_id')
    elif change=='missing_metadata_key':label.metadata=label.metadata.drop(columns='event_id')
    with pytest.raises((KeyError,ValueError)):assemble_dataset(f,label,layout='team_match')

@pytest.mark.parametrize('change',['missing_home','missing_away','same_teams','null_team','wrong_away','team_in_identity','side_in_identity'])
def test_invalid_match_label_identities_raise(change):
    h,f=inputs();label=create_labels(h,{'target':MatchTotal(STAT)})['target']
    if change=='missing_home':label.metadata=label.metadata.drop(columns='home_id')
    elif change=='missing_away':label.metadata=label.metadata.drop(columns='away_id')
    elif change=='same_teams':label.metadata['away_id']=label.metadata.home_id.copy()
    elif change=='null_team':label.metadata.iloc[0,label.metadata.columns.get_loc('home_id')]=pd.NA
    elif change=='wrong_away':label.metadata.iloc[0,label.metadata.columns.get_loc('away_id')]=TEAM+88
    elif change=='team_in_identity':label.identity_columns+=('team_id',)
    elif change=='side_in_identity':label.identity_columns+=('side',)
    with pytest.raises((KeyError,ValueError)):assemble_dataset(f,label,layout='match')

@pytest.mark.parametrize('parameter,selection',[(p,x) for p in ['feature_columns','target_columns'] for x in [[],['form','form'],[''],[1],'absent']])
def test_invalid_column_selections(parameter,selection):
    h,f=inputs();label=create_labels(h,{'target':MatchTotal(STAT)})['target']
    with pytest.raises((ValueError,KeyError)):assemble_dataset(f,label,layout='match',**{parameter:selection})

@pytest.mark.parametrize('change',['layout','unit','label_type','feature_type','drop_flag','y_index','settlement_index','settlement_column','settlement_collision','duplicate_columns'])
def test_invalid_container_and_output_contracts(change):
    h,f=inputs();label=create_labels(h,{'target':BetOption(MatchTotal(STAT),'over',line=10)})['target'];kwargs={'layout':'match'}
    if change=='layout':kwargs['layout']='team'
    elif change=='unit':kwargs['layout']='team_match'
    elif change=='label_type':label={'target':label}
    elif change=='feature_type':f=f.to_numpy()
    elif change=='drop_flag':kwargs['drop_missing_targets']=1
    elif change=='y_index':label.y=label.y.reset_index(drop=True)
    elif change=='settlement_index':label.settlement=label.settlement.reset_index(drop=True)
    elif change=='settlement_column':label.settlement=label.settlement.rename(columns={'target':'other'})
    elif change=='settlement_collision':label.metadata['settlement::target']='existing'
    elif change=='duplicate_columns':f.columns=['x','x']
    with pytest.raises((TypeError,ValueError,KeyError)):assemble_dataset(f,label,**kwargs)
