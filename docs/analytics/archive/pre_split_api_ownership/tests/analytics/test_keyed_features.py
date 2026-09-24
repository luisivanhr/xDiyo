"""Opt-in feature identity indexes retain legacy values and default behavior."""
import pandas as pd
import pytest
from xdiyo_analytics.features import IsHome,Lag,RollingMean,evaluate_features
from label_samples import STAT
from dataset_samples import inputs,KEYS

def test_keyed_feature_values_and_attrs_match_default_with_duplicate_input_index():
    h,_=inputs();definitions={'home':IsHome(),'lag':Lag(STAT),'mean':RollingMean(STAT,2)}
    default=evaluate_features(h,definitions);explicit=evaluate_features(h,definitions,keyed=False)
    keyed=evaluate_features(h,definitions,keyed=True)
    pd.testing.assert_frame_equal(default,explicit)
    assert default.index.equals(h.index) and list(keyed.index.names)==KEYS
    pd.testing.assert_frame_equal(keyed.reset_index(drop=True),default.reset_index(drop=True))
    assert keyed.attrs['identity_columns']==tuple(KEYS)
    for key,value in default.attrs.items():assert keyed.attrs[key]==value
    assert keyed.index.get_level_values('team_id').tolist()==h.team_id.tolist()
    assert keyed.index.get_level_values('event_id').tolist()==h.event_id.tolist()

@pytest.mark.parametrize('keyed',[None,0,1,'yes'])
def test_keyed_flag_requires_boolean(keyed):
    h,_=inputs()
    with pytest.raises(TypeError,match='keyed must be a boolean'):evaluate_features(h,{'home':IsHome()},keyed=keyed)

@pytest.mark.parametrize('change',['null_id','duplicate_id','bad_side','missing_event'])
def test_keyed_output_rejects_invalid_identity(change):
    h,_=inputs()
    if change=='null_id':h.iloc[0,h.columns.get_loc('event_id')]=pd.NA
    elif change=='duplicate_id':h=pd.concat([h,h.iloc[[0]]])
    elif change=='bad_side':h.iloc[0,h.columns.get_loc('side')]='neutral'
    elif change=='missing_event':h=h.drop(columns='event_id')
    with pytest.raises((ValueError,KeyError)):evaluate_features(h,{'home':IsHome()},keyed=True)

def test_empty_keyed_feature_frame_keeps_named_levels():
    h,_=inputs();h=h.iloc[:0]
    result=evaluate_features(h,{'home':IsHome()},keyed=True)
    assert result.empty and isinstance(result.index,pd.MultiIndex) and list(result.index.names)==KEYS
