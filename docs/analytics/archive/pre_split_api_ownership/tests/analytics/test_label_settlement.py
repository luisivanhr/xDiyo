"""Independent settlement truth tables, encodings and composition restrictions."""
import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.labels import TeamValue, MatchTotal, Outcome, Above, BetOption, create_labels
from label_samples import label_history, STAT, ALL_PERIODS

def check(result,states,values):
    assert result.settlement.iloc[:,0].tolist()==states
    np.testing.assert_allclose(result.y.iloc[:,0].to_numpy(dtype=float),values,equal_nan=True,rtol=0,atol=0)
    pd.testing.assert_index_equal(result.y.index,result.settlement.index)
    assert list(result.y)==list(result.settlement)

@pytest.mark.parametrize('selection,states,encoded',[
    ('yes',['loss','loss','win','missing','missing','missing'],[0,0,1,np.nan,np.nan,np.nan]),
    ('no',['win','win','loss','missing','missing','missing'],[1,1,0,np.nan,np.nan,np.nan]),
])
def test_yes_no_obey_strict_above(selection,states,encoded):
    result=create_labels(label_history(),{'bet':BetOption(Above(MatchTotal(STAT),11),selection)})['bet']
    check(result,states,encoded)
    assert result.unit=='match' and result.perspective=='total'

@pytest.mark.parametrize('selection,states,encoded',[
    ('win',['win','loss','loss','missing','missing','missing'],[1,0,0,np.nan,np.nan,np.nan]),
    ('draw',['loss','win','loss','missing','missing','missing'],[0,1,0,np.nan,np.nan,np.nan]),
    ('loss',['loss','loss','win','missing','missing','missing'],[0,0,1,np.nan,np.nan,np.nan]),
])
def test_wdl_settlement_default(selection,states,encoded):
    result=create_labels(label_history(),{'bet':BetOption(Outcome(perspective='home'),selection)})['bet']
    check(result,states,encoded)

@pytest.mark.parametrize('selection',['win','loss'])
@pytest.mark.parametrize('push_value',[None,0.,.5,1.,-2.])
def test_draw_push_is_distinct_from_loss_even_with_same_numeric_mapping(selection,push_value):
    result=create_labels(label_history(),{'bet':BetOption(Outcome(perspective='home'),selection,draw='push',push_value=push_value)})['bet']
    assert result.settlement.iloc[1,0]=='push'
    assert (pd.isna(result.y.iloc[1,0]) if push_value is None else result.y.iloc[1,0]==push_value)
    assert result.settlement.iloc[3,0]=='missing' and pd.isna(result.y.iloc[3,0])

def test_draw_selection_wins_on_draw_even_when_draw_policy_push():
    result=create_labels(label_history(),{'bet':BetOption(Outcome(perspective='away'),'draw',draw='push')})['bet']
    assert result.settlement.iloc[1,0]=='win' and result.y.iloc[1,0]==1

@pytest.mark.parametrize('selection,states,encoded',[
    ('over',['push','loss','win','missing','missing','missing'],[np.nan,0,1,np.nan,np.nan,np.nan]),
    ('under',['push','win','loss','missing','missing','missing'],[np.nan,1,0,np.nan,np.nan,np.nan]),
])
def test_over_under_push_default(selection,states,encoded):
    result=create_labels(label_history(),{'bet':BetOption(MatchTotal(STAT),selection,line=11)})['bet']
    check(result,states,encoded)

@pytest.mark.parametrize('selection',['over','under'])
def test_equality_loss_is_explicit(selection):
    result=create_labels(label_history(),{'bet':BetOption(MatchTotal(STAT),selection,line=11,on_equal='loss',push_value=.5)})['bet']
    assert result.settlement.iloc[0,0]=='loss' and result.y.iloc[0,0]==0

@pytest.mark.parametrize('line,expected',[(-2.25,1),(9.25,1),(10.25,1),(11.25,0)])
def test_fractional_lines_are_literal_thresholds(line,expected):
    result=create_labels(label_history(),{'bet':BetOption(MatchTotal(STAT),'over',line=line)})['bet']
    assert result.y.iloc[0,0]==expected
    assert result.settlement.iloc[0,0]==('win' if expected else 'loss')

@pytest.mark.parametrize('void_value',[None,0.,.5,1.,-3.])
def test_only_explicit_void_status_overrides_missing(void_value):
    h=label_history()
    node=BetOption(MatchTotal(STAT),'over',line=11,void_statuses=['cancelled'],void_value=void_value)
    assert node.void_statuses==('cancelled',) and hash(node)
    result=create_labels(h,{'bet':node})['bet']
    assert result.settlement.iloc[:,0].tolist()==['push','loss','win','missing','void','missing']
    assert (pd.isna(result.y.iloc[4,0]) if void_value is None else result.y.iloc[4,0]==void_value)
    assert result.settlement.iloc[3,0]=='missing' and pd.isna(result.y.iloc[3,0])

def test_explicit_void_overrides_win_loss_push_and_every_expanded_column():
    result=create_labels(label_history(),{'bet':BetOption(TeamValue(ALL_PERIODS,'both'),'over',line=3,void_statuses=['finished'],void_value=.25)})['bet']
    assert result.y.shape==(12,4) and result.unit=='team_match' and result.perspective=='team'
    assert result.settlement.iloc[:6].eq('void').all().all()
    assert result.y.iloc[:6].eq(.25).all().all()
    assert result.settlement.iloc[6:10].eq('missing').all().all()
    assert result.settlement.iloc[10:].eq('void').all().all()

def test_void_list_is_normalized_without_mutating_caller():
    statuses=['cancelled']; node=BetOption(Above(TeamValue(STAT),3),void_statuses=statuses)
    statuses.append('finished')
    assert node.void_statuses==('cancelled',)
    create_labels(label_history(),{'bet':node})

@pytest.mark.parametrize('bad',['cancelled',None,[''],[1],{'cancelled'},('cancelled',None)])
def test_rejects_invalid_void_status_collection(bad):
    with pytest.raises(TypeError,match='tuple/list'):BetOption(Above(TeamValue(STAT),1),void_statuses=bad)

@pytest.mark.parametrize('node',[
    BetOption(TeamValue(STAT),'yes'),BetOption(MatchTotal(STAT),'no'),
    BetOption(Above(TeamValue(STAT),3),'win'),BetOption(MatchTotal(STAT),'draw'),
    BetOption(Outcome(),'over',line=1),BetOption(Above(TeamValue(STAT),3),'under',line=1),
    BetOption(MatchTotal(STAT),'over'),BetOption(MatchTotal(STAT),'over',line=np.nan),
    BetOption(MatchTotal(STAT),'under',line=np.inf),BetOption(MatchTotal(STAT),'under',line=True),
    BetOption(Outcome(),'win',line=1),BetOption(Above(TeamValue(STAT),3),'unknown'),
    BetOption(Outcome(),'win',draw='void'),BetOption(MatchTotal(STAT),'over',line=1,on_equal='win'),
    BetOption(Outcome(),'win',push_value=np.inf),BetOption(Outcome(),'win',void_value=np.nan),
    BetOption(Outcome(),'win',push_value=True),BetOption(Outcome(),'win',void_value='zero'),
])
def test_rejects_invalid_settlement_compositions_and_parameters(node):
    with pytest.raises((TypeError,ValueError)):create_labels(label_history(),{'bet':node})

def test_duplicate_child_evaluations_do_not_share_mutable_settlement_outputs():
    node=BetOption(Outcome(perspective='away'),'win',draw='push')
    labels=create_labels(label_history(),{'a':node,'b':node})
    labels['a'].settlement.iloc[0,0]='changed';labels['a'].y.iloc[0,0]=99
    assert labels['b'].settlement.iloc[0,0]=='loss' and labels['b'].y.iloc[0,0]==0
