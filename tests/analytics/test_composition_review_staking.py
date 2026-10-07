import numpy as np
import pandas as pd
from xdiyo_analytics.evaluation import *
from xdiyo_analytics.evaluation.stake_policy import *
from xdiyo_analytics.evaluation.bankroll import BankrollLedger

def offers(n=2):
    return pd.DataFrame(dict(fold_id=[0]*n,row_position=range(n),event_id=range(n),competition_id=[1]*n,
        season_id=[1]*n,source_season=['24_25']*n,round=[1]*n,stage=['main']*n,
        kickoff_at=pd.date_range('2025-01-02',periods=n,freq='h',tz='UTC'),
        bet=['over']*n,take=[True]*n,odds=[2.]*n,stake=[20.]*n,
        settlement=['win']*n,p_win=[.8]*n,p_push=[0.]*n,p_loss=[.2]*n,
        decision_at=[pd.Timestamp('2025-01-01',tz='UTC')]*n,quote_at=[pd.Timestamp('2024-12-01',tz='UTC')]*n,
        quote_id=[f'q{i}' for i in range(n)]))

def gate_offers():
    f=offers()
    for name in ['a','b']:
        f[name+'::probability']=.9
        for field,time in [('issued_at','2025-01-01'),('artifact_vintage','2024-12-30'),('trained_through','2024-12-29')]:
            f[name+'::'+field]=pd.Timestamp(time,tz='UTC')
    return f

def gate(**kw):
    return AllCombinations(legs=2,stage_column='stage',probability_mode='independent',payoff='binary',
        probability_columns={'a':'a::probability','b':'b::probability'},ticket_gate=DecisionLayer(('a','b'),**kw))

"""Read-only acceptance probes against commit 25eb79a; failure tests assert required behavior."""
import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.evaluation import *
from xdiyo_analytics.evaluation.stake_policy import *

@pytest.mark.parametrize('field,bad',[('issued_at',pd.Timestamp('2024-12-20',tz='UTC')),('trained_through',pd.NaT)])
def test_each_gate_leg_must_have_valid_own_temporal_provenance(field,bad):
    f=gate_offers(); f.loc[0,'a::'+field]=bad
    with pytest.raises(ValueError):
        compose_bets(f,gate())

def test_native_or_honors_missing_reject():
    f=gate_offers(); f.loc[0,'a::probability']=np.nan
    result=compose_bets(f,gate(gate='or',missing='reject'))
    assert len(result[0])==1
    audit=pd.DataFrame(result[0].attrs['decision_policy_audit'])
    assert audit.loc[audit.model.eq('a'),'reason'].eq('missing_model').all()

def test_native_source_league_cap_cannot_be_ignored():
    f=offers().drop(columns=['competition_id','season_id']).assign(source_league='league-A')
    result=compose_bets(f,Parlay(size=1),stake_policy=FixedStake(10,'u'),stake_context=StakeContext(100,100,'u','2025-01-01'),risk_limits=RiskLimits(exposure_caps=(('league_keys',0.),)))
    assert result[0].stake.eq(0).all()

def test_missing_exposure_cannot_bypass_zero_cap():
    frame=pd.DataFrame({'nominal_stake':[1.], 'decision_at':['2025-01-01'], 'fixture':[np.nan]},index=['a'])
    with pytest.raises(ValueError):
        allocate_batch(FrozenTable(frame),FixedStake(10,'u'),StakeContext(100,100,'u','2025-01-01'),RiskLimits(exposure_caps=(('fixture',0.),)))

def test_quote_must_be_available_at_candidate_decision():
    f=pd.DataFrame({'economic_key':['e'],'quote_id':['q'],'quote_at':['2025-01-02'],'decision_at':['2025-01-01'],'odds':[2.],'payoff':['binary']},index=['a'])
    predictions=f.assign(probability=.9,issued_at='2024-12-31',artifact_vintage='2024-12-30',trained_through='2024-12-29')
    with pytest.raises(ValueError):
        DecisionLayer(('a',)).decide({'a':predictions},DecisionContext('2025-01-03',FrozenTable(f)))

def test_legacy_default_and_multibet_total_golden():
    f=offers(3); f.loc[2,'settlement']='loss'
    tickets,members,metrics=compose_bets(f,BetSlip({'single':Parlay(size=1),'system':MultiBet(size=3,sizes=(2,3),stake=8,stake_mode='total')}))
    assert tickets.stake.tolist()==[1.,1.,1.,2.,2.,2.,2.]
    assert tickets.payout.tolist()==[2.,2.,0.,8.,0.,0.,0.]
    assert tickets.profit.sum()==1.
    assert members.stake.eq(20).all()

def test_batch_replacement_callback_is_outcome_free_and_order_invariant():
    f=offers(3); f.loc[2,'settlement']='loss'
    class Inspect:
        def allocate(self,batch,context):
            table=batch.frame
            assert not {'settlement','result','payout','profit'} & set(table)
            return FixedStake(10,'u').allocate(batch,context)
    context=StakeContext(100,100,'u','2025-01-01')
    p=BetSlip({'single':Parlay(size=1,stake=7),'system':MultiBet(size=3,sizes=(2,3),stake=8,stake_mode='total')})
    a=compose_bets(f,p,stake_policy=Inspect(),stake_context=context,risk_limits=RiskLimits(exposure_caps=(('league_keys',35.),)))[0]
    b=compose_bets(f.iloc[::-1].assign(settlement='loss'),p,stake_policy=Inspect(),stake_context=context,risk_limits=RiskLimits(exposure_caps=(('league_keys',35.),)))[0]
    assert a.nominal_stake.tolist()==[7,7,7,2,2,2,2]
    assert a.stake.tolist()==[5]*7
    pd.testing.assert_frame_equal(a[['ticket_id','stake','nominal_stake']],b[['ticket_id','stake','nominal_stake']])
    assert a.attrs['allocation_audit']==b.attrs['allocation_audit']

def test_zero_funding_counts_and_payouts_use_actual_stakes():
    f=offers()
    tickets,members,metrics=compose_bets(f,Parlay(size=1,stake=7),stake_policy=FixedStake(0,'u'),stake_context=StakeContext(100,100,'u','2025-01-01'))
    assert tickets.accounting_status.eq('unfunded').all()
    assert tickets.payout.eq(0).all() and tickets.profit.eq(0).all()
    values=metrics.set_index('metric').value
    assert values.selected_tickets==2 and values.funded_tickets==0 and values.unfunded_tickets==2 and values.settled_stakes==0

def test_replay_simultaneous_batch_reserve_and_settlement_tie():
    tickets=pd.DataFrame({'nominal_stake':[1.]*4,'decision_at':['2025-01-01','2025-01-01','2025-01-02','2025-01-03']},index=['a','b','c','d'])
    settlements=pd.DataFrame({'available_at':['2025-01-03','2025-01-04','2025-01-04','2025-01-05'],'return_multiplier':[2.,0.,1.,1.]},index=tickets.index)
    r=replay_bankroll(FrozenTable(tickets),settlements,FixedFraction(.75,'u'),initial_capital=100,currency='u')
    assert r.allocations.actual_stake.tolist()==[50.,50.,0.,100.]
    assert r.ledger.cash.min()==0 and r.summary['ending_wealth']==100 and r.summary['outstanding_reserve']==0
    assert r.ledger.loc[r.ledger.time.eq(pd.Timestamp('2025-01-03',tz='UTC')),'kind'].tolist()==['settle','place']

def test_binary_kelly_and_historical_cutoff_enforced():
    frame=pd.DataFrame({'nominal_stake':[1.],'decision_at':['2025-01-01'],'odds':[2.],'probability':[.6],'probability_provenance':['heldout'],'probability_issued_at':['2024-12-31'],'payoff':['binary'],'selection_id':['rule']},index=['a'])
    context=StakeContext(100,100,'u','2025-01-01')
    result=allocate_batch(FrozenTable(frame),FractionalKelly('u',alpha=.5,max_fraction=1),context)
    assert result.amounts.iloc[0]==pytest.approx(10,abs=.011)
    with pytest.raises(ValueError):
        allocate_batch(FrozenTable(frame.assign(payoff='push_void')),FractionalKelly('u'),context)
    h=pd.DataFrame({'selection_id':['rule']*2,'selected':[True]*2,'won':[True,False],'available_at':['2024-12-29','2025-01-02'],'prediction_origin':['chronological_oof']*2,'issued_at':['2024-12-28','2025-01-01'],'trained_through':['2024-12-27','2024-12-30']})
    rate=HistoricalRateSource('rule','2024-12-31',exchangeable_within_strata=True).fit(h)
    assert rate.audit_[0]['samples']==1
    assert rate.probabilities(FrozenTable(frame),context).iloc[0]==pytest.approx(2/3)
    with pytest.raises(ValueError):
        HistoricalRateSource('rule','2024-12-31').fit(h.assign(trained_through='2024-12-28'))


@pytest.mark.parametrize('model', ['a','b'])
@pytest.mark.parametrize('leg', [0,1])
@pytest.mark.parametrize('field,bad', [('issued_at','2024-12-20'),('trained_through',None),
    ('artifact_vintage','2025-01-02'),('issued_at',None)])
def test_provenance_validation_keeps_model_leg_associations(model,leg,field,bad):
    f=gate_offers()
    f.loc[leg,f'{model}::{field}']=pd.Timestamp(bad,tz='UTC') if bad else pd.NaT
    for source in (f,f.iloc[::-1]):
        with pytest.raises(ValueError,match='provenance|issue time'):
            compose_bets(source,gate())


@pytest.mark.parametrize('missing', ['column','one_leg','all_legs'])
@pytest.mark.parametrize('mode', ['and','or'])
def test_native_missing_probability_abstains_as_configured(missing,mode):
    f=gate_offers()
    if missing=='column':
        f=f.drop(columns=['a::probability','a::issued_at','a::trained_through','a::artifact_vintage'])
    elif missing=='one_leg':
        f.loc[0,'a::probability']=np.nan
    else:
        f['a::probability']=np.nan
    tickets,_,_=compose_bets(f,gate(gate=mode,missing='reject'))
    assert len(tickets)==int(mode=='or')
    audit=pd.DataFrame(tickets.attrs['decision_policy_audit'])
    assert audit.loc[audit.model.eq('a'),'reason'].eq('missing_model').all()
    with pytest.raises(ValueError,match='prediction|required'):
        compose_bets(f,gate(gate=mode,missing='error'))


@pytest.mark.parametrize('bad', [-.1,1.1,np.inf,-np.inf])
def test_invalid_supplied_probability_is_not_abstention(bad):
    f=gate_offers()
    f.loc[0,'a::probability']=bad
    f.loc[1,'a::probability']=np.nan
    with pytest.raises(ValueError,match='finite'):
        compose_bets(f,gate(gate='or',missing='reject'))


@pytest.mark.parametrize('identity', [None,np.nan,(),(None,),((1,None),)])
def test_incomplete_exposure_identity_fails_closed(identity):
    f=pd.DataFrame({'nominal_stake':[1.],'decision_at':['2025-01-01'],'fixture':[identity]},index=['one'])
    with pytest.raises(ValueError,match='identities'):
        allocate_batch(FrozenTable(f),FixedStake(10,'u'),StakeContext(100,100,'u','2025-01-01'),RiskLimits(exposure_caps=(('fixture',0.),)))


@pytest.mark.parametrize('column', ['league_keys','round_keys'])
def test_source_identity_caps_account_for_open_exposure(column):
    f=offers().drop(columns=['competition_id','season_id']).assign(source_league='league-A')
    identity='league-A' if column=='league_keys' else ('league-A','24_25',1)
    result=compose_bets(f,Parlay(size=1),stake_policy=FixedStake(10,'u'),
        stake_context=StakeContext(100,100,'u','2025-01-01',open_exposure=((column,identity,5.),)),
        risk_limits=RiskLimits(exposure_caps=((column,7.),)))
    assert result[0].stake.tolist()==[1.,1.]


def test_safe_duplicate_decisions_ignore_conflicting_outcomes_and_mark_unresolved():
    f=gate_offers()
    duplicate=f.iloc[[0]].assign(row_position=100)
    original=pd.concat([f,duplicate],ignore_index=True)
    changed=original.copy()
    changed.loc[2,'settlement']='loss'
    a=compose_bets(original,gate())[0]
    b=compose_bets(changed,gate())[0]
    pd.testing.assert_frame_equal(a[['ticket_id','stake','probability']],b[['ticket_id','stake','probability']])
    assert a.attrs['decision_policy_audit']==b.attrs['decision_policy_audit']
    assert b.accounting_status.eq('unresolved').all()
    # The inherited ordinary composition behavior is unchanged.
    with pytest.raises(ValueError,match='Conflicting'):
        compose_bets(changed,AllCombinations(legs=2,stage_column='stage'))


def test_settlement_overflow_is_atomic():
    from copy import deepcopy
    book=BankrollLedger(1e308,'u')
    table=FrozenTable(pd.DataFrame({'nominal_stake':[1.],'decision_at':['2025-01-01']},index=['one']))
    book.place(table,pd.Series([1.],index=['one']),'2025-01-01')
    before=deepcopy(vars(book))
    with pytest.raises(ValueError,match='conservation'):
        book.settle('one',gross_return=1e308,time='2025-01-02',available_at='2025-01-02')
    assert vars(book)==before
    book.settle('one',gross_return=0,time='2025-01-02',available_at='2025-01-02')
    assert not book.open_tickets


def test_missing_open_exposure_identity_cannot_bypass_cap():
    table=FrozenTable(pd.DataFrame({'nominal_stake':[1.],'decision_at':['2025-01-01'],'fixture':[1]},index=['one']))
    context=StakeContext(100,100,'u','2025-01-01',open_exposure=(('fixture',np.nan,10.),))
    with pytest.raises(ValueError,match='Open exposure'):
        allocate_batch(table,FixedStake(10,'u'),context,RiskLimits(exposure_caps=(('fixture',5.),)))
