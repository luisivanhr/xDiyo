from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.evaluation.decision_layer import FrozenTable, DecisionContext, DecisionLayer
from xdiyo_analytics.evaluation.stake_policy import *
from xdiyo_analytics.evaluation.bankroll import BankrollLedger, replay_bankroll
from xdiyo_analytics.evaluation import Parlay, MultiBet, BetSlip, compose_bets
from test_bet_tickets import ledger


def batch():
    return pd.DataFrame(dict(nominal_stake=[1.,3.], odds=[2.,4.], probability=[.6,.4],payoff=['binary']*2,
        probability_provenance=['heldout']*2,probability_issued_at=['2024-12-31']*2,decision_at=['2025-01-01']*2,selection_id=['chosen']*2),index=pd.Index(['a','b'],name='ticket_id'))


def test_kelly_known_values_and_final_cash_cap():
    context = StakeContext(100,100,'u','2025-01-01')
    result = allocate_batch(FrozenTable(batch()),FractionalKelly('u',alpha=.5,max_fraction=1),context)
    np.testing.assert_allclose(result.amounts,[10,10],atol=.011)
    assert result.audit.nominal_stake.tolist()==[1,3]
    capped = allocate_batch(FrozenTable(batch()),FixedStake(100,'u'),context,RiskLimits(per_ticket=30))
    assert capped.amounts.tolist()==[30,30]
    zero = allocate_batch(FrozenTable(batch()),FractionalKelly('u',alpha=0),context)
    assert zero.amounts.eq(0).all()
    for key,value in [('odds',1),('probability',np.nan),('probability',1.2),('payoff','push_void')]:
        with pytest.raises(ValueError):
            FractionalKelly('u').allocate(FrozenTable(batch().assign(**{key:value})),context)


def test_whole_slip_replacement_outcome_independence_and_nominal_total():
    offers=ledger(('win','win','loss'))
    policy=BetSlip({'singles':Parlay(size=1,stake=7), 'system':MultiBet(size=3,sizes=(2,3),stake=8,stake_mode='total')})
    context=StakeContext(100,100,'u','2025-01-01')
    a=compose_bets(offers,policy,stake_policy=FixedStake(4,'u'),stake_context=context)
    b=compose_bets(offers.assign(settlement='loss'),policy,stake_policy=FixedStake(4,'u'),stake_context=context)
    assert a[0].stake.tolist()==[4]*7
    assert a[0].nominal_stake.tolist()==[7]*3+[2]*4
    assert a[0].payout.tolist()==[8,8,0,16,0,0,0]
    pd.testing.assert_frame_equal(a[0][['ticket_id','stake','nominal_stake']],b[0][['ticket_id','stake','nominal_stake']])
    assert a[0].attrs['allocation_audit']==b[0].attrs['allocation_audit']


def test_reserve_release_refund_fees_and_no_premature_reinvestment():
    tickets=batch()
    tickets.loc['b','decision_at']='2025-01-02'
    settlements=pd.DataFrame({'available_at':['2025-01-03','2025-01-04'],'return_multiplier':[2.,0.]},index=tickets.index)
    result=replay_bankroll(FrozenTable(tickets),settlements,FixedFraction(.75,'u'),initial_capital=100,currency='u')
    assert result.allocations.actual_stake.tolist()==[75,25]
    assert result.summary['ending_wealth']==150
    assert result.ledger.reserve.max()==100
    assert result.ledger.cash.min()==0
    changed=settlements.assign(return_multiplier=9.)
    other=replay_bankroll(FrozenTable(tickets),changed,FixedFraction(.75,'u'),initial_capital=100,currency='u')
    pd.testing.assert_frame_equal(result.allocations,other.allocations)
    book=BankrollLedger(100,'u')
    one=FrozenTable(batch().iloc[:1])
    book.place(one,pd.Series([20.],index=one.frame.index),'2025-01-01')
    assert (book.wealth,book.reserve,book.cash)==(100,20,80)
    with pytest.raises(ValueError,match='availability'):
        book.settle('a',gross_return=20,time='2025-01-02',available_at='2025-01-03')
    assert book.cash==80
    book.settle('a',gross_return=20,time='2025-01-03',available_at='2025-01-03',fee=1)
    assert (book.wealth,book.reserve,book.cash)==(99,0,99)


def test_history_rate_cutoff_selection_and_shrinkage():
    history=pd.DataFrame(dict(selection_id=['chosen']*4,available_at=pd.date_range('2024-01-01',periods=4,tz='UTC'),
                              selected=[True]*4,won=[1,1,0,0]))
    source=HistoricalRateSource('chosen','2024-01-02',learned_selection=False,exchangeable_within_strata=True).fit(history)
    context=StakeContext(100,100,'u','2025-01-01')
    assert source.probabilities(FrozenTable(batch()),context).tolist()==[.75,.75]
    with pytest.raises(ValueError,match='provenance'):
        HistoricalRateSource('chosen','2024-01-02').fit(history)
    with pytest.raises(ValueError,match='identity'):
        source.probabilities(FrozenTable(batch().assign(selection_id='different')),context)


def test_gates_align_economics_and_strict_zero_ev():
    offers=batch().assign(economic_key=['one','two'],quote_id=['q1','q2'],quote_at='2024-12-31')
    context=DecisionContext('2025-01-01',FrozenTable(offers))
    first=offers.assign(probability=[.5,.25],issued_at='2024-12-31',trained_through='2024-12-01',artifact_vintage='2024-12-02')
    second=first.assign(probability=[.7,.3])
    assert DecisionLayer(('a','b')).decide({'a':first,'b':second},context).selected.empty
    assert len(DecisionLayer(('a','b'),strict=False).decide({'a':first.iloc[::-1],'b':second},context).selected)==2
    assert len(DecisionLayer(('a','b'),gate='or').decide({'a':first,'b':second},context).selected)==2
    with pytest.raises(ValueError,match='identity'):
        DecisionLayer(('a','b')).decide({'a':first,'b':second.assign(quote_id='wrong')},context)
    with pytest.raises(ValueError,match='unavailable'):
        DecisionLayer(('a',)).decide({'a':first.assign(trained_through='2026-01-01')},context)
