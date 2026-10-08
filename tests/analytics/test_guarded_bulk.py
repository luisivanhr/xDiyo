"""Exact repaired-reference parity plus independent scalar gate/settlement oracles."""
from dataclasses import replace
from itertools import product, combinations
from math import prod
import pickle
import runpy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.evaluation import compose_bets, BetSlip, QuoteAvailability
from xdiyo_analytics.evaluation import bulk_tickets
from xdiyo_analytics.evaluation.stake_policy import FixedStake, StakeContext
from test_quote_availability import research_legs, policy
from xdiyo_analytics.reporting.tickets import ticket_html


def population(n=6):
    base = research_legs().iloc[:1]
    data = pd.concat([base]*n, ignore_index=True) if n else base.iloc[:0].copy()
    data['event_id'] = np.arange(n, dtype=np.uint64) + 2**63 + 7
    data['row_position'] = range(n)
    data['quote_id'] = [f'q{i}' for i in range(n)]
    data['odds'] = np.array([2.,3.,4.,5.,6.,7.][:n])
    data['a::probability'] = [.3,.8,.2,.6,.45,.7][:n]
    data['b::probability'] = [.4,.7,.2,.5,.45,.8][:n]
    sync(data, policy().quote_availability)
    return data


def sync(data, contract):
    for model in ('a','b'):
        for field in contract.identities(data):
            data[f'{model}::{field}'] = data[field]


def equal_outputs(a, b):
    from test_consensus_performance import _benchmark
    _benchmark['assert_outputs_equal'](a, b)


def both(data, p, monkeypatch, **kwargs):
    before = data.copy(deep=True)
    actual = compose_bets(data, p, **kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(bulk_tickets, 'supports_bulk', lambda *a: False)
        expected = compose_bets(data, p, **kwargs)
    equal_outputs(actual, expected)
    pd.testing.assert_frame_equal(data, before, check_exact=True)
    return actual, expected

@pytest.mark.parametrize('k', [1,2,3,4])
@pytest.mark.parametrize('n', [0,1,4,6])
@pytest.mark.parametrize('observed', [False,True])
def test_full_exact_parity(k,n,observed,monkeypatch):
    data = population(n); p = replace(policy(), legs=k)
    if observed:
        c = QuoteAvailability(); data['quote_at'] = data.decision_at
        data['assumed_available_at'] = pd.NaT
        for key,value in c.labels.items(): data[key] = value
        sync(data,c); p = replace(p,quote_availability=c)
    assert bulk_tickets.supports_bulk(data, {'tickets':p}, ('event_id',))
    a,b = both(data,p,monkeypatch)
    assert ticket_html(a[0],a[1],{}) == ticket_html(b[0],b[1],{})
    equal_outputs(pickle.loads(pickle.dumps(a)), b)

@pytest.mark.parametrize('states', list(product(['win','loss','missing','push','void'], repeat=2)))
def test_all_settlement_pairs_scalar_oracle(states,monkeypatch):
    data = population(2); data['settlement'] = states
    data['a::probability'] = data['b::probability'] = .9
    (t,m,_),_ = both(data,policy(),monkeypatch)
    if 'loss' in states: expected,state = 0.,'loss'
    elif 'missing' in states: expected,state = np.nan,'missing'
    else:
        expected = prod(float(o) for o,s in zip(data.odds,states) if s=='win')
        state = 'win' if 'win' in states else 'void' if all(s=='void' for s in states) else 'push'
    assert t.settlement.iloc[0] == state
    assert (pd.isna(t.payout.iloc[0]) and pd.isna(expected)) or t.payout.iloc[0] == expected

@pytest.mark.parametrize('k',[1,2,3,4])
def test_independent_ordered_probability_and_ev_oracle(k,monkeypatch):
    data=population(); data['a::probability']=data['a::probability'].astype('float32')
    (t,m,_),_=both(data,replace(policy(),legs=k),monkeypatch)
    audit=t.attrs['decision_policy_audit']; candidates=t.attrs['ticket_candidates']
    for candidate, indices in zip(candidates,combinations(range(len(data)),k)):
        expected=[]
        for model in ('a','b'):
            probability=prod(float(data[f'{model}::probability'].iloc[i]) for i in indices)
            odds=prod(float(data.odds.iloc[i]) for i in indices)
            ballot=next(row for row in audit if row['candidate_id']==candidate['ticket_id'] and row['model']==model)
            assert ballot['probability']==probability
            assert ballot['value']==probability*odds-1.
            expected.append(probability*odds-1.>0)
        assert (candidate['ticket_id'] in set(t.ticket_id)) == all(expected)

@pytest.mark.parametrize('change', ['summary','float32','object','strings','stake','threshold','push','or','index','custom','identity','allocation','slip'])
def test_capability_fallback_is_predictable(change,monkeypatch):
    data=population(4); p=policy(); kwargs={}
    if change=='summary': p=replace(p,audit_level='summary')
    elif change in ('float32','object','strings'):
        data['odds']=data.odds.astype({'float32':'float32','object':object,'strings':str}[change]); sync(data,p.quote_availability)
        if change=='strings':
            for model in ('a','b'): data[f'{model}::odds']=pd.to_numeric(data.odds)
    elif change=='stake': p=replace(p,stake=.5)
    elif change=='threshold': p=replace(p,ticket_gate=replace(p.ticket_gate,threshold=.1))
    elif change=='push': p=replace(p,on_push='refund')
    elif change=='or': p=replace(p,ticket_gate=replace(p.ticket_gate,gate='or'))
    elif change=='index': data.index=[7,7,2,3]
    elif change=='identity': p=replace(p,ticket_gate=replace(p.ticket_gate,identity_columns=(*p.ticket_gate.identity_columns,'payoff')))
    elif change=='custom':
        class Custom(type(p.ticket_gate)): pass
        p=replace(p,ticket_gate=Custom(**p.ticket_gate.__dict__))
    elif change=='allocation': kwargs=dict(stake_policy=FixedStake(0,'u'),stake_context=StakeContext(10,10,'u','2025-01-01'))
    templates={'tickets':p} if change!='slip' else {'other':p}
    assert not bulk_tickets.supports_bulk(data,templates,('event_id',),kwargs.get('stake_policy'),kwargs.get('stake_context'))
    monkeypatch.setattr(bulk_tickets,'execute_bulk',lambda *a:pytest.fail('unsupported dispatch'))
    compose_bets(data,BetSlip(templates) if change=='slip' else p,**kwargs)

@pytest.mark.parametrize('field,bad',[('a::quote_id','forged'),('a::issued_at',pd.Timestamp('2030-01-01',tz='UTC')),('a::probability',np.nan),('odds',np.inf)])
@pytest.mark.parametrize('selected',[False,True])
def test_invalid_original_evidence_cannot_hide_in_rejected_or_small_pools(field,bad,selected):
    data=population(1);data['take']=selected;data[field]=data[field].astype(object);data.loc[0,field]=bad
    with pytest.raises(ValueError):compose_bets(data,policy())

@pytest.mark.parametrize('prices',[[2**40,2**40],[2**63,2],[1e200,1e200]])
def test_overflow_rejected_in_both_backends(prices,monkeypatch):
    data=population(2);data['odds']=prices;sync(data,policy().quote_availability)
    with pytest.raises(ValueError,match='overflow'):compose_bets(data,policy())
    monkeypatch.setattr(bulk_tickets,'supports_bulk',lambda *a:False)
    with pytest.raises(ValueError,match='overflow'):compose_bets(data,policy())

def test_outcome_mutation_cannot_change_any_decision(monkeypatch):
    data=population(); a=compose_bets(data,policy());data['settlement']=['win','loss','push','void','missing','win'];b=compose_bets(data,policy())
    for fields in [['ticket_id','odds','probability','stake']]:pd.testing.assert_frame_equal(a[0][fields],b[0][fields],check_exact=True)
    pd.testing.assert_frame_equal(a[1].drop(columns='settlement'),b[1].drop(columns='settlement'),check_exact=True)
    assert a[0].attrs==b[0].attrs

def test_call_local_evidence_and_cutoff_checks():
    data=population(4);compose_bets(data,policy())
    data['round']=[1,1,2,2]
    data.loc[2:,'decision_at']+=pd.Timedelta(hours=1);data.loc[2:,'kickoff_at']+=pd.Timedelta(hours=1)
    sync(data,policy().quote_availability)
    data.loc[0,'a::issued_at']=data.decision_at.iloc[0]+pd.Timedelta(minutes=30)
    with pytest.raises(ValueError):compose_bets(data,policy())


def test_mixed_timestamp_units_and_empty_unknown_outcomes(monkeypatch):
    data=population(4)
    data['decision_at']=data.decision_at.dt.as_unit('ms')
    data['kickoff_at']=data.kickoff_at.dt.as_unit('us')
    sync(data,policy().quote_availability)
    both(data,policy(),monkeypatch)

@pytest.mark.parametrize('selected',[False,True])
def test_reserved_quote_extensions_reject_before_selection(selected):
    data=population(1);data['take']=selected;data['quote_result']='win'
    for model in ('a','b'):data[f'{model}::quote_result']='win'
    with pytest.raises(ValueError,match='Outcome-bearing'):compose_bets(data,policy())


@pytest.mark.parametrize('k',[1,2,3,4])
def test_native_fixture_gate_and_cap_boundaries(k,monkeypatch):
    from test_consensus_performance import _benchmark
    data,pmfs,contract=_benchmark['population'](1)
    for model in pmfs:
        pmfs[model]['draw']=[.2,.19,.21,.9,.4,.3,.8,.7]
        pmfs[model]['non_draw']=1.-pmfs[model]['draw']
    api=runpy.run_path(str(Path(__file__).resolve().parents[2]/'examples/research_quote_consensus.py'))
    actual=api['consensus'](data,pmfs,contract,ticket_legs=k)
    with monkeypatch.context() as scoped:
        scoped.setattr(bulk_tickets,'supports_bulk',lambda *a:False)
        reference=api['consensus'](data,pmfs,contract,ticket_legs=k)
    equal_outputs(actual,reference)
    ballots=actual[3]
    for model in pmfs:
        expected=(1.-pmfs[model]['non_draw']) >= 1.-.80
        np.testing.assert_array_equal(ballots.loc[ballots.model.eq(model),'take'],expected)
    count=len(actual[0].attrs['ticket_candidates'])
    api['consensus'](data,pmfs,contract,ticket_legs=k,max_tickets=count)
    with pytest.raises(ValueError,match='max_tickets'):
        api['consensus'](data,pmfs,contract,ticket_legs=k,max_tickets=count-1)

@pytest.mark.parametrize('dtype',['int64','uint64','float32','object'])
def test_valid_price_representations_keep_output_bits(dtype,monkeypatch):
    data=population(4);data['odds']=data.odds.astype(dtype);sync(data,policy().quote_availability)
    both(data,policy(),monkeypatch)

def test_undefined_unselected_outcomes_not_consulted(monkeypatch):
    data=population(2);data['a::probability']=0.;data['settlement']='unsupported'
    (t,_,_),_=both(data,policy(),monkeypatch)
    assert t.empty
