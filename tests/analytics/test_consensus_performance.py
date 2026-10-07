"""Structural performance regressions; no machine-dependent timing assertions."""
from pathlib import Path
import runpy
import pandas as pd
import pytest

from xdiyo_analytics.evaluation import compose_bets
from xdiyo_analytics.evaluation import ticket_allocation
from xdiyo_analytics.evaluation.stake_policy import FixedStake, StakeContext
from test_quote_availability import research_legs, policy

_benchmark = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'examples/benchmark_quote_consensus.py'))
population, consensus = _benchmark['population'], _benchmark['consensus']


def test_membership_index_preserves_exact_ids_all_rows_and_order():
    members = pd.DataFrame({'ticket_id': ['a:1', 'a:10', 'a:1', 'b:1', 'a:1'],
                            'quote_id': ['q1', 'q2', 'q3', 'q4', 'conflicting-q1'],
                            'event_id': [1, 2, 3, 4, 1]}, index=[9, 2, 9, 1, 0])
    indexed = ticket_allocation._index_members(members)
    assert list(indexed) == ['a:1', 'a:10', 'b:1']
    for key, frame in indexed.items():
        pd.testing.assert_frame_equal(frame, members.loc[members.ticket_id.eq(key)])
    assert len(indexed['a:1']) == 3  # even conflicting duplicates reach validation


@pytest.mark.parametrize('funded', [False, True])
def test_compose_publishes_audits_after_internal_pandas_operations(monkeypatch, funded):
    original = pd.DataFrame.__finalize__

    def check_finalize(self, other, *args, **kwargs):
        assert not {'decision_policy_audit', 'allocation_audit', 'ticket_candidates',
                    'ticket_selection_summary'} & set(getattr(other, 'attrs', {}))
        return original(self, other, *args, **kwargs)

    data = research_legs()
    kwargs = dict(stake_policy=FixedStake(3, 'u'),
                  stake_context=StakeContext(100, 100, 'u', data.decision_at.iloc[0])) if funded else {}
    with monkeypatch.context() as scoped:
        scoped.setattr(pd.DataFrame, '__finalize__', check_finalize)
        tickets, members, metrics = compose_bets(data, policy(), **kwargs)
    assert len(tickets) == 1 and len(members) == 2 and not metrics.empty
    assert len(tickets.attrs['decision_policy_audit']) == 2
    assert tickets.attrs['ticket_candidates']
    assert tickets.attrs['ticket_selection_summary']
    assert tickets.attrs['quote_assumptions']
    if funded:
        assert tickets.attrs['allocation_audit']
        assert tickets.stake.iloc[0] == 3


def test_membership_is_indexed_once_and_not_rescanned_per_ticket(monkeypatch):
    original_index = ticket_allocation._index_members
    original_eq = pd.Series.eq
    calls = []

    def index(members):
        calls.append(len(members))
        return original_index(members)

    def eq(series, other, *args, **kwargs):
        assert series.name != 'ticket_id', 'Do not rescan membership for individual tickets.'
        return original_eq(series, other, *args, **kwargs)

    monkeypatch.setattr(ticket_allocation, '_index_members', index)
    monkeypatch.setattr(pd.Series, 'eq', eq)
    tickets, members, _, _ = consensus(*population(1))
    assert calls == [len(members)]
    assert len(tickets) == 28 and len(members) == 56


def test_direct_finalization_returns_full_audits():
    data = research_legs()
    tickets, members, _ = compose_bets(data, policy())
    tickets.attrs.clear()
    members.attrs.clear()
    finalized, returned_members = ticket_allocation.finalize_tickets(
        tickets, members, {'tickets': policy()}, ('event_id',), None, None, None)
    assert len(finalized.attrs['decision_policy_audit']) == 2
    pd.testing.assert_frame_equal(returned_members, members)


def test_index_does_not_hide_conflicting_quote_leg_evidence():
    data = research_legs()
    tickets, members, _ = compose_bets(data, policy())
    conflicting = pd.concat([members, members.iloc[:1].assign(quote_id='conflict')], ignore_index=True)
    with pytest.raises(ValueError):
        ticket_allocation.finalize_tickets(tickets, conflicting, {'tickets': policy()},
                                           ('event_id',), None, None, None)
