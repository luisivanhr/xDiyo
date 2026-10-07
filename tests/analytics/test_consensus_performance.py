"""Structural performance regressions; no machine-dependent timing assertions."""
from pathlib import Path
import runpy
from dataclasses import replace
import numpy as np
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


def test_ticket_batch_checks_all_original_quote_rows_once(monkeypatch):
    from xdiyo_analytics.evaluation import QuoteAvailability
    tickets, members, _, _ = consensus(*population(1))
    tickets.attrs.clear(); members.attrs.clear()
    api = runpy.run_path(str(Path(__file__).resolve().parents[2]/'examples/research_quote_consensus.py'))
    _, _, contract = api['synthetic_example']()
    p = replace(policy(), quote_availability=contract)
    original = QuoteAvailability.validate
    calls = []
    def validate(self, frame, time, **kwargs):
        calls.append(len(frame))
        return original(self, frame, time, **kwargs)
    monkeypatch.setattr(QuoteAvailability, 'validate', validate)
    batch = ticket_allocation.ticket_batch(tickets, members, {'tickets': p}, ('event_id',))
    assert len(batch) == 28 and calls == [56]


@pytest.mark.parametrize('size', [2, 3, 5, 8])
@pytest.mark.parametrize('level', ['full', 'summary'])
def test_batched_model_product_preserves_original_order_and_bits(size, level):
    from test_composition_review_staking import offers, gate
    frame = offers(size)
    probabilities = np.array([np.nextafter(.2, 1), .9, .03, .78, .123456789, .999, 1., .81])[:size]
    for model in ('a', 'b'):
        frame[f'{model}::probability'] = probabilities if model == 'a' else probabilities[::-1]
        for field, value in [('issued_at','2025-01-01'),('trained_through','2024-12-29'),('artifact_vintage','2024-12-30')]:
            frame[f'{model}::{field}'] = pd.Timestamp(value, tz='UTC')
    tickets, _, _ = compose_bets(frame, replace(gate(threshold=-2), legs=size, audit_level=level))
    assert len(tickets) == 1
    audit = (pd.DataFrame(tickets.attrs['decision_policy_audit']) if level == 'full' else
             pd.DataFrame(**tickets.attrs['selected_model_values']))
    for model in ('a', 'b'):
        expected = float(np.prod(frame[f'{model}::probability'].astype(float)))
        ballot = audit.loc[audit.model.eq(model)].iloc[0]
        assert ballot.probability.hex() == expected.hex()
        assert ballot.value == expected * float(np.prod(frame.odds)) - 1.


def test_batched_model_validation_uses_each_ticket_cutoff():
    from test_composition_review_staking import offers, gate
    frame = offers(4)
    frame['round'] = [1, 1, 2, 2]
    frame.loc[2:, 'decision_at'] += pd.Timedelta(hours=1)
    for model in ('a', 'b'):
        frame[f'{model}::probability'] = .9
        frame[f'{model}::issued_at'] = pd.Timestamp('2025-01-01', tz='UTC')
        frame[f'{model}::trained_through'] = pd.Timestamp('2024-12-29', tz='UTC')
        frame[f'{model}::artifact_vintage'] = pd.Timestamp('2024-12-30', tz='UTC')
    compose_bets(frame, gate())
    frame.loc[0, 'a::issued_at'] += pd.Timedelta(minutes=30)
    with pytest.raises(ValueError, match='provenance'):
        compose_bets(frame, gate())


def test_quote_and_report_serialization_exclude_unused_wide_columns(monkeypatch):
    from xdiyo_analytics.reporting.tickets import ticket_html
    tickets, members, _ = compose_bets(research_legs(), policy())
    expected = ticket_html(tickets, members, {})
    members['irrelevant_payload'] = ['unused'] * len(members)
    original = pd.DataFrame.to_dict
    def to_dict(frame, *args, **kwargs):
        assert 'irrelevant_payload' not in frame, 'Do not serialize unused membership columns.'
        return original(frame, *args, **kwargs)
    monkeypatch.setattr(pd.DataFrame, 'to_dict', to_dict)
    batch = ticket_allocation.ticket_batch(tickets, members, {'tickets': policy()}, ('event_id',))
    assert len(batch) == 1
    assert 'quote_vendor_scheduled_at' in batch.quote_legs.iloc[0]
    assert ticket_html(tickets, members, {}) == expected
