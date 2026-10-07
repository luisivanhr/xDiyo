from dataclasses import replace
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.evaluation import compose_bets
from xdiyo_analytics.evaluation import DecisionContext, FrozenTable
from xdiyo_analytics.evaluation.stake_policy import FixedStake, StakeContext
from xdiyo_analytics.evaluation.audit_storage import fingerprint
from xdiyo_analytics.reporting.contracts import StudyResult
from xdiyo_analytics.reporting.tickets import add_tickets
from xdiyo_analytics.reporting.tickets import ticket_html
from test_quote_availability import research_legs, policy


def no_attrs(frame):
    copy = frame.copy()
    copy.attrs.clear()
    return copy


@pytest.mark.parametrize('case', ['accepted', 'rejected', 'mixed', 'settled', 'empty', 'undersized'])
def test_summary_preserves_computation_and_counts(case, tmp_path):
    data = research_legs()
    p = policy()
    if case == 'rejected': p = replace(p, ticket_gate=replace(p.ticket_gate, threshold=100))
    if case == 'mixed': data['a::probability'] = [.1, .9]
    if case == 'settled': data['settlement'] = ['win', 'loss']
    if case == 'empty': data = data.iloc[:0]
    if case == 'undersized': data = data.iloc[:1]
    full = compose_bets(data, p)
    summary = compose_bets(data, replace(p, audit_level='summary'))
    for i, (a, b) in enumerate(zip(full, summary)):
        if i == 0: a = a.drop(columns=['quote_legs'], errors='ignore')
        pd.testing.assert_frame_equal(no_attrs(a), no_attrs(b), check_exact=True)
    t = summary[0]
    manifest = t.attrs['audit_manifest']
    assert manifest['candidate_count'] == len(full[0].attrs['ticket_candidates'])
    assert manifest['selected_count'] == len(t)
    assert manifest['rejected_count'] == manifest['candidate_count'] - len(t)
    assert manifest['input_sha256'] == fingerprint(data)
    assert manifest['audit_level'] == 'summary' and not manifest['complete_row_audit']
    assert 'decision_policy_audit' not in t.attrs and 'ticket_candidates' not in t.attrs
    assert manifest['omitted_details'] and manifest['runtime_source_sha256'] and manifest['models']
    if 'decision_policy_audit' in full[0].attrs:
        old = pd.DataFrame(full[0].attrs['decision_policy_audit'])
        counts = old.groupby(['model', 'reason'], sort=False).size().to_dict()
        assert counts == {(r['model'], r['reason']): r['count'] for r in manifest['gate_reasons']}
        selected = t.attrs['selected_model_values']
        new = pd.DataFrame(selected['data'], columns=selected['columns']).drop(columns='template')
        expected = old.loc[old.candidate_id.isin(t.ticket_id), list(new)].reset_index(drop=True)
        # Empty constructors infer object; nonempty original values are exact.
        pd.testing.assert_frame_equal(new, expected, check_dtype=len(new) > 0, check_exact=True)
    path = tmp_path/'summary.parquet'
    t.to_parquet(path)
    restored = pd.read_parquet(path)
    assert restored.attrs['audit_manifest'] == manifest


@pytest.mark.parametrize('field,bad', [
    ('decision_at', pd.NaT), ('quote_at', '2030-01-01T00:00:00Z'),
    ('assumed_available_at', '2020-01-01'), ('a::quote_id', 'wrong'),
    ('b::quote_crosswalk_hash', 'wrong'), ('a::quote_snapshot_hash', 'wrong'),
    ('a::market', 'wrong'), ('a::selection', 'wrong'), ('b::odds', 1.1),
    ('a::issued_at', pd.NaT), ('b::trained_through', '2030-01-01T00:00:00Z'),
    ('a::artifact_vintage', '2030-01-01T00:00:00Z'), ('a::probability', np.nan)])
@pytest.mark.parametrize('level', ['full', 'summary'])
def test_modes_reject_same_invalid_original_evidence(level, field, bad):
    data = research_legs()
    data[field] = data[field].astype(object)
    data.loc[0, field] = bad
    with pytest.raises(ValueError):
        compose_bets(data, replace(policy(), audit_level=level))


def test_reused_ids_different_asof_and_dynamic_pin_do_not_reuse_evidence():
    data = research_legs()
    for level in ('full', 'summary'):
        p = replace(policy(), audit_level=level)
        for field in ('quote_vendor', 'line', 'period'):
            data[field] = 'original'
            for model in ('a', 'b'): data[f'{model}::{field}'] = data[field]
        compose_bets(data, p)
        for field in ('quote_vendor', 'line', 'period'):
            bad = data.copy()
            bad[f'a::{field}'] = 'different'
            with pytest.raises(ValueError): compose_bets(bad, p)
        bad = data.copy()
        bad['a::issued_at'] = pd.to_datetime(bad.decision_at, utc=True) + pd.Timedelta(days=1)
        with pytest.raises(ValueError): compose_bets(bad, p)


def test_summary_report_explains_omissions_instead_of_empty_audit():
    result = add_tickets(StudyResult('Summary', tables={'ledger': research_legs()}),
                         replace(policy(), audit_level='summary'), SimpleNamespace(match_columns=('event_id',)))
    assert 'decision_policy_audit' not in result.tables
    assert 'ticket_candidates' not in result.tables
    assert {'audit_manifest', 'selected_model_values', 'gate_summary'} <= set(result.tables)
    manifest = json.loads(result.tables['audit_manifest'].manifest.iloc[0])
    assert manifest['candidate_count'] == 1
    assert any('Summary audit:' in note for note in result.notes)


def test_fingerprints_distinguish_neighboring_binary_probabilities():
    a = pd.DataFrame({'p': [.3]})
    b = pd.DataFrame({'p': [np.nextafter(.3, 1.)]})
    assert fingerprint(a) != fingerprint(b)


def test_summary_consensus_keeps_fixture_votes_and_original_probabilities():
    api = runpy.run_path(str(Path(__file__).resolve().parents[2]/'examples/research_quote_consensus.py'))
    inputs = api['synthetic_example']()
    full = api['consensus'](*inputs)
    summary = api['consensus'](*inputs, audit_level='summary')
    fields = ['candidate_id', 'model', 'probability', 'value', 'take', 'reason', 'comparator']
    pd.testing.assert_frame_equal(no_attrs(full[3][fields]), no_attrs(summary[3]), check_exact=True)
    assert summary[0].attrs['audit_manifest']['fixture_gate']['threshold'] == 1.-.80
    assert summary[0].attrs['audit_manifest']['original_pmf_fingerprints'] == {m: fingerprint(p) for m, p in inputs[1].items()}


def test_renderer_does_not_propagate_manifest_per_row(monkeypatch):
    t, m, _ = compose_bets(research_legs(), replace(policy(), audit_level='summary'))
    original = pd.DataFrame.__finalize__
    copies = []
    def traced(self, other, *args, **kwargs):
        if 'audit_manifest' in getattr(other, 'attrs', {}): copies.append(1)
        return original(self, other, *args, **kwargs)
    monkeypatch.setattr(pd.DataFrame, '__finalize__', traced)
    html = ticket_html(t, m, {})
    assert 'Summary audit' in html and len(copies) == 2
    assert 'audit_manifest' in t.attrs and 'audit_manifest' in m.attrs


@pytest.mark.parametrize('amount', [0, 3])
def test_summary_preserves_allocated_and_unfunded_accounting(amount):
    data = research_legs()
    kwargs = dict(stake_policy=FixedStake(amount, 'u'), stake_context=StakeContext(100, 100, 'u', data.decision_at.iloc[0]))
    full = compose_bets(data, policy(), **kwargs)
    summary = compose_bets(data, replace(policy(), audit_level='summary'), **kwargs)
    for i, (a, b) in enumerate(zip(full, summary)):
        if i == 0: a = a.drop(columns=['quote_legs'])
        pd.testing.assert_frame_equal(no_attrs(a), no_attrs(b), check_exact=True)
    assert full[0].attrs['allocation_audit'] == summary[0].attrs['allocation_audit']


def test_summary_outcome_model_fixture_order_and_strict_zero_boundary():
    data = research_legs()
    p = replace(policy(), audit_level='summary')
    baseline = compose_bets(data, p)[0]
    changed = data.iloc[::-1].assign(settlement='loss', profit=-1.e9, payout=0, actual=999)
    reversed_policy = replace(p, ticket_gate=replace(p.ticket_gate, models=('b', 'a')),
                              probability_columns={'b': 'b::probability', 'a': 'a::probability'})
    mutated = compose_bets(changed, reversed_policy)[0]
    fields = ['ticket_id', 'odds', 'stake', 'take']
    pd.testing.assert_frame_equal(no_attrs(baseline[fields]), no_attrs(mutated[fields]), check_exact=True)
    for table in (baseline, mutated):
        values = table.attrs['selected_model_values']
        ballots = pd.DataFrame(values['data'], columns=values['columns']).sort_values(['candidate_id', 'model']).reset_index(drop=True)
        if table is baseline: expected = ballots
        else: pd.testing.assert_frame_equal(expected, ballots, check_exact=True)
    data['a::probability'] = [1., 1./6.]
    result = compose_bets(data, p)[0]
    assert result.empty and result.attrs['audit_manifest']['rejected_count'] == 1


def test_summary_expansion_ev_rejections_are_counted_and_disclosed():
    data = research_legs().assign(p_win=.5, p_push=0.)
    p = replace(policy(), audit_level='summary', min_ev=100.)
    result = add_tickets(StudyResult('No tickets', tables={'ledger': data}), p,
                         SimpleNamespace(match_columns=('event_id',)))
    manifest = result.tables['tickets'].attrs['audit_manifest']
    assert manifest['candidate_count'] == 1 and manifest['rejected_count'] == 1
    assert manifest['expansion_summary'][0]['rejected_by_ev'] == 1
    assert 'ticket_candidates' not in result.tables


def test_summary_multi_template_pins_each_original_model_mapping():
    from xdiyo_analytics.evaluation import BetSlip
    data = research_legs().assign(alternate_probability=.8)
    first = policy()
    second = replace(first, probability_columns={'a': 'alternate_probability', 'b': 'b::probability'})
    full = compose_bets(data, BetSlip({'first': first, 'second': second}))
    summary = compose_bets(data, BetSlip({name: replace(p, audit_level='summary')
                                        for name, p in [('first', first), ('second', second)]}))
    for i, (a, b) in enumerate(zip(full, summary)):
        if i == 0: a = a.drop(columns='quote_legs')
        pd.testing.assert_frame_equal(no_attrs(a), no_attrs(b), check_exact=True)
    models = summary[0].attrs['audit_manifest']['models']
    assert models['first']['a']['source'] == 'a::probability'
    assert models['second']['a']['source'] == 'alternate_probability'
    assert models['first']['a']['stream_sha256'] != models['second']['a']['stream_sha256']


def test_summary_gate_counts_exclude_templates_without_a_gate():
    from xdiyo_analytics.evaluation import BetSlip
    from test_composition_review_staking import gate_offers, gate
    p = replace(gate(), audit_level='summary')
    plain = replace(p, ticket_gate=None, probability_columns=None)
    tickets, _, _ = compose_bets(gate_offers(), BetSlip({'gated': p, 'plain': plain}))
    manifest = tickets.attrs['audit_manifest']
    assert manifest['selected_count'] == 2
    assert manifest['gate_counts'] == {'candidates': 1, 'selected': 1, 'rejected': 0}


def test_batched_quote_validation_keeps_per_ticket_formats_and_asof():
    from xdiyo_analytics.evaluation.ticket_allocation import ticket_batch
    p = policy()
    tickets, members, _ = compose_bets(research_legs(), p)
    first = ticket_batch(tickets, members, {'tickets': p}, ('event_id',))
    second = first.copy()
    second.index = ['second-ticket']
    delta = pd.Timedelta(microseconds=123000)
    records = json.loads(second.quote_legs.iloc[0])
    for leg in records:
        for field in ('decision_at', 'assumed_available_at'):
            leg[field] = (pd.Timestamp(leg[field]) + delta).isoformat()
    second['quote_legs'] = json.dumps(records)
    for field in ('decision_at', 'assumed_available_at'):
        second[field] = second[field] + delta
    combined = pd.concat([first, second])
    time = combined.decision_at.max()
    DecisionContext(time, FrozenTable(combined), quote_availability=p.quote_availability)
    # A global max cutoff cannot authorize later evidence on the first ticket.
    bad = combined.copy()
    records = json.loads(bad.quote_legs.iloc[0])
    records[0]['assumed_available_at'] = time.isoformat()
    bad.loc[bad.index[0], 'quote_legs'] = json.dumps(records)
    with pytest.raises(ValueError):
        DecisionContext(time, FrozenTable(bad), quote_availability=p.quote_availability)


@pytest.mark.parametrize('level', ['full', 'summary'])
def test_model_time_batch_preserves_per_ticket_string_formats(level):
    from test_composition_review_staking import gate_offers, gate
    from xdiyo_analytics.evaluation.ticket_allocation import finalize_tickets
    data = gate_offers()
    second = data.copy()
    second['event_id'] += 10
    second['row_position'] += 10
    second['round'] += 1
    for field in second:
        if isinstance(second[field].dtype, pd.DatetimeTZDtype):
            second[field] += pd.Timedelta(days=7, microseconds=123000)
    p = replace(gate(), audit_level=level)
    tickets, members, _ = compose_bets(pd.concat([data, second], ignore_index=True), p)
    tickets, members = no_attrs(tickets), no_attrs(members)
    expected = finalize_tickets(tickets.copy(), members.copy(), {'tickets': p}, ('event_id',), None, None, None)
    for model in ('a', 'b'):
        for field in ('issued_at', 'trained_through', 'artifact_vintage'):
            key = f'{model}::{field}'
            members[key] = members[key].map(lambda value: value.isoformat())
    actual = finalize_tickets(tickets.copy(), members, {'tickets': p}, ('event_id',), None, None, None)
    pd.testing.assert_frame_equal(actual[0], expected[0], check_exact=True)
    if level == 'full': assert actual[0].attrs == expected[0].attrs


def test_direct_model_batch_does_not_parse_unreferenced_members():
    from test_composition_review_staking import gate_offers, gate
    from xdiyo_analytics.evaluation.ticket_allocation import finalize_tickets
    p = gate()
    t, m, _ = compose_bets(gate_offers(), p)
    t, m = no_attrs(t), no_attrs(m)
    expected = finalize_tickets(t.copy(), m.copy(), {'tickets': p}, ('event_id',), None, None, None)[0]
    orphan = m.iloc[:1].assign(ticket_id='unreferenced', **{'a::probability':'unconsumed', 'a::issued_at':'unconsumed'})
    extended = pd.concat([m, orphan], ignore_index=True)
    actual = finalize_tickets(t.copy(), extended, {'tickets': p}, ('event_id',), None, None, None)[0]
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.attrs == expected.attrs


def test_quote_batch_preserves_legacy_column_oriented_json():
    from xdiyo_analytics.evaluation.ticket_allocation import ticket_batch
    p = policy()
    t, m, _ = compose_bets(research_legs(), p)
    batch = ticket_batch(t, m, {'tickets': p}, ('event_id',))
    records = pd.DataFrame(json.loads(batch.quote_legs.iloc[0]))
    batch['quote_legs'] = json.dumps(records.to_dict('list'))
    DecisionContext(batch.decision_at.max(), FrozenTable(batch), quote_availability=p.quote_availability)
