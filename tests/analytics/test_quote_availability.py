"""Explicit research quotes and exact two-stage consensus, using native decisions."""
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.evaluation import (
    QuoteAvailability, DecisionContext, DecisionLayer, FrozenTable, AllCombinations, compose_bets,
)
from xdiyo_analytics.evaluation.quote_availability import ASSUMPTION_FIELDS, QUOTE_IDENTITY
from xdiyo_analytics.reporting.tickets import ticket_html
from test_composition_review_staking import gate_offers, gate


def assumption():
    return QuoteAvailability('research_assumed', 'opening-round-minus-1h-v1',
        'Unverified opening availability by league/season/stage/round first kickoff minus one hour.',
        'research-protocol/quote-assumption-v1')


def research_legs():
    f = gate_offers().assign(market='1x2', selection='draw', quote_at=pd.NaT,
        quote_snapshot_hash='snapshot-pin', quote_crosswalk_hash='crosswalk-pin',
        assumed_available_at=pd.Timestamp('2025-01-01', tz='UTC'),
        quote_timing_certainty='vendor_label_only_unknown_timestamp',
        quote_vendor_scheduled_at='2025-01-02 12:00', line=np.nan)
    f.loc[1, 'odds'] = 3.  # Different fixtures keep their own distinct prices.
    f = assumption().annotate(f)
    for model in ('a', 'b'):
        for field in assumption().identities(f):
            f[f'{model}::{field}'] = f[field]
    return f


def policy(**kwargs):
    return replace(gate(**kwargs), quote_availability=assumption())


def candidate_inputs(f):
    fields = list(assumption().identities(f))
    c = f[fields].copy()
    c.index = pd.Index(f.event_id, name='fixture')
    c['economic_key'] = [f'fixture-{k}' for k in c.index]
    c['payoff'] = 'binary'
    streams = {}
    for model in ('a', 'b'):
        v = c.copy()
        for field in fields:
            v[field] = f[f'{model}::{field}'].to_numpy()
        for field in ('probability', 'issued_at', 'trained_through', 'artifact_vintage'):
            v[field] = f[f'{model}::{field}'].to_numpy()
        streams[model] = v
    return c, streams


def test_opt_in_preserves_unknown_observed_times_and_reports_assumption():
    f = research_legs()
    before = f.copy(deep=True)
    tickets, members, metrics = compose_bets(f, policy())
    assert len(tickets) == 1 and tickets.odds.iloc[0] == 6.
    assert tickets.stake.eq(1.).all()
    assert tickets.quote_at.isna().all() and members.quote_at.isna().all()
    assert json.loads(tickets.quote_legs.iloc[0])[1]['odds'] == 3.
    assert all(r['quote_at'] is None for r in json.loads(tickets.quote_legs.iloc[0]))
    for frame in (tickets, members, metrics):
        assert frame.quote_availability_mode.eq('research_assumed').all()
        assert frame.quote_assumption_id.eq(assumption().assumption_id).all()
    for name in ('ticket_candidates', 'ticket_selection_summary', 'decision_policy_audit'):
        assert all(r['quote_availability_mode'] == 'research_assumed' for r in tickets.attrs[name])
    html = ticket_html(tickets, members, {})
    assert 'Research simulation only' in html and assumption().reference in html
    pd.testing.assert_frame_equal(f, before)
    with pytest.raises(ValueError):
        compose_bets(f, gate())


@pytest.mark.parametrize('change', ['unknown', 'future', 'assumed', 'mixed'])
def test_observed_default_remains_strict(change):
    c = pd.DataFrame(dict(quote_at=['2025-01-01'], decision_at=['2025-01-02']))
    if change == 'unknown': c['quote_at'] = pd.NaT
    if change == 'future': c['quote_at'] = '2025-01-03'
    if change == 'assumed': c['assumed_available_at'] = '2025-01-01'
    if change == 'mixed': c['quote_availability_mode'] = 'research_assumed'
    with pytest.raises(ValueError):
        DecisionContext('2025-01-02', FrozenTable(c))


@pytest.mark.parametrize('field,value', [
    ('quote_at', pd.Timestamp('2025-01-02', tz='UTC')),
    ('assumed_available_at', pd.Timestamp('2025-01-02', tz='UTC')),
    ('assumed_available_at', '2025-01-01'),
    ('quote_availability_mode', 'observed'),
    ('quote_assumption_id', 'different'),
    ('quote_snapshot_hash', ''),
    ('a::quote_id', 'different'),
    ('a::odds', 2.5),
    ('b::selection', 'home'),
    ('b::market', 'corners'),
    ('b::quote_snapshot_hash', 'other'),
    ('b::quote_crosswalk_hash', 'other'),
    ('b::quote_availability_mode', 'observed'),
    ('a::assumed_available_at', pd.Timestamp('2025-01-02', tz='UTC')),
    ('a::quote_at', pd.Timestamp('2025-01-02', tz='UTC')),
    ('a::decision_at', pd.Timestamp('2024-12-31', tz='UTC')),
    ('a::quote_assumption_reference', 'other'),
    ('a::probability', np.nan),
    ('b::probability', np.nan),
    ('a::issued_at', pd.Timestamp('2025-01-02', tz='UTC')),
])
@pytest.mark.parametrize('undersized', [False, True])
def test_each_leg_fails_closed_before_aggregation(field, value, undersized):
    f = research_legs()
    f[field] = f[field].astype(object)
    f.loc[0, field] = value
    with pytest.raises((ValueError, TypeError)):
        compose_bets(f.iloc[:1] if undersized else f, policy())


def test_observed_evidence_cannot_be_overridden_even_before_decision():
    f = research_legs()
    f['assumed_available_at'] = pd.Timestamp('2024-12-30', tz='UTC')
    f['quote_at'] = pd.Timestamp('2024-12-31', tz='UTC')
    with pytest.raises(ValueError, match='contradictory'):
        compose_bets(f, policy())


def test_assumptions_require_attestation_and_missing_error():
    for field in ('assumption_id', 'rationale', 'reference'):
        with pytest.raises(ValueError): replace(assumption(), **{field:' '})
    with pytest.raises(ValueError): policy(gate='or', missing='reject')
    with pytest.raises(ValueError): QuoteAvailability('observed', assumption_id='x')


@pytest.mark.parametrize('a', [np.nextafter(.8, 0), .8, np.nextafter(.8, 1)])
@pytest.mark.parametrize('b', [np.nextafter(.8, 0), .8, np.nextafter(.8, 1)])
def test_or_exact_complement_boundary_and_truth_table(a, b):
    c, streams = candidate_inputs(research_legs())
    streams['a']['probability'] = 1. - a
    streams['b']['probability'] = 1. - b
    decision = DecisionLayer(('a', 'b'), gate='or', metric='probability', threshold=1. - .8, strict=False)
    result = decision.decide(streams, DecisionContext('2025-01-01', FrozenTable(c), quote_availability=assumption()))
    assert len(result.selected) == (len(c) if a <= .8 or b <= .8 else 0)


@pytest.mark.parametrize('missing', ['a', 'b'])
def test_or_requires_both_streams(missing):
    c, streams = candidate_inputs(research_legs())
    streams.pop(missing)
    decision = DecisionLayer(('a', 'b'), gate='or', metric='probability', threshold=1. - .8, strict=False)
    with pytest.raises(ValueError):
        decision.decide(streams, DecisionContext('2025-01-01', FrozenTable(c), quote_availability=assumption()))


def test_zero_ev_rejected_for_either_original_model():
    f = research_legs()
    # Odds 2*3; exact p product 1/6 yields EV=0, despite other model approval.
    f['a::probability'] = [1., 1./6.]
    tickets, _, _ = compose_bets(f, policy())
    assert tickets.empty
    a = [r for r in tickets.attrs['decision_policy_audit'] if r['model'] == 'a'][0]
    assert a['value'] == 0 and not a['take']
    assert 'Research simulation only' in ticket_html(tickets, pd.DataFrame(), {})


def test_model_order_fixture_order_and_outcomes_cannot_change_decisions():
    f = research_legs()
    a = compose_bets(f, policy())
    mutated = f.iloc[::-1].assign(settlement='loss', profit=-1.e9, payout=0, actual=999)
    p = replace(policy(), ticket_gate=DecisionLayer(('b', 'a')),
                probability_columns={'b':'b::probability', 'a':'a::probability'})
    b = compose_bets(mutated, p)
    cols = ['ticket_id', 'odds', 'stake', 'quote_legs', 'quote_at', *ASSUMPTION_FIELDS]
    pd.testing.assert_frame_equal(a[0][cols], b[0][cols])
    assert a[0].profit.iloc[0] != b[0].profit.iloc[0]


def test_conflicting_duplicate_quote_and_nested_late_evidence_rejected():
    f = research_legs()
    duplicate = f.iloc[:1].copy().assign(quote_id='conflict')
    with pytest.raises(ValueError): compose_bets(pd.concat([f, duplicate], ignore_index=True), policy())
    tickets, members, _ = compose_bets(f, policy())
    from xdiyo_analytics.evaluation.ticket_allocation import ticket_batch
    c = ticket_batch(tickets, members, {'tickets':policy()}, ('event_id',))
    evidence = json.loads(c.quote_legs.iloc[0])
    evidence[0]['quote_at'] = '2025-01-02T00:00:00Z'
    c['quote_legs'] = json.dumps(evidence)
    with pytest.raises(ValueError):
        DecisionContext('2025-01-01', FrozenTable(c), quote_availability=assumption())


def test_native_recipe_catalog_roundtrip():
    from xdiyo_analytics.ui.recipe import catalog_for_ui, node
    catalog = catalog_for_ui()
    spec = node('evaluation.AllCombinations', stage_column='stage', probability_mode='independent', payoff='binary',
        probability_columns={'a':'a::probability', 'b':'b::probability'},
        ticket_gate=node('evaluation.DecisionLayer', models=['a', 'b']),
        quote_availability=node('evaluation.QuoteAvailability', **assumption().__dict__))
    restored = catalog.build(json.loads(json.dumps(spec)))
    assert restored.quote_availability == assumption()
    assert catalog.build(json.loads(json.dumps(catalog.encode(restored)))).quote_availability == assumption()
    assert len(compose_bets(research_legs(), restored)[0]) == 1


def test_explicit_observed_contract_and_mismatched_model_quote():
    f = research_legs().drop(columns=[c for c in research_legs() if c in ASSUMPTION_FIELDS or '::' in c and c.split('::', 1)[1] in ASSUMPTION_FIELDS])
    f['quote_at'] = pd.Timestamp('2024-12-31', tz='UTC')
    contract = QuoteAvailability()
    f = contract.annotate(f)
    for model in ('a', 'b'):
        for field in contract.identities(f):
            f[f'{model}::{field}'] = f[field]
    p = replace(gate(), quote_availability=contract)
    assert len(compose_bets(f, p)[0]) == 1
    f.loc[0, 'a::quote_crosswalk_hash'] = 'wrong'
    with pytest.raises(ValueError): compose_bets(f, p)


def test_nested_quote_membership_and_outcome_payload_fail_closed():
    from xdiyo_analytics.evaluation.ticket_allocation import ticket_batch
    f = research_legs()
    tickets, members, _ = compose_bets(f, policy())
    c = ticket_batch(tickets, members, {'tickets':policy()}, ('event_id',))
    for mode in ('duplicate', 'nested', 'outcome', 'price'):
        mutated = c.copy()
        evidence = json.loads(c.quote_legs.iloc[0])
        if mode == 'duplicate': evidence[1]['fixture_identity'] = evidence[0]['fixture_identity']
        if mode == 'nested': evidence[0]['quote_legs'] = '[]'
        if mode == 'outcome': evidence[0]['profit'] = 999
        if mode == 'price': evidence[0]['odds'] = 20
        mutated['quote_legs'] = json.dumps(evidence)
        with pytest.raises(ValueError):
            DecisionContext('2025-01-01', FrozenTable(mutated), quote_availability=assumption())


def test_outcome_metadata_never_reaches_decision_policy():
    c, _ = candidate_inputs(research_legs())
    for field in ('status', 'is_awarded', 'profit', 'result'):
        with pytest.raises(ValueError): FrozenTable(c.assign(**{field:'known'}))


@pytest.mark.parametrize('field', ['quote_assumption_id', 'quote_availability_mode', 'a::quote_timing_certainty', 'a::quote_assumption_reference'])
def test_nullable_string_missing_evidence_is_not_silently_accepted(field):
    f = research_legs()
    f[field] = f[field].astype('string')
    f.loc[0, field] = pd.NA
    with pytest.raises(ValueError): compose_bets(f, policy())


def test_example_uses_original_draw_pmf_and_class_labels_under_permutations():
    from pathlib import Path
    import runpy
    example = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'examples/research_quote_consensus.py'))
    consensus, synthetic_example = example['consensus'], example['synthetic_example']
    data, pmfs, contract = synthetic_example()
    expected = consensus(data, pmfs, contract)
    reversed_pmfs = {m:p.iloc[::-1, ::-1] for m,p in reversed(list(pmfs.items()))}
    actual = consensus(data.iloc[::-1], reversed_pmfs, contract)
    pd.testing.assert_frame_equal(expected[0], actual[0])
    assert set(expected[1].event_id) == {1, 2}
    audit = pd.DataFrame(expected[0].attrs['decision_policy_audit'])
    assert audit.loc[audit.model.eq('xgb'), 'probability'].iloc[0] == .20 * .30
    for missing in ('xgb', 'lgbm'):
        bad = dict(pmfs)
        bad[missing] = bad[missing].drop(index=3)
        with pytest.raises(ValueError): consensus(data, bad, contract)


@pytest.mark.parametrize('reject_all', [False, True])
def test_report_parquet_and_performance_reuse_preserve_research_disclosure(tmp_path, monkeypatch, reject_all):
    from types import SimpleNamespace
    from xdiyo_analytics.reporting import BetPerformanceReporter
    from xdiyo_analytics.reporting.contracts import StudyResult, StudyRun, AnalysisReport
    from xdiyo_analytics.reporting.tickets import add_tickets
    f = research_legs()
    p = policy(threshold=100. if reject_all else 0.)
    for column in f:
        if pd.api.types.is_datetime64_any_dtype(f[column]):
            f[column] = f[column].dt.as_unit('ns')
    context = SimpleNamespace(match_columns=('event_id',))
    source = add_tickets(StudyResult('Synthetic', tables={'ledger':f}), p, context)
    source.tables['alternatives'] = pd.DataFrame()  # required source contract, no recomputation
    restored = {}
    for name, table in source.tables.items():
        path = tmp_path / f'{name}.parquet'
        table.to_parquet(path)
        restored[name] = pd.read_parquet(path)
        expected = table.copy()
        comparable = restored[name].copy()
        # Parquet has no seconds timestamp unit. pandas 3 can infer seconds
        # for constructed report columns; Arrow stores those as milliseconds.
        # Compare exact instants in nanoseconds, including missing masks, and
        # normalize pandas 3's string inference without touching numerical types
        # or changing the reloaded report data.
        for column in table:
            if pd.api.types.is_datetime64_any_dtype(table[column]):
                expected[column] = expected[column].dt.as_unit('ns')
                comparable[column] = comparable[column].dt.as_unit('ns')
            elif table[column].dtype == object and isinstance(comparable[column].dtype, pd.StringDtype):
                comparable[column] = comparable[column].astype(object)
        pd.testing.assert_frame_equal(expected, comparable)
        assert table.attrs == restored[name].attrs
    context.previous_results = {'cached':StudyResult('Restored', tables=restored)}
    def forbidden(*args, **kwargs): raise AssertionError('Saved report must not recompose or refit.')
    monkeypatch.setattr('xdiyo_analytics.reporting.tickets.compose_bets', forbidden)
    reused = BetPerformanceReporter(type='overall', partition='test', source='cached').run(context)
    assert any('Research simulation only' in note for note in reused.notes)
    pd.testing.assert_frame_equal(reused.tables['quote_assumptions'], source.tables['quote_assumptions'])
    assert reused.tables['tickets'].quote_at.isna().all()
    report = AnalysisReport([StudyRun('Tickets', 'overall', 'test', None, 'match', np.array([], dtype=int), 0, reused)])
    html = report.to_html()
    assert 'Research simulation only' in html and assumption().reference in html
    assert 'data:text/csv' in html
