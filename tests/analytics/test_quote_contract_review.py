"""Independent review repairs: valid empties, complete timing and named sources."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import runpy
import pandas as pd
import pytest

from xdiyo_analytics.evaluation import compose_bets
from xdiyo_analytics.reporting.contracts import StudyResult
from xdiyo_analytics.reporting.tickets import add_tickets
from test_quote_availability import research_legs, policy, assumption
from test_composition_review_staking import gate, gate_offers


def example():
    return runpy.run_path(str(Path(__file__).resolve().parents[2] / 'examples/research_quote_consensus.py'))


@pytest.mark.parametrize('kind', ['empty', 'unselected', 'undersized'])
def test_valid_empty_native_results_keep_schemas_declarations_and_exports(kind, tmp_path):
    data = research_legs()
    if kind == 'empty': data = data.iloc[:0]
    elif kind == 'unselected': data = data.assign(take=False)
    else: data = data.iloc[:1]
    tickets, members, metrics = compose_bets(data, policy())
    assert tickets.empty and members.empty
    assert {'ticket_id', 'bet', 'stake', 'profit', 'accounting_status'} <= set(tickets)
    assert {'event_id', 'ticket_id', 'template', 'leg_number'} <= set(members)
    for table in (tickets, members, metrics):
        assert set(assumption().labels) <= set(table)
        assert table.attrs['quote_assumptions'][0]['quote_assumption_id'] == assumption().assumption_id
    result = add_tickets(StudyResult('Empty research', tables={'ledger':data}), policy(),
                         SimpleNamespace(match_columns=('event_id',)))
    assert result.tables['quote_assumptions'].quote_assumption_id.iloc[0] == assumption().assumption_id
    assert any('Research simulation only' in n for n in result.notes)
    for name in ('tickets', 'ticket_legs', 'quote_assumptions', 'ticket_candidates', 'ticket_selection_summary'):
        table = result.tables[name]
        assert set(assumption().labels) <= set(table) or 'quote_availability_declarations' in table
        path = tmp_path / f'{name}.parquet'
        table.to_parquet(path)
        restored = pd.read_parquet(path)
        pd.testing.assert_frame_equal(table, restored)
        assert table.attrs == restored.attrs


@pytest.mark.parametrize('kind', ['empty', 'all_or_rejected', 'undersized'])
def test_consensus_valid_empty_audits_every_supplied_fixture(kind):
    api = example()
    data, pmfs, contract = api['synthetic_example']()
    if kind == 'all_or_rejected':
        pmfs = {m:p.assign(draw=.01, non_draw=.99) for m,p in pmfs.items()}
    else:
        n = 0 if kind == 'empty' else 1
        data = data.iloc[:n]
        pmfs = {m:p.iloc[:n] for m,p in pmfs.items()}
    tickets, members, _, audit = api['consensus'](data, pmfs, contract)
    assert tickets.empty and members.empty
    assert len(audit) == 2 * len(data)
    assert set(audit.candidate_id) == set(data.event_id)
    assert set(contract.labels) <= set(audit)


@pytest.mark.parametrize('field', ['decision_at', 'xgb::decision_at', 'lgbm::decision_at',
                                   'xgb::issued_at', 'lgbm::artifact_vintage', 'xgb::trained_through'])
@pytest.mark.parametrize('value', [None, 'not-a-date'])
def test_ineligible_fixture_bad_time_cannot_disappear_before_or(field, value):
    api = example()
    data, pmfs, contract = api['synthetic_example']()
    data[field] = data[field].astype(object)
    data.loc[data.event_id.eq(3), field] = value  # fixture 3 would fail OR
    if field == 'decision_at':
        for model in pmfs:
            data[f'{model}::decision_at'] = data[f'{model}::decision_at'].astype(object)
            data.loc[data.event_id.eq(3), f'{model}::decision_at'] = value
    with pytest.raises(ValueError): api['consensus'](data, pmfs, contract)


@pytest.mark.parametrize('field', ['won', 'outcome', 'profit', 'payout', 'actual', 'settlement', 'model::target'])
@pytest.mark.parametrize('research', [False, True])
def test_reserved_probability_source_names_rejected_before_reading(field, research):
    p = policy() if research else gate()
    with pytest.raises(ValueError, match='Outcome-bearing'):
        replace(p, probability_columns={'a':field, 'b':'b::probability'})
    # Frozen dataclasses can still contain mutable dicts; validate again at use.
    p.probability_columns['a'] = field
    data = research_legs() if research else gate_offers()
    data[field] = 1.
    for frame in (data, data.iloc[:0]):
        with pytest.raises(ValueError, match='Outcome-bearing'):
            compose_bets(frame, p)


def test_ordinary_quote_status_metadata_remains_accepted():
    data = research_legs().assign(quote_status='available')
    for model in ('a', 'b'): data[f'{model}::quote_status'] = data.quote_status
    assert len(compose_bets(data, policy())[0]) == 1


def test_finalization_rechecks_mutated_source_mapping():
    from xdiyo_analytics.evaluation.ticket_allocation import finalize_tickets
    p = policy()
    tickets, members, _ = compose_bets(research_legs(), p)
    p.probability_columns['a'] = 'won'
    members['won'] = 1.
    with pytest.raises(ValueError, match='Outcome-bearing'):
        finalize_tickets(tickets, members, {'tickets':p}, ('event_id',), None, None, None)
