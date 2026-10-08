"""Consolidated A1-A4: independent public-boundary counterexamples."""
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from test_quote_availability import research_legs, policy
from xdiyo_analytics.evaluation import compose_bets, DecisionContext, FrozenTable
from xdiyo_analytics.evaluation.quote_availability import QUOTE_IDENTITY, ASSUMPTION_FIELDS, _QuoteLegEvidence
from xdiyo_analytics.evaluation.ticket_allocation import ticket_batch, finalize_tickets, _index_members
from xdiyo_analytics.evaluation.timestamps import timestamps

@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('field', list(dict.fromkeys((*QUOTE_IDENTITY, *ASSUMPTION_FIELDS, 'fixture_identity'))))
def test_each_nested_leg_requires_its_own_schema(compact, field):
    p = replace(policy(), audit_level='summary' if compact else 'full')
    t, m, _ = compose_bets(research_legs(), p)
    b = ticket_batch(t, m, {'tickets': p}, ('event_id',))
    other = b.copy(); other.index = ['other']
    b = pd.concat([b, other])
    evidence = b.quote_legs.iloc[0]
    records = evidence.dictionaries() if compact else json.loads(evidence)
    records[0].pop(field)
    b.at[b.index[0], 'quote_legs'] = (_QuoteLegEvidence(tuple(tuple(r.items()) for r in records)) if compact else json.dumps(records))
    with pytest.raises(ValueError):
        DecisionContext('2025-01-01', FrozenTable(b), quote_availability=p.quote_availability)

@pytest.mark.parametrize('label', ['unknown', '', None])
def test_consumed_labels_cannot_hide_quote_identity(label):
    p = policy(); t, m, _ = compose_bets(research_legs(), p)
    t.attrs.clear(); m.attrs.clear(); m['template'] = label
    m.loc[m.index[0], 'a::quote_id'] = 'forged'
    for fn in [lambda: ticket_batch(t, m, {'tickets': p}, ('event_id',)),
               lambda: finalize_tickets(t, m, {'tickets': p}, ('event_id',), None, None, None)]:
        with pytest.raises(ValueError, match='template'):
            fn()

def test_direct_helper_does_not_require_template_or_consume_orphans():
    p = policy(); t, m, _ = compose_bets(research_legs(), p)
    t.attrs.clear(); m.attrs.clear()
    orphan = m.iloc[:1].assign(ticket_id='orphan', **{'a::quote_id': 'unused'})
    m = pd.concat([m, orphan], ignore_index=True).drop(columns='template')
    out, _ = finalize_tickets(t, m, {'tickets': p}, ('event_id',), None, None, None)
    assert len(out) == 1

@pytest.mark.parametrize('bad', ['', 'not-a-date', 123, object()])
def test_strict_timestamp_parser_never_coerces_bad_required_values(bad):
    with pytest.raises((ValueError, TypeError)):
        timestamps(pd.Series(['2024-12-01T00:00:00Z', bad], dtype=object))

def test_inventory_mixed_formats_unknown_and_unsupported_values():
    values = pd.Series(['2024-12-01T00:00:00+00:00', '2024-12-02T00:00:00.123000+00:00', None, '', 123], dtype=object)
    parsed = timestamps(values, errors='coerce')
    assert parsed.max() == pd.Timestamp('2024-12-02T00:00:00.123000Z')
    assert int((parsed.isna() & values.notna()).sum()) == 2
    assert int(values.isna().sum()) == 1
    assert timestamps(pd.Series([], dtype=object)).empty
    assert timestamps(pd.Series([None, pd.NaT])).isna().all()
    assert timestamps(pd.Series(['2024-12-02T09:00:00+09:00', '2024-12-02'])).nunique() == 1

def test_unknown_observed_time_cannot_support_outer_observed_time():
    p = policy(); t, m, _ = compose_bets(research_legs(), p)
    b = ticket_batch(t, m, {'tickets': p}, ('event_id',))
    DecisionContext('2025-01-01', FrozenTable(b), quote_availability=p.quote_availability)
    b['quote_at'] = pd.Timestamp('2024-12-31', tz='UTC')
    with pytest.raises(ValueError, match='quote_at'):
        DecisionContext('2025-01-01', FrozenTable(b), quote_availability=p.quote_availability)

@pytest.mark.parametrize('keys', [['x','y','x'], [2**63+1, 2**63+2, 2**63+1], pd.Categorical(['x','y','x'])])
def test_index_contains_uses_positions_only(monkeypatch, keys):
    index = _index_members(pd.DataFrame({'ticket_id': keys}, index=[7,7,2]))
    monkeypatch.setattr(type(index), '__getitem__', lambda *a: pytest.fail('discarded slice'))
    assert keys[0] in index
    assert 'absent' not in index
