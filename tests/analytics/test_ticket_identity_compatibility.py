"""166196c review: lossless boxing and ticket-specific fixture equality."""
from dataclasses import replace
from itertools import combinations
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
import pytest
from test_guarded_bulk import both, population, sync
from test_guarded_repairs import ID, composite_population
from test_guarded_boundaries import retained, invoke
from test_quote_availability import policy
from xdiyo_analytics.evaluation import compose_bets, Parlay, MultiBet, bulk_tickets
from xdiyo_analytics.evaluation.ticket_allocation import ticket_batch, finalize_tickets
from xdiyo_analytics.evaluation.stake_policy import FixedStake, StakeContext
from xdiyo_analytics.reporting.tickets import ticket_html


@pytest.mark.parametrize('value', [1.1, 2.2, 3.3])
@pytest.mark.parametrize('column', ['fold_id', 'competition_id', 'season_id', 'round', 'stage',
                                    'event_id', 'source_league', 'source_season'])
@pytest.mark.parametrize('level', ['full', 'summary'])
def test_nullable_float32_native_identity_survives_boxing(value, column, level, monkeypatch):
    data = composite_population().iloc[:3].copy()
    data[column] = pd.Series([value, value + 1., value + 2.] if column == 'event_id' else [value]*3, dtype='Float32')
    data['a::probability'] = data['b::probability'] = 1.
    sync(data, policy().quote_availability)
    p = replace(policy(), audit_level=level)
    actual, reference = both(data, p, monkeypatch, match_columns=ID)
    canonical = lambda values: json.dumps([str(v) for v in values], ensure_ascii=False, separators=(',', ':'))
    original_group = canonical(next(iter(data[k])) for k in GROUPS)
    original_events = [canonical(v) for v in zip(*(data[k] for k in ID))]
    expected_ids = {'tickets:' + hashlib.sha256(json.dumps(
        ['tickets', original_group, sorted(pair)], separators=(',', ':')).encode()).hexdigest()
        for pair in combinations(original_events, 2)}
    assert set(actual[0].ticket_id) == expected_ids
    assert ticket_html(actual[0], actual[1], {}) == ticket_html(reference[0], reference[1], {})
    for output in (actual, reference):
        t, m, _ = pickle.loads(pickle.dumps(output))
        invoke('batch', t, m, p)
        nt, nm = invoke('finalize', t.copy(), m.copy(), p)
        pd.testing.assert_frame_equal(nt, t, check_exact=True)
        pd.testing.assert_frame_equal(nm, m, check_exact=True)
        assert nt.attrs == t.attrs
        for helper in ('batch', 'finalize'):
            changed = m.copy()
            # A different exactly representable binary32 value is still a
            # different identity, not an eligible boxing representation.
            changed[column] = changed[column] + float(np.float32(.25))
            with pytest.raises(ValueError): invoke(helper, t.copy(), changed, p)


def test_original_float32_evidence_supports_multiple_columns(monkeypatch):
    data = composite_population().iloc[:2].copy()
    for field in ['fold_id', 'competition_id', 'season_id', 'round', 'stage']:
        data[field] = pd.Series([1.1]*2, dtype='Float32')
    data['event_id'] = pd.Series([2.2, 3.3], dtype='Float32')
    data['a::probability'] = data['b::probability'] = 1.
    actual, _ = both(data, policy(), monkeypatch, match_columns=ID)
    t, m, _ = actual
    invoke('batch', t, m, policy())
    for value in [1.1, np.nextafter(float(np.float32(2.2)), np.inf)]:
        changed = m.copy(); changed.loc[changed.index[0], 'event_id'] = value
        with pytest.raises(ValueError): invoke('batch', t, changed, policy())


@pytest.mark.parametrize('kind', ['parlay', 'multi'])
@pytest.mark.parametrize('keys', [
    [1, '1', 3], [pd.Timestamp('2024-01-01'), '2024-01-01 00:00:00', 3],
    [1, 2, 3], [True, 1, 2, 3], [1, 1, 2, 3],
])
@pytest.mark.parametrize('multi_column', [False, True])
def test_generic_equality_in_allocation_and_both_helpers(kind, keys, multi_column):
    data = population(len(keys))
    data = data.drop(columns=[c for c in data if '::' in c or c.startswith(('quote_', 'assum'))])
    data['event_id'] = pd.Series(keys, dtype=object)
    data['quote_at'] = data.decision_at
    data['extra_key'] = ['same', 'other'] + ['same'] * (len(data) - 2)
    match_columns = ('extra_key', 'event_id') if multi_column else ('event_id',)
    p = Parlay(size=2) if kind == 'parlay' else MultiBet(size=3, sizes=(2, 3))
    t, m, _ = compose_bets(data, p, match_columns=match_columns)
    unique = len(data.drop_duplicates(list(match_columns)))
    assert len(t) == (unique // 2 if kind == 'parlay' else 4 * (unique // 3))
    funded, _, _ = compose_bets(data, p, match_columns=match_columns, stake_policy=FixedStake(1., 'u'),
                               stake_context=StakeContext(100., 100., 'u', data.decision_at.max()))
    assert funded.ticket_id.tolist() == t.ticket_id.tolist()
    for helper in ('batch', 'finalize'):
        def call(members):
            if helper == 'batch': return ticket_batch(t, members, {'tickets': p}, match_columns)
            return finalize_tickets(t.copy(), members, {'tickets': p}, match_columns, None, None, None)
        call(m)
        duplicate = m.copy()
        positions = np.flatnonzero(m.ticket_id.eq(t.ticket_id.iloc[0]).to_numpy())
        for field in match_columns:
            duplicate.iloc[positions[1], duplicate.columns.get_loc(field)] = duplicate.iloc[positions[0]][field]
        with pytest.raises(ValueError, match='duplicate fixtures'): call(duplicate)


GROUPS = ['fold_id', 'competition_id', 'season_id', 'stage', 'round']


@pytest.mark.parametrize('omitted', [None, *GROUPS])
@pytest.mark.parametrize('level', ['full', 'summary'])
@pytest.mark.parametrize('helper', ['batch', 'finalize'])
def test_each_available_group_assertion_checked(omitted, level, helper):
    t, m, p = retained(level, False)
    if omitted is not None: t = t.drop(columns=omitted)
    invoke(helper, t.copy(), m, p)
    invoke(helper, t.drop(columns=[k for k in GROUPS if k in t]), m, p)
    for field in GROUPS:
        if field not in t: continue
        changed = t.copy()
        changed[field] = 'wrong-stage' if field == 'stage' else 987
        with pytest.raises(ValueError, match='Ticket grouping differs'):
            invoke(helper, changed, m, p)
