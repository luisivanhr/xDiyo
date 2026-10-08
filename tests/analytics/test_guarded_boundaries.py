"""7714579 review: scalar inference, categorical PMFs and retained bindings."""
from dataclasses import replace
import json
import pickle
import numpy as np
import pandas as pd
import pytest
from test_guarded_bulk import both, equal_outputs, sync
from test_guarded_repairs import composite_population, ID, EXAMPLE
from test_quote_availability import policy
from xdiyo_analytics.evaluation import compose_bets
from xdiyo_analytics.evaluation.ticket_allocation import ticket_batch, finalize_tickets
from xdiyo_analytics.reporting.tickets import ticket_html


@pytest.mark.parametrize('column', ['fold_id', 'competition_id', 'season_id', 'round', 'stage'])
@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
                                  'uint32', 'uint64', 'float32', 'float64', 'Int16',
                                  'UInt64', 'int16[pyarrow]', 'category'])
def test_group_scalar_inference_matches_reference(column, dtype, monkeypatch):
    data = composite_population()
    data['stage'] = 1
    data[column] = pd.Series([1] * 3 + [2] * 3, dtype=dtype)
    sync(data, policy().quote_availability)
    a, b = both(data, replace(policy(), stage_column='stage'), monkeypatch, match_columns=ID)
    assert ticket_html(a[0], a[1], {}) == ticket_html(b[0], b[1], {})
    equal_outputs(pickle.loads(pickle.dumps(a)), b)


@pytest.mark.parametrize('identity', [('event_id',), ID])
@pytest.mark.parametrize('case', ['high_uint64', 'rejected', 'empty', 'undersized'])
def test_group_dtype_boundary_outputs(identity, case, monkeypatch):
    data = composite_population()
    data['fold_id'] = np.uint64(2**64 - 1)
    data['round'] = np.float32(1.25)
    if identity == ('event_id',):
        data['event_id'] = np.arange(len(data), dtype=np.uint64) + 2**63
        sync(data, policy().quote_availability)
    if case == 'rejected': data['take'] = False
    if case == 'empty': data = data.iloc[:0]
    if case == 'undersized': data = data.iloc[[0, 3]]
    both(data, policy(), monkeypatch, match_columns=identity)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('channels', [('draw',), ('non_draw',), ('draw', 'non_draw')])
def test_categorical_pmf_native_precision_and_adapter(dtype, channels):
    legs, pmfs, contract = EXAMPLE['synthetic_example']()
    for pmf in pmfs.values():
        for col in pmf:
            pmf[col] = pmf[col].astype(dtype)
    expected = EXAMPLE['consensus'](legs, pmfs, contract)
    for pmf in pmfs.values():
        for col in channels: pmf[col] = pmf[col].astype('category')
    before = {k: v.copy(deep=True) for k, v in pmfs.items()}
    actual = EXAMPLE['consensus'](legs, pmfs, contract)
    # The adapter may retain categorical input channels in memberships; native
    # numerical decisions, IDs, PMF values and evidence remain unchanged.
    for a, b in zip(actual, expected):
        if isinstance(a, pd.DataFrame):
            for col in a:
                if isinstance(a[col].dtype, pd.CategoricalDtype):
                    a = a.copy(); a[col] = a[col].astype(a[col].cat.categories.dtype)
            pd.testing.assert_frame_equal(a, b, check_exact=True)
            assert a.attrs == b.attrs
    for k, pmf in pmfs.items(): pd.testing.assert_frame_equal(pmf, before[k], check_exact=True)
    raw = pd.DataFrame({'draw': np.array([.5], dtype='float32'),
                        'non_draw': np.array([np.nextafter(np.float32(.5), np.float32(1))], dtype='float32')})
    EXAMPLE['_validate_pmf'](raw.astype('category'))
    with pytest.raises(ValueError): EXAMPLE['_validate_pmf'](raw.astype('float64').astype('category'))


@pytest.mark.parametrize('values', [[.4, .5], [np.nan, .5], [np.inf, .5], [-.1, 1.1], ['bad', '.5']])
def test_invalid_categorical_pmf_deliberate_error(values):
    pmf = pd.DataFrame([values], columns=['draw', 'non_draw']).astype('category')
    with pytest.raises(ValueError): EXAMPLE['_validate_pmf'](pmf)


def retained(level, explicit):
    data = composite_population().iloc[:2].copy()
    data['a::probability'] = data['b::probability'] = 1.
    if explicit:
        for model in ('a', 'b'):
            for key in ID: data[f'{model}::{key}'] = data[key]
    p = replace(policy(), audit_level=level)
    t, m, _ = compose_bets(data, p, match_columns=ID)
    return t, m, p


def invoke(helper, t, m, p):
    if helper == 'batch': return ticket_batch(t, m, {'tickets': p}, ID)
    return finalize_tickets(t, m, {'tickets': p}, ID, None, None, None)


@pytest.mark.parametrize('level', ['full', 'summary'])
@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('helper', ['batch', 'finalize'])
@pytest.mark.parametrize('key', ID)
def test_original_ticket_binds_each_fixture_key(level, explicit, helper, key):
    t, m, p = retained(level, explicit)
    m.loc[m.index[0], key] = 'forged' if key.startswith('source_') else 42
    before = t.copy(deep=True)
    with pytest.raises(ValueError): invoke(helper, t, m, p)
    pd.testing.assert_frame_equal(t, before, check_exact=True)


@pytest.mark.parametrize('helper', ['batch', 'finalize'])
@pytest.mark.parametrize('change', ['ticket_id', 'duplicate', 'group', 'quote_fixture', 'quote_price', 'quote_order', 'quote_duplicate', 'quote_malformed'])
def test_stale_identity_or_retained_nested_evidence_rejected(helper, change):
    t, m, p = retained('full', False)
    if change == 'ticket_id':
        t['ticket_id'] = 'tickets:forged'; m['ticket_id'] = 'tickets:forged'
    elif change == 'duplicate': m = pd.concat([m, m.iloc[[0]]])
    elif change == 'group': t['round'] += 1
    else:
        records = json.loads(t.quote_legs.iloc[0])
        if change == 'quote_fixture': records[0]['fixture_identity'] = '["forged"]'
        if change == 'quote_price': records[0]['odds'] = 999.
        if change == 'quote_order': records.reverse()
        if change == 'quote_duplicate': records[1] = records[0]
        t['quote_legs'] = 'null' if change == 'quote_malformed' else json.dumps(records)
    with pytest.raises(ValueError): invoke(helper, t, m, p)


@pytest.mark.parametrize('level', ['full', 'summary'])
@pytest.mark.parametrize('explicit', [False, True])
def test_valid_refinalization_and_outcome_independence(level, explicit):
    t, m, p = retained(level, explicit)
    batch = invoke('batch', t, m, p)
    new_t, new_m = invoke('finalize', t.copy(), m.copy(), p)
    pd.testing.assert_frame_equal(t, new_t, check_exact=True)
    pd.testing.assert_frame_equal(m, new_m, check_exact=True)
    altered = m.assign(settlement='loss', actual=999., profit=-99.)
    pd.testing.assert_frame_equal(batch, invoke('batch', t, altered, p), check_exact=True)
    if level == 'full':
        legacy = t.copy()
        legacy['quote_legs'] = pd.DataFrame(json.loads(t.quote_legs.iloc[0])).to_json()
        # Default pandas JSON converts None to null, matching native evidence.
        invoke('batch', legacy, m, p)


@pytest.mark.parametrize('helper', ['batch', 'finalize'])
@pytest.mark.parametrize('shift_seconds', [0, -1])
def test_retained_quote_times_compare_instants(helper, shift_seconds):
    t, m, p = retained('full', False)
    for field in ('decision_at', 'assumed_available_at'):
        values = m[field]
        if field == 'assumed_available_at': values = values + pd.Timedelta(seconds=shift_seconds)
        m[field] = values.map(lambda v: v.tz_convert('Asia/Tokyo').isoformat())
    sync(m, p.quote_availability)
    if shift_seconds:
        with pytest.raises(ValueError, match='Retained ticket quote evidence'):
            invoke(helper, t, m, p)
    else:
        invoke(helper, t, m, p)
