"""39aba81 audit: dtype-independent evidence and composite dispatch regressions."""
from dataclasses import replace
import json
import runpy
from pathlib import Path
from math import fsum
import numpy as np
import pandas as pd
import pytest
from test_guarded_bulk import population, sync, both, equal_outputs
from test_quote_availability import policy
from xdiyo_analytics.evaluation import compose_bets, bulk_tickets
from xdiyo_analytics.evaluation.timestamps import timestamps
from xdiyo_analytics.evaluation.ticket_allocation import ticket_batch, finalize_tickets


@pytest.mark.parametrize('dtype', ['object', 'string', 'string[pyarrow]', 'category'])
@pytest.mark.parametrize('bad', ['', ' ', '\t'])
@pytest.mark.parametrize('case', ['contract', 'undersized', 'rejected', 'selected', 'helper'])
@pytest.mark.parametrize('engine', ['auto', 'reference'])
def test_empty_timestamp_rejected_before_filtering(dtype, bad, case, engine, monkeypatch):
    if engine == 'reference':
        monkeypatch.setattr(bulk_tickets, 'supports_bulk', lambda *a: False)
    data = population(1 if case == 'undersized' else 2)
    p = policy()
    if case == 'helper':
        data['a::probability'] = data['b::probability'] = 1.
        t, data, _ = compose_bets(data, p)
    data['quote_at'] = pd.Series([bad] * len(data), index=data.index, dtype=dtype)
    sync(data, p.quote_availability)
    if case == 'rejected':
        data['take'] = False
    with pytest.raises(ValueError, match='Timestamps require'):
        if case == 'contract':
            p.quote_availability.validate(data, data.decision_at.max(), complete=True)
        elif case == 'helper':
            ticket_batch(t, data, {'tickets': p}, ('event_id',))
        else:
            compose_bets(data, p)


@pytest.mark.parametrize('dtype', ['object', 'string', 'string[pyarrow]', 'Float64', 'Int64',
                                  'datetime64[ns]', 'datetime64[ns, UTC]'])
def test_explicit_null_timestamps_remain_missing(dtype):
    values = pd.Series([None, None], dtype=dtype)
    assert timestamps(values).isna().all()
    data = population(2)
    data['quote_at'] = values
    sync(data, policy().quote_availability)
    compose_bets(data, policy())
    data['a::issued_at'] = values
    with pytest.raises(ValueError):
        compose_bets(data, policy())


@pytest.mark.parametrize('dtype', ['Int64', 'Float64', 'int64[pyarrow]', 'boolean'])
def test_numeric_extension_epochs_reject_or_count_as_invalid(dtype):
    values = pd.Series([1, None], dtype=dtype)
    with pytest.raises(ValueError):
        timestamps(values)
    assert timestamps(values, errors='coerce').isna().all()


@pytest.mark.parametrize('dtype', ['object', 'string', 'string[pyarrow]', 'category'])
def test_mixed_iso_precision_and_inventory_counts(dtype):
    values = pd.Series(['2024-12-01T00:00:00Z', '2024-12-02T09:00:00.123+09:00', None, ''], dtype=dtype)
    parsed = timestamps(values, errors='coerce')
    assert parsed.max() == pd.Timestamp('2024-12-02T00:00:00.123Z')
    assert (parsed.isna() & values.notna()).sum() == 1
    assert values.isna().sum() == 1


ID = ('source_league', 'source_season', 'competition_id', 'season_id', 'event_id')


def composite_population():
    data = population(6)
    data['source_league'] = ['League A'] * 3 + ['League B'] * 3
    data['source_season'] = '24_25'
    data['competition_id'] = [1] * 3 + [2] * 3
    data['event_id'] = np.array([2**64 - 3, 2**64 - 2, 2**64 - 1] * 2, dtype='uint64')
    sync(data, policy().quote_availability)
    return data


@pytest.mark.parametrize('case', ['base', 'labels', 'multiindex', 'duplicate', 'folds', 'permuted',
                                  'column_order', 'nullable', 'empty', 'undersized', 'rejected', 'cutoffs'])
def test_composite_dispatch_exact_parity(case, monkeypatch):
    data = composite_population()
    if case == 'labels': data.index = ['same', 'same', 'x', 'x', 'x', 'other']
    if case == 'multiindex': data.index = pd.MultiIndex.from_frame(data[list(ID)])
    if case == 'duplicate': data = pd.concat([data, data.iloc[[0]]])
    if case == 'folds': data = pd.concat([data, data.assign(fold_id=1)])
    if case == 'permuted': data = data.iloc[[5, 2, 1, 4, 0, 3]]
    if case == 'column_order': data = data[list(reversed(data.columns))]
    if case == 'nullable':
        data['source_league'] = data.source_league.astype('string[pyarrow]')
        data['event_id'] = data.event_id.astype('UInt64')
    if case == 'empty': data = data.iloc[:0]
    if case == 'undersized': data = data.iloc[[0, 3]]
    if case == 'rejected': data['take'] = False
    if case == 'cutoffs':
        for field in ['decision_at', 'kickoff_at']:
            data.loc[3:, field] += pd.Timedelta(days=1)
    sync(data, policy().quote_availability)
    calls = []
    original = bulk_tickets.execute_bulk
    def execute(*args):
        calls.append(args[3])
        return original(*args)
    monkeypatch.setattr(bulk_tickets, 'execute_bulk', execute)
    (t, m, metrics), _ = both(data, policy(), monkeypatch, match_columns=ID)
    assert calls == [ID]
    if len(t):
        for _, row in t.iterrows():
            evidence = json.loads(row.quote_legs)
            members = m.loc[m.ticket_id.eq(row.ticket_id)]
            assert [json.loads(e['fixture_identity']) for e in evidence] == [
                list(v) for v in members[list(ID)].itertuples(index=False, name=None)]
        assert max(m.event_id) == 2**64 - 1


@pytest.mark.parametrize('case', ['missing', 'null', 'conflicting', 'model_key', 'model_key_float', 'model_missing', 'model_permuted'])
@pytest.mark.parametrize('engine', ['auto', 'reference'])
def test_composite_invalid_evidence_rejects(case, engine, monkeypatch):
    data = composite_population()
    if engine == 'reference': monkeypatch.setattr(bulk_tickets, 'supports_bulk', lambda *a: False)
    if case == 'missing': data = data.drop(columns='source_league')
    if case == 'null': data.loc[0, 'source_league'] = None
    if case == 'conflicting': data = pd.concat([data, data.iloc[[0]].assign(**{'a::probability': .99})])
    if case == 'model_key': data.loc[0, 'a::event_id'] = 42
    if case == 'model_key_float': data['a::event_id'] = data.event_id.astype(float)
    if case == 'model_missing': data.loc[0, 'b::probability'] = np.nan
    if case == 'model_permuted':
        cols = [c for c in data if c.startswith('b::')]
        data[cols] = data[cols].iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError): compose_bets(data, policy(), match_columns=ID)


def test_composite_mixed_keys_fall_back(monkeypatch):
    data = composite_population()
    data['event_id'] = data.event_id.astype(object)
    data.loc[0, 'event_id'] = 'text-id'
    sync(data, policy().quote_availability)
    assert not bulk_tickets.supports_bulk(data, {'tickets': policy()}, ID)
    monkeypatch.setattr(bulk_tickets, 'execute_bulk', lambda *a: pytest.fail('mixed identity dispatch'))
    compose_bets(data, policy(), match_columns=ID)


def test_composite_forged_membership_and_outcome_independence(monkeypatch):
    data = composite_population()
    data['a::probability'] = data['b::probability'] = 1.
    p = policy()
    original = compose_bets(data, p, match_columns=ID)
    mutated = compose_bets(data.assign(settlement='loss'), p, match_columns=ID)
    pd.testing.assert_frame_equal(original[0][['ticket_id', 'stake', 'odds']], mutated[0][['ticket_id', 'stake', 'odds']], check_exact=True)
    assert original[0].attrs == mutated[0].attrs
    t, m, _ = original
    m = m.copy()
    m.loc[m.index[0], 'a::source_league'] = 'forged'
    with pytest.raises(ValueError): finalize_tickets(t, m, {'tickets': p}, ID, None, None, None)


EXAMPLE = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'examples/research_quote_consensus.py'))


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'mixed'])
def test_original_pmf_precision_is_validated_without_modification(dtype):
    # Actual retained XGB outputs (events 10388697, 10388702, 10388705).
    values = [[.294831246137619, .7051687240600586],
              [.269776850938797, .7302231788635254],
              [.2847467362880707, .7152532339096069]]
    pmf = pd.DataFrame(values, columns=['draw', 'non_draw'], dtype='float32')
    if dtype == 'float64':
        pmf = pd.DataFrame([[.29483125, .70516875], [.26977685, .73022315], [.28474674, .71525326]], columns=pmf.columns)
    if dtype == 'mixed': pmf['draw'] = pmf.draw.astype('float64')
    before = pmf.copy(deep=True)
    EXAMPLE['_validate_pmf'](pmf)
    pd.testing.assert_frame_equal(pmf, before, check_exact=True)
    assert all(pmf[c].to_numpy().tobytes() == before[c].to_numpy().tobytes() for c in pmf)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('bad', [[.4, .5], [np.nan, .5], [np.inf, .5], [-.01, 1.01], [.5, .500001]])
def test_malformed_pmf_still_rejected(dtype, bad):
    with pytest.raises(ValueError):
        EXAMPLE['_validate_pmf'](pd.DataFrame([bad], columns=['draw', 'non_draw'], dtype=dtype))


def test_pmf_native_roundoff_boundary_and_no_float64_relaxation():
    pmf = pd.DataFrame({'draw': np.array([.5], dtype='float32'),
                        'non_draw': np.array([np.nextafter(np.float32(.5), np.float32(1))], dtype='float32')})
    EXAMPLE['_validate_pmf'](pmf)
    with pytest.raises(ValueError): EXAMPLE['_validate_pmf'](pmf.astype('float64'))
    pmf['non_draw'] = np.nextafter(pmf.non_draw, np.float32(1))
    with pytest.raises(ValueError): EXAMPLE['_validate_pmf'](pmf)
    EXAMPLE['_validate_pmf'](pd.DataFrame([[0., 1.], [1., 0.]], columns=['draw', 'non_draw']))


@pytest.mark.parametrize('engine', ['auto', 'reference'])
def test_finite_accounting_total_cannot_publish_infinity(engine, monkeypatch):
    if engine == 'reference': monkeypatch.setattr(bulk_tickets, 'supports_bulk', lambda *a: False)
    prices = [7.39709913948858e307, 8.505824920175633e307, 2.0740072889589442e307]
    data = population(3)
    data['event_id'] = [0, 1, 2]
    data['odds'] = prices
    data['a::probability'] = data['b::probability'] = 1.
    data['settlement'] = 'win'
    sync(data, policy().quote_availability)
    t, m, metrics = compose_bets(data, replace(policy(), legs=1))
    metric = metrics.set_index('metric')
    for name in ('profit', 'payout'):
        assert metric.loc[name, 'value'] == fsum(prices)
        assert metric.loc[name, 'status'] == 'ok'
    assert metric.loc['roi', 'value'] == fsum(prices) / 3


def test_normal_metric_rounding_and_unrepresentable_total():
    from xdiyo_analytics.evaluation.tickets import _finite_accounting_sum
    values = pd.Series([1e16, 1., -1e16, 3.])
    assert _finite_accounting_sum(values) == values.sum()
    with pytest.raises(ValueError, match='overflow'):
        _finite_accounting_sum(pd.Series([1e308, 1e308]))
