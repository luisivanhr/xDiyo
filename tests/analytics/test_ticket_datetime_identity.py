"""Known datetime boxing is lossless; text and timezone changes are not."""
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from io import StringIO
import hashlib
import json
import pickle
import pandas as pd
import pytest
from test_guarded_bulk import sync
from test_guarded_repairs import composite_population, ID
from test_guarded_boundaries import invoke
from test_quote_availability import policy
from xdiyo_analytics.evaluation import compose_bets, bulk_tickets
from xdiyo_analytics.evaluation import ticket_identity as identity
from xdiyo_analytics.evaluation.stake_policy import FixedStake, StakeContext


@pytest.mark.parametrize('zone', [None, timezone.utc])
@pytest.mark.parametrize('floating', ['none', 'group', 'key'])
@pytest.mark.parametrize('level', ['full', 'summary'])
@pytest.mark.parametrize('engine', ['auto', 'reference'])
def test_datetime_boxing_and_retained_helpers(zone, floating, level, engine, monkeypatch):
    data = composite_population().iloc[:2].copy()
    data['event_id'] = pd.Series([datetime(2020, 1, day, 3, 4, 5, 123456, tzinfo=zone)
                                 for day in (1, 2)], dtype=object)
    if floating != 'none':
        data['round' if floating == 'group' else 'source_season'] = pd.Series([1.1, 1.1], dtype='Float32')
    data['a::probability'] = data['b::probability'] = 1.
    p = replace(policy(), audit_level=level)
    sync(data, p.quote_availability)
    before = data.copy(deep=True)
    calls = []
    original = bulk_tickets.execute_bulk
    def counted(*a, **kw): calls.append(1); return original(*a, **kw)
    monkeypatch.setattr(bulk_tickets, 'execute_bulk', counted)
    if engine == 'reference': monkeypatch.setattr(bulk_tickets, 'supports_bulk', lambda *a: False)
    canonical = lambda values: json.dumps([str(v) for v in values], ensure_ascii=False, separators=(',', ':'))
    group = canonical(next(iter(data[k])) for k in ['fold_id', 'competition_id', 'season_id', 'stage', 'round'])
    events = sorted(canonical(v) for v in zip(*(data[k] for k in ID)))
    expected = 'tickets:' + hashlib.sha256(json.dumps(['tickets', group, events], separators=(',', ':')).encode()).hexdigest()
    t, m, _ = compose_bets(data, p, match_columns=ID)
    assert t.ticket_id.tolist() == [expected]
    assert not calls  # Object datetime fixture keys intentionally use guarded fallback.
    assert all(type(v) is pd.Timestamp for v in m.event_id)
    outputs = [(t, m)]
    funded = compose_bets(data, p, match_columns=ID, stake_policy=FixedStake(1., 'u'),
                         stake_context=StakeContext(100., 100., 'u', data.decision_at.max()))
    outputs.append(funded[:2])
    for tickets, members in outputs:
        tickets, members = pickle.loads(pickle.dumps((tickets, members)))
        assert tickets.ticket_id.tolist() == [expected]
        for helper in ('batch', 'finalize'):
            invoke(helper, tickets.copy(), members.copy(), p)
            changes = [members.event_id + pd.Timedelta(microseconds=1),
                       members.event_id + pd.Timedelta(nanoseconds=1),
                       members.event_id.astype(str),
                       members.event_id.dt.tz_localize(timezone.utc) if zone is None
                       else members.event_id.dt.tz_localize(None)]
            if zone is not None:
                # Same instant and offset but a different explicit timezone name.
                changes += [members.event_id.dt.tz_convert(timezone(timedelta(0), 'renamed-UTC')),
                            members.event_id.dt.tz_convert(timezone(timedelta(hours=1)))]
            for changed_values in changes:
                changed = members.copy(); changed['event_id'] = changed_values
                with pytest.raises(ValueError): invoke(helper, tickets.copy(), changed, p)
    pd.testing.assert_frame_equal(data, before, check_exact=True)


def test_legacy_datetime_cells_keep_their_existing_strict_contract():
    stamp = pd.Timestamp('2020-01-01', tz='UTC')
    old = ['other', type(stamp).__module__ + '.' + type(stamp).__qualname__, str(stamp)]
    assert identity._same(old, stamp)
    assert not identity._same(old, str(stamp))
    assert not identity._same(old, stamp.to_pydatetime())


def test_unknown_objects_do_not_gain_string_aliases():
    class First:
        def __str__(self): return '2020-01-01 00:00:00'
    class Second(First): pass
    cell = identity._scalar(First())
    assert identity._same(cell, First())
    assert not identity._same(cell, Second())
    assert not identity._same(cell, str(First()))
    assert not identity._same(cell, pd.Timestamp('2020-01-01'))


@pytest.mark.parametrize('cell', [
    ['datetime', 'NaT', 'null', '0'],
    ['datetime', '2020-01-01', 'null', '0'],
    ['datetime', '2020-01-01 00:00:00', '["fake","UTC"]', '0'],
    ['datetime', '2020-01-01 00:00:00+00:00', 'null', '0'],
    ['datetime', '2020-01-01 00:00:00', 'null', '2'],
])
def test_malformed_datetime_encoding_rejects(cell):
    with pytest.raises(ValueError): identity._token(cell)


def test_fold_and_submicrosecond_identity_are_preserved():
    date = datetime(2020, 1, 1, fold=1)
    assert identity._same(identity._scalar(date), pd.Timestamp(date))
    assert not identity._same(identity._scalar(date), date.replace(fold=0))
    stamp = pd.Timestamp('2020-01-01 00:00:00.123456789')
    assert identity._token(identity._scalar(stamp)) == str(stamp)
    assert not identity._same(identity._scalar(stamp), stamp.floor('us'))


def test_csv_float32_group_assertions_need_explicit_dtype():
    data = composite_population().iloc[:2].copy()
    data['round'] = pd.Series([1.1, 1.1], dtype='Float32')
    data['a::probability'] = data['b::probability'] = 1.
    p = replace(policy(), audit_level='summary')
    t, m, _ = compose_bets(data, p, match_columns=ID)
    for precision in (None, 'round_trip'):
        plain = pd.read_csv(StringIO(t.to_csv(index=False)), float_precision=precision)
        restored = pd.read_csv(StringIO(t.to_csv(index=False)), dtype={'round': 'Float32'},
                               float_precision=precision)
        # CSV also needs the existing timestamp schema restored; pandas 3
        # correctly disallows assigning Timestamp values to inferred strings.
        for column in t.select_dtypes(include=['datetime', 'datetimetz']):
            for frame in (plain, restored):
                frame[column] = pd.to_datetime(frame[column], utc=isinstance(t[column].dtype, pd.DatetimeTZDtype))
        for helper in ('batch', 'finalize'):
            with pytest.raises(ValueError): invoke(helper, plain.copy(), m.copy(), p)
            invoke(helper, restored.copy(), m.copy(), p)
