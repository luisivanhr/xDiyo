"""Snapshot indexing must preserve source order, exact IDs and copy semantics."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from test_temporal_graph import sample, config
from test_graph_speed import reference, assert_exact
from xdiyo_analytics.graphs import TemporalEventStore


def compare_snapshot(actual, expected):
    pd.testing.assert_frame_equal(actual.edges, expected.edges, check_exact=True)
    assert actual.edges.attrs == expected.edges.attrs
    for name in ('issuance', 'cutoff', 'teams', 'receiver', 'sender', 'log_weight',
                 'support', 'gate', 'league', 'entry', 'store_version'):
        assert_exact(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize('dtype,offset', [
    ('int64', 0), ('Int64', 2**53), ('UInt64', 2**63),
    ('uint64[pyarrow]', 2**63), ('object', 2**70)])
def test_snapshot_exact_id_codes_order_memberships_and_metadata(dtype, offset):
    base = sample()
    h, m, s = base.history, base.team_seasons, base.schedule
    # Stable source order matters to floating point support accumulation.
    h.loc[h.event_id.isin([2, 3]), 'kickoff_at'] = pd.Timestamp('2020-01-04T00:00Z')
    h = h.sample(frac=1, random_state=38).reset_index(drop=True)
    h['result_available_at'] = pd.Series(pd.NaT, index=h.index, dtype='datetime64[ns, UTC]')
    h.loc[h.event_id.eq(2), 'result_available_at'] = pd.Timestamp('2020-01-15T00:00Z')
    h.loc[h.event_id.eq(4), 'status'] = 'notstarted'
    # Entries with no intervening matches, plus an entry exactly at kickoff.
    additions = []
    for season, date, league in [(901, '2020-01-04T00:00Z', 11),
                                (902, '2020-02-01T01:00Z', 12),
                                (903, '2020-02-01T02:00Z', 13),
                                (904, '2020-05-01T00:00Z', 14)]:
        row = m.iloc[0].to_dict()
        row.update(season_id=season, entry_at=pd.Timestamp(date), competition_id=league,
                   movement='promoted')
        additions.append(row)
    m = pd.concat([m, pd.DataFrame(additions)], ignore_index=True)
    for frame, columns in ((h, ('team_id','opponent_id','competition_id','season_id','event_id')),
                           (m, ('team_id','competition_id','season_id')),
                           (s, ('home_id','away_id','competition_id','season_id','event_id'))):
        for col in columns:
            frame[col] = pd.Series([int(x)+offset for x in frame[col]], dtype=dtype)
    h.attrs = {'lineage': {'columns': ['corners', 'goals'], 'sources': {'revision': 7}}}
    native = TemporalEventStore(h, m, schedule=s)
    old = reference()['store'].TemporalEventStore(h, m, schedule=s)
    for time in ('2019-01-01T00:00Z', '2020-01-15T00:00Z',
                 '2020-02-01T01:00Z', '2020-02-01T02:00Z', '2020-04-01T05:00+09:00'):
        for teams in ((), (offset+3, offset+999)):
            actual, expected = native.snapshot(time, config(), teams), old.snapshot(time, config(), teams)
            compare_snapshot(actual, expected)
            cols = ['team_id', 'opponent_id', 'kickoff_at']
            slim = native.snapshot(time, config(), teams, _columns=cols)
            pd.testing.assert_frame_equal(slim.edges, expected.edges[cols], check_exact=True)
            assert slim.edges.attrs == expected.edges.attrs
    # Callers may mutate returned metadata or values without changing the store.
    original_attrs = deepcopy(native.history.attrs)
    actual.edges.attrs['lineage']['sources']['revision'] = 999
    actual.edges.iloc[0, actual.edges.columns.get_loc('goals_for')] = 999
    assert native.history.attrs == original_attrs
    assert native.history.goals_for.max() != 999
    compare_snapshot(native.snapshot(time, config(), teams), old.snapshot(time, config(), teams))


def test_snapshot_cutoff_nanoseconds_missing_status_and_empty_source():
    base = sample()
    h = base.history
    cutoff = pd.Timestamp('2020-03-28T23:00Z')  # Europe/Berlin DST boundary
    for event, kick in [(1, cutoff-pd.Timedelta(1, 'ns')), (2, cutoff),
                        (3, cutoff-pd.Timedelta(days=1)), (4, cutoff-pd.Timedelta(days=2))]:
        h.loc[h.event_id.eq(event), 'kickoff_at'] = kick
    h['result_available_at'] = pd.Series(pd.NaT, index=h.index, dtype='datetime64[ns, UTC]')
    h.loc[h.event_id.eq(3), 'result_available_at'] = cutoff + pd.Timedelta(1, 'ns')
    h.loc[h.event_id.eq(4), 'result_available_at'] = cutoff
    h['status'] = h.status.astype('string')
    h.loc[h.event_id.eq(5), 'status'] = pd.NA
    for source in (h, h.iloc[:0]):
        native = TemporalEventStore(source, base.team_seasons, schedule=base.schedule)
        old = reference()['store'].TemporalEventStore(source, base.team_seasons, schedule=base.schedule)
        for teams in ((), (3, 999)):
            compare_snapshot(native.snapshot('2020-03-29T16:00Z', config(), teams),
                             old.snapshot('2020-03-29T16:00Z', config(), teams))
