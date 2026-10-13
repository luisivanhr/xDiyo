"""Fixture preparation retains the frozen scalar rest-day arithmetic."""
import numpy as np
import pandas as pd
import pytest

from test_temporal_graph import sample, config, context
from test_graph_speed import reference
from test_graph_preparation_memory import assert_exact
from xdiyo_analytics.graphs import TemporalEventStore
from xdiyo_analytics.graphs.streaming import GroupPlan


@pytest.mark.parametrize('issuance', [
    '2019-12-01T12:00:00.000000999Z',
    '2020-01-08T12:00:00.000000001Z',
    '2020-03-01T12:00:00.000000999Z',
])
@pytest.mark.parametrize('cold_team', [False, True])
def test_raw_rest_nanoseconds_eligible_rows_and_large_ids(issuance, cold_team):
    original = sample()
    h, m = original.history, original.team_seasons
    h['kickoff_at'] += pd.Timedelta(nanoseconds=999)
    h['result_available_at'] = h.kickoff_at + pd.Timedelta(hours=3)
    h.loc[h.event_id.eq(2), 'result_available_at'] = pd.Timestamp('2020-02-01T00:00Z')
    h.loc[h.event_id.eq(3), 'status'] = 'scheduled'
    for frame, columns in ((h, ('team_id', 'opponent_id')), (m, ('team_id',))):
        for column in columns:
            frame[column] = pd.array([int(x)+2**63 for x in frame[column]], dtype='UInt64')
    store = TemporalEventStore(h, m)
    ctx = context(store, [4, 1])
    ctx.metadata['neutral_venue'] = [0., np.nan]
    if cold_team:
        ctx.metadata['home_id'] = pd.array([2**63+999, 2**63+998], dtype='UInt64')
    issue = pd.Series(pd.Timestamp(issuance), index=ctx.metadata.index)
    native = GroupPlan(store, config(), ctx, issue)
    old = reference()['streaming'].GroupPlan(store, config(), ctx, issue)
    a, b = native.raw(0), old.raw(0)
    for position in (0, 2, 3, 5):
        assert_exact(a[position], b[position])
    pd.testing.assert_frame_equal(a[1].edges, b[1].edges, check_exact=True)
    pd.testing.assert_frame_equal(a[4], b[4], check_exact=True)
    for key in ('teams', 'receiver', 'sender', 'support', 'gate', 'league', 'entry', 'cutoff'):
        assert_exact(getattr(a[1], key), getattr(b[1], key))
    assert_exact(native.fitting_inputs(), old.fitting_inputs())


def test_raw_metadata_attrs_preserved_and_isolated():
    store = sample()
    ctx = context(store, [1, 2, 3])
    ctx.metadata.attrs = {'source': {'items': ['original']}, 'stat_columns': ['a', 'b']}
    plan = GroupPlan(store, config(), ctx, store.issue_times(ctx.metadata, config()))
    expected = reference()['streaming'].GroupPlan(
        store, config(), ctx, store.issue_times(ctx.metadata, config())).fitting_inputs()
    assert_exact(plan.fitting_inputs(), expected)
    returned = plan.raw(0)[4]
    assert returned.attrs == ctx.metadata.attrs == plan.metadata.attrs
    returned.attrs['source']['items'].append('changed')
    assert plan.metadata.attrs['source']['items'] == ['original']
    assert ctx.metadata.attrs['source']['items'] == ['original']
    from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor
    old_pre = reference()['preprocessing'].GraphPreprocessor(config())
    pre = GraphPreprocessor(config())
    numeric, leagues = plan.fitting_inputs()
    for obj in (old_pre, pre):
        obj.fit(store.history, (), ctx.metadata, numeric, extra_leagues=leagues)
    assert_exact(pre.fixture(returned, numeric[:len(returned)]),
                 old_pre.fixture(returned, numeric[:len(returned)]))
    assert returned.attrs['source']['items'] == ['original', 'changed']
