"""PR #2 regressions: scaler release bounds and cross-segment gap provenance."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')

from test_temporal_graph import sample, config, context, history
from xdiyo_analytics.graphs import (
    BoundaryMemory, GATResearchBNV1, TemporalEventStore, TemporalGraphModel,
)
from xdiyo_analytics.graphs.adapter import _Backend
from xdiyo_analytics.graphs.memory import BoundaryReplay
from xdiyo_analytics.graphs.observed_gap import encode_observed_gap
from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor, MaskedScale
from xdiyo_analytics.graphs.research_profiles import RETAINED_STATISTICS
from xdiyo_analytics.graphs.streaming import GroupPlan, PreparedGroups
from xdiyo_analytics.training import TrainingControl
from xdiyo_analytics.training.residency import InputCachePolicy

STAT = RETAINED_STATISTICS[0]


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(params=['cpu', 'cuda:0'])
def device(request):
    if request.param.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


def released_store(value=7., release='2020-02-01T00:00Z', *, metadata=True):
    original = sample()
    h = original.history
    suffixes = sorted({c.split('::', 1)[1] for c in RETAINED_STATISTICS})
    for j, suffix in enumerate(suffixes):
        h['team::'+suffix] = 2.*h.team_id + h.event_id + j
        h['opponent::'+suffix] = 2.*h.opponent_id + h.event_id + j
    if metadata:
        h['available_at::'+STAT] = h.kickoff_at + pd.Timedelta(hours=1)
        h.loc[h.event_id.eq(1), 'available_at::'+STAT] = pd.Timestamp(release)
    h.loc[h.event_id.eq(1), STAT] = value
    h.loc[h.event_id.eq(1), STAT.replace('team::', 'opponent::')] = value
    schedule = original.schedule
    schedule['round_stage'] = 'regular'
    return TemporalEventStore(h, original.team_seasons, schedule=schedule)


def release_config(mode):
    return GATResearchBNV1() if mode == 'retained' else config(
        statistics=(STAT,), temporal_mode=mode, deterministic=True,
        statistic_availability_policy='per_stat_release_v1')


def initialized(store, cfg, device='cpu', frozen=None):
    target = context(store, [2, 3, 4])
    target.validation = context(store, [6, 7])
    backend = _Backend(cfg, store, 17, device, preprocessor_state=frozen)
    backend.initialize(target)
    return backend


def logits(backend):
    with torch.no_grad():
        return [backend.model(*backend.train_groups[i][1:]).detach().cpu()
                for i in range(len(backend.train_groups))]


@pytest.mark.parametrize('mode', ['snapshot', 'time_features', 'historical', 'retained'])
@pytest.mark.parametrize('release', ['2020-02-01T00:00Z', None])
def test_unreleased_values_leave_fitted_scalers_inputs_and_logits_invariant(mode, release, device):
    a_store, b_store = released_store(release=release), released_store(1e9, release)
    before = b_store.history
    a, b = initialized(a_store, release_config(mode), device), initialized(b_store, release_config(mode), device)
    assert a.preprocessor.state() == b.preprocessor.state()
    for i in range(len(a.train_groups)):
        torch.testing.assert_close(a.train_groups[i][1]['x'], b.train_groups[i][1]['x'], rtol=0, atol=0)
    for x, y in zip(logits(a), logits(b)):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    pd.testing.assert_frame_equal(before, b_store.history, check_exact=True)
    # Explicit frozen preprocessing remains authoritative, without refitting.
    frozen = deepcopy(a.preprocessor.state())
    restored = initialized(b_store, release_config(mode), device, frozen)
    assert restored.preprocessor.state() == frozen
    for x, y in zip(logits(a), logits(restored)):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_public_retained_fresh_fit_is_invariant_to_unreleased_statistic(device):
    adapters = []
    for value in (7., 1e9):
        store = released_store(value)
        adapter = TemporalGraphModel(GATResearchBNV1(), control=TrainingControl(max_steps=1, seed=17)).build(store)
        adapter.configure_device(device)
        adapter.fit(context(store, [2, 3, 4]))
        adapters.append(adapter)
    a, b = adapters
    assert a.backend_.preprocessor.state() == b.backend_.preprocessor.state()
    for name, value in a.backend_.model.state_dict().items():
        torch.testing.assert_close(value, b.backend_.model.state_dict()[name], rtol=0, atol=0)
    pd.testing.assert_frame_equal(a.predict(context(a.event_store, [7, 8], fit=False))['predict_proba'],
                                b.predict(context(b.event_store, [7, 8], fit=False))['predict_proba'], check_exact=True)


def test_scaler_fit_uses_training_cutoff_not_validation_or_label_boundary():
    store = released_store(release='2020-01-15T00:00Z')
    cfg = release_config('snapshot')
    b = initialized(store, cfg)
    cut = max(cfg.history_cutoff(t) for t, _ in b.train_groups.plan.descriptors)
    assert cut < pd.Timestamp('2020-01-15T00:00Z') < b.trained_through
    frame = store.eligible(cut)
    expected = b.preprocessor.numeric_edges(frame).copy()
    expected[frame.event_id.eq(1), b.preprocessor.edge_columns.index(STAT)] = np.nan
    assert b.preprocessor.edge_scale.state() == MaskedScale().fit(expected).state()


def test_release_fit_exact_boundaries_mixed_columns_goals_and_empty_history():
    cut = pd.Timestamp('2020-01-12T23:00Z')
    h = released_store().history.iloc[:4].copy()
    h['available_at::'+STAT] = pd.Series(
        [cut-pd.Timedelta(1, 'ns'), cut, cut+pd.Timedelta(1, 'ns'), pd.NaT], index=h.index)
    other = RETAINED_STATISTICS[1]
    cfg = config(statistics=(STAT, other), statistic_availability_policy='per_stat_release_v1')
    raw = GraphPreprocessor(cfg).numeric_edges(h)
    expected = raw.copy()
    expected[2:, 2] = np.nan
    before = h.copy(deep=True)
    pre = GraphPreprocessor(cfg).fit(h, (), sample().schedule, np.zeros((9, 5)), cutoff=cut)
    assert pre.edge_scale.state() == MaskedScale().fit(expected).state()
    np.testing.assert_array_equal(pre._encode_edges(h, cutoff=cut)[:, :8], pre.edge_scale.transform(expected))
    pd.testing.assert_frame_equal(h, before, check_exact=True)
    # All unavailable statistics use missing-value fallback; goals and a column
    # without release metadata retain their ordinary fitting arithmetic.
    h['available_at::'+STAT] = pd.NaT
    pre = GraphPreprocessor(cfg).fit(h, (), sample().schedule, np.zeros((9, 5)), cutoff=cut)
    assert pre.edge_scale.mean[2] == 0 and pre.edge_scale.scale[2] == 1
    np.testing.assert_array_equal(pre.edge_scale.mean[[0, 1, 3]], np.mean(raw[:, [0, 1, 3]], axis=0))
    empty = GraphPreprocessor(cfg).fit(h.iloc[:0], (), sample().schedule, np.zeros((9, 5)), cutoff=cut)
    np.testing.assert_array_equal(empty.edge_scale.mean, np.zeros(4))
    np.testing.assert_array_equal(empty.edge_scale.scale, np.ones(4))


@pytest.mark.parametrize('cut', [None, pd.NaT, '2020-01-01'])
def test_release_aware_direct_fit_requires_nonmissing_aware_cutoff(cut):
    with pytest.raises(ValueError, match='cutoff'):
        GraphPreprocessor(release_config('snapshot')).fit(
            released_store().history, (), sample().schedule, np.zeros((9, 5)), cutoff=cut)


def test_legacy_and_absent_release_columns_keep_exact_scaler_arithmetic():
    h = released_store().history
    cfg = replace(release_config('snapshot'), statistic_availability_policy='legacy_unmasked_v1')
    old = GraphPreprocessor(cfg).fit(h, (), sample().schedule, np.zeros((9, 5)))
    assert old.edge_scale.state() == MaskedScale().fit(old.numeric_edges(h)).state()
    h = h.drop(columns='available_at::'+STAT)
    new = GraphPreprocessor(replace(cfg, statistic_availability_policy='per_stat_release_v1')).fit(
        h, (), sample().schedule, np.zeros((9, 5)))
    assert old.edge_scale.state() == new.edge_scale.state()
    np.testing.assert_array_equal(old._encode_edges(h), new._encode_edges(h))


def gap_store(*, in_segment=False, empty=False, large_ids=False):
    events = [(1, '2020-01-01T00:00', 1, 2, 1, 0, 10, 900, 1),
              (2, '2020-01-04T00:00:00.000000123', 1, 2, 1, 1, 10, 900, 2)]
    if in_segment:
        events.append((3, '2020-01-06T12:00', 1, 2, 0, 1, 10, 901, 3))
    h = history(events)
    if large_ids:
        for col in ('team_id', 'opponent_id'):
            h[col] = pd.array([2**63+int(x) for x in h[col]], dtype='UInt64')
    m = sample().team_seasons
    newer = m.copy()
    newer['entry_at'], newer['season_id'] = pd.Timestamp('2020-01-06T00:00Z'), 901
    m = pd.concat([m, newer], ignore_index=True)
    if large_ids:
        m['team_id'] = pd.array([2**63+int(x) for x in m.team_id], dtype='UInt64')
    return TemporalEventStore(h.iloc[:0] if empty else h, m)


def gap_config(policy, *, bounded=True):
    return replace(config(), timezone='UTC', boundary_memory=BoundaryMemory(policy, 5, '2020-01-01T00:00Z'),
                   fixture_gap_encoding='bounded_observed_gap_v1' if bounded else 'legacy_standardized_gap_v1')


def gap_plan(store, cfg, time, *, teams=(1, 2)):
    target = context(sample(), [5])
    target.metadata['home_id'], target.metadata['away_id'] = teams
    target.metadata['kickoff_at'] = pd.Timestamp('2020-02-01T12:00Z')
    target.metadata['issuance_at'] = pd.Timestamp(time)
    issue = pd.Series(pd.Timestamp(time), index=target.metadata.index)
    return GroupPlan(store, cfg, target, issue)


def assert_full_gap_oracle(plan, raw):
    _, snap, _, _, meta, numeric = raw
    full = plan.store.eligible(snap.cutoff)
    latest = full.groupby('team_id').kickoff_at.max()
    for side, column in [('home', 1), ('away', 2)]:
        team = meta[side+'_id'].iloc[0]
        expected = latest.get(team, pd.NaT)
        actual = meta[side+'_latest_eligible_observed_at'].iloc[0]
        assert (pd.isna(actual) and pd.isna(expected)) or actual == expected
        assert bool(meta[side+'_has_eligible_observation'].iloc[0]) == pd.notna(expected)
        days = (snap.issuance-expected).total_seconds()/86400 if pd.notna(expected) else np.nan
        np.testing.assert_equal(numeric[0, column], days)
    pre = GraphPreprocessor(plan.config).fit(full, (), meta, numeric)
    encoded = pre.fixture(meta, numeric)
    for side, column in [('home', 1), ('away', 2)]:
        names = [f'{side}_{suffix}' for suffix in ('observed_gap_r7', 'no_eligible_history', 'no_eligible_observation_30d')]
        indices = [pre.fixture_encoded_columns.index(n) for n in names]
        np.testing.assert_array_equal(encoded[:, indices], encode_observed_gap(numeric[:, column], np.isfinite(numeric[:, column])))
    return pre, encoded


@pytest.mark.parametrize('policy', ['fixed_days', 'season'])
@pytest.mark.parametrize('in_segment', [False, True])
def test_bounded_gap_keeps_full_history_across_boundary(policy, in_segment):
    store, cfg = gap_store(in_segment=in_segment), gap_config(policy)
    plan = gap_plan(store, cfg, '2020-01-07T12:00Z')
    raw = plan.raw(0, extra_teams=(999,))
    assert_full_gap_oracle(plan, raw)
    segment = store.snapshot(plan.descriptors[0][0], cfg, (1, 2, 999), _since=pd.Timestamp('2020-01-06T00:00Z'))
    pd.testing.assert_frame_equal(raw[1].edges, segment.edges, check_exact=True)
    np.testing.assert_array_equal(raw[1].support, segment.support)
    assert len(raw[1].edges) == (2 if in_segment else 0)


@pytest.mark.parametrize('policy', ['fixed_days', 'season'])
@pytest.mark.parametrize('scenario', ['empty', 'new_team', 'late_result', 'large_ids'])
def test_bounded_gap_full_history_handles_availability_and_team_identity(policy, scenario):
    store, cfg = gap_store(empty=scenario == 'empty', large_ids=scenario == 'large_ids'), gap_config(policy)
    teams = (2**63+1, 2**63+2) if scenario == 'large_ids' else (1, 999) if scenario == 'new_team' else (1, 2)
    if scenario == 'late_result':
        h = store.history
        h['result_available_at'] = h.kickoff_at + pd.Timedelta(days=1)
        h.loc[h.event_id.eq(2), 'result_available_at'] = pd.Timestamp('2020-01-08T00:00Z')
        future = history([(9, '2020-02-01', 1, 2, 9, 9, 10, 901, 9)])
        future['result_available_at'] = pd.Timestamp('2020-02-02T00:00Z')
        store = TemporalEventStore(pd.concat([h, future], ignore_index=True), store.team_seasons)
    for time in ['2020-01-08T12:00Z', '2020-01-07T12:00Z', '2020-01-08T12:00Z']:
        plan = gap_plan(store, cfg, time, teams=teams)
        assert_full_gap_oracle(plan, plan.raw(0, extra_teams=(998,)))


@pytest.mark.parametrize('policy', ['fixed_days', 'season'])
@pytest.mark.parametrize('delta', [-1, 0, 1])
def test_boundary_nanosecond_ties_use_full_history_without_changing_segments(policy, delta):
    time = pd.Timestamp('2020-01-06T00:00Z') + pd.Timedelta(delta, 'ns')
    plan = gap_plan(gap_store(), gap_config(policy), time)
    raw = plan.raw(0)
    assert_full_gap_oracle(plan, raw)
    assert len(raw[1].edges) == (4 if delta < 0 else 0)


@pytest.mark.parametrize('policy', ['fixed_days', 'season'])
def test_bounded_gap_replay_and_resident_lease_keep_segment_and_carry(policy, device):
    store, cfg = gap_store(), gap_config(policy)
    backend = _Backend(cfg, store, 17, device, InputCachePolicy(16*1024**2, 0))
    backend.initialize(context(store, [1]))
    plan = gap_plan(store, cfg, '2020-01-07T12:00Z')
    raw = plan.raw(0)
    _, expected = assert_full_gap_oracle(plan, raw)
    # Use the backend's fitted transform for the other fixture coordinates.
    expected = backend.preprocessor.fixture(raw[-2], raw[-1])
    groups = PreparedGroups(plan, backend.preprocessor, device, InputCachePolicy(16*1024**2, 0))
    replay = BoundaryReplay(backend, plan.descriptors[-1][0])
    ordinary = replay.group(groups, 0)
    assert replay.start == pd.Timestamp('2020-01-06T00:00Z') and set(replay.state) == {1, 2}
    carry = {k: v.clone() for k, v in replay.state.items()}
    for _ in range(2):
        with replay.lease(groups, 0) as leased:
            assert leased[1]['x'].shape[0] == 0
            torch.testing.assert_close(leased[-1], torch.tensor(expected, dtype=torch.float32, device=device), rtol=0, atol=0)
            for key, value in ordinary[1].items():
                torch.testing.assert_close(value, leased[1][key], rtol=0, atol=0)
            for key, value in carry.items():
                torch.testing.assert_close(value, replay.state[key], rtol=0, atol=0)
    if device.startswith('cuda'):
        assert groups._resident.pool is not None


@pytest.mark.parametrize('policy', ['fixed_days', 'season'])
def test_legacy_segment_gap_and_bounded_no_memory_controls(policy):
    store = gap_store()
    old = gap_plan(store, gap_config(policy, bounded=False), '2020-01-07T12:00Z').raw(0)
    assert np.isnan(old[-1][0, 1:3]).all() and old[1].edges.empty
    assert not any('observed_gap' in col for col in old[-2])
    plan = gap_plan(store, replace(gap_config(policy), boundary_memory=None), '2020-01-07T12:00Z')
    raw = plan.raw(0)
    assert len(raw[1].edges) == 4
    assert_full_gap_oracle(plan, raw)
