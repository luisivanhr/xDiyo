"""Exact native speed-path comparisons against frozen 11ffa446 source."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import types
import weakref

import numpy as np
import pandas as pd
import pytest
torch = pytest.importorskip('torch')

from test_temporal_graph import sample, context
from test_graph_preparation_memory import config
from test_graph_preparation_memory import assert_exact
from xdiyo_analytics.graphs import TemporalEventStore, GraphProfile
from xdiyo_analytics.graphs.adapter import _Backend
from xdiyo_analytics.graphs.streaming import GroupPlan, PreparedGroups, prepare_group


def reference():
    modules = {}
    for name in ('store', 'preprocessing', 'streaming', 'adapter'):
        fullname = f'xdiyo_analytics.graphs._speed_{name}'
        module = types.ModuleType(fullname)
        module.__package__ = 'xdiyo_analytics.graphs'
        sys.modules[fullname] = module
        path = Path(__file__).parent/'reference'/f'gat_{name}_11ffa446.txt'
        exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
        modules[name] = module
    adapter = modules['adapter']
    # Keep the frozen implementation's pre-option fingerprint for disabled mode.
    original_asdict = adapter.asdict
    def legacy_asdict(value):
        result = original_asdict(value)
        for name in ('boundary_memory', 'statistics_forecast', 'goal_auxiliary'):
            if result.get(name) is None:
                result.pop(name, None)
        if result.get('deterministic') is False:
            result.pop('deterministic')
        if 'temporal_mode' in result:
            assert result.pop('temporal_mode') == 'snapshot'
            result.pop('time_scales_days')
            for name in ('time_encoding', 'age_scale_days', 'gap_scale_days'):
                result.pop(name)
        for name in ('fixture_gap_encoding', 'observed_gap_scale_days', 'observed_gap_recent_days', 'sender_state_mode'):
            result.pop(name, None)
        return result
    adapter.asdict = legacy_asdict
    adapter.GraphPreprocessor = modules['preprocessing'].GraphPreprocessor
    adapter.GroupPlan = modules['streaming'].GroupPlan
    adapter.PreparedGroups = modules['streaming'].PreparedGroups
    # Frozen functions carry their own globals; replacing only class bindings
    # leaves the entire old execution path independent of the optimization.
    return modules


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_snapshot_original_boundaries_dtypes_and_large_ids():
    original = sample()
    h, m, s = original.history, original.team_seasons, original.schedule
    h['result_available_at'] = h.kickoff_at + pd.Timedelta(hours=3)
    h.loc[h.event_id.eq(1), 'result_available_at'] = pd.Timestamp('2020-01-11T00:00Z')
    for f, cols in ((h, ('team_id','opponent_id')), (m, ('team_id',)), (s, ('home_id','away_id'))):
        for c in cols:
            f[c] = pd.array([int(x)+2**63 for x in f[c]], dtype='UInt64')
    extra = m.iloc[[0]].copy()
    extra['entry_at'] = pd.Timestamp('2020-01-10T07:00Z')
    extra['season_id'], extra['competition_id'], extra['movement'] = 202, 12, 'promoted'
    m = pd.concat([m, extra], ignore_index=True)
    native = TemporalEventStore(h, m, schedule=s)
    old = reference()['store'].TemporalEventStore(h, m, schedule=s)
    assert native.version == old.version
    for time in ('2019-12-01T00:00Z', '2020-01-10T06:00Z', '2020-01-10T07:00Z',
                 '2020-01-11T00:00Z', '2020-03-01T00:00Z'):
        a, b = native.snapshot(time, config(), (2**63+999,)), old.snapshot(time, config(), (2**63+999,))
        pd.testing.assert_frame_equal(a.edges, b.edges, check_exact=True)
        for key in ('teams','receiver','sender','log_weight','support','gate','league','entry','cutoff'):
            assert_exact(getattr(a,key), getattr(b,key))


def test_cache_row_parity_invalidation_budget_and_no_future_validation():
    store = sample()
    h = store.history
    col, paired = 'team::value', 'opponent::value'
    h[col] = np.where(h.side.eq('home'), 0., np.nan)
    h[paired] = np.where(h.side.eq('away'), 0., np.nan)
    store = TemporalEventStore(h, store.team_seasons, schedule=store.schedule)
    cfg = config(statistics=(col,paired))
    backend = _Backend(cfg, store, 1, 'cpu')
    backend.initialize(context(store, [0,1,2,3]))
    pre = backend.preprocessor
    plan = backend.train_groups.plan
    for i in range(len(plan)):
        assert_exact(backend.train_groups[i], prepare_group(plan.raw(i), pre, 'cpu'))
    cache = pre._edge_cache
    assert cache[1].nbytes + cache[2].nbytes <= pre._edge_cache_limit
    for i in range(len(plan)):
        backend.train_groups[i]
    assert pre._edge_cache is cache
    pre.edge_scale.mean += .1
    assert_exact(backend.train_groups[-1], prepare_group(plan.raw(-1), pre, 'cpu'))
    assert pre._edge_cache is not cache
    pre._edge_cache_limit = 0
    assert_exact(backend.train_groups[-1], prepare_group(plan.raw(-1), pre, 'cpu'))
    assert pre._edge_cache is None
    pre._edge_cache_limit = 128*1024**2
    # All-source eager encoding would incorrectly fail this early forecast.
    h.loc[h.event_id.eq(9), [col,paired]] = np.inf
    changed = TemporalEventStore(h, store.team_seasons, schedule=store.schedule)
    newplan = GroupPlan(changed,cfg,context(changed,[1,2]), changed.issue_times(context(changed,[1,2]).metadata,cfg))
    groups = PreparedGroups(newplan, pre, 'cpu')
    assert_exact(groups[0], prepare_group(newplan.raw(0),pre,'cpu'))
    assert pre._edge_cache[0][1] == changed.version
    with pytest.raises(ValueError, match='different event store'):
        pre.cached_edges(store.snapshot('2020-01-05T00:00Z',cfg), changed)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_cached_cast_extremes_masks_and_signed_zero(device):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    original=sample(); h=original.history
    values=[0.,-0.,np.nan,1e-44,-1e-44,1e-300,1e30,1.0000000596046448,1.0000001788139343]
    h['team::value']=np.repeat(values,2)
    h['opponent::value']=h['team::value']
    store=TemporalEventStore(h,original.team_seasons,schedule=original.schedule)
    backend=_Backend(config(statistics=('team::value','opponent::value')),store,17,device)
    backend.initialize(context(store,[1,2]))
    pre=backend.preprocessor
    pre.edge_scale.mean[:]=0.;pre.edge_scale.scale[:]=1.
    snap=store.snapshot('2020-03-01T00:00Z',backend.config)
    expected=pre.tensors(snap,device)['x']
    actual=torch.as_tensor(pre.cached_edges(snap,store),device=device)
    assert_exact(actual,expected)


@pytest.mark.parametrize('device', ['cpu','cuda'])
@pytest.mark.parametrize('reconstruction', [None, 'mse', 'kl'])
def test_full_update_gradient_optimizer_rng_and_continuation(device, reconstruction):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from test_heatmap_cnn import graph_setup
    from xdiyo_analytics.neural import Reconstruction
    if reconstruction is None:
        store, cfg = sample(), config()
    else:
        spec, store = graph_setup(Reconstruction(loss=reconstruction))
        cfg = replace(spec.config, fixture_gap_encoding='legacy_standardized_gap_v1')
    oldmod = reference()
    oldstore = oldmod['store'].TemporalEventStore(store.history,store.team_seasons,
                                                schedule=store.schedule, provenance=store.provenance)
    train = context(store,[3,1,2,4])
    train.validation = context(store,[6,5])
    train.sample_weight = pd.Series([1.,0.,2.,.5], index=train.X.index)
    native, old = _Backend(cfg,store,17,device), oldmod['adapter']._Backend(cfg,oldstore,17,device)
    # Enforce identical device reductions for a bitwise test. Production CUDA
    # atomic scatter may be nondeterministic; benchmarks disclose that separately.
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        for backend in (native,old):
            backend.initialize(train)
        assert_exact(list(native.train_groups), list(old.train_groups))
        for step in range(3):
            results, gradients, rng = [], [], []
            for backend in (native,old):
                backend.optimizer.zero_grad(set_to_none=True)
                loss = backend._loss(backend.train_groups,backend.train_targets,backend.train_weights,True)
                gradients.append([p.grad.clone() if p.grad is not None else None for p in backend.model.parameters()])
                values = backend.step()
                results.append((loss,backend.checkpoint_metrics('train_loss',values)))
                rng.append(torch.get_rng_state().clone())
            assert_exact(results[0],results[1])
            assert_exact(gradients[0],gradients[1])
            assert_exact(rng[0],rng[1])
            assert_exact(native.model.state_dict(),old.model.state_dict())
            assert_exact(native.optimizer.state_dict(),old.optimizer.state_dict())
            if step == 1:
                # Full-state continuation after discarding all transient caches.
                state = deepcopy((native.model.state_dict(),native.optimizer.state_dict()))
                native = _Backend(cfg,store,99,device)
                native.initialize(train)
                native.model.load_state_dict(state[0]); native.optimizer.load_state_dict(state[1])
        with torch.no_grad():
            for ng,og in zip(native.validation_groups,old.validation_groups):
                assert_exact(native.model(*ng[1:]),old.model(*og[1:]))
    finally:
        torch.use_deterministic_algorithms(deterministic)


def test_profile_is_opt_in_and_does_not_change_state():
    store = sample()
    a,b = _Backend(config(),store,17,'cpu'),_Backend(config(),store,17,'cpu')
    for backend in (a,b):
        backend.initialize(context(store,[1,2,3]))
    normal = a.checkpoint_metrics('train_loss',a.step())
    with GraphProfile() as profile:
        observed = b.checkpoint_metrics('train_loss',b.step())
    assert_exact(normal,observed)
    assert_exact(a.optimizer.state_dict(),b.optimizer.state_dict())
    assert {'snapshot_index','row_encoding','tensor_transfer','forward','backward','loss_transfer',
            'optimizer','fingerprint','metric_replay'} <= profile.timings.keys()
    before = deepcopy(profile.timings)
    b.step()
    assert before == profile.timings


def test_research_full_state_bridge_and_completed_budget_zero_updates(tmp_path, monkeypatch):
    """Exercise the supplied research wrapper, not a newly invented resume loop."""
    from xdiyo_analytics.graphs import TemporalGraphAdapter
    from xdiyo_analytics.training import TrainingControl
    modules = {}
    for name in ('replay_bridge','recovery_state_latest_draft'):
        module = types.ModuleType(name)
        path = Path(__file__).parent/'reference'/f'gat_{name}_11ffa446.txt'
        exec(compile(path.read_text(),str(path),'exec'), module.__dict__)
        monkeypatch.setitem(sys.modules,name,module)
        modules[name] = module
    recovery = modules['recovery_state_latest_draft']
    store, cfg = sample(), config()
    train = context(store,[1,2,3])
    old = reference()['adapter']._Backend(cfg,store,17,'cpu')
    old.initialize(train)
    records = []
    for step in range(1,3):
        records.append(dict(step=step,metrics=old.checkpoint_metrics('train_loss',old.step())))
    recovery.save_state(tmp_path/'interrupted',old,{'audit':'speed'},records,3)
    payload = torch.load(tmp_path/'interrupted/training-state.pt',weights_only=True)
    final_metrics = old.checkpoint_metrics('train_loss',old.step())
    model = TemporalGraphAdapter(cfg,store,control=TrainingControl(max_steps=3,seed=17,restore_best=False))
    recovery.fit_fixed(model,train,payload)
    assert_exact(model.backend_.model.state_dict(),old.model.state_dict())
    assert_exact(model.backend_.optimizer.state_dict(),old.optimizer.state_dict())
    records.append(dict(step=3,metrics=final_metrics))
    recovery.save_state(tmp_path/'completed',model.backend_,{'audit':'speed'},records,3)
    completed = torch.load(tmp_path/'completed/training-state.pt',weights_only=True)
    def must_not_update(*args,**kwargs):
        raise AssertionError('Completed budget must not perform updates or replay metrics.')
    monkeypatch.setattr(_Backend,'step',must_not_update)
    monkeypatch.setattr(_Backend,'checkpoint_metrics',must_not_update)
    recovery.fit_fixed(model,train,completed)
    assert_exact(model.backend_.model.state_dict(),old.model.state_dict())
    assert_exact(model.backend_.optimizer.state_dict(),old.optimizer.state_dict())
