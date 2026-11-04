"""Behavioral contracts for the opt-in time-indexed attention encoder."""
import json
from pathlib import Path
import types
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')

from test_temporal_graph import sample, config, context, history
from xdiyo_analytics.graphs import TemporalGraphAdapter, TemporalGraphSerializer, TemporalEventStore
from xdiyo_analytics.graphs.adapter import _Backend
from xdiyo_analytics.graphs.network import TemporalNetwork
from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor
from xdiyo_analytics.graphs.temporal import TemporalHistoryIndex, encode_time_features, rational_decay
from xdiyo_analytics.training import TrainingControl

MODES = ('snapshot', 'time_features', 'historical')


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def components(store, cfg):
    snap = store.snapshot(pd.Timestamp('2020-03-01T12:00Z'), cfg, (1, 2, 3))
    pre = GraphPreprocessor(cfg).fit(store.history, (snap,), store.schedule,
                                   np.zeros((len(store.schedule), 5)))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(47)
        model = TemporalNetwork(cfg, len(pre.leagues), pre.edge_dim, pre.fixture_dim)
    return pre, model


def encode(store, cfg, pre, model, time='2020-02-01T12:00Z', teams=(1, 2, 3)):
    snap = store.snapshot(pd.Timestamp(time), cfg, teams)
    data = pre.native_tensors(snap, 'cpu', store)
    return snap, data, model.encode(data)


def chain():
    h = history([(1, '2020-01-01T18:00', 2, 3, 1, 0, 10, 900, 1),
                 (2, '2020-01-10T18:00', 1, 2, 2, 1, 10, 900, 2)])
    return TemporalEventStore(h, sample().team_seasons)


def extended_chain(store):
    extra = history([(3, '2020-01-20T18:00', 2, 3, 0, 4, 10, 900, 3)])
    return TemporalEventStore(pd.concat([store.history, extra]), store.team_seasons)


@pytest.mark.parametrize('mode', MODES)
def test_later_opponent_form_preserves_historical_encounter_only(mode):
    store = chain()
    cfg = config(temporal_mode=mode)
    pre, model = components(store, cfg)
    old, _, a = encode(store, cfg, pre, model)
    new, _, b = encode(extended_chain(store), cfg, pre, model)
    a, b = a[old.teams.index(1)], b[new.teams.index(1)]
    if mode == 'historical':
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert new.evidence(1, 2) == ((10, 900, 1), (10, 900, 2))
    else:
        assert not torch.allclose(a, b, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize('mode', ('time_features', 'historical'))
def test_fixed_historical_state_unavailable_and_future_invariance(mode):
    store = chain()
    cfg = config(temporal_mode=mode)
    pre, model = components(store, cfg)
    _, _, before = encode(store, cfg, pre, model, '2020-01-10T18:00Z')
    extra = history([(3, '2020-01-09T18:00', 2, 3, 99, 0, 10, 900, 3),
                     (4, '2020-01-10T01:00', 1, 2, 88, 0, 10, 900, 4),
                     (5, '2020-03-01T18:00', 1, 3, 77, 0, 10, 900, 5)])
    extra['result_available_at'] = pd.Timestamp('2020-03-02T00:00Z')
    h = store.history
    h['result_available_at'] = pd.Series(pd.NaT, index=h.index, dtype='datetime64[ns, UTC]')
    changed = TemporalEventStore(pd.concat([h, extra]), store.team_seasons)
    _, _, after = encode(changed, cfg, pre, model, '2020-01-10T18:00Z')
    torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize('mode', MODES)
def test_target_result_cannot_change_its_own_forecast(mode):
    store = sample()
    cfg = config(temporal_mode=mode)
    adapter = TemporalGraphAdapter(cfg, store, control=TrainingControl(max_steps=1))
    adapter.fit(context(store, [0, 1, 2]))
    target = context(store, [7], fit=False)
    before = adapter.predict(target)['predict_proba']
    h = store.history
    mask = h.event_id.eq(8)
    h.loc[mask, 'goals_for'] = np.where(h.loc[mask, 'side'].eq('home'), 99, 77)
    h.loc[mask, 'goals_against'] = np.where(h.loc[mask, 'side'].eq('home'), 77, 99)
    altered = TemporalEventStore(h, store.team_seasons, schedule=store.schedule)
    # Test fixed parameters/preprocessing against counterfactual source labels;
    # public attach_event_store intentionally forbids revising old evidence.
    adapter.backend_.store = altered
    after = adapter.predict(target)['predict_proba']
    pd.testing.assert_frame_equal(before, after, check_exact=True)


@pytest.mark.parametrize('mode', MODES)
def test_inconsistent_schedule_cannot_admit_the_target(mode):
    store = sample()
    cfg = config(temporal_mode=mode)
    schedule = store.schedule
    schedule['round_issue_at'] = schedule.kickoff_at
    schedule.loc[0, ['kickoff_at', 'round_issue_at']] = pd.Timestamp('2020-03-01T12:00Z')
    changed = TemporalEventStore(store.history, store.team_seasons, schedule=schedule)
    with pytest.raises(ValueError, match='target is already present'):
        _Backend(cfg, changed, 1, 'cpu')._validate(context(changed, [0], fit=False))
    schedule.loc[0, 'round_issue_at'] = store.schedule.kickoff_at.iloc[0]
    valid = TemporalEventStore(store.history, store.team_seasons, schedule=schedule)
    assert _Backend(cfg, valid, 1, 'cpu')._validate(context(valid, [0], fit=False)).iloc[0] == schedule.round_issue_at.iloc[0]


def test_label_boundary_uses_later_history_release_and_validation_rejects_it():
    store = sample()
    h = store.history
    h['result_available_at'] = pd.Series(pd.NaT, index=h.index, dtype='datetime64[ns, UTC]')
    h.loc[h.event_id.eq(3), 'result_available_at'] = pd.Timestamp('2020-01-15T00:00Z')
    delayed = TemporalEventStore(h, store.team_seasons, schedule=store.schedule)
    cfg = config(temporal_mode='historical')
    train = context(delayed, [0, 1, 2])
    # Only history declares the later release; target metadata has the old proxy.
    assert delayed.label_availability(train.metadata, cfg).max() == pd.Timestamp('2020-01-15T00:00Z')
    train.validation = context(delayed, [4])
    with pytest.raises(ValueError, match='before validation issuance'):
        _Backend(cfg, delayed, 1, 'cpu').initialize(train)


def test_release_index_matches_direct_store_cutoffs_including_dst():
    store = sample()
    h = store.history
    h['result_available_at'] = h.kickoff_at + pd.Timedelta(hours=4)
    h.loc[h.event_id.eq(1), 'result_available_at'] = pd.Timestamp('2020-03-28T23:00Z')
    h.loc[h.event_id.eq(2), 'status'] = 'scheduled'
    store = TemporalEventStore(h, store.team_seasons)
    cfg = config(temporal_mode='historical')
    index = TemporalHistoryIndex(store, cfg)
    for query in ('2020-01-04T12:00Z', '2020-03-28T12:00Z', '2020-03-29T16:00Z', '2020-03-30T16:00Z'):
        snap = store.snapshot(pd.Timestamp(query), cfg)
        for team in (1, 2, 999):
            expected = snap.edges.index[snap.edges.team_id.eq(team)].to_numpy()
            np.testing.assert_array_equal(index.eligible_rows(team, snap.cutoff.value), expected)
    assert 0 in index.eligible_rows(1, pd.Timestamp('2020-03-28T23:00Z').value)


def test_time_features_are_explicit_masked_and_sensitive_to_dates():
    store = chain()
    cfg = config(temporal_mode='time_features')
    pre, model = components(store, cfg)
    snap, data, _ = encode(store, cfg, pre, model)
    names = pre.temporal_columns
    values = data['x'][:, -len(names):].numpy()
    row = np.flatnonzero(snap.edges.event_id.eq(2) & snap.edges.team_id.eq(1))[0]
    assert values[row, names.index('receiver_gap::missing')] == 1
    assert values[row, names.index('opponent_gap::missing')] == 0
    assert values[row, names.index('opponent_gap::rational_decay_7_days')] == pytest.approx(7/16)
    assert len(names) == 14
    h = store.history
    h.loc[h.event_id.eq(1), 'kickoff_at'] += pd.Timedelta(days=2)
    shifted = TemporalEventStore(h, store.team_seasons)
    _, moved, _ = encode(shifted, cfg, pre, model)
    assert not torch.equal(data['x'], moved['x'])
    assert moved['x'][row, -len(names)+names.index('opponent_gap::rational_decay_7_days')] == pytest.approx(.5)


def test_rational_numerics_layout_and_fractional_days():
    cfg = config(temporal_mode='historical')
    assert config().temporal_mode == 'snapshot'
    assert cfg.time_encoding == 'rational_decay_v1'
    assert (cfg.age_scale_days, cfg.gap_scale_days) == (180., 7.)
    for scale in (7., 180., 2.5):
        days = np.array([0., .125, 1., scale, 365., 36525., 1e9])
        actual = rational_decay(days, scale)
        np.testing.assert_allclose(actual, scale/(scale+days), rtol=2e-16)
        assert actual[0] == 1. and actual[3] == .5
        assert np.isfinite(actual).all() and ((actual > 0) & (actual <= 1)).all()
        assert (np.diff(actual) < 0).all()
    # Overflow-safe evaluation without clipping, even outside practical ranges.
    assert rational_decay([1e308], 1e308)[0] == .5
    assert rational_decay([1e308], .1)[0] > 0
    age, receiver, opponent = [0., 180., .25], [0., np.nan, 1.5], [np.nan, 7., .125]
    actual = encode_time_features(age, receiver, opponent, cfg)
    assert actual.shape == (3, 14) and actual.dtype == np.float32
    np.testing.assert_array_equal(actual[0, :4], [1, 1, 1, 1])
    np.testing.assert_array_equal(actual[0, 4:9], [1, 1, 1, 1, 0])
    np.testing.assert_array_equal(actual[0, 9:], [0, 0, 0, 0, 1])
    np.testing.assert_array_equal(actual[1, 4:9], [0, 0, 0, 0, 1])
    assert actual[1, 0] == .5 and actual[1, 9] == .5
    for days, start, scale in ((.25, 0, 180), (1.5, 4, 7), (.125, 9, 7)):
        np.testing.assert_array_equal(actual[2, start:start+4],
            np.asarray([scale/(scale+days)] + [2**(-days/s) for s in (7, 30, 180)], dtype=np.float32))
    legacy = encode_time_features(age, receiver, opponent, replace(cfg, time_encoding='log1p_v1'))
    keep = [i for i in range(14) if i not in (0, 4, 9)]
    np.testing.assert_array_equal(legacy[:, keep], actual[:, keep])
    np.testing.assert_array_equal(legacy[2, [0, 4, 9]], np.log1p([.25, 1.5, .125]).astype(np.float32))


@pytest.mark.parametrize('bad', [-1., np.inf, -np.inf, np.nan])
def test_invalid_known_durations_rejected(bad):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        rational_decay([bad], 7.)
    for encoding in ('rational_decay_v1', 'log1p_v1'):
        cfg = config(time_encoding=encoding)
        for position in range(3):
            if position and np.isnan(bad):
                continue  # NaN is the explicit missing-gap sentinel only.
            values = [[0.], [0.], [0.]]
            values[position] = [bad]
            with pytest.raises(ValueError, match='finite and nonnegative'):
                encode_time_features(*values, cfg)


@pytest.mark.parametrize('bad', [0., -1., np.inf, -np.inf, np.nan, True, '7'])
def test_invalid_rational_scales_rejected(bad):
    with pytest.raises(ValueError, match='finite and positive'):
        rational_decay([0.], bad)
    for name in ('age_scale_days', 'gap_scale_days'):
        with pytest.raises(ValueError, match='finite and positive'):
            config(**{name: bad})


@pytest.mark.parametrize('encoding', ['rational_decay_v1', 'log1p_v1'])
def test_root_and_recursive_features_agree_for_same_queries(encoding):
    store = extended_chain(chain())
    cfg = config(temporal_mode='historical', time_encoding=encoding, age_scale_days=90., gap_scale_days=3.5)
    pre, model = components(store, cfg)
    snap, data, _ = encode(store, cfg, pre, model)
    plan = snap._temporal_plan
    dependency_edges = 0
    for depth, (batch, tensors) in enumerate(zip(plan.batches, data['temporal_layers'])):
        torch.testing.assert_close(tensors['x'][tensors['root_edge_index']], data['x'], rtol=0, atol=0)
        for position, (team, time_ns) in enumerate(plan.queries[depth+1]):
            mask = batch['receiver'] == position
            if time_ns == snap.issuance.value or not mask.any():
                continue
            query = store.snapshot(pd.Timestamp(time_ns, tz='UTC'), cfg, (team,))
            root = pre.native_tensors(query, 'cpu', store)
            root_rows = {int(row): i for i, row in enumerate(query.edges.index)}
            selected = [root_rows[int(row)] for row in batch['rows'][mask]]
            torch.testing.assert_close(tensors['x'][mask], root['x'][selected], rtol=0, atol=0)
            dependency_edges += len(selected)
    assert dependency_edges > 0


@pytest.mark.parametrize('mode', ['time_features', 'historical'])
def test_temporal_projection_connections_have_finite_nonzero_gradients(mode):
    store, cfg = sample(), config(temporal_mode=mode)
    pre, model = components(store, cfg)
    _, _, states = encode(store, cfg, pre, model)
    states.square().sum().backward()
    for layer in model.layers:
        for projection in (layer.e, layer.u):
            gradient = projection.weight.grad[:, -14:]
            assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize('mode', ['time_features', 'historical'])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_encoding_cache_and_fingerprints_are_distinct(mode, device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from xdiyo_analytics.training.residency import InputCachePolicy
    from xdiyo_analytics.experiments.store import configuration_hash
    store, cfg = sample(), config(temporal_mode=mode, deterministic=True)
    variants = [cfg, replace(cfg, time_encoding='log1p_v1'), replace(cfg, age_scale_days=90.),
                replace(cfg, gap_scale_days=2.5), replace(cfg, time_scales_days=(8., 30., 180.))]
    keys, versions, hashes, features, backends = [], [], [], [], []
    for variant in variants:
        backend = _Backend(variant, store, 17, device, InputCachePolicy(16*2**20, 2**30))
        backend.initialize(context(store, [1, 2, 3]))
        backends.append(backend)  # Keep all cache owners alive together.
        backend.refresh_version()
        keys.append(backend.train_groups._time_encoding_key)
        hashes.append(configuration_hash(asdict(variant)))
        versions.append(backend.model_version)
        with backend.train_groups.lease(2) as group:
            features.append(group[1]['x'].detach().cpu().clone())
        with backend.train_groups.lease(2) as group:
            torch.testing.assert_close(group[1]['x'].cpu(), features[-1], rtol=0, atol=0)
        if device.startswith('cuda'):
            owner = backend.train_groups._resident
            assert any(token[0]() is owner and token[1] == (keys[-1], tuple(backend.preprocessor.fixture_encoded_columns), 2) for token in owner.pool.entries)
    assert len(set(keys)) == len(set(versions)) == len(set(hashes)) == len(variants)
    assert all(not torch.equal(features[0], other) for other in features[1:])
    # Direct use of an already prepared snapshot cannot reinterpret equal-width inputs.
    snap = store.snapshot(pd.Timestamp('2020-02-01T12:00Z'), cfg)
    base, other = backends[0].preprocessor, backends[1].preprocessor
    base.native_tensors(snap, 'cpu', store)
    with pytest.raises(ValueError, match='native_tensors'):
        other.tensors(snap, 'cpu')
    fresh = other.native_tensors(snap, 'cpu', store)
    assert fresh['x'].shape[1] == base.edge_dim
    for backend in backends:
        backend.train_groups._resident.close()


def test_historical_membership_priors_and_deduplicated_queries():
    store = chain()
    m = store.team_seasons
    newer = m.loc[m.team_id.eq(2)].copy()
    newer['entry_at'] = pd.Timestamp('2020-01-15T00:00Z')
    newer['season_id'], newer['competition_id'], newer['movement'] = 901, 20, 'promoted'
    store = TemporalEventStore(store.history, pd.concat([m, newer]))
    cfg = config(temporal_mode='historical', layers=3)
    pre, model = components(store, cfg)
    snap, data, _ = encode(store, cfg, pre, model)
    plan = snap._temporal_plan
    old = plan.queries[0].index((2, pd.Timestamp('2020-01-10T18:00Z').value))
    current = plan.queries[0].index((2, pd.Timestamp('2020-02-01T12:00Z').value))
    assert data['temporal_base']['league'][old].item() == pre.leagues[10]
    assert data['temporal_base']['league'][current].item() == pre.leagues[20]
    assert data['temporal_base']['entry'][old].item() == 0
    assert data['temporal_base']['entry'][current].item() == 1
    assert all(len(queries) == len(set(queries)) for queries in plan.queries)


def slow_encode(team, time, depth, model, store, pre, cfg):
    """Small recursive oracle using direct snapshots, independent of the DAG/index."""
    snap = store.snapshot(time, cfg, (team,))
    i = snap.teams.index(team)
    if depth == 0:
        return model.prior(torch.tensor([pre.leagues.get(snap.league[i], 0)]),
                           torch.tensor([('retained', 'promoted', 'relegated', 'unknown').index(snap.entry[i])]))
    own = slow_encode(team, time, depth-1, model, store, pre, cfg)
    mask = snap.edges.team_id.eq(team).to_numpy()
    edges = snap.edges.loc[mask]
    if not len(edges):
        return own
    features, senders = [], []
    for event in edges.itertuples():
        before = store.snapshot(event.kickoff_at, cfg)
        gaps = []
        for t in (event.team_id, event.opponent_id):
            last = before.edges.loc[before.edges.team_id.eq(t), 'kickoff_at'].max()
            gaps.append((event.kickoff_at-last).total_seconds()/86400 if pd.notna(last) else None)
        values = []
        for age, masked in (((time-event.kickoff_at).total_seconds()/86400, False),
                            (gaps[0], True), (gaps[1], True)):
            scale = cfg.gap_scale_days if masked else cfg.age_scale_days
            first = (np.log1p(age) if cfg.time_encoding == 'log1p_v1' else scale/(scale+age)) if age is not None else 0.
            values += ([first] + [2**(-age/scale) for scale in cfg.time_scales_days]
                       if age is not None else [0.] * (len(cfg.time_scales_days)+1))
            if masked:
                values.append(float(age is None))
        features.append(values)
        senders.append(slow_encode(event.opponent_id, event.kickoff_at, depth-1, model, store, pre, cfg))
    x = torch.tensor(np.column_stack((pre._encode_edges(edges), features)), dtype=torch.float32)
    return model.layers[depth-1](own, x, torch.zeros(len(edges), dtype=torch.long),
        torch.zeros(len(edges), dtype=torch.long), torch.tensor(snap.log_weight[mask], dtype=torch.float64),
        torch.tensor([snap.gate[i]], dtype=torch.float32), sender_state=torch.cat(senders))


@pytest.mark.parametrize('layers', [1, 2, 3])
@pytest.mark.parametrize('encoding', ['rational_decay_v1', 'log1p_v1'])
def test_batched_queries_match_recursive_definition_and_attention_order(layers, encoding):
    store = extended_chain(chain())
    cfg = config(temporal_mode='historical', layers=layers, time_encoding=encoding)
    pre, model = components(store, cfg)
    snap, data, actual = encode(store, cfg, pre, model)
    for i, team in enumerate(snap.teams):
        expected = slow_encode(team, snap.issuance, layers, model, store, pre, cfg)
        torch.testing.assert_close(actual[i:i+1], expected, rtol=2e-6, atol=2e-7)
    result, attention = model.encode(data, return_attention=True)
    torch.testing.assert_close(result, actual)
    for alpha in attention:
        assert alpha.shape == (len(snap.edges), cfg.heads)
        for i in np.unique(snap.receiver):
            torch.testing.assert_close(alpha[snap.receiver == i].sum(0), torch.ones(cfg.heads))


def test_gradient_reaches_historical_sender_and_no_learned_cache_survives_pass():
    store = extended_chain(chain())
    cfg = config(temporal_mode='historical')
    pre, model = components(store, cfg)
    snap, data, _ = encode(store, cfg, pre, model)
    states = []
    def retain(module, args, output):
        output.retain_grad()
        states.append(output)
    hook = model.layers[0].register_forward_hook(retain)
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        model.encode(data)[snap.teams.index(1)].square().sum().backward()
        query = snap._temporal_plan.queries[1].index((2, pd.Timestamp('2020-01-10T18:00Z').value))
        assert states[-1].grad[query].abs().sum() > 0
        assert model.layers[0].u.weight.grad.abs().sum() > 0
    hook.remove()
    before = model.encode(data).detach().clone()
    with torch.no_grad():
        model.global_prior.add_(0.1)
    after = model.encode(data).detach()
    assert not torch.equal(before, after)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_historical_cuda_matches_cpu_values_and_gradients():
    from copy import deepcopy
    from xdiyo_analytics.graphs.execution import prepare_determinism
    store = extended_chain(chain())
    cfg = config(temporal_mode='historical', deterministic=True)
    prepare_determinism(cfg, 'cuda:0')
    pre, cpu = components(store, cfg)
    snap, inputs, expected = encode(store, cfg, pre, cpu)
    gpu = deepcopy(cpu).to('cuda:0')
    actual = gpu.encode(pre.native_tensors(snap, 'cuda:0', store))
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for (name, a), (_, b) in zip(cpu.named_parameters(), gpu.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(b.grad.cpu(), a.grad, rtol=2e-5, atol=2e-6, msg=name)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_resident_inputs_preserve_exact_updates(mode):
    from xdiyo_analytics.training.residency import InputCachePolicy
    store = sample()
    cfg = config(temporal_mode=mode, deterministic=True)
    train = context(store,[1,2,3])
    train.validation = context(store,[5,6])
    plain = _Backend(cfg,store,17,'cuda:0')
    cached = _Backend(cfg,store,17,'cuda:0',InputCachePolicy(16*2**20,2**30))
    plain.initialize(train)
    cached.initialize(train)
    for _ in range(2):
        assert plain.step() == cached.step()
        for name, value in plain.model.state_dict().items():
            torch.testing.assert_close(value,cached.model.state_dict()[name],rtol=0,atol=0)
        for a,b in zip(plain.model.parameters(),cached.model.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
    pool = cached.train_groups._resident.pool
    assert pool.hits > 0 and pool.transient == 0 and pool.resident <= 16*2**20


@pytest.mark.parametrize('mode,encoding', [('snapshot', 'rational_decay_v1'),
    ('time_features', 'rational_decay_v1'), ('historical', 'rational_decay_v1'),
    ('time_features', 'log1p_v1'), ('historical', 'log1p_v1')])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_training_save_load_predictions_and_inspection(mode, encoding, device, tmp_path):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    store = sample()
    cfg = config(temporal_mode=mode, time_encoding=encoding, age_scale_days=90., gap_scale_days=2.5,
                 deterministic=True)  # Exact CUDA reload checks require deterministic reductions.
    adapter = TemporalGraphAdapter(cfg, store, control=TrainingControl(max_steps=2, seed=7))
    adapter.configure_device(device)
    adapter.fit(context(store, [1, 2, 3]))
    target = context(store, [7, 8], fit=False)
    expected = adapter.predict(target)['predict_proba']
    ser = TemporalGraphSerializer(embed_event_store=True, device=device)
    ser.save(adapter, tmp_path)
    path = tmp_path/'graph.json'
    metadata = json.loads(path.read_text())
    assert metadata['config']['time_encoding'] == encoding
    assert metadata['config']['age_scale_days'] == 90.
    assert metadata['config']['gap_scale_days'] == 2.5
    if mode == 'snapshot':
        del metadata['config']['temporal_mode'], metadata['config']['time_scales_days']
        for name in ('time_encoding', 'age_scale_days', 'gap_scale_days'):
            del metadata['config'][name]
        path.write_text(json.dumps(metadata))
    restored = ser.load(tmp_path)
    if mode != 'snapshot':
        assert restored.config == cfg
    pd.testing.assert_frame_equal(expected, restored.predict(target)['predict_proba'], check_exact=True)
    q = pd.DataFrame({'team_id':[1, 999], 'issuance_at':pd.to_datetime(['2020-03-01T12:00Z']*2)})
    pd.testing.assert_frame_equal(adapter.encode_teams(q), restored.encode_teams(q), check_exact=True)
    weights = restored.attention_weights(q)
    assert len(weights) > 0
    if mode == 'historical':
        assert (weights.sender_history_cutoff <= weights.sender_query_at).all()
        assert weights.sender_query_at.equals(weights.kickoff_at)


@pytest.mark.parametrize('mode', ['time_features', 'historical'])
def test_legacy_logarithmic_checkpoint_and_reject_ambiguous_metadata(mode, tmp_path):
    store = sample()
    cfg = config(temporal_mode=mode, time_encoding='log1p_v1')
    adapter = TemporalGraphAdapter(cfg, store, control=TrainingControl(max_steps=1, seed=7))
    adapter.fit(context(store, [1, 2, 3]))
    target = context(store, [7, 8], fit=False)
    expected = adapter.predict(target)['predict_proba']
    ser = TemporalGraphSerializer(store)
    ser.save(adapter, tmp_path)
    path = tmp_path/'graph.json'
    metadata = json.loads(path.read_text())
    for name in ('time_encoding', 'age_scale_days', 'gap_scale_days'):
        del metadata['config'][name]
    path.write_text(json.dumps(metadata))
    restored = ser.load(tmp_path)
    assert restored.config.time_encoding == 'log1p_v1'
    assert restored.backend_.model_version == adapter.backend_.model_version
    pd.testing.assert_frame_equal(expected, restored.predict(target)['predict_proba'], check_exact=True)
    # Explicit metadata must be complete and agree with the stored feature order.
    metadata['config']['time_encoding'] = 'log1p_v1'
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='incomplete'):
        ser.load(tmp_path)
    metadata['config'].update(time_encoding='rational_decay_v1', age_scale_days=180., gap_scale_days=7.)
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='feature order'):
        ser.load(tmp_path)
    # An unrecognized or stripped legacy feature order must never select a default.
    for name in ('time_encoding', 'age_scale_days', 'gap_scale_days'):
        del metadata['config'][name]
    del metadata['temporal_feature_order']
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='legacy temporal artifacts'):
        ser.load(tmp_path)


@pytest.mark.parametrize('mode', ('time_features', 'historical'))
def test_empty_history_and_unknown_team_keep_contextual_priors(mode):
    store = sample()
    memberships = store.team_seasons
    large_league = 2**63 + 99
    memberships['competition_id'] = pd.array([large_league]*len(memberships), dtype='UInt64')
    cold = TemporalEventStore(store.history.iloc[:0], memberships, schedule=store.schedule)
    adapter = TemporalGraphAdapter(config(temporal_mode=mode), cold,
                                   control=TrainingControl(max_steps=1))
    adapter.fit(context(store, [1, 2]))
    queries = pd.DataFrame({'team_id':[1, 999],
                           'issuance_at':pd.to_datetime(['2020-02-01T12:00Z']*2)})
    output = adapter.encode_teams(queries)
    assert output.league.iloc[0] == large_league and pd.isna(output.league.iloc[1])
    assert output.support.eq(0).all()
    np.testing.assert_array_equal(output.filter(regex='^z_'), output.filter(regex='^prior_'))
    assert adapter.attention_weights(queries).empty


def test_default_network_matches_frozen_baseline_outputs_and_gradients():
    module = types.ModuleType('frozen_network')
    path = Path(__file__).parent/'reference'/'gat_network_de32880.txt'
    exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
    store, cfg = sample(), config()
    pre, model = components(store, cfg)
    legacy = module.TemporalNetwork(cfg, len(pre.leagues), pre.edge_dim, pre.fixture_dim)
    legacy.load_state_dict(model.state_dict())
    _, data, actual = encode(store, cfg, pre, model)
    expected = legacy.encode(data)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for (name, a), (_, b) in zip(model.named_parameters(), legacy.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize('kwargs', [dict(temporal_mode='unknown'), dict(time_encoding='unknown'), dict(time_scales_days=()),
    dict(time_scales_days=(0,)), dict(time_scales_days=(np.nan,)), dict(time_scales_days=(True,)),
    dict(time_scales_days=(7, 7))])
def test_invalid_temporal_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


def test_temporal_modes_reject_unvalidated_extensions():
    from xdiyo_analytics.graphs import BoundaryMemory, GraphHeatmaps
    for mode in ('time_features', 'historical'):
        for settings in (dict(boundary_memory=BoundaryMemory()), dict(heatmaps=GraphHeatmaps())):
            with pytest.raises(ValueError, match='require.*disabled'):
                config(temporal_mode=mode, **settings)
