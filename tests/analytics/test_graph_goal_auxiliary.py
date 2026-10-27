"""Focused contracts for parallel goal supervision and its detached control."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
from test_temporal_graph import config, context, sample
from xdiyo_analytics.graphs import GoalAuxiliary, TemporalEventStore, TemporalGraphAdapter, TemporalGraphSerializer
from xdiyo_analytics.graphs.adapter import _Backend
from xdiyo_analytics.graphs.goals import goal_labels, poisson_goal_loss, goal_loss
from xdiyo_analytics.training import TrainingControl


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
    from xdiyo_analytics.graphs.execution import prepare_determinism
    prepare_determinism(config(deterministic=True), request.param)
    return request.param


def backend(auxiliary, device='cpu', seed=17):
    store = sample()
    ctx = context(store, [2, 3, 4])
    ctx.validation = context(store, [6, 7])
    model = _Backend(config(goal_auxiliary=auxiliary, deterministic=True), store, seed, device)
    model.initialize(ctx)
    return model


def test_raw_poisson_sum_mean_zeros_asymmetry_and_constant_optimum():
    y = torch.tensor([[0., 0.], [3., 1.], [1., 4.]], dtype=torch.float64)
    rates = torch.tensor([[.2, .5], [3., 1.], [1.4, 2.7]], dtype=torch.float64)
    independent = -torch.distributions.Poisson(rates).log_prob(y)
    full = poisson_goal_loss(rates, y, full=True)
    torch.testing.assert_close(full, independent.sum(1), rtol=1e-14, atol=1e-14)
    loss = poisson_goal_loss(rates, y)
    torch.testing.assert_close(loss, full-torch.lgamma(y+1).sum(1), rtol=1e-14, atol=1e-14)
    assert loss[0] == .7
    torch.testing.assert_close(loss.mean(), (rates-y*rates.log()).sum()/len(y))
    assert not torch.isclose(loss.mean(), (rates-y*rates.log()).mean())
    assert loss[1] < poisson_goal_loss(rates[1:2].flip(1), y[1:2])[0]
    mean = y.mean(0).requires_grad_()
    optimum = poisson_goal_loss(mean.expand_as(y), y).mean()
    torch.testing.assert_close(torch.autograd.grad(optimum, mean)[0], torch.zeros(2, dtype=mean.dtype), atol=1e-15, rtol=0)
    for factor in (.5, 1.5):
        assert optimum < poisson_goal_loss((mean*factor).expand_as(y), y).mean()


def test_extreme_softplus_outputs_remain_finite(device):
    z = torch.tensor([[-1000., 1000.], [-100., 100.], [0., 0.]], device=device, requires_grad=True)
    rates = torch.nn.functional.softplus(z)+GoalAuxiliary().rate_floor
    loss = poisson_goal_loss(rates, torch.tensor([[0., 3.], [3., 0.], [1., 2.]], device=device))
    loss.mean().backward()
    assert torch.isfinite(rates).all() and (rates > 0).all()
    assert torch.isfinite(loss).all() and torch.isfinite(z.grad).all()


def test_unequal_issuance_groups_preserve_global_loss_and_shared_gradient(device):
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Linear(3, 4)
            self.head = torch.nn.Linear(4, 3)
            self.goal_head = torch.nn.Linear(4, 2)
        def forward(self, data, home, away, fixture, return_goals=False):
            h = self.shared(data['x']).tanh()
            return self.head(h), torch.nn.functional.softplus(self.goal_head(h))+1e-6
    torch.manual_seed(91)
    initial = Toy().to(device)
    x = torch.randn(6, 3, device=device)
    y = torch.tensor([0, 1, 2, 0, 2, 1], device=device)
    goals = torch.tensor([[3, 1], [0, 0], [1, 2], [1, 0], [0, 4], [2, 2]], device=device, dtype=torch.float32)
    weights = torch.tensor([1., .5, .25, 1., .75, .1], device=device)
    results = []
    for partitions in ([np.arange(6)], [np.array([0]), np.array([1, 2, 3]), np.array([4, 5])]):
        b = _Backend(config(goal_auxiliary=GoalAuxiliary()), None, 17, device)
        b.model = deepcopy(initial)
        b.train_groups = [(p, {'x':x[p]}, None, None, None) for p in partitions]
        b.train_goals = goals
        outcome = goal_loss(b, b.train_groups, y, weights, True)
        results.append((outcome, b._goal_loss, [p.grad.clone() for p in b.model.parameters()]))
    np.testing.assert_allclose(results[0][:2], results[1][:2], rtol=2e-6, atol=2e-7)
    for a, b in zip(results[0][2], results[1][2]):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)


def test_task_specific_gradient_routing_and_no_parameter_updates(device):
    for detached in (False, True):
        b = backend(GoalAuxiliary(detach_encoder=detached), device)
        state = b.snapshot()
        names, params = zip(*b.model.named_parameters())
        with b.train_groups.lease(2) as group:
            positions, data, home, away, fixture = group
            ix = torch.as_tensor(positions, device=device)
            logits, rates = b.model(data, home, away, fixture, return_goals=True)
            go = torch.autograd.grad(poisson_goal_loss(rates, b.train_goals[ix]).mean(), params, retain_graph=True, allow_unused=True)
            out = torch.autograd.grad(torch.nn.functional.cross_entropy(logits, b.train_targets[ix]), params, allow_unused=True)
        shared = [i for i, n in enumerate(names) if not n.startswith(('goal_head.', 'head.'))]
        goal = [i for i, n in enumerate(names) if n.startswith('goal_head.')]
        outcome = [i for i, n in enumerate(names) if n.startswith('head.')]
        assert all(go[i] is None for i in outcome)
        assert all(out[i] is None for i in goal)
        assert sum(go[i].abs().sum() for i in goal) > 0
        assert sum(out[i].abs().sum() for i in shared if out[i] is not None) > 0
        if detached:
            assert all(go[i] is None for i in shared)
        else:
            assert sum(go[i].abs().sum() for i in shared if go[i] is not None) > 0
        assert all(torch.isfinite(g).all() for g in (*go, *out) if g is not None)
        assert all(p.grad is None for p in params)
        assert not b.optimizer.state
        for name, value in b.model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)


def test_zero_beta_and_detached_short_trajectory_match_reference(device):
    reference = backend(None, device)
    zero = backend(GoalAuxiliary(beta=0.), device)
    detached = backend(GoalAuxiliary(beta=.1, detach_encoder=True), device, seed=73)
    common = reference.model.state_dict()
    # Append-only module creation must preserve reference initialization too.
    for name, value in common.items():
        torch.testing.assert_close(value, zero.model.state_dict()[name], rtol=0, atol=0)
    target = zero.model.state_dict()
    target.update(common)
    zero.model.load_state_dict(target)
    detached.model.load_state_dict(deepcopy(zero.model.state_dict()))
    initial_goal = deepcopy(zero.model.goal_head.state_dict())
    for b in (zero, detached):
        with b.train_groups.lease(2) as group:
            args = group[1:]
            torch.testing.assert_close(b.model(*args), reference.model(*args), rtol=0, atol=0)
            # Arbitrarily modifying the unused outputs cannot change logits.
            before = b.model(*args).detach().clone()
            with torch.no_grad(): b.model.goal_head.bias.add_(1.)
            torch.testing.assert_close(before, b.model(*args), rtol=0, atol=0)
            with torch.no_grad(): b.model.goal_head.bias.sub_(1.)
    detached.model.load_state_dict(deepcopy(zero.model.state_dict()))
    for _ in range(3):
        a, z, d = reference.step(), zero.step(), detached.step()
        assert z['train_objective'] == pytest.approx(z['train_loss']+z['train_regularization'])
        np.testing.assert_allclose([a['train_loss'], a['validation_loss']], [z['train_loss'], z['validation_loss']], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose([z['train_loss'], z['validation_loss']], [d['train_loss'], d['validation_loss']], rtol=1e-5, atol=1e-6)
        for name, value in reference.model.state_dict().items():
            for b in (zero, detached):
                torch.testing.assert_close(value, b.model.state_dict()[name], rtol=1e-5, atol=1e-6)
        for (name, p), (_, q) in zip(reference.model.named_parameters(), zero.model.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=2e-6)
        assert all(p.grad is None for p in zero.model.goal_head.parameters())
    for name, value in zero.model.goal_head.state_dict().items():
        torch.testing.assert_close(value, initial_goal[name], rtol=1e-6, atol=1e-7)


def test_goal_labels_orientation_availability_and_population_are_strict():
    store = sample()
    h = store.history
    home = h.event_id.eq(4) & h.side.eq('home')
    away = h.event_id.eq(4) & h.side.eq('away')
    h.loc[home, ['goals_for', 'goals_against']] = [3, 1]
    h.loc[away, ['goals_for', 'goals_against']] = [1, 3]
    store = TemporalEventStore(h, store.team_seasons)
    meta = store.schedule.iloc[[3]]
    bound = pd.Timestamp('2020-02-01T00:00Z')
    np.testing.assert_array_equal(goal_labels(store, meta, bound, [1]), [[3, 1]])
    wrong = meta.copy()
    wrong[['home_id', 'away_id']] = wrong[['away_id', 'home_id']].to_numpy()
    with pytest.raises(ValueError, match='identities'): goal_labels(store, wrong, bound)
    with pytest.raises(ValueError, match='outcomes'): goal_labels(store, meta, bound, [-1])
    with pytest.raises(ValueError, match='unique'): goal_labels(store, pd.concat([meta, meta]), bound)
    with pytest.raises(ValueError, match='both available'): goal_labels(store, meta, pd.Timestamp('2019-01-01T00:00Z'))
    unavailable = store.history
    unavailable['result_available_at'] = pd.Timestamp('2020-03-01T00:00Z')
    delayed = TemporalEventStore(unavailable, store.team_seasons)
    with pytest.raises(ValueError, match='both available'): goal_labels(delayed, meta, bound)


def test_persistence_public_boundaries_and_target_exclusion(device, tmp_path):
    store = sample()
    cfg = config(goal_auxiliary=GoalAuxiliary(), deterministic=True)
    model = TemporalGraphAdapter(cfg, store, control=TrainingControl(max_steps=2, seed=17), device=device)
    model.fit(context(store, [1, 2, 3]))
    target = context(store, [7, 8], fit=False)
    expected, goals = model.predict(target)['predict_proba'], model.predict_goals(target)
    serializer = TemporalGraphSerializer(store, device=device)
    serializer.save(model, tmp_path)
    restored = serializer.load(tmp_path)
    assert restored.config == cfg
    pd.testing.assert_frame_equal(expected, restored.predict(target)['predict_proba'], check_exact=True)
    pd.testing.assert_frame_equal(goals, restored.predict_goals(target), check_exact=True)
    for operation in (restored.predict, restored.predict_goals):
        with pytest.raises(ValueError, match='boundary'): operation(context(store, [1], fit=False))
    # Counterfactual revision of the target result is never its own input.
    h = store.history
    selected = h.event_id.eq(9)
    h.loc[selected, 'goals_for'] = np.where(h.loc[selected, 'side'].eq('home'), 99, 77)
    h.loc[selected, 'goals_against'] = np.where(h.loc[selected, 'side'].eq('home'), 77, 99)
    restored.backend_.store = TemporalEventStore(h, store.team_seasons, schedule=store.schedule)
    pd.testing.assert_frame_equal(expected, restored.predict(target)['predict_proba'], check_exact=True)
    pd.testing.assert_frame_equal(goals, restored.predict_goals(target), check_exact=True)


def test_pre_goal_artifact_loads_and_ui_round_trips(tmp_path):
    store = sample()
    model = TemporalGraphAdapter(config(), store, control=TrainingControl(max_steps=1))
    model.fit(context(store, [1, 2, 3]))
    serializer = TemporalGraphSerializer(store)
    serializer.save(model, tmp_path)
    path = tmp_path/'graph.json'
    meta = json.loads(path.read_text())
    assert meta['config'].pop('goal_auxiliary') is None
    path.write_text(json.dumps(meta))
    restored = serializer.load(tmp_path)
    assert restored.config.goal_auxiliary is None and restored.backend_.model.goal_head is None
    target = context(store, [8], fit=False)
    pd.testing.assert_frame_equal(model.predict(target)['predict_proba'], restored.predict(target)['predict_proba'], check_exact=True)
    with pytest.raises(ValueError, match='enabled'): restored.predict_goals(target)
    from xdiyo_analytics.ui.recipe import catalog_for_ui, node
    from dataclasses import fields
    catalog = catalog_for_ui()
    schema = {s['id']:s for s in catalog.schema()}
    assert {f['name'] for f in schema['graphs.GoalAuxiliary']['fields']} == {f.name for f in fields(GoalAuxiliary)}
    assert catalog.build(node('graphs.GoalAuxiliary', beta=.1, detach_encoder=True)) == GoalAuxiliary(detach_encoder=True)


@pytest.mark.parametrize('settings', [dict(beta=-1), dict(beta=np.nan), dict(beta=True), dict(rate_floor=0), dict(rate_floor=np.inf), dict(architecture='mlp'), dict(detach_encoder=1)])
def test_invalid_goal_configuration_rejected(settings):
    with pytest.raises(ValueError): GoalAuxiliary(**settings)
