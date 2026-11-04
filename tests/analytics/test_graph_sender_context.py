"""Matched learned-prior sender intervention in both K and V, CPU and CUDA."""
from dataclasses import replace
import copy
import os
import pytest
import torch

from xdiyo_analytics.graphs import TemporalGraphConfig
from xdiyo_analytics.graphs.network import TemporalNetwork
from xdiyo_analytics.graphs.extensions import GoalAuxiliary
from xdiyo_analytics.graphs.execution import deterministic_execution


@pytest.fixture(params=['cpu', 'cuda:0'])
def device(request):
    if request.param.startswith('cuda') and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    return request.param


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def models(device, layers=2):
    cfg = TemporalGraphConfig('Europe/Berlin', embedding_dim=4, heads=2, layers=layers,
        hidden_width=8, temporal_mode='time_features', deterministic=True, goal_auxiliary=GoalAuxiliary(beta=0))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(143)
        full = TemporalNetwork(cfg, 3, 17, 9).to(device)
        prior = TemporalNetwork(replace(cfg, sender_state_mode='layer_zero'), 3, 17, 9).to(device)
        prior.load_state_dict(full.state_dict())
        # A=0, B=1, J=2, K=3, L=4. Extra A-L gives attention a genuine choice.
        x = torch.randn(6, 17, device='cpu').to(device)
    data = dict(x=x, receiver=torch.tensor([0, 0, 2, 2, 3, 4], device=device),
        sender=torch.tensor([2, 4, 0, 3, 2, 0], device=device),
        league=torch.tensor([1, 2, 3, 1, 2], device=device), entry=torch.tensor([1, 2, 0, 3, 1], device=device),
        log_weight=torch.tensor([-.2, -.7, -.2, -.5, -.5, -.7], dtype=torch.float64, device=device),
        gate=torch.tensor([.8, 0, .7, .6, .9], device=device))
    inputs = (torch.tensor([0], device=device), torch.tensor([1], device=device), torch.zeros((1, 9), device=device))
    return cfg, full, prior, data, inputs


def gradients(model):
    return {k: None if p.grad is None else p.grad.clone() for k, p in model.named_parameters()}


def same_gradients(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert (a[key] is None) == (b[key] is None), key
        if a[key] is not None: torch.testing.assert_close(a[key], b[key], atol=1e-6, rtol=1e-5, msg=key)


def test_updated_route_matches_original_encoder_output_and_gradients(device):
    cfg, full, _, data, inputs = models(device)
    reference = copy.deepcopy(full)
    with deterministic_execution(cfg, device):
        actual = full.encode(data)
        expected = reference.prior(data['league'], data['entry'])
        for layer in reference.layers:
            expected = layer(expected, data['x'], data['receiver'], data['sender'], data['log_weight'], data['gate'])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.square().sum().backward(); expected.square().sum().backward()
        same_gradients(gradients(full), gradients(reference))


def test_one_layer_output_attention_and_gradient_parity(device):
    cfg, full, prior, data, inputs = models(device, 1)
    with deterministic_execution(cfg, device):
        a, aa = full.encode(data, return_attention=True)
        b, ba = prior.encode(data, return_attention=True)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(aa[0], ba[0], atol=0, rtol=0)
        left = full(data, *inputs); right = prior(data, *inputs)
        torch.testing.assert_close(left, right, atol=0, rtol=0)
        left.square().sum().backward(); right.square().sum().backward()
        same_gradients(gradients(full), gradients(prior))


def trace(model, data):
    seen = {'state': [], 'attention': [], 'key': [], 'value': []}
    handles = []
    for layer in model.layers:
        handles.append(layer.register_forward_hook(lambda module, args, output:
            (seen['state'].append(output[0].detach().clone()), seen['attention'].append(output[1].detach().clone())) and None))
        handles.append(layer.k.register_forward_pre_hook(lambda module, args: seen['key'].append(args[0].detach().clone())))
        handles.append(layer.v.register_forward_pre_hook(lambda module, args: seen['value'].append(args[0].detach().clone())))
    try: result = model.encode(data, return_attention=True)
    finally:
        for handle in handles: handle.remove()
    return result, seen


def test_two_layer_first_layer_parity_and_both_sender_routes(device):
    cfg, full, prior, data, _ = models(device)
    for key in full.state_dict(): torch.testing.assert_close(full.state_dict()[key], prior.state_dict()[key], rtol=0, atol=0)
    with deterministic_execution(cfg, device):
        (a, _), ta = trace(full, data); (b, _), tb = trace(prior, data)
        torch.testing.assert_close(ta['state'][0], tb['state'][0], rtol=0, atol=0)
        torch.testing.assert_close(ta['attention'][0], tb['attention'][0], rtol=0, atol=0)
        assert (a-b).abs().max() > 1e-5
        h0 = prior.prior(data['league'], data['entry']).detach()
        for route in ('key', 'value'):
            for layer in range(2): torch.testing.assert_close(tb[route][layer], h0[data['sender']], rtol=0, atol=0)
            torch.testing.assert_close(ta[route][1], ta['state'][0][data['sender']], rtol=0, atol=0)
        # Advancing a forward pass cannot overwrite h0 or cache a learned prior.
        with torch.no_grad(): prior.global_prior.add_(.2)
        _, changed = trace(prior, data)
        torch.testing.assert_close(changed['value'][1], prior.prior(data['league'], data['entry'])[data['sender']], rtol=0, atol=0)
        assert (changed['value'][1]-tb['value'][1]).abs().max() > .19


def test_opponent_only_evidence_intervention_and_trainable_own_history(device):
    cfg, full, prior, data, inputs = models(device)
    changed = dict(data, x=data['x'].clone())
    # Only the two J-K perspectives' goal coordinates change. All A edges,
    # topology, dates, recency, time coordinates, fixture inputs and priors stay fixed.
    changed['x'][3, :2] += torch.tensor([5., -3.], device=device)
    changed['x'][4, :2] += torch.tensor([-3., 5.], device=device)
    with deterministic_execution(cfg, device):
        pa = prior.encode(data); pb = prior.encode(changed)
        torch.testing.assert_close(pa[0], pb[0], rtol=0, atol=0)
        torch.testing.assert_close(prior(data, *inputs), prior(changed, *inputs), rtol=0, atol=0)
        fa = full.encode(data); fb = full.encode(changed)
        assert (fa[0]-fb[0]).abs().max() > 1e-5
        assert (full(data, *inputs)-full(changed, *inputs)).abs().max() > 1e-6
        # The intervention disappears through both the sender-key and value routes.
        _, ta = trace(prior, data); _, tb = trace(prior, changed)
        for route in ('key', 'value'):
            torch.testing.assert_close(ta[route][1], tb[route][1], rtol=0, atol=0)
        output, rates = prior(data, *inputs, return_goals=True)
        torch.nn.functional.cross_entropy(output, torch.tensor([0], device=device)).backward()
        assert prior.global_prior.grad.abs().sum() > 0
        assert prior.league.weight.grad.abs().sum() > 0
        assert prior.entry.weight.grad.abs().sum() > 0
        for layer in prior.layers:
            assert layer.u.weight.grad.abs().sum() > 0
            assert layer.e.weight.grad.abs().sum() > 0
            assert layer.u.weight.grad[:, -14:].abs().sum() > 0
            assert layer.e.weight.grad[:, -14:].abs().sum() > 0
        assert prior.head[0].weight.grad.abs().sum() > 0
        assert all(parameter.grad is None for parameter in prior.goal_head.parameters())


def test_layer_zero_rejects_carried_memory(device):
    _, _, model, data, _ = models(device)
    with pytest.raises(ValueError, match='ordinary priors'):
        model.encode(dict(data, initial_state=torch.zeros(5, 4, device=device)))
