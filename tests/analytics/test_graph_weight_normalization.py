"""GAT weight-scale regressions from Ayre's d22aae1 follow-up audit."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
from xdiyo_analytics.graphs.adapter import _Backend
from xdiyo_analytics.neural import Reconstruction
from test_cnn_audit_repairs import fixture, graph_context
from test_heatmap_cnn import one_thread
from test_temporal_graph import context


def setup(kind):
    history, memberships, points, spatial, spec = fixture()
    if kind == 'scalar':
        cfg = replace(spec.config, heatmaps=None)
    else:
        reconstruction = None if kind == 'disabled' else Reconstruction(loss=kind)
        cfg = replace(spec.config, heatmaps=replace(spatial,
            encoder=replace(spatial.encoder, reconstruction=reconstruction)))
    spec = replace(spec, config=cfg)
    store = spec.event_store(history, memberships, points)
    fitting = graph_context(store)
    fitting.validation = context(store, [3, 4])
    return cfg, store, fitting


def evaluate(cfg, store, fitting, scale, population, device='cpu'):
    fitting = replace(fitting, sample_weight=pd.Series(scale*np.array([1., 2.]), index=fitting.X.index))
    fitting.validation = replace(fitting.validation,
        sample_weight=pd.Series(scale*np.array([2., 1.]), index=fitting.validation.X.index))
    backend = _Backend(cfg, store, 17, device)
    backend.initialize(fitting)
    groups, targets, weights = [getattr(backend, f'{population}_{name}') for name in ('groups', 'targets', 'weights')]
    primary = backend._loss(groups, targets, weights, True)
    losses = np.array([primary, backend._reconstruction_loss])
    gradients = {name: p.grad.clone() for name, p in backend.model.named_parameters() if p.grad is not None}
    assert np.isfinite(losses).all() and primary > 0
    assert all(torch.isfinite(g).all() for g in gradients.values())
    assert any(g.count_nonzero() for g in gradients.values())
    assert torch.isfinite(weights).all() and (weights > 0).all()
    return losses, gradients


@pytest.mark.parametrize('kind', ['scalar', 'disabled', 'mse', 'kl'])
@pytest.mark.parametrize('population', ['train', 'validation'])
def test_loss_and_gradient_invariance_over_extreme_scales(kind, population):
    cfg, store, fitting = setup(kind)
    expected, expected_gradients = evaluate(cfg, store, fitting, 1., population)
    for scale in (1000., 1e-32, 5e37, 1e-50, 1e100, 1.2e38, 5e307, np.nextafter(0., 1.)):
        actual, gradients = evaluate(cfg, store, fitting, scale, population)
        np.testing.assert_allclose(actual, expected, rtol=5e-7, atol=1e-8)
        assert gradients.keys() == expected_gradients.keys()
        for name, gradient in gradients.items():
            torch.testing.assert_close(gradient, expected_gradients[name], rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize('kind', ['mse', 'kl'])
def test_issuance_share_and_zero_weight_groups(kind):
    cfg, store, fitting = setup(kind)
    fitting.sample_weight = pd.Series([0., 1e308], index=fitting.X.index)
    b = _Backend(cfg, store, 17, 'cpu'); b.initialize(fitting)
    primary = b._loss(b.train_groups, b.train_targets, b.train_weights, True)
    reconstruction = b._reconstruction_loss
    assert len(b.train_groups) == 2
    # Only the second issuance contributes. Reconstruction is weighted by
    # issuance fixture weight, not by historical edge count or map count.
    positions, data, home, away, features = b.train_groups[1]
    with torch.no_grad():
        logits, auxiliary = b.model(data, home, away, features, return_auxiliary=True)
        expected = torch.nn.functional.cross_entropy(logits, b.train_targets[positions])
    assert primary == pytest.approx(float(expected), abs=1e-7)
    assert reconstruction == pytest.approx(float(auxiliary), abs=1e-7)
    assert all(torch.isfinite(p.grad).all() for p in b.model.parameters() if p.grad is not None)


def test_float64_sum_overflow_is_safe_and_input_weights_unchanged():
    cfg, store, fitting = setup('mse')
    fitting.sample_weight = pd.Series([1.7e308, 1.7e308], index=fitting.X.index)
    before = fitting.sample_weight.copy()
    b = _Backend(cfg, store, 17, 'cpu'); b.initialize(fitting)
    assert torch.isfinite(b.train_weights.sum())
    assert (b.train_weights > 0).all()
    assert np.isfinite(b._loss(b.train_groups, b.train_targets, b.train_weights, True))
    pd.testing.assert_series_equal(before, fitting.sample_weight)


@pytest.mark.parametrize('weights', [[0., 0.], [-1., 1.], [np.inf, 1.], [np.nan, 1.], [1e-300, 1e300]])
def test_invalid_or_unrepresentable_relative_weights_rejected(weights):
    cfg, store, fitting = setup('scalar')
    fitting.sample_weight = pd.Series(weights, index=fitting.X.index)
    with pytest.raises(ValueError, match='weights'):
        _Backend(cfg, store, 17, 'cpu').initialize(fitting)


def test_weight_total_computed_once_per_replay(monkeypatch):
    cfg, store, fitting = setup('mse')
    b = _Backend(cfg, store, 17, 'cpu'); b.initialize(fitting)
    sums = []
    original = torch.Tensor.sum
    def summed(tensor, *args, **kwargs):
        if tensor is b.train_weights:
            sums.append(1)
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'sum', summed)
    b._loss(b.train_groups, b.train_targets, b.train_weights, True)
    assert len(sums) == 1


@pytest.mark.parametrize('kind', ['scalar', 'mse', 'kl'])
@pytest.mark.parametrize('population', ['train', 'validation'])
def test_cuda_weight_scale_invariance(kind, population):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable.')
    cfg, store, fitting = setup(kind)
    expected, expected_gradients = evaluate(cfg, store, fitting, 1., population, 'cuda:0')
    for scale in (1e-50, 1.2e38, 1e100):
        actual, gradients = evaluate(cfg, store, fitting, scale, population, 'cuda:0')
        np.testing.assert_allclose(actual, expected, rtol=5e-7, atol=1e-8)
        for name, gradient in gradients.items():
            torch.testing.assert_close(gradient, expected_gradients[name], rtol=2e-5, atol=2e-7)
