"""Bounded numerical checks, distinct from full-training quality acceptance."""
from dataclasses import replace
import os
import numpy as np
import pytest
import torch
from graph_execution_fixture import execution_fixture
from xdiyo_analytics.graphs.adapter import _Backend
from xdiyo_analytics.graphs.packing import stream_batches
from xdiyo_analytics.training import TrainingControl


@pytest.fixture(autouse=True)
def single_thread():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


def backend(mode, statistics=True, layers=2, ratings=True):
    adapter, train, query = execution_fixture(mode, statistics, layers, ratings)
    b = _Backend(adapter.config, adapter.event_store, 19, os.environ.get('XDIYO_EXECUTION_DEVICE', 'cpu'))
    b.initialize(train)
    return b, train, query


def scored(b, groups):
    logits = torch.empty((len(groups.plan.metadata), 3))
    forecasts = torch.empty((len(logits), 222)) if b.config.statistics_forecast else None
    if b.config.execution.mode == 'native':
        iterator = ((groups[i], (i,), None) for i in range(len(groups)))
    else:
        iterator = stream_batches(b, groups)
    with torch.no_grad():
        for group, _, _ in iterator:
            positions, data, home, away, fixture = group
            result, _, predicted = b.model(data, home, away, fixture, return_statistics=True)
            logits[positions] = result.cpu()
            if forecasts is not None:
                forecasts[positions] = torch.cat([p['quantiles'] for p in predicted], dim=1).cpu()
            del group, data, home, away, fixture, result, predicted
    return logits, forecasts


@pytest.mark.parametrize('mode', ['dependency', 'dependency_lean', 'dependency_packed'])
@pytest.mark.parametrize('layers', [1, 2, 3])
@pytest.mark.parametrize('statistics,ratings', [(False, False), (True, False), (True, True)])
def test_bounded_fixed_weights_losses_gradients_and_adam(mode, layers, statistics, ratings):
    native, _, _ = backend('native', statistics, layers, ratings)
    candidate, _, _ = backend(mode, statistics, layers, ratings)
    a, sa = scored(native, native.train_groups)
    b, sb = scored(candidate, candidate.train_groups)
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(a.softmax(-1), b.softmax(-1), atol=5e-7, rtol=2e-6)
    if statistics:
        assert sa.shape[1] == 222
        torch.testing.assert_close(sa, sb, atol=2e-6, rtol=2e-5)
        assert native.statistics_scale.state() == candidate.statistics_scale.state()
    if mode != 'dependency':
        assert candidate._execution_trace['fallback_groups'] == 0
    if mode == 'dependency_packed':
        assert candidate._execution_trace['batches'] < candidate._execution_trace['groups']
    va, vb = native.step(), candidate.step()
    for key in va:
        assert vb[key] == pytest.approx(va[key], abs=3e-6, rel=3e-5), key
    for (name, p), (other, q) in zip(native.model.named_parameters(), candidate.model.named_parameters()):
        assert name == other
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-3, msg=name)
        # A diagnostic bound, NOT interchangeability: near-zero AdamW gradients
        # can amplify small float32 differences into an order-lr update.
        torch.testing.assert_close(p, q, atol=2.1e-3, rtol=1e-5, msg=name)
        for key, value in native.optimizer.state[p].items():
            torch.testing.assert_close(value, candidate.optimizer.state[q][key], atol=2e-6, rtol=5e-3, msg=name+key)


@pytest.mark.parametrize('mode', ['native', 'dependency', 'dependency_lean', 'dependency_packed'])
def test_repeatability_three_updates_within_mode(mode):
    a, _, _ = backend(mode)
    b, _, _ = backend(mode)
    for _ in range(3):
        assert a.step() == b.step()
        for p, q in zip(a.model.parameters(), b.model.parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
        for p, q in zip(a.optimizer.state.values(), b.optimizer.state.values()):
            for key in p:
                torch.testing.assert_close(p[key], q[key], atol=0, rtol=0)


def test_prediction_restores_original_order_and_policy_records(tmp_path):
    from xdiyo_analytics.graphs import TemporalGraphSerializer
    a, train, query = execution_fixture()
    a.configure_device(os.environ.get('XDIYO_EXECUTION_DEVICE', 'cpu'))
    a.fit_controlled(train, TrainingControl(max_steps=1, seed=19))
    expected = a.predict(query)
    assert expected['statistics'].shape[1] == 222
    reversed_query = replace(query, X=query.X.iloc[::-1], metadata=query.metadata.iloc[::-1])
    actual = a.predict(reversed_query)
    for key in expected:
        np.testing.assert_allclose(actual[key].iloc[::-1], expected[key], atol=0, rtol=0)
    serializer = TemporalGraphSerializer(device=a.device, embed_event_store=True)
    serializer.save(a, tmp_path)
    restored = serializer.load(tmp_path)
    assert restored.config.execution == a.config.execution
    assert restored.training_summary_['graph_execution']['mode'] == 'dependency_packed'
    np.testing.assert_array_equal(restored.predict(query)['predict_proba'], expected['predict_proba'])


def test_selection_and_patience_reject_auxiliary_monitor():
    a, train, _ = execution_fixture()
    with pytest.raises(ValueError, match='validation outcome cross-entropy'):
        a.fit_controlled(train, TrainingControl(max_steps=1, monitor='validation_objective'))


@pytest.mark.parametrize('route', ['auxiliary', 'bottleneck'])
def test_additional_quantile_routes_keep_complete_loss_and_update(route):
    adapter, train, _ = execution_fixture()
    statistics = replace(adapter.config.statistics_forecast, route=route)
    cfg = replace(adapter.config, statistics_forecast=statistics)
    native = _Backend(replace(cfg, execution=replace(cfg.execution, mode='native')),
                      adapter.event_store, 19, os.environ.get('XDIYO_EXECUTION_DEVICE', 'cpu'))
    candidate = _Backend(cfg, adapter.event_store, 19, native.device)
    native.initialize(train)
    candidate.initialize(train)
    expected = native.step()
    values = candidate.step()
    for key, value in expected.items():
        assert values[key] == pytest.approx(value, abs=3e-6, rel=3e-5)
    for p, q in zip(native.model.parameters(), candidate.model.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-3)
        torch.testing.assert_close(p, q, atol=2.1e-3, rtol=1e-5)


def test_validation_with_no_observed_statistics_stays_unavailable():
    native, _, _ = backend('native')
    candidate, _, _ = backend('dependency_packed')
    for b in (native, candidate):
        b.validation_statistics.fill_(float('nan'))
        b._statistics_observations.clear()
    a, b = native.step(), candidate.step()
    assert np.isnan(a['validation_statistics_loss']) and np.isnan(b['validation_statistics_loss'])
    assert np.isnan(a['validation_objective']) and np.isnan(b['validation_objective'])
    assert b['validation_loss'] == pytest.approx(a['validation_loss'], abs=3e-6)


def test_reported_adam_amplification_uses_unchanged_epsilon():
    ga, gb = 3.3023966e-8, -5.8954406e-8
    lr, epsilon = .001, 1e-8
    predicted_difference = abs(-lr*ga/(abs(ga)+epsilon) + lr*gb/(abs(gb)+epsilon))
    assert predicted_difference == pytest.approx(.00162255, abs=1e-8)
    parameters = []
    for gradient in (ga, gb):
        p = torch.nn.Parameter(torch.tensor(.028, dtype=torch.float64))
        opt = torch.optim.AdamW([p], lr=lr, eps=epsilon, weight_decay=.001)
        p.grad = torch.tensor(gradient, dtype=p.dtype)
        opt.step()
        parameters.append(p.item())
    assert abs(parameters[0]-parameters[1]) == pytest.approx(predicted_difference, abs=1e-12)
