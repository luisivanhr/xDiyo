"""Independent known-trace decisions for optional iterative controls."""
import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.training import EarlyStopping, ReduceOnPlateau, TrainingControl, run_iterations
from training_controls_samples import ScriptBackend, fit_context


def run(sequence, **kwargs):
    events = []
    backend, history, summary = run_iterations(lambda seed: ScriptBackend(sequence),
        fit_context(observer=events.append), TrainingControl(max_steps=len(sequence), **kwargs))
    return backend, history, summary, events


def test_patience_stops_and_restores_exact_best_without_dropping_history():
    model, history, summary, events = run([10., 8., 9., 9., 0.], early_stopping=EarlyStopping(patience=2))
    assert model.position == 2
    assert history.step.tolist() == [1, 2, 3, 4]
    assert history.value.tolist() == [10, 8, 9, 9]
    assert summary['retained_step'] == 2 and summary['termination_reason'] == 'early_stopping'
    assert summary['monitored_value'] == 8
    assert [event.final for event in events] == [False] * 4 + [True]
    assert events[-1].step == 4 and events[-1].metrics == {'train_loss': 9.}


def test_min_delta_governs_patience_but_checkpoint_uses_exact_improvement():
    model, history, summary, _ = run([10., 9.8, 9.7, 9.6, 9.4], early_stopping=EarlyStopping(3, .5))
    assert len(history) == 4 and model.position == summary['best_step'] == 4
    assert summary['monitored_value'] == 9.6


def test_improvement_exactly_at_min_delta_does_not_reset_patience():
    model, history, summary, _ = run([10., 9.5, 9.], early_stopping=EarlyStopping(1, .5))
    assert len(history) == 2 and model.position == 2
    assert summary['termination_reason'] == 'early_stopping'


def test_maximizing_named_metric_and_ties_retain_first_best():
    events = []
    backend, history, summary = run_iterations(lambda seed: ScriptBackend([
        {'accuracy': .4, 'loss': 4}, {'accuracy': .7, 'loss': 3}, {'accuracy': .7, 'loss': 2}]),
        fit_context(observer=events.append), TrainingControl(max_steps=3, monitor='accuracy', direction='maximize'))
    assert backend.position == 2 and summary['monitored_value'] == .7
    assert history.metric.tolist() == ['accuracy', 'loss'] * 3
    assert history.fold_id.tolist() == [7] * 6


@pytest.mark.parametrize('has_validation,expected', [(False, 1), (True, 2)])
def test_implicit_monitor_uses_only_the_declared_population(has_validation, expected):
    backend, _, summary = run_iterations(lambda seed: ScriptBackend([
        {'train_loss': 1, 'validation_loss': 8}, {'train_loss': 4, 'validation_loss': 2}]),
        fit_context(validation=has_validation), TrainingControl(max_steps=2))
    assert backend.position == expected
    assert summary['monitor'] == ('validation_loss' if has_validation else 'train_loss')


@pytest.mark.parametrize('restore,attempt,step,value', [(True, 0, 2, 1), (False, 1, 3, 2)])
def test_restarts_select_retained_state_and_receive_consecutive_seeds(restore, attempt, step, value):
    seeds = []
    sequences = {9: [5, 1, 4], 10: [4, 2, 2]}
    def factory(seed):
        seeds.append(seed)
        return ScriptBackend(sequences[seed])
    model, history, summary = run_iterations(factory, fit_context(),
        TrainingControl(max_steps=3, restarts=1, seed=9, restore_best=restore))
    assert seeds == [9, 10] and history.attempt.tolist() == [0, 0, 0, 1, 1, 1]
    assert summary['selected_attempt'] == attempt and model.position == step
    assert summary['monitored_value'] == value


def test_equal_restart_scores_keep_first_attempt():
    model, _, summary = run_iterations(lambda seed: ScriptBackend([2, 1]), fit_context(),
                                       TrainingControl(max_steps=2, restarts=2))
    assert summary['selected_attempt'] == 0 and model.position == 2
    assert len(summary['attempts']) == 3


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_monitor_retains_only_a_checkpointed_earlier_state(bad):
    model, history, summary, events = run([3, bad, 1])
    assert model.position == 1 and len(history) == 2
    assert summary['termination_reason'] == 'nonfinite_monitor' and events[-1].final
    with pytest.raises(ValueError, match='No training attempt'):
        run([3, bad, 1], restore_best=False)


def test_invalid_attempt_can_be_followed_by_valid_restart():
    model, history, summary = run_iterations(lambda seed: ScriptBackend([np.nan] if seed == 0 else [4]),
        fit_context(), TrainingControl(max_steps=1, restarts=1))
    assert summary['selected_attempt'] == 1 and summary['attempts'][0]['monitored_value'] is None
    assert len(history) == 2 and model.position == 1


def test_scheduler_records_rate_used_and_stopping_precedes_next_rate_change():
    model, history, summary, _ = run([2, 2, 2, 2], restore_best=False,
        early_stopping=EarlyStopping(2), scheduler=ReduceOnPlateau(1, .5, .1))
    assert history.learning_rate.tolist() == [1, 1, .5]
    assert model.rate_changes == [.5] and summary['termination_reason'] == 'early_stopping'


@pytest.mark.parametrize('initial,expected', [(1., [.5, .25, .2]), (.05, [.05, .05, .05])])
def test_plateau_schedule_obeys_floor_without_increasing_existing_low_rate(initial, expected):
    model, history, _ = run_iterations(lambda seed: ScriptBackend([3, 3, 3, 3], rate=initial), fit_context(),
        TrainingControl(max_steps=4, restore_best=False, scheduler=ReduceOnPlateau(1, .5, .2)))
    assert model.rate_changes == expected
    assert history.learning_rate.tolist() == [initial, initial, *expected[:2]]


def test_restart_inputs_and_nested_definitions_are_independent():
    context = fit_context(validation=True)
    backends = []
    def factory(seed):
        model = ScriptBackend([{'train_loss': 1, 'validation_loss': 2}], mutate_inputs=True)
        backends.append(model)
        return model
    run_iterations(factory, context, TrainingControl(max_steps=1, restarts=1))
    assert context.X.x.tolist() == [0, 1, 2] and context.validation.X.x.tolist() == [3, 4]
    assert context.definitions == {'nested': ['clean']}
    for model in backends:
        pd.testing.assert_frame_equal(model.before.X, context.X)
        pd.testing.assert_frame_equal(model.before.y, context.y)
        pd.testing.assert_frame_equal(model.before.metadata, context.metadata)
        assert model.before.definitions == context.definitions


@pytest.mark.parametrize('reused', ['backend', 'estimator', 'preprocessor'])
def test_restart_factory_cannot_share_model_internals(reused):
    shared = ScriptBackend([1]) if reused == 'backend' else object()
    def factory(seed):
        model = shared if reused == 'backend' else ScriptBackend([1])
        if reused != 'backend': setattr(model, reused, shared)
        return model
    with pytest.raises(ValueError, match='fresh'):
        run_iterations(factory, fit_context(), TrainingControl(max_steps=1, restarts=1))


@pytest.mark.parametrize('method,kwargs', [('snapshot', {}), ('restore', {}),
    ('get_learning_rate', {'scheduler': ReduceOnPlateau()}), ('set_learning_rate', {'scheduler': ReduceOnPlateau()})])
def test_missing_optional_protocol_capabilities_fail_clearly(method, kwargs):
    model = ScriptBackend([1]); setattr(model, method, None)
    with pytest.raises(TypeError):
        run_iterations(lambda seed: model, fit_context(), TrainingControl(max_steps=1, **kwargs))


def test_without_restoration_backend_needs_no_checkpoint_protocol():
    model = ScriptBackend([4, 1, 3]); model.snapshot = model.restore = None
    retained, _, summary = run_iterations(lambda seed: model, fit_context(), TrainingControl(max_steps=3, restore_best=False))
    assert retained.position == 3 and summary['monitored_value'] == 3


@pytest.mark.parametrize('metrics', [{'other': 1}, {'train_loss': [1]}, {'train_loss': 1, '': 2}])
def test_malformed_step_metrics_are_rejected(metrics):
    with pytest.raises(ValueError): run([metrics])


def test_arbitrary_backend_exception_is_not_treated_as_restartable():
    original = RuntimeError('scripted failure'); seeds = []
    def factory(seed):
        seeds.append(seed); return ScriptBackend([original])
    with pytest.raises(RuntimeError) as caught:
        run_iterations(factory, fit_context(), TrainingControl(max_steps=1, restarts=2))
    assert caught.value is original and seeds == [0]


@pytest.mark.parametrize('kwargs', [dict(max_steps=0), dict(max_steps=True), dict(restarts=-1),
    dict(seed=1.5), dict(direction='up'), dict(restore_best=1), dict(monitor=''),
    dict(early_stopping=EarlyStopping(0)), dict(early_stopping=EarlyStopping(1, -1)),
    dict(scheduler=ReduceOnPlateau(1, 1, .1)), dict(scheduler=ReduceOnPlateau(1, .5, 0))])
def test_invalid_controls_fail_before_training(kwargs):
    with pytest.raises(ValueError): TrainingControl(**kwargs)
