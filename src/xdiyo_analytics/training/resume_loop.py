"""Opt-in resumable equivalent of the native iteration loop.

Keep step/observer/selection/scheduler order in parity with control.run_iterations.
Checkpoint offers occur only after a completed update and its metric evaluation.
"""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pandas as pd
from .._identity import IdentityRegistry
from .control import TrainingEvent
from .native_resume import capture_backend, restore_backend, rng_state, restore_rng, _cpu


def run_resumable_iterations(backend_factory, context, control, *, state=None, save):
    """Shared loop used by IterativeAdapter and available to custom adapters.

    backend_factory(seed) returns a fresh backend exposing initialize(context),
    step() -> scalar metric dict and predict(PredictionContext). Checkpointing
    additionally requires snapshot()/restore(state); a scheduler requires
    get_learning_rate()/set_learning_rate(rate). The backend owns model/device
    details and learns preprocessing only from context.X/y. context.validation
    supplies separate monitoring data, never outer test rows.

    Metrics used for selection must describe the state returned by snapshot().
    A backend that measures loss while computing gradients may implement
    checkpoint_metrics(monitor, values) to re-evaluate training monitors after
    the update, before observers, scheduling, selection and checkpointing.

    Returns (selected_backend, history_frame, summary). Observers receive
    TrainingEvent objects at completed steps and a final event per attempt.
    """
    monitor = control.monitor or ("validation_loss" if context.validation is not None else "train_loss")
    direction = 1 if control.direction == "minimize" else -1
    rows, attempts = [], []
    seen, owned_states = IdentityRegistry(), IdentityRegistry()
    winner, winning_value, winner_info = None, np.inf, None
    delta = control.early_stopping.min_delta if control.early_stopping else 0.0
    def copy_context(inputs):
        return replace(inputs, X=inputs.X.copy(deep=True), y=inputs.y.copy(deep=True),
                       metadata=inputs.metadata.copy(deep=True), definitions=deepcopy(inputs.definitions),
                       fold_metadata=deepcopy(inputs.fold_metadata), sample_weight=deepcopy(inputs.sample_weight),
                       validation=copy_context(inputs.validation) if inputs.validation is not None else None)
    if state is not None:
        rows, attempts = state['rows'], state['attempts']
        winning_value, winner_info = state['winning_value'], state['winner_info']
        if state['winner'] is not None:
            winner = backend_factory(control.seed+winner_info['attempt'])
            winner.initialize(copy_context(context))
            restore_backend(winner, state['winner'])
        restore_rng(state['rng'])
    def publish(attempt, active=None):
        save(dict(attempt=attempt, restarts=control.restarts, rows=deepcopy(rows), attempts=deepcopy(attempts),
                  winner=None if winner is None else capture_backend(winner), winning_value=winning_value,
                  winner_info=deepcopy(winner_info), active=active,
                  rng=rng_state(backend.device)))
    for attempt in range(0 if state is None else state['attempt'], control.restarts+1):
        backend = backend_factory(control.seed+attempt)
        if backend in seen:
            raise ValueError("backend_factory must create a fresh backend for each attempt.")
        seen.add(backend)
        for name in ("estimator", "preprocessor"):
            owned = getattr(backend, name, None)
            if owned is not None:
                if owned in owned_states:
                    raise ValueError("Restart backends must own fresh estimators and preprocessing.")
                owned_states.add(owned)
        if control.restore_best and not all(callable(getattr(backend, method, None)) for method in ("snapshot", "restore")):
            raise TypeError("Checkpoint restoration requires backend snapshot()/restore().")
        if control.scheduler and not all(callable(getattr(backend, method, None)) for method in ("get_learning_rate", "set_learning_rate")):
            raise TypeError("Learning-rate scheduling requires backend get/set_learning_rate().")
        if context.sample_weight is not None and not getattr(backend, "supports_sample_weight", False):
            raise TypeError("Weighted iterative fitting requires a backend declaring supports_sample_weight.")
        def copy_inputs(inputs):
            return replace(inputs, X=inputs.X.copy(deep=True), y=inputs.y.copy(deep=True),
                           metadata=inputs.metadata.copy(deep=True), definitions=deepcopy(inputs.definitions),
                           fold_metadata=deepcopy(inputs.fold_metadata), sample_weight=deepcopy(inputs.sample_weight),
                           validation=copy_inputs(inputs.validation) if inputs.validation is not None else None)
        backend.initialize(copy_inputs(context))
        best, best_step, checkpoint = np.inf, None, None
        progress_best, stale, schedule_stale = np.inf, 0, 0
        last_value, reason, last_metrics = np.inf, "max_steps", {}
        first_step = 1
        if state is not None and attempt == state['attempt'] and state['active'] is not None:
            active = state['active']
            restore_backend(backend, active['backend'])
            best, best_step, checkpoint = active['best'], active['best_step'], active['checkpoint']
            progress_best, stale, schedule_stale = active['progress_best'], active['stale'], active['schedule_stale']
            last_value, reason, last_metrics = active['last_value'], active['reason'], active['last_metrics']
            step = active['step']
            first_step = control.max_steps+1 if active['stopped'] else step+1
            restore_rng(state['rng'])
            state = None
        def offer(stopped=False):
            publish(attempt, dict(backend=capture_backend(backend), best=best, best_step=best_step,
                    checkpoint=_cpu(checkpoint), progress_best=progress_best, stale=stale,
                    schedule_stale=schedule_stale, last_value=last_value, reason=reason,
                    last_metrics=deepcopy(last_metrics), step=step, stopped=stopped))
        for step in range(first_step, control.max_steps+1):
            rate = float(backend.get_learning_rate()) if control.scheduler else None
            if rate is not None and (not np.isfinite(rate) or rate <= 0):
                raise ValueError("The backend learning rate must be positive and finite.")
            values = backend.step()
            checkpoint_metrics = getattr(backend, 'checkpoint_metrics', None)
            if callable(checkpoint_metrics):
                values = checkpoint_metrics(monitor, values)
            if not isinstance(values, dict) or monitor not in values:
                raise ValueError(f"Backend step must expose monitored metric {monitor!r}.")
            if any(not isinstance(name, str) or not name or not np.isscalar(value) for name, value in values.items()):
                raise ValueError("Backend step metrics must be named scalars.")
            values = {name: float(value) for name, value in values.items()}
            for name, value in values.items():
                rows.append(dict(fold_id=context.fold_id, attempt=attempt, step=step, metric=name,
                                 value=value, learning_rate=rate))
            last_metrics = values
            event = TrainingEvent(context.fold_id, attempt, step, values.copy(), monitor, rate)
            if context.observer is not None:
                context.observer(event)
            last_value = direction*values[monitor]
            if not np.isfinite(last_value):
                reason = "nonfinite_monitor"
                offer(stopped=True)
                break
            if last_value < best:
                best, best_step = last_value, step
                if control.restore_best:
                    checkpoint = backend.snapshot()
            # Significant progress governs patience; exact best governs checkpoint.
            if last_value < progress_best-delta:
                progress_best, stale, schedule_stale = last_value, 0, 0
            else:
                stale += 1
                schedule_stale += 1
            if control.early_stopping and stale >= control.early_stopping.patience:
                reason = "early_stopping"
                offer(stopped=True)
                break
            if control.scheduler and schedule_stale >= control.scheduler.patience:
                rate = float(backend.get_learning_rate())
                backend.set_learning_rate(max(min(rate, control.scheduler.min_lr), rate*control.scheduler.factor))
                schedule_stale = 0
            offer()
        if control.restore_best and best_step is not None:
            backend.restore(checkpoint)
        retained_value = best if control.restore_best else last_value
        retained_step = best_step if control.restore_best else step
        info = dict(attempt=attempt, seed=control.seed+attempt, steps=step, best_step=best_step,
                    retained_step=retained_step, monitored_value=(direction*retained_value if np.isfinite(retained_value) else None),
                    termination_reason=reason)
        attempts.append(info)
        if np.isfinite(retained_value) and retained_value < winning_value:
            winner, winning_value, winner_info = backend, retained_value, info.copy()
        if context.observer is not None:
            context.observer(TrainingEvent(context.fold_id, attempt, step, last_metrics.copy(), monitor, final=True))
        publish(attempt+1)
        state = None
        del backend, checkpoint
        checkpoint_metrics = None
    if winner is None:
        raise ValueError("No training attempt produced a finite monitored value.")
    summary = dict(monitor=monitor, direction=control.direction, restore_best=control.restore_best,
                   selected_attempt=winner_info["attempt"], best_step=winner_info["best_step"],
                   retained_step=winner_info["retained_step"], termination_reason=winner_info["termination_reason"],
                   monitored_value=winner_info["monitored_value"], attempts=attempts)
    return winner, pd.DataFrame(rows), summary

