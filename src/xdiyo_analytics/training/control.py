"""Optional training controls and reusable iteration monitoring."""

from copy import deepcopy
from dataclasses import dataclass, replace
from numbers import Integral, Real

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EarlyStopping:
    """Stop after patience consecutive steps without improvement > min_delta."""
    patience: int = 10
    min_delta: float = 0.0


@dataclass(frozen=True)
class ReduceOnPlateau:
    """Multiply the backend's learning rate after patience stalled steps."""
    patience: int = 5
    factor: float = 0.5
    min_lr: float = 1e-6


@dataclass(frozen=True)
class TrainingControl:
    """Model-supported iteration control, not hyperparameter search or refitting.

    max_steps is per attempt. restarts adds fresh attempts with seeds seed+i;
    the best monitored attempt is retained. monitor=None chooses validation_loss
    when validation is supplied, otherwise train_loss. Both are minimized by
    default; direction='maximize' supports other explicitly named metrics.
    restore_best restores each attempt's best checkpoint. Without it, attempts
    are compared by their final finite metric and their final state is retained.
    A nonfinite monitor ends an attempt; usable earlier best states are retained
    only with restore_best. Arbitrary training errors propagate.
    """
    max_steps: int = 100
    early_stopping: EarlyStopping | None = None
    scheduler: ReduceOnPlateau | None = None
    restarts: int = 0
    seed: int = 0
    monitor: str | None = None
    direction: str = "minimize"
    restore_best: bool = True

    def __post_init__(self):
        for name, value, minimum in (("max_steps", self.max_steps, 1), ("restarts", self.restarts, 0), ("seed", self.seed, 0)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if self.direction not in {"minimize", "maximize"} or not isinstance(self.restore_best, bool):
            raise ValueError("direction must be minimize/maximize and restore_best boolean.")
        if self.monitor is not None and (not isinstance(self.monitor, str) or not self.monitor):
            raise ValueError("monitor must be a nonempty metric name or None.")
        for policy in (self.early_stopping, self.scheduler):
            if policy is not None and (isinstance(policy.patience, bool) or not isinstance(policy.patience, Integral) or policy.patience < 1):
                raise ValueError("Policy patience must be a positive integer.")
        if self.early_stopping is not None:
            delta = self.early_stopping.min_delta
            if not isinstance(delta, Real) or not np.isfinite(delta) or delta < 0:
                raise ValueError("min_delta must be finite and nonnegative.")
        if self.scheduler is not None:
            policy = self.scheduler
            if not (np.isfinite(policy.factor) and 0 < policy.factor < 1 and
                    np.isfinite(policy.min_lr) and policy.min_lr > 0):
                raise ValueError("Scheduler needs 0 < factor < 1 and positive finite min_lr.")


@dataclass(frozen=True)
class ValidationTail:
    """Reserve the latest fraction of distinct kickoff batches inside training.

    Whole matches and equal-time batches stay together. This is a pooled temporal
    tail, not an automatic competition calendar or gap policy. Explicit validation
    positions can be supplied instead for another prepared training-only split.
    """
    fraction: float = 0.2
    time_column: str = "kickoff_at"

    def select(self, dataset, train_positions):
        if isinstance(self.fraction, bool) or not isinstance(self.fraction, Real) or not 0 < self.fraction < 1:
            raise ValueError("Validation fraction must lie strictly between zero and one.")
        times = pd.to_datetime(dataset.metadata.iloc[train_positions][self.time_column], utc=True, errors="raise")
        if times.isna().any():
            raise ValueError("ValidationTail needs nonmissing timestamps.")
        batches = times.drop_duplicates().sort_values()
        size = int(np.ceil(len(batches)*self.fraction))
        if size >= len(batches):
            raise ValueError("ValidationTail must leave at least one fitting kickoff batch.")
        return np.asarray(train_positions)[times.ge(batches.iloc[-size]).to_numpy()]


@dataclass(frozen=True)
class TrainingEvent:
    """One completed optimization step. Metrics are scalar values, not a model."""
    fold_id: int
    attempt: int
    step: int
    metrics: dict
    monitor: str
    learning_rate: float | None = None
    final: bool = False


def run_iterations(backend_factory, context, control):
    """Shared loop used by IterativeAdapter and available to custom adapters.

    backend_factory(seed) returns a fresh backend exposing initialize(context),
    step() -> scalar metric dict and predict(PredictionContext). Checkpointing
    additionally requires snapshot()/restore(state); a scheduler requires
    get_learning_rate()/set_learning_rate(rate). The backend owns model/device
    details and learns preprocessing only from context.X/y. context.validation
    supplies separate monitoring data, never outer test rows.

    Returns (selected_backend, history_frame, summary). Observers receive
    TrainingEvent objects at completed steps and a final event per attempt.
    """
    monitor = control.monitor or ("validation_loss" if context.validation is not None else "train_loss")
    direction = 1 if control.direction == "minimize" else -1
    rows, attempts, seen, owned_states = [], [], [], []
    winner, winning_value, winner_info = None, np.inf, None
    delta = control.early_stopping.min_delta if control.early_stopping else 0.0
    for attempt in range(control.restarts+1):
        backend = backend_factory(control.seed+attempt)
        if any(backend is other for other in seen):
            raise ValueError("backend_factory must create a fresh backend for each attempt.")
        seen.append(backend)
        for name in ("estimator", "preprocessor"):
            state = getattr(backend, name, None)
            if state is not None:
                if any(state is other for other in owned_states):
                    raise ValueError("Restart backends must own fresh estimators and preprocessing.")
                owned_states.append(state)
        if control.restore_best and not all(callable(getattr(backend, method, None)) for method in ("snapshot", "restore")):
            raise TypeError("Checkpoint restoration requires backend snapshot()/restore().")
        if control.scheduler and not all(callable(getattr(backend, method, None)) for method in ("get_learning_rate", "set_learning_rate")):
            raise TypeError("Learning-rate scheduling requires backend get/set_learning_rate().")
        def copy_inputs(inputs):
            return replace(inputs, X=inputs.X.copy(deep=True), y=inputs.y.copy(deep=True),
                           metadata=inputs.metadata.copy(deep=True), definitions=deepcopy(inputs.definitions),
                           fold_metadata=deepcopy(inputs.fold_metadata),
                           validation=copy_inputs(inputs.validation) if inputs.validation is not None else None)
        backend.initialize(copy_inputs(context))
        best, best_step, checkpoint = np.inf, None, None
        progress_best, stale, schedule_stale = np.inf, 0, 0
        last_value, reason, last_metrics = np.inf, "max_steps", {}
        for step in range(1, control.max_steps+1):
            rate = float(backend.get_learning_rate()) if control.scheduler else None
            if rate is not None and (not np.isfinite(rate) or rate <= 0):
                raise ValueError("The backend learning rate must be positive and finite.")
            values = backend.step()
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
                break
            if control.scheduler and schedule_stale >= control.scheduler.patience:
                rate = float(backend.get_learning_rate())
                backend.set_learning_rate(max(min(rate, control.scheduler.min_lr), rate*control.scheduler.factor))
                schedule_stale = 0
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
    if winner is None:
        raise ValueError("No training attempt produced a finite monitored value.")
    summary = dict(monitor=monitor, direction=control.direction, restore_best=control.restore_best,
                   selected_attempt=winner_info["attempt"], best_step=winner_info["best_step"],
                   retained_step=winner_info["retained_step"], termination_reason=winner_info["termination_reason"],
                   monitored_value=winner_info["monitored_value"], attempts=attempts)
    return winner, pd.DataFrame(rows), summary


class IterativeAdapter:
    """ModelAdapter using the common training loop with an arbitrary backend.

    The runner supplies TrainingControl optionally; a direct fit uses its default
    bounded loop. Custom neural/graph backends can use the same protocol without
    adopting sklearn. This is an opt-in capability, never applied to plain fit().
    """
    def __init__(self, backend_factory):
        self.backend_factory = backend_factory

    def fit(self, context):
        self.fit_controlled(context, TrainingControl())

    def fit_controlled(self, context, control):
        self.backend_, self.training_history_, self.training_summary_ = run_iterations(
            self.backend_factory, context, control or TrainingControl())

    def predict(self, context):
        return self.backend_.predict(context)

    def coefficient_table(self):
        return self.backend_.coefficient_table()
