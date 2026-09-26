# Training controls API reference

This reference covers the preserved controls source revision recorded in
[training_controls_check.json](training_controls_check.json). The
[guide](training_controls.md) supplies executable examples and the
[equations](training_controls_equations.md) define decisions. Existing data,
prediction, selection and metric contracts remain in the
[training reference](training_reference.md) and
[post-training reference](post_training_reference.md).

## Public imports and compatibility

All control, backend, observer and refit names below are exported from
`xdiyo_analytics.training`. `LearningCurveReporter` and `CoefficientReporter`
are exported from `xdiyo_analytics.reporting`. Store and leaderboard options
extend the existing `ExperimentStore` and `ExperimentLeaderboardReporter`.

Plain `EstimatorAdapter` accepts no controlled fitting or validation interface.
Passing either to it raises before fitting. Its ordinary `fit` remains supported
and records native diagnostics after fitting. `IterativeAdapter` implements the
optional `fit_controlled(context, control)` method. With `control=None`, it uses
`TrainingControl()`; thus the adapter choice itself opts into the iteration loop.

## Policies and events

| API / argument | Default | Contract |
| --- | --- | --- |
| `EarlyStopping.patience` | `10` | Positive integer stalled steps before stopping. |
| `EarlyStopping.min_delta` | `0.0` | Finite nonnegative improvement threshold; equality does not reset patience. |
| `ReduceOnPlateau.patience` | `5` | Positive integer stalled steps before rate reduction. |
| `ReduceOnPlateau.factor` | `0.5` | Finite multiplier strictly between zero and one. |
| `ReduceOnPlateau.min_lr` | `1e-6` | Positive finite lower bound; an already lower rate is never raised. |
| `TrainingControl.max_steps` | `100` | Positive integer per-attempt limit. |
| `early_stopping` | `None` | Optional `EarlyStopping`; otherwise no patience-based stop. |
| `scheduler` | `None` | Optional `ReduceOnPlateau`; otherwise backend rate management. |
| `restarts` | `0` | Nonnegative integer extra fresh attempts. |
| `seed` | `0` | Nonnegative integer; attempt i receives seed+i. |
| `monitor` | `None` | Nonempty metric key; automatic validation/train loss selection when omitted. |
| `direction` | `'minimize'` | `'minimize'` or `'maximize'`; applies to progress and attempt selection. |
| `restore_best` | `True` | Boolean; restore each attempt's exact best snapshot before comparing attempts. |
| `ValidationTail.fraction` | `0.2` | Strictly between zero and one; fraction of distinct UTC kickoff batches. |
| `ValidationTail.time_column` | `'kickoff_at'` | Metadata timestamp column, required and nonmissing. |

`TrainingControl.__post_init__` validates policy fields when assembling controls;
the simple `EarlyStopping` and `ReduceOnPlateau` records do not validate themselves
at construction. Integer budget/seed/patience arguments reject booleans.
`ValidationTail.select(dataset, train_positions)` validates its fraction on use,
reserves the latest ceiling-rounded number of distinct timestamps, and leaves at
least one fitting batch. It preserves input row order. This is a pooled temporal
tail, without automatic competition calendars or embargoes.

`TrainingEvent(fold_id, attempt, step, metrics, monitor, learning_rate=None,
final=False)` describes one completed step. Attempts are zero-based and steps
one-based. Metrics are a scalar mapping. `learning_rate` is the rate used for that
step when the external scheduler is enabled. One final event follows each
attempt, using its last observed metrics and step, with no extra history row.
That event does not substitute the restored best metric.

## Iteration loop and backend protocol

`run_iterations(backend_factory, context, control)` returns
`(retained_backend, history_dataframe, summary_dict)`. Its nested `copy_inputs`
helper copies fitting and
validation frames plus nested metadata/definitions for each attempt. The observer
callback remains shared. Backend factories receive a seed and must create fresh
state; reused backend, estimator or preprocessor identities across attempts are
rejected. Callers remain responsible for freshness across separate runs.

| Backend method | Required when | Meaning |
| --- | --- | --- |
| `initialize(context)` | Always | Initialize from copied fitting context and optional validation context. |
| `step()` | Always | Perform one update; return a dict of named scalar metrics including the monitor. |
| `predict(context)` | Adapter prediction | Return named DataFrames with the exact requested row index. |
| `snapshot()` | `restore_best=True` | Return an independent restorable state after an exact improvement. |
| `restore(state)` | `restore_best=True` | Restore that state before attempt comparison. |
| `get_learning_rate()` | Scheduler enabled | Return a positive finite scalar used for the next update. |
| `set_learning_rate(value)` | Scheduler enabled | Apply the requested next-update rate. |
| `coefficient_table()` | Coefficient inspection | Return the validated long-form table described below. |

Metric values are converted to floats. A missing monitor, malformed metrics or
missing required methods raises. NaN or infinity in the monitored value ends the
attempt. An earlier finite best is usable only when `restore_best=True`.
All-nonfinite unusable attempts raise after all attempts; arbitrary exceptions
propagate immediately. Exact-value ties preserve the earlier step/attempt.
Early stopping is checked before a rate reduction at the same step. Significant
progress resets both stale counters; a rate reduction resets its own counter.

`IterativeAdapter(backend_factory)` exposes `fit`, `fit_controlled`, `predict` and
`coefficient_table`. It retains `backend_`, `training_history_` and
`training_summary_`. `fit` delegates to controlled fitting with default controls.
There is no persisted checkpoint, resume-from-disk or continuation API: snapshots
are in-memory restoration points; restarts begin fresh.

## PartialFitBackend

| Argument | Default | Meaning |
| --- | --- | --- |
| `estimator` | Required | Fresh estimator exposing `partial_fit`. |
| `loss` | `'mse'` | Existing metric specification; exactly one scalar target/metric result. |
| `preprocessor` | `None` | Fresh fit/transform object; fitted once on fitting rows per attempt. |
| `partial_fit_kwargs` | `None` | Keyword mapping forwarded to each update. |
| `prediction_methods` | `('predict',)` | Named estimator prediction methods returned at prediction time. |

`initialize` supports exactly one selected target. Classifier classes default to
the unique fitting labels; validation/test labels are not used to discover them.
Pass an explicit `classes` value through `partial_fit_kwargs` when the application
has a justified complete class vocabulary. Probability methods required by the
loss are added internally for metric calculation. `step` runs one full-population
`partial_fit`, then calculates `train_loss` and optional `validation_loss` through
the existing metric layer. Loss eligibility and missing-value rules remain those
of [evaluate_metrics](post_training_reference.md).

`_transform` is the private preprocessing helper. DataFrame output must preserve
the index; array output uses `get_feature_names_out` when available, otherwise
`transformed_i` names; sparse outputs become sparse DataFrames. `_outputs` creates
the metric prediction mapping. `predict` transforms inputs and uses
`EstimatorAdapter`'s named-output conversion. `snapshot`/`restore` deep-copy the
estimator; the fixed preprocessor is shared within that attempt. `steps_` counts
updates performed, so use summary `retained_step` for the restored model step.
Rate access requires an estimator with `learning_rate='constant'` and `eta0`;
native schedules such as `invscaling` reject the external scheduler. The supplied
estimator still owns its update rule, regularization and randomness.

## Runner, contexts and feature selection

`TrainingRunner(model_factory, feature_columns=None, target_columns=None,
control=None, validation=None, observer=None)` adds the last three options to
fixed-configuration training. `model_factory()` returns a fresh adapter.
Column selections are explicit names or all available columns when `None`.
Validation accepts positions, an object with `select(dataset, train_positions)`,
or a runner-only dict keyed by original fold ID. Missing dict entries raise.
An observer is a callable taking `TrainingEvent`; it does not enable controls on
an ordinary adapter by itself.

`run(dataset, split_plan, *, fold_ids=None, analysis_report=None,
features_from=None)` retains original fold IDs and consumes a named feature
selector result from the supplied analysis report. `selection_plan(dataset,
split_plan)` returns a copied split plan whose training arrays contain fitting
rows only. Run the selector on this derived plan, then pass its report and study
name to `run` with the original plan. Consuming a selector calculated on validation
rows raises. Test/score populations, metadata and path definitions are retained.

`split_training_rows(dataset, train_positions, validation=None)` returns
`(fit_positions, validation_positions)`. Positions must be unique, one-dimensional
valid integer locations; validation must be a subset of development and fitting
must remain nonempty. Both populations preserve complete matches. It does not
change the dataset or infer information availability.

`fit_predict(dataset, fold, model_factory, *, fold_id=0, feature_columns=None,
target_columns=None, validation=None, control=None,
observer=None)` exposes the same optional fitting controls for one prepared fold.
The runner handles named selector consumption and compatibility checks.

| Record | Added fields / behavior |
| --- | --- |
| `FitContext` | `validation=None`: separate FitContext; `observer=None`: optional callback. `X/y/metadata` contain fitting rows. |
| `PredictionContext` | Remains target-free; no observed `y` is supplied to prediction. |
| `FoldResult` | `train_positions` is the full declared development scope. `fit_positions=None`, `validation_positions=None`, `training_history=empty DataFrame`, `training_summary={}` are backward-compatible additions. |
| `TrainingResult` | Contains those fold records; existing indexed prediction access and scoring behavior remain intact. |
| `PostTrainingContext` | `fold_results` maps selected original fold IDs to their retained records for model diagnostics; treat models/records as read-only. |

Private runner helpers `_columns`, `_positions` and `_frame` check selection,
integer positions and copying/indexing. `_fit_model` constructs copied fit and
validation contexts, validates development targets, selects controlled or ordinary
fit, and returns the adapter plus fit/validation membership. `_validation_for`
resolves runner dicts. Existing held-out identity and whole-match checks remain.

## Histories and model inspection

Controlled history columns are `fold_id`, `attempt`, `step`, `metric`, `value`,
`learning_rate`. Every performed step is retained even when an earlier snapshot
is restored. Summary fields are `monitor`, `direction`, `restore_best`,
`selected_attempt`, `best_step`, `retained_step`, `termination_reason`,
`monitored_value`, and `attempts`. Each attempt records its index, seed, performed
steps, best/retained step, usable monitored value and termination reason.
Reasons are `max_steps`, `early_stopping` and `nonfinite_monitor`.

`estimator_history(estimator, fold_id)` is an internal inspection helper called by
ordinary `EstimatorAdapter.fit`. It reads the final pipeline estimator's
`loss_curve_` as `train_loss` and `evals_result_` as `partition.metric`. It records
available `n_iter_` and `best_iteration_`, with `history_source` equal to
`native_estimator` or `unavailable` and `termination_reason='estimator_managed'`.
These native step counts/metrics keep their estimator-specific meaning. It never
reconstructs an optimization curve from predictions.

`estimator_coefficients(estimator, feature_names, target_names)` is the internal
coefficient extractor exposed through `EstimatorAdapter.coefficient_table` and
`PartialFitBackend.coefficient_table`. It reads `coef_` and `intercept_` from the
fitted estimator, including sparse coefficients, pipeline-transformed feature
names, multiple regression outputs and classifier labels. Pipeline transforms
must expose output names; original feature names are never guessed after an
unidentified transformation. Missing `coef_` raises TypeError; dimensions and
distinct feature names must agree. An absent intercept attribute yields zeros.

## Model reporters

Both reporters require keyword `type='per_fold'` or `'overall'` and default to
`partition='model'`; other types/partitions raise. Their `run(context)` returns
`StudyResult` tables, artifacts and availability notes. `PostTrainingAnalysis.run`
builds a model context with selected models/fold records and empty prediction,
target and row-scope frames; prediction pooling is irrelevant here. These reports
can run on a result containing fitted models and no prediction population.

| Reporter option | Default | Behavior |
| --- | --- | --- |
| `LearningCurveReporter.metrics` | `None` | All metrics; a string or sequence filters names. |
| `LearningCurveReporter.attempts` | `None` | All attempts; `'selected'` uses each fold's selected attempt, or supply attempt IDs. |
| `CoefficientReporter.tolerance` | `0.0` | Finite nonnegative threshold; boolean rejected. |
| `CoefficientReporter.include_zeros` | `False` | Include all feature bars when true; otherwise only coefficients exceeding tolerance. |
| `CoefficientReporter.survival_frequency` | `True` | Add across-inspected-fold frequency table. |

Learning-curve tables are `summary`, `attempts` and `history`. Unequal histories
remain separate traces for each fold, attempt and metric. Nonfinite values create
gaps. Filtering the plotted history does not rewrite retained training records.
Missing curves produce a note and summary rather than a fabricated line.

Coefficient tables are `coefficients` (all terms), `surviving_coefficients`,
`intercepts` and optional `survival_frequency`. A custom adapter's
`coefficient_table()` must return a DataFrame with `target`, `label`, `feature`,
`coefficient`, `term`; term is `feature` or `intercept`. Each
target/label/feature/term combination must be unique and all coefficients finite.
The reporter adds `fold_id` and `survives`. Missing method or TypeError/AttributeError
means unavailable; other errors remain visible. Bars retain signed values and
sort by absolute magnitude in fitted units. Frequency reports `folds_present`,
`folds_survived`, `folds_inspected` and `survival_frequency` for each target/class
and feature. Every inspected output counts in the denominator, including outputs
containing only an intercept; an unsupported/uninspected output does not count.

## Optional live view

`LiveLossPlot(*, interval=1.0, metrics=('train_loss', 'validation_loss'),
max_points=500)` accepts a positive finite interval and integer point budget >=2.
`metrics` is a name or sequence. `__call__(event)` records selected nonfinal event
metrics and refreshes one IPython display handle on the interval or any final event.
`to_html()` returns the current static SVG. It escapes names, keeps folds/attempts
separate, preserves nonfinite gaps even during downsampling, and shows a waiting
message if no finite points exist. `series` retains all observed selected points.
Rendering has no server, widget or JavaScript requirement; frontend display-update
support remains necessary for live use. No live frontend is claimed by this batch.

## Explicit final refit

`refit_model(dataset, model_factory, *, train_positions, feature_columns=None,
target_columns=None, validation=None, control=None, observer=None)` returns
`FittedModel`. Development positions are mandatory. Validation/control semantics
match `_fit_model`; pass the already selected feature names explicitly. It fits
fresh preprocessing and model state. No model selection, evaluation or future
scheduling is implicit.

`FittedModel` retains `model`, `train_positions`, `fit_positions`,
`validation_positions`, `feature_columns`, `target_columns`, `layout`,
`identity_columns`, `match_columns`, `target_perspective` and copied `definitions`.
`predict(dataset, *, positions=None)`
uses all rows when omitted, requires compatible layout/match columns and selected
features, and returns copied named DataFrames with exact prediction indexes.

`evaluate(dataset, *, test_positions, score_positions=None, same_dataset=True)`
returns one-fold `TrainingResult` without fitting. Score defaults to all supplied
test rows and may be empty; score must be a subset of test. Both preserve whole
matches. Same-dataset evaluation excludes all declared development rows. For a
different dataset, the caller explicitly sets `same_dataset=False`: local
train/fit/validation arrays are empty, with original development positions stored
in fold metadata. Evaluation does not prove that two datasets are independent.

## Saved groups, roles and leaderboard

| Method / new argument | Default | Contract |
| --- | --- | --- |
| `ExperimentStore.start_run(name, *, config=None)` | — | Create a named UUID group manifest; no computation or selection. |
| `save_run(..., role=...)` | `'final'` | `'final'` or `'trial'`. Existing required training/report/name/config and save flags remain. |
| `save_run(..., run_group=...)` | `None` | Trials require an explicit existing group; standalone finals use their own run ID as group. |
| `save_run(..., selected_trial_id=...)` | `None` | Final-only link to a successful trial in the same explicit group. |
| `save_failure(..., role=..., run_group=...)` | `'final'`, `None` | Explicit failed record; same group/role rules. No selected-trial argument. |
| `read_runs(*, role=None, run_group=None)` | All published runs | Filter optionally; legacy records without roles count as final and use run ID as group. |
| `ExperimentLeaderboardReporter.include_trials` | `False` | Finals only unless explicitly enabled. |
| `ExperimentLeaderboardReporter.run_group` | `None` | Optional exact group filter before unchanged numerical ranking. |

One explicit group permits one final record, including a failed final. Callers
serialize same-group writes. A selected-trial link copies no metric or model.
Legacy defaults apply to filtering/display without rewriting old JSON. Leaderboard
tables add `role`, `run_group`, `selected_trial_id`; required metric compatibility,
scaling and missing-value rules remain in the post-training reference.

Every successful save writes `training.json` as a list of per-fold records,
including when predictions are disabled: development/fit/validation positions,
selected columns, fold metadata, summary and long-form history. Ordinary histories
retain pandas JSON table format; histories with MultiIndex columns use
`{"encoding": "xdiyo.data-only.v1", "value": ...}` containing a data-only frame.
JSON-native fold metadata stays unchanged. Other fold metadata uses a data-only
representation in `fold_metadata`, with sibling
`fold_metadata_encoding: "xdiyo.data-only.v1"`; this preserves integer block keys,
arrays, timestamps and durations. `ExperimentStore.load_run` and
`FootballExperiment.load` automatically read both original and tagged forms,
including runs without `recovery.json`.

Missing legacy fit membership falls back to development rows. Nonfinite numbers
in ordinary diagnostics serialize as null; tagged fold metadata retains its
explicit missing-value representation. Summaries and optional Parquet/HTML retain
prior behavior. Models and resumable checkpoints are not serialized. Private
`_start` validates role/group and trial linkage before staging; configuration
`_json` validation remains strict. `_write` and `_publish` retain exclusive writes
and bounded Windows publication retry behavior.

## Scope and evidence

The new tests cover independent stopping/rate decisions, exact SGD updates,
validation leakage boundaries, restored models, coefficient calculations,
intercept-only survival denominators, refit populations, persisted roles and live
SVG geometry. See the [coverage checklist](training_controls_documentation_checklist.md).
Grid search, Optuna, automated new-match retraining, persisted resume and real
football performance are outside this increment.
