# Model-selection API and helper reference

The public module is `xdiyo_analytics.selection`. It exports the twelve names
below. This reference describes the frozen implementation verified in
[model_selection_check.json](model_selection_check.json); source ownership
remains with the primary implementation task.

## Candidate definitions

### Candidate

`Candidate(name, model_factory, config={}, feature_columns=None, target_columns=None, pre_analysis=None, features_from=None, control=None, validation=None, observer=None, complexity=None, fit_statistics=None)`

| Argument | Contract |
|---|---|
| `name` | Nonblank string; distinct within one search. |
| `model_factory` | Zero-argument callable returning a fresh adapter and fresh fitted preprocessing/backend state. |
| `config` | Explicit description mapping accepted by `configuration_hash`. Finite primitives, string-keyed mappings, sequences, supported dates/paths/dataclasses are canonicalized. Arbitrary objects and callable representations are rejected. Describe all meaningful model, features, controls, validation and seed choices. |
| `feature_columns` | Fixed input column selection passed to `TrainingRunner`; `None` uses available columns. Mutually exclusive with `features_from`. |
| `target_columns` | Selected target names passed to the runner; `None` uses all. |
| `pre_analysis` | Optional `PreTrainingAnalysis`; every reporter must have `partition="train"`. Deep-copied and rerun on the runner's fitting populations. Use per-fold studies for prospective inner-fold evaluation. |
| `features_from` | Name of a selector study in `pre_analysis`; its selected columns feed fitting. Requires that study and excludes fixed `feature_columns`. |
| `control` | Optional existing `TrainingControl`, passed through to the runner and compatible adapter. |
| `validation` | Optional explicit positions, fold-to-positions mapping, or object with `select(dataset, train_positions)`. Explicit positions supplied to `run` use original dataset indices and are mapped locally. Custom selectors receive the development-only dataset and must return local rows. Validation must satisfy the runner's train-subset, whole-match and nonempty-fit requirements. |
| `observer` | Optional existing training observer passed through. A factory should avoid sharing hidden mutable fitted state. |
| `complexity` | Optional `callback(fitted_adapter) -> mapping` of nonempty string names to finite nonnegative measures. Called after each fit. Arithmetic means across all inner fits are retained; a missing measure in one fit stays missing in the aggregate. |
| `fit_statistics` | Optional `callback(fitted_adapter) -> FitStatistics`. When information criteria are requested, this overrides the adapter's `fit_statistics()` method. No call occurs otherwise. |

`Candidate.__post_init__()` validates name/factory/config and the basic
`features_from` dependency. Preparation scope is checked before candidate error
recording and again at fitting. Runtime estimator/numerical errors can still
arise later.

### GridCandidates

`GridCandidates(parameters, build)` is a finite iterable.

- `parameters`: one mapping, or an iterable of conditional mappings. Keys are
  strings; each value is a nonempty iterable of choices. String/bytes/dict values
  are not choice sequences. An empty mapping yields one configuration.
- `build(parameters)`: returns a `Candidate`; the builder receives a deep copy.
  Parameter application remains the builder's job.
- `__iter__()`: generates ordered Cartesian products, copies parameters into
  `config.grid_parameters`, and appends ` [1]`, ` [2]`, etc. to names.
  Use reusable lists/tuples for grids needed more than once; embedded one-shot
  iterables can be consumed.

A normal finite candidate iterable is also accepted. Callable sources returning
fresh iterables support repeated runs. An ordinary one-shot source passed to
`run` is consumed; `run_nested` materializes a noncallable source once so every
outer fold sees its configurations.

## Decision rules

Each rule implements `decide(trials) -> SelectionDecision`. It reads retained
numerical evidence and does not fit models.

| Class/signature | Options and result |
|---|---|
| `SelectionDecision(winner_id, table)` | Trial ID and reusable DataFrame containing `run_id`. A custom rule must choose a successful trial; it owns its calculation, comparability and table semantics. |
| `MetricSelection(metric, direction=None, selector={})` | Select best scalar metric. Default selector chooses the named overall record. `selector` can distinguish `target`, `study`, `output`, `partition`, `type` or `fold_id`. Direction comes from the record unless overridden by `"minimize"`/`"maximize"`. Exact ties follow candidate order. |
| `WeightedSelection(weights, selectors={}, scaling="percentile", reference_scales={}, directions={})` | Named nonnegative weights, at least one positive; normalized over positive entries. Per-key selectors and direction overrides; `"percentile"` is relative to comparable candidates, `"fixed"` needs finite increasing `(low, high)` bounds for each positive key. Raw/utility/contribution columns remain visible. |
| `ParsimonySelection(metric, direction=None, selector={}, complexity="n_parameters", tolerance=0.0)` | Starts from the scalar metric, retains candidates within the finite nonnegative absolute tolerance, then minimizes the named aggregate complexity. Remaining ties prefer better metric, then candidate order. Missing complexity excludes the candidate from this choice. |

The built-in rules reuse `rank_runs`. Every positive-weight metric must resolve
to one finite, status-`ok` record with a sample hash. Ambiguous records raise.
Missing, partial or undefined metrics leave candidates unranked; failed trials
remain visible. Multiple eligible comparison groups raise instead of choosing
across incompatible samples. The comparison identity includes metric,
calculation, target, output, parameters, sample hash, execution type/partition,
fold, layout, scope label and direction.

Common comparison columns are `run_id`, `name`, `config_hash`, `status`,
`comparison_group`, `score`, `rank`, `raw::<key>`, `utility::<key>`,
`contribution::<key>`, `selected`, `candidate_order`; the search adds
`saved_run_id` and `error`. Parsimony adds `complexity` and
`within_tolerance`. Its selected winner can have a raw metric rank above 1.
Only metrics used by the rule become its raw columns.

## Fitted evidence

`FitStatistics(log_likelihood, n_parameters, n_observations, likelihood_id)`
is a frozen record. `information_criteria(statistics)` returns a dict with
`aic` and `bic`.

| Field | Meaning and validation |
|---|---|
| `log_likelihood` | Finite real fitted, unpenalized data log likelihood; include constants needed for comparison. |
| `n_parameters` | Finite nonnegative real count/effective degrees of freedom under the adapter's justified convention, including appropriate nuisance/intercept terms. |
| `n_observations` | Positive integer-valued finite real sampling-unit count. It need not equal feature-row count; the provider declares the convention. |
| `likelihood_id` | Nonempty string describing response units, normalization and parameter-count convention; compatible model families use the same convention ID. |

Booleans and nonfinite values are rejected. The helper requires a
`FitStatistics` instance. Prediction loss, a regularization penalty and
nonzero coefficient count do not automatically supply these fields.
Provider mathematical applicability cannot be proved by schema validation.
See [equations](model_selection_equations.md#fitted-likelihood-evidence).

## Search orchestration

### ModelSelection

`ModelSelection(candidates, metrics=(), decision=None, pooling=None, information_criteria=False, on_error="raise", evidence_reporters={})`

| Argument | Behavior |
|---|---|
| `candidates` | Finite candidate iterable or zero-argument callable producing one. Names must be unique per run. |
| `metrics` | A registered name, a `Metric` request, or a sequence. Computed per fold and once over pooled score predictions. With one request and no explicit decision, its name/key becomes `MetricSelection`; multiple targets can still require a disambiguating selector. |
| `decision` | Object implementing `decide(trials)`; required unless exactly one prediction metric is requested. IC-only and custom-evidence-only searches require an explicit rule. |
| `pooling` | `None`, `"occurrences"`, `"first"`, `"last"` or `"mean"`. Repeated score rows require an explicit policy. First/last follow retained fold order; mean needs numeric outputs and preserves missing cells. It is not a class-label voting rule. |
| `information_criteria` | If true, request fitted likelihood statistics, retain each fit, and expose arithmetic mean per-fit AIC/BIC. Fit response identities, conventions and counts participate in comparability. |
| `on_error` | `"raise"` or `"record"`. Applies to exceptions raised during candidate fitting or numerical evidence calculation. Declarative scope/preparation, final-decision and storage errors propagate. |
| `evidence_reporters` | Explicit extra reporter mapping; every reporter must use `partition="score"`. Names `selection_metrics`, `fold_metrics`, `fit_evidence` are reserved. Tables named `metrics` use the shared numerical schema. |

`run(dataset, split_plan, *, development_positions, experiment=None, run_group=None, name="Model selection")`

- Requires aligned `ModelDataset`, a nonempty original-indexed `SplitPlan`
  with the original row count, and a nonempty unique development population.
  Development and train/test/score partitions retain whole matches. Inner
  train/test are disjoint; score is a test subset; every inner row is in development.
- Copies development frames/definitions, resets local indices, maps split and
  explicit validation positions, runs fresh candidates, and restores original
  row positions in retained evidence.
- Reuses `TrainingRunner.selection_plan` and `TrainingRunner.run`. Monitoring
  validation is excluded from fitted preprocessing and consumed feature selection.
  Scope-only validation does not certify chronology: choose a suitable splitter.
- Needs at least prediction metrics, requested IC, or explicit extra evidence.
  Extra reporter tables and fitted evidence remain available on each trial.
- `experiment`: optional `ExperimentStore`. If no `run_group` is supplied,
  creates one named group recording development positions. Passing a group
  without a store raises. Successful and failed candidates save as `role="trial"`.
  An existing group must satisfy the store's contract.
- Returns `SelectionResult`. No final run, deployment fit or model serialization
  is implicit. Fitting failure in raise mode is saved before propagating when a
  store is supplied; persistent store errors propagate in either policy.

`run_nested(dataset, outer_plan, inner_plan_factory, *, experiment=None, name="Nested selection")`

- Requires a nonempty original-indexed outer plan with the dataset row count.
  For each outer fold, the factory sees only a copied local outer-training
  dataset and must return a `SplitPlan` for its local row count.
- Maps that plan to original rows, invokes search, then freshly evaluates that
  fold's winner. Original outer test/score identities are retained.
- Returns `NestedSelectionResult`; `training` contains outer predictions only.
  Fitted fold/history/selector IDs correspond to outer folds. Metadata also
  records `outer_fold_id`.
- A shared freshness guard spans outer iterations. Callable candidate sources
  are invoked for each search; noncallable finite sources are materialized once.
- Optional storage creates separate trial groups. One final nested record can
  be saved explicitly, with each outer winner in its configuration. There is
  no global winner inferred from outer outcomes.

## Result records

### TrialResult

`TrialResult(trial_id, candidate, record, training=None, report=None, complexity={}, saved_run_id=None, error=None)`

The in-memory UUID identifies the decision table. `candidate` is its definition;
`record` contains status/config/hash/numerical metrics; `training` retains
fitted inner models and original-indexed predictions; `report` contains
numerical studies plus `fit_evidence` tables `complexity` and
`fit_statistics` (and IC `metrics` when requested).
`complexity` is the aggregate mapping. `saved_run_id` is the store's separate
run UUID or None. `error` contains the exception type/message on failure.
A numerical failure after fitting may leave training evidence on the failed
trial; its metric record is cleared and it cannot win.

### SelectionResult

`SelectionResult(trials, decision, development_positions, n_rows, run_group=None, layout="selection", n_matches=0)`

| Member | Behavior |
|---|---|
| `trials`, `decision` | Retained candidate evidence and final selection decision. |
| `development_positions`, `n_rows` | Original development row positions and original dataset row count. |
| `run_group`, `layout`, `n_matches` | Optional saved search group, dataset layout and number of development matches. |
| `winner` | Returns the trial named by the decision's winner ID. |
| `comparison` | Deep copy of the decision table. |
| `to_report()` | One overall selection-scoped comparison study; scope table lists all development positions. Reuses `AnalysisReport.to_html/show/to_notebook`. Notes distinguish internal selection from final evaluation. |
| `evaluate(dataset, fold, *, validation=UNSET, control=UNSET)` | Freshly fits the winner on the outer train and predicts the outer test. Dataset row count must match, train must lie in development, and every test row must lie outside development. Returns a `TrainingResult`. Omitted overrides inherit candidate settings; explicit `None` removes them. |

The evaluation method guards against reuse of retained inner fitted state.
It reruns feature selection and preprocessing. It checks row count and scope,
not a full dataset content fingerprint: keep original row order/identities and
do not substitute another dataset with the same length.

### NestedSelectionResult

`NestedSelectionResult(selections, training)`: `selections` maps outer fold
IDs to their separate `SelectionResult`; `training` contains only independent
outer predictions. `comparison` concatenates inner comparison tables with
`outer_fold`. There is no single winner property.

## Internal helper map

These are implementation details, documented for maintainers rather than public
extension APIs.

| Module/helper | Responsibility |
|---|---|
| `candidates.Candidate.__post_init__` | Validate basic definition and stable explicit config. |
| `candidates.GridCandidates.__iter__` | Validate grid values, copy products and append stable names. |
| `criteria._comparable(table)` | Require eligible evidence in exactly one comparison group. |
| `criteria._finish(table, winner, order)` | Copy comparison and add selected flag/candidate order. |
| `core._development(dataset, positions)` | Validate alignment and complete matches, copy frames and definitions, return local dataset plus original positions. |
| `core._restrict_plan(plan, rows, n)` | Validate original-index plan and map every fold plus row order to development coordinates. |
| `core._local_validation(validation, lookup)` | Deep-copy selectors or recursively map explicit original validation positions/dictionaries. |
| `core._fitted_objects(root)` | Identity-deduplicated, cycle-safe walk through declared model/preprocessor/backend links and sklearn composite steps/transformers. Skips string sentinels. |
| `core._validate_preparation(candidate)` | Validate train-scoped pre-analysis, named selector and fixed/dynamic feature exclusivity before error recording. |
| `core._FreshModels.__init__ / factory(candidate)` | Retain object identities; wrapped `create()` rejects reused adapters/estimators/preprocessors. Wrapped iterative `backend_factory(seed)` also checks backend components. |
| `core._fit_candidate(dataset, plan, candidate, guard)` | Construct runner, execute copied train preparation on its selection plan, fit with the consumed selector. |
| `core._original_positions(training, positions)` | Restore train/test/score/fit/validation positions, prediction/truth/metadata indices and selector rows/optional scope. |
| `core._metric_records(report)` | Flatten each metrics table with study, type, partition, fold, layout and scope identity. |
| `core._fitted_evidence(candidate, training, dataset, use_ic)` | Validate callback complexity, compute per-fit criteria/sample signatures, retain tables and explicit aggregate metrics. |
| `ModelSelection.run_nested.candidates / original` | Wrap candidate factories with the shared outer guard and map locally built inner rows to originals. |

The freshness walk follows `estimator`, `preprocessor`, `backend_`,
`remainder`, `steps`, `transformers`, `transformers_`, `transformer_list`.
It does not scan arbitrary attributes. Custom factories remain responsible for
fresh hidden framework state. Callbacks are not sandboxed, and stochastic
reproducibility requires explicit model-specific seed configuration.

This increment supplies finite search. Adaptive Optuna execution,
`ExperimentRunner`, scheduled retraining and real-data pilot evaluation are
outside this verified implementation.
