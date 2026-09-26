# Post-training API, persistence and helper reference

Public entry points live in `xdiyo_analytics.analysis`, `evaluation`, `reporting`
and `experiments`. See [practical examples](post_training.md) and
[calculation definitions](post_training_equations.md). All prediction populations
come from already-computed training results; post-training APIs perform no fit,
search, feature construction or automatic experiment publication.

## PostTrainingAnalysis

```text
PostTrainingAnalysis(reporters=<new dict>, title='Post-training analysis')
.run(training=None, *, fold_ids=None, experiment=None) -> AnalysisReport
```

| Argument | Contract |
| --- | --- |
| `reporters` | Mapping from distinct nonempty study names to reporter objects. An empty mapping returns an empty report. |
| `title` | Display title of the returned shared report. |
| `training` | Existing `TrainingResult`; may be None for experiment-only studies. The caller owns its data/provenance consistency. |
| `fold_ids` | None selects all fitted folds in stored order. Otherwise distinct existing fitted IDs in the requested order; IDs are retained. Prediction studies require a nonempty selection. |
| `experiment` | Optional `ExperimentStore` supplied to experiment reporters. No implicit save or model reload. |

Prediction reporters expose `type`, `partition`, `supported_types`, optional
`pooling` and `run(context) -> StudyResult`. They may subclass `PredictionReporter`
or satisfy this structural contract themselves. `per_fold` invokes once per fold;
`overall` and `timeline` invoke once over selected folds. `score` selects each
fold's score positions; `test` selects all its held-out positions. A timeline
requires nonmissing parseable kickoff timestamps and sorts all frames together,
stably. `experiment` requires overall type and a supplied store.

Named prediction outputs and target column order must agree across pooled folds.
Different class columns are union-aligned and remain missing where absent.
Repeated row positions require `occurrences`, `first`, `last` or `mean`; no-repeat
populations permit `pooling=None`. First/last follow selected-fold order. Mean
requires numeric output columns, propagates missing cells and records fold `-1`.
It is unsuitable for numeric category decisions despite their numeric dtype.
Conflicting repeated targets/metadata raise under every pooling rule.

The orchestrator copies frames and nested definitions per study. Fitted model
objects remain shared references for read-only inspection. As elsewhere, pandas
deep copies do not recursively clone arbitrary Python objects inside cells.
Reporter exceptions retain their type and gain a study/type/partition/fold note;
scope-validation errors occur before invocation. A non-`StudyResult` return raises.

## PostTrainingContext and shared result scope

```text
PostTrainingContext(y, predictions, metadata, layout, identity_columns,
                    match_columns, type, partition, fold_id=None,
                    pooling='occurrences', models=<new dict>,
                    definitions=<new dict>, experiment=None)
```

| Field/property | Meaning |
| --- | --- |
| `y` | Selected true targets; rows have a two-level `(fold_id, row_position)` index. |
| `predictions` | Named output DataFrames with the same index/order as y. |
| `metadata` | Selected identity/context rows aligned to y; no automatic predictor/decision construction. |
| `layout` | Declared match/team_match layout; experiment-only contexts use experiment. |
| `identity_columns` | Full observation keys, including team identity in team_match layout. |
| `match_columns` | Whole-match identity keys. |
| `type`, `partition` | Actual execution arrangement and test/score/experiment population. |
| `fold_id` | Specific fitted ID for per-fold execution; None for pooled/experiment contexts. |
| `pooling` | Effective occurrence policy, defaulting to occurrences when no explicit rule was needed. |
| `models` | Selected fold ID to fitted model/adapter reference; read-only use is a caller obligation. |
| `definitions` | Independent deep copy of training definitions. |
| `experiment` | Supplied store object, or None. |
| `row_positions` | Original row-position index level, including repeated occurrences. |
| `n_matches` | Distinct match-key rows; zero for empty contexts. |

`StudyRun` keeps its existing name/type/partition/fold/layout/row-position/count/
result fields and adds `scope=None` and `scope_label=None`. Post-training scope is
a DataFrame containing the actual fold/row pairs, captured before reporter
execution. Prediction scope labels state the pooling rule; experiment scope says
`saved experiment runs` and is empty. This preserves the existing pre-training
construction contract.

`AnalysisReport.studies` contains these records. Its existing `to_html(path=None,
renderers=None)`, `to_notebook(height=800, renderers=None)`, `show(...)` and rich
display use stored results only. The viewer downloads `StudyRun.scope` when
present; old reports retain row-position-only scope downloads. Custom leaderboard
legends override the earlier correlation legend, while signed bars and full CSV
tables remain available. HTML/custom renderers are trusted code; ordinary labels
are escaped. See the [shared reporting reference](reporting_reference.md).

## Metric requests and registry

```text
Metric(name, target=None, output=None, parameters=<new dict>, key=None,
       direction=None)
MetricDefinition(function, kind, direction)
register_metric(name, function, *, kind, direction=None, replace=False) -> None
list_metrics() -> DataFrame[name, kind, direction]
evaluate_metrics(y, predictions, metrics, *, metadata=None) -> DataFrame
```

`Metric` and `MetricDefinition` are frozen dataclasses, but supplied parameter
containers are ordinary mutable objects. A string in `metrics` is shorthand for
`Metric(name)`. `target=None` expands over y columns in order. `output=None`
chooses `predict` for numeric/label metrics or `predict_proba` for probability/
uncertainty metrics. `key` names the configured result; keys must be distinct per
target/output. `direction` is minimize/maximize/None; None uses the definition's
direction. Parameters are forwarded to the calculation and must also serialize
as JSON for recorded metric identity. Array-valued parameters are not implicitly
filtered alongside complete cases.

The process-local registry accepts a nonempty name and callable
`function(y, prediction, **parameters) -> scalar`. `kind` is numeric, label,
probability or uncertainty. Probability/uncertainty functions receive a DataFrame
with class labels as columns; numeric/label functions receive Series. Uncertainty
does not require observed y values, although its target column must exist.
Duplicate registration requires `replace=True`. Replacement affects that registry
name; initially registered aliases are separate entries pointing to the same
definition. Registration performs no file I/O or persistence.

### Built-in names, aliases and parameters

| Names | Kind; default direction | Parameters and conventions |
| --- | --- | --- |
| `mse`, `mean_squared_error` | numeric; minimize | Mean squared residual; no parameters. |
| `mae`, `mean_absolute_error` | numeric; minimize | Mean absolute residual; no parameters. |
| `rmse` | numeric; minimize | Square root of MSE; no parameters. |
| `r2` | numeric; maximize | Supported sklearn r2_score parameters; this layer always marks small/constant targets undefined first. |
| `accuracy`, `accuracy_score` | label; maximize | Supported accuracy_score parameters, ordinarily normalized accuracy. |
| `balanced_accuracy` | label; maximize | Supported balanced_accuracy_score parameters, including adjusted. |
| `precision`, `precision_score` | label; maximize | `average='macro'`, `zero_division=0` unless overridden; binary mode uses explicit pos_label. |
| `recall`, `recall_score` | label; maximize | Same averaging/zero-division defaults. |
| `f1`, `f1_score` | label; maximize | Same averaging/zero-division defaults; configure averaging to return a scalar. |
| `mcc` | label; maximize | Supported matthews_corrcoef parameters. |
| `log_loss`, `cross_entropy` | probability; minimize | `eps=1e-15`; optional `binary=False`. Uses observed class labels directly. |
| `binary_cross_entropy` | probability; minimize | Exactly two class columns; optional eps. |
| `binary_entropy` | uncertainty; None | Exactly two class columns; no parameters; zero log zero is zero. |
| `brier_score`, `brier_score_loss` | probability; minimize | Binary only; `positive_label=1`. |
| `roc_auc` | probability; maximize | `positive_label=1`, `multi_class='ovr'`, `average='macro'`. Binary selection uses the declared positive label; multiclass uses declared class order. |

See [equations and conventions](post_training_equations.md) and
[scikit-learn metric parameters](https://scikit-learn.org/stable/modules/model_evaluation.html).
The library's Brier implementation is explicitly binary, regardless of additional
capabilities in a particular installed backend version.

### Input checks and metric table

All prediction frames and optional metadata must have exactly y's index/order.
Probability columns require a unique two-level `(target, class)` MultiIndex.
Complete vectors need at least two classes, finite values in [0,1] and row sums
within absolute `1e-6` of one. Unknown observed classes and malformed settings
raise; incomplete vectors are excluded and counted. Numeric truth/prediction
values convert to numbers; nonfinite pairs are excluded. Labels require nonmissing
values. A custom non-scalar result raises; a nonfinite scalar becomes undefined.

| Output column | Meaning |
| --- | --- |
| `metric`, `calculation` | Configured key/name and registered calculation name. |
| `target`, `output` | Actual target and prediction mapping key. |
| `value`, `direction` | Ordinary metric value and optimization preference; descriptive direction may be None. |
| `n`, `n_total`, `n_missing` | Evaluated complete cases, input rows and excluded rows. |
| `status` | ok; no_valid_observations; undefined_constant_or_small_target; undefined_single_observed_class; or undefined. |
| `parameters` | Sorted JSON encoding of explicitly supplied parameters. Equivalent omitted/explicit defaults are not normalized into one identity. |
| `sample_hash` | SHA-256 binding evaluated index, metadata, dtypes and observed target; predictions excluded. Uncertainty substitutes a constant target. |

The metadata/index order and all supplied metadata columns affect the sample hash.
It identifies the evaluated population, not training data, feature provenance or
custom function source code. Caller configuration must record those additional
choices. No valid observations may mean malformed values/settings are never passed
to a metric function; the coverage status is not a validation certificate for
unused inputs.

## Prediction reporters

All constructors are keyword-only. `PredictionReporter(*, type, partition,
pooling=None)` supplies the shared scope settings. Its supported types are
per_fold/overall/timeline. Concrete reporters additionally accept:

| Reporter | Additional fields/defaults | Numerical tables and figures |
| --- | --- | --- |
| `PerformanceReporter` | `metrics` required | metrics table and its table artifact, with coverage note. |
| `ResidualAnalysisReporter` | `targets=None`, `output='predict'`, `bins='auto'` | pairs::target (fold/row, observed/predicted/residual); histogram::target (left/right/count); coverage. Scatter, residual scatter and histogram. |
| `PredictionDistributionReporter` | `targets=None`, `output='predict'`, `mode='kde'`, `bandwidth='scott'`, `grid_size=256` | distribution::target (distribution/x/value); coverage (target/distribution/n/n_missing/status). KDE, ECDF or categorical frequency overlay. |
| `LabelPredictionDistributionReporter` | Exact alias of PredictionDistributionReporter | Same constructor, behavior and result. |
| `CalibrationReporter` | `targets=None`, `output='predict_proba'`, `n_bins=10`, `strategy='uniform'`, `classes=None` | calibration (target/label/bin/n/mean_probability/observed_frequency); coverage. One-vs-rest reliability curves and diagonal. |
| `PredictionTimelineReporter` | `targets=None`, `output='predict'`, `group_by=None` | timeline::target retains fold/row, grouping keys, kickoff_at, observed/predicted/residual. Separate traces by fold and requested metadata groups. |

`targets` accepts None/all, a string, or ordered distinct existing column names.
Each `run(context)` returns `StudyResult` with `Artifact` figures and reusable
tables. Numeric diagnostics require numeric point outputs and paired finite
values. Residual bins follow NumPy histogram conventions. KDE bandwidth accepts
Scott/Silverman or a positive finite scalar factor; grid_size is an integer at
least two, excluding bool. Its coverage status is ok, empty or constant_marker.
No histogram bars appear in KDE mode. Frequency mode is categorical and uses
the same paired nonmissing population for both distributions.

Calibration n_bins is a positive integer, excluding bool; strategy is uniform or
quantile. `classes=None` uses all class columns; an explicit sequence selects
existing class labels. Incomplete probability vectors are excluded. Tied quantiles
collapse edges and empty bins are omitted; no recalibration is fitted. Timeline
group_by accepts a metadata name or sequence; None groups only by fold. It requires
nonmissing kickoff_at, preserves missing-value gaps and performs no aggregation.
In match layout, team grouping requires explicit home/away metadata columns.

## BetSpec and evaluate_bets

```text
BetSpec(option, odds, take, stake=1.0,
        policy='provided_decisions_fixed_stake')
evaluate_bets(context, bets, *, labels=None, history=None) -> (ledger, metrics)
```

`bets` maps nonempty names to `BetSpec` or dictionaries accepted by its constructor.
`option` must be an existing `BetOption`. Supply exactly one of a same-name mapping
of `LabelData` in `labels`, or paired history for `create_labels(history, options)`.
An empty bets mapping returns empty tables after the exclusive-source check.

The label's definition must equal the requested option, its unit must match
context.layout, and its settlement must have exactly one column aligned to label
metadata. Both match and team_match are supported. Full label identity keys join
settlements; team_match includes team identity. Source identities must be unique.
Context observations must also be unique, so repeated bets require per-fold
execution or explicit unique-row pooling. Missing joined settlements become
missing; accepted categories are win/loss/push/void/missing.

| Field | Contract |
| --- | --- |
| `odds` | Decimal odds scalar or identity/occurrence-indexed Series. Selected finite values must exceed one; missing selected quotes are unresolved. |
| `take` | Explicit boolean scalar or Series; optionally callable(context) returning Series. Missing/nonboolean decisions raise. No automatic prediction threshold or eligibility inference. |
| `stake` | Nonnegative finite scalar or indexed Series; default one unit. Missing selected stakes are unresolved; zero is allowed. |
| `policy` | Caller-supplied decision/staking-rule identifier for comparison; no inspection of rule equivalence or information timing. |

Series must have the exact current occurrence index/order, or a unique MultiIndex
whose names equal the label identity columns in order. Identity Series are
reindexed to observations, preserving exact integer identifiers. Arrays and
unnamed/order-only Series are rejected. A take callable must avoid realized
outcomes and use eligible information; the API cannot establish that property.

The ledger contains original metadata plus fold_id, row_position, bet, odds, take,
actual stake, settlement, accounting_status, payout and profit. Accounting status
is not_placed/settled/unresolved. Unselected rows have zero actual stake and profit;
unresolved selected profit remains missing. Metric rows use the shared metric
schema and include profit, roi, bets_placed, settled_bets, unresolved_bets,
settled_stakes, payout and win/loss/push/void counts. `n` is settled bets, `n_total`
is available observations and `n_missing` is unresolved placed bets. Profit/ROI
are partial while unresolved bets exist; zero settled stake makes ROI undefined.

Comparison fingerprints include offered observations/metadata, odds and settlement;
they exclude the actual decision mask. Parameters record the option, policy,
scalar stake (or supplied_by_policy), ROI denominator and zero fees. The caller
must use a meaningful policy identifier for supplied stake/decision rules.
The [accounting equations](post_training_equations.md#bet-accounting) define all
denominators and distinguish kickoff summaries from realized cash timing.

## BetPerformanceReporter

```text
BetPerformanceReporter(*, type, partition, pooling=None, bets,
                       labels=None, history=None, time_column='kickoff_at')
```

Uses evaluate_bets and returns metrics/ledger tables, a metrics table artifact and
one cumulative-known-profit figure. The ledger is stably sorted by time_column,
which must contain nonmissing parseable timestamps. Per named option, the added
cumulative_known_profit sums known profit, treating unresolved contributions as
zero while retaining their missing row-level profit. Pass actual settlement times
when that ordering is available; this remains accounting, without bankroll or
dynamic strategy simulation. The standard scope fields apply.

## ExperimentStore and configuration_hash

```text
ExperimentStore(root, name)
.save_run(training, report, *, name, config,
          save_predictions=True, save_html=False) -> dict
.save_failure(*, name, config, error) -> dict
.read_runs() -> list[dict]
configuration_hash(config) -> str
```

Root is a parent directory. A nonempty display name maps to a safe slug plus short
name hash; `.path` is the resulting directory and `.manifest` holds schema,
experiment_id and name. Opening the same name reuses its UUID. Names do not select
arbitrary filesystem paths. Corrupt/incompatible manifests and run records raise.

Config is caller-supplied descriptive configuration, ordinarily a meaningful
mapping. Supported values include None, strings, booleans, integers, finite floats,
NumPy scalars, datetime/Timestamp (tagged ISO encoding), Path, dataclass instances,
string-key dictionaries and lists/tuples. Missing pandas sentinels become None;
unsupported objects and nonfinite configuration numbers raise. Dataclasses retain
their qualified type and fields. Dictionary order does not affect SHA-256 hashing;
list order does. Custom models/functions are not introspected or pickled.

NumPy datetime64/timedelta64 configuration scalars use a `__numpy_temporal__`
mapping with `dtype` and integer `ticks`. Temporal arrays use
`__numpy_temporal_array__` with `dtype`, `shape` and nested integer `ticks`.
These descriptors retain units, multipliers, array shape and the `NaT` sentinel
when hashing; ordinary NumPy numeric scalar/list conversion remains unchanged.

Each call creates a new UUID run_id even for identical config_hash. The store
records creation time, status and config. Successful records additionally retain
layout, target perspective, identity columns, numerical metrics and artifact paths.
The caller must supply matching training/report objects; saving does not recompute
or independently verify their provenance. Experiment-only studies are excluded
from saved tables/metrics to avoid recursively recording a leaderboard as results.

| Artifact | Contents |
| --- | --- |
| `experiment.json` | Schema 1, experiment UUID and exact display name. |
| `runs/<run_id>/run.json` | Run/config IDs, config, UTC creation time, status, metric records and artifact mapping. Failure records additionally include explicit error text. |
| `tables.json` | Study/name/fold metadata. Ordinary tables use pandas table-oriented JSON, whose float formatting can round values. Tables whose axes, dtypes or values are not safely represented there use `{"encoding": "xdiyo.data-only.v1", "value": ...}` containing a data-only frame representation for typed labels and values. Metric records retain ordinary JSON scalar values. |
| `fold-<id>/output-<number>.parquet` | Indexed prediction DataFrames; output names are mapped to safe numbered filenames. MultiIndex class columns are retained. |
| `fold-<id>/targets.parquet`, `metadata.parquet` | Indexed held-out targets and metadata. The run record also retains train/test/score positions and selected columns. |
| `report.html` | Optional standalone report, rendered without recomputing studies or predictions. |

Tagged tables include MultiIndex, tuple-valued or duplicate columns, non-string
column labels, named column axes, and index layouts with duplicate labels or
conflicting field names. Temporal indexes and columns, dtypes that table JSON
would change, and supported typed cells such as tuples, dates and NumPy temporal
scalars also use this wrapper, including nested values. `ExperimentStore.load_run`
and `FootballExperiment.load` recognize both table forms, including runs without
`recovery.json`. Existing artifacts remain readable without rewriting them.
Unsigned and narrow integer row indexes also use typed storage when table JSON
would alter their dtype or values. Complex-valued table data, including columns,
row indexes and categories, are unsupported and raise before a completed run is
published.

In data-only payloads, NumPy date and duration scalars use a `numpy_temporal` tag
with their dtype (including unit and multiplier) and signed integer value; `NaT`
retains its sentinel. Temporal arrays retain the existing array tag, dtype and
shape, with integer counts as values. Older payloads remain readable.

DatetimeIndex and TimedeltaIndex payloads additionally record `freq`: either null
or a mapping with an allowlisted offset `name`, `n`, `normalize` and packed `kwds`.
This retains offset parameters, including custom business calendars, alongside
index values, dtype and timezone. Explicit null or absent legacy frequency stays
unset; the loader does not infer a frequency. User-defined offset subclasses are
unsupported, and decoding uses fixed constructors rather than imports named by
the payload.

`save_predictions=False` skips all fold Parquet artifacts; `save_html=False`
skips HTML. Models and training feature matrices are not saved. `save_failure`
records a candidate failure explicitly; it does not catch runner exceptions.
`read_runs` reads published records, sorted by creation time then run ID, without
loading predictions or models. It does not verify all referenced artifact hashes.

Successful publication renames one `.pending-<run_id>` directory after writes
finish. `_publish` retries only OSError winerror 5/32/33, using delays 0.05, 0.1,
0.2, 0.4 and 0.8 seconds: six attempts, 1.55 seconds of total backoff. Other errors
raise immediately. Permanent errors preserve the caught exception, add the pending
path as a note and retain unpublished artifacts. There is no partial-copy fallback,
permission change or automatic cleanup. `read_runs` ignores pending directories.

## rank_runs and ExperimentLeaderboardReporter

```text
rank_runs(records, weights, *, selectors=None, scaling='percentile',
          reference_scales=None, directions=None) -> DataFrame
ExperimentLeaderboardReporter(*, weights, selectors=<new dict>,
    scaling='percentile', reference_scales=<new dict>, directions=<new dict>,
    type='overall', partition='experiment')
```

| Argument | Contract |
| --- | --- |
| `records` | Published numerical record dictionaries, ordinarily from read_runs. No model loading. |
| `weights` | Nonempty display-key to finite nonnegative weight mapping with positive total; zero weights are discarded, positives normalized once. |
| `selectors` | Per display key, exact metric-record filters such as metric/study/target/output/type/partition/fold_id. Default metric name is the key; default type is overall, or per_fold when a non-None fold_id is explicitly supplied. |
| `scaling` | percentile within comparison group, or fixed declared bounds. |
| `reference_scales` | Fixed mode requires finite increasing (low, high) for every positive-weight key. Values outside bounds are clipped. |
| `directions` | Per-key minimize/maximize overrides. Metrics with no universal direction require an explicit choice. |

Each key must resolve to exactly one metric; multiple matches raise rather than
choosing a target/study or averaging folds. Required values need status ok, a
finite number and a sample hash. Missing/undefined/partial metrics leave a run
unranked; failed statuses remain visible. Comparison identity includes each
metric's name/calculation, target/output, serialized parameters, sample hash,
type/partition/fold/layout/scope label and effective direction. It does not inspect
custom metric function code or infer comparable policy implementations.

Output retains run_id/name/config_hash/status/comparison_group, raw::key,
utility::key, contribution::key, score and rank. Higher score is better inside one
group. Ties use average metric ranks and minimum final score rank. Missing values
do not cause candidate-specific reweighting. The normalized weights are attached
as DataFrame attrs for nonempty results. Empty inputs return an empty base-schema
table. See [exact transformations](post_training_equations.md#weighted-experiment-comparison).

The leaderboard reporter is intrinsically overall/experiment and needs
context.experiment. It can run with training=None through PostTrainingAnalysis.
It reads saved records once, adds a shared leaderboard artifact with a custom
contribution legend, and returns the same table under `leaderboard`. Its notes
explain cohorts, scaling and missing metrics. It performs no automatic selection,
fit, tuning or persistence.

## Internal helper inventory

| Helper | Responsibility |
| --- | --- |
| `analysis.post_training._population` | Align fold outputs/targets/metadata and apply explicit occurrence pooling. |
| `evaluation.metrics._sklearn` | Call a named sklearn metric with ordinary NumPy scalar label arrays. |
| `evaluation.metrics._classification` and nested `calculate` | Wrap classification calls, supplying macro and zero-division defaults where applicable. |
| `evaluation.metrics._probabilities` | Validate complete class vectors, range, minimum vocabulary and row sums. |
| `evaluation.metrics._log_loss`, `_entropy`, `_brier`, `_auc` | Implement labeled probability/uncertainty calculations and their specific checks. |
| `evaluation.metrics.metric_inputs` | Resolve one configured target/output and its complete-case mask, preserving index/order. |
| `evaluation.metrics._sample_hash` | Hash evaluated identities/metadata/dtypes and target; never model predictions. |
| `evaluation.betting._aligned` | Expand scalars, evaluate supplied callables, or align exact occurrence/identity Series. |
| `reporting.post_training._pairs` | Produce paired finite observed/predicted/residual rows for numeric diagnostics. |
| `experiments.store._json` | Canonical supported configuration/record conversion; missing record values become JSON null. |
| `experiments.store._write` | Exclusive JSON file creation using supported record conversion. |
| `experiments.store._publish` | Bounded Windows-error retry around atomic directory publication. |
| `ExperimentStore._start` | Create run UUID/config record and unpublished stage. |
| Existing `reporting.studies._columns`, `_numeric`, `_plotting`, `_scipy`, `_style` | Shared selection/conversion, lazy optional dependencies and figure styling; unchanged by this increment. |
| Existing `reporting.viewer._csv_link`, `_table`, `render_report` | CSV payloads, sortable tables/bars and report HTML; render_report now uses optional scope/legend fields. |

## Verification and limits

The [coverage checklist](post_training_documentation_checklist.md) maps these
contracts to numerical, persistence, browser, notebook and preservation checks.
Evidence is recorded in [post_training_check.json](post_training_check.json).
All demonstration fits and accounting use synthetic data. No source-data rewrite,
real pilot, package installation, search, scheduled refitting, feature-importance
report or live Jupyter trust claim is part of this increment.
