# Analysis, reporting and selection API reference

Companion to the [practical guide and equations](reporting.md),
[coverage checklist](reporting_documentation_checklist.md) and
[minimal notebook](../../notebooks/08_reporting_quickstart.ipynb).

## Imports and optional dependencies

Import `PreTrainingAnalysis` from `xdiyo_analytics.analysis`. The public exports
from `xdiyo_analytics.reporting` are `Artifact`, `StudyResult`, `AnalysisContext`,
`Reporter`, `StudyRun`, `AnalysisReport`, `FeatureSelection`, `FeatureSelector`,
`vote_selections`, `FeatureDistributionReporter`, `CorrelationAnalysis`,
`FeatureTimeline` and `TopKCorrelationSelector`.

The `reporting` package extra declares `scipy>=1.14,<2` and `plotly>=5.24,<7`.
Plot construction loads Plotly when needed; KDE/Kendall load SciPy when needed.
Plain report contracts, custom reporters and empty reports do not require a model
library. Notebook display additionally uses IPython from the notebook environment.
The verification environment already supplied these dependencies; no install was
performed. There is no sklearn dependency in the implementation.

## Orchestration and scope

```text
PreTrainingAnalysis(reporters=<new empty dict>, title='Pre-training analysis')
PreTrainingAnalysis.run(dataset, *, split_plan=None, fold_ids=None) -> AnalysisReport
```

| Parameter | Meaning |
| --- | --- |
| `reporters` | Ordered mapping of distinct nonempty names to reporter objects. Empty means no studies. Named dependencies must appear first. |
| `title` | Report heading and document title. |
| `dataset` | Assembled `ModelDataset`; `X`, `y` and metadata indexes must align. |
| `split_plan` | Optional plan referring to this dataset's original row order. Its `n_rows` must match. Same-length reordering cannot be detected. |
| `fold_ids` | `None` uses every fold; otherwise distinct valid zero-based integer fold numbers, in requested execution order. Booleans are rejected. Requires a plan. |

Every reporter explicitly declares `type`, `partition`, `supported_types` and
`run(context)`. Built-in constructors require keyword-only `type` and `partition`.

| `type` | Population/order |
| --- | --- |
| `per_fold` | One invocation per selected fold, positions sorted in original order. |
| `overall` | One invocation over unique selected positions, in original order. |
| `timeline` | One invocation over that population in stable UTC kickoff order. |

`partition` is `train`, `test`, `score` or `all`. Per-fold `all` is the train/test
union and excludes unassigned gaps. Overall/timeline `all` is the entire supplied
dataset; other overall/timeline partitions are unions over selected folds.
Overlapping occurrences are deduplicated by original row position. Without a
plan, only supported overall/timeline `all` is available. Scopes requiring folds
reject an empty selected fold list; an individual empty partition is retained.

Fold arrays must be one-dimensional integer positions in range. Repeated
positions are deduplicated. Analysis consumes the plan's memberships; it does not
revalidate whole-match construction or train/test separation of manually edited
plans. It does not filter targets, transform features or fit anything.

`features_from='name'` is optional on the four built-in study/selector classes
and may be provided by custom reporters. It uses one earlier selection from the
same fold, falling back to an explicitly named overall selection. Input names
are selected in the selection's order. Unconfigured later reporters receive their
usual feature population. Source scores are reused only with matching fold,
partition and row positions; choosing columns across train/test scopes is a
different operation from reusing scores.

## Shared records and reporter protocol

All list/dictionary defaults below are newly allocated per instance. Records and
results are mutable; they are not tamper-proof provenance records.

```text
Artifact(kind, data, title='', options=<new empty dict>)
FeatureSelection(columns, ranking)
StudyResult(title, artifacts=<new empty list>, tables=<new empty dict>,
            notes=<new empty list>, selection=None)
StudyRun(name, type, partition, fold_id, layout, row_positions, n_matches, result)
AnalysisReport(studies=<new empty list>, title='Pre-training analysis')
```

| Record field | Contract |
| --- | --- |
| `Artifact.kind` | Built-in or custom renderer key. |
| `Artifact.data` | Text, DataFrame, Plotly figure, trusted HTML, or custom payload. |
| `Artifact.title` | Collapsible artifact heading; empty falls back to kind. |
| `Artifact.options` | Kind-specific presentation settings, not scope controls. |
| `FeatureSelection.columns` | Ordered tuple of distinct selected input feature names. Empty is valid. |
| `FeatureSelection.ranking` | DataFrame retaining the algorithm's ranking/decision evidence. |
| `StudyResult.title` | Per-view study heading. |
| `StudyResult.artifacts` | Any ordered arrangement of display items. |
| `StudyResult.tables` | Named full numerical DataFrames, available independently of rendering. |
| `StudyResult.notes` | Escaped explanatory strings. |
| `StudyResult.selection` | `None` or `FeatureSelection`; no automatic input reduction. |
| `StudyRun.name` | Name from the ordered reporter mapping. |
| `StudyRun.type` | Requested execution type. |
| `StudyRun.partition` | Requested row partition. |
| `StudyRun.fold_id` | Zero-based fold number or `None` for a combined population. |
| `StudyRun.layout` | Declared match/team-match observation unit. |
| `StudyRun.row_positions` | Copy of the exact ordered original positions used. |
| `StudyRun.n_matches` | Distinct supplied match tuples in that scope. |
| `StudyRun.result` | The computed `StudyResult`. |
| `AnalysisReport.studies` | Ordered run collection; per-fold views retain their separate runs. |
| `AnalysisReport.title` | Viewer and document title. |

`Reporter` is a structural `Protocol`; inheritance/registration is unnecessary.
`supported_types` declares permitted arrangements. `run(context)` must return a
`StudyResult`. Computation exceptions propagate with a note identifying study,
type, partition and fold. No fallback silently drops a failed study.

### AnalysisContext

```text
AnalysisContext(X, y, metadata, layout, match_columns, row_positions, type, partition,
                fold_id=None, fold_metadata=<new dict>, definitions=<new dict>,
                previous_results=<new dict>, fold_rows=<new dict>,
                fold_results=<new dict>, fold_metadata_by_id=<new dict>)
```

| Field/property | Meaning |
| --- | --- |
| `X` | Copied selected feature rows, restricted by explicit `features_from` if supplied. |
| `y` | Copied target rows, including missing values. |
| `metadata` | Copied aligned metadata with exact identifier dtypes. |
| `layout` | `match` or `team_match`; no inferred reshaping. |
| `match_columns` | Tuple of metadata columns identifying whole matches. |
| `row_positions` | Copied original positions; also the integer `row_position` index of all three frames. |
| `type` | This invocation's execution type. |
| `partition` | This invocation's partition. |
| `fold_id` | Current fold or `None`. |
| `fold_metadata` | Independent copy of this fold's metadata; empty for a pooled context. |
| `definitions` | Deep-copied feature/label definitions from the dataset. |
| `previous_results` | Deep copies of earlier results matching this exact fold/partition/ordered rows. |
| `fold_rows` | Selected fold ID to original positions for the current partition. |
| `fold_results` | Earlier study name to matching fold ID/result copies, for consensus consumers. |
| `fold_metadata_by_id` | Deep-copied metadata catalogue for selected folds. |
| `n_matches` | Computed distinct match-tuple count, separate from `len(X)`. |

Consensus children receive their own `fold_metadata`, `definitions` and previous
results. Their fold catalogues are cleared to avoid recursive aggregation. Custom
objects remain responsible for any mutable state stored outside these contexts.

## FeatureDistributionReporter

```text
FeatureDistributionReporter(*, type, partition, features=None, features_from=None,
                            bins='auto', bandwidth='scott', grid_size=256)
FeatureDistributionReporter.run(context) -> StudyResult
supported_types = ('per_fold', 'overall')
```

| Setting | Contract |
| --- | --- |
| `features` | `None` means all available input features; string or ordered sequence selects distinct names. |
| `features_from` | Optional earlier selector; applied before this feature subset. |
| `bins` | NumPy histogram bin count, supported bin-rule string, or explicit edges. Edges must be finite, strictly increasing and cover every finite observation. |
| `bandwidth` | `None`, `scott`, `silverman`, or a positive finite real scalar covariance factor. Booleans, callables, unsupported objects/strings, zero and negative/nonfinite factors are rejected. |
| `grid_size` | Integer at least two; defaults to 256. |

Numeric strings are converted; nonnumeric, missing and infinite values are
excluded. Each retained observation has equal weight. KDE uses sample variance
and the specified SciPy factor; its grid spans the histogram edges. The bandwidth
guard runs even for constant, empty or feature-empty inputs.

| Table | Columns |
| --- | --- |
| `summary` | `feature`, `rows`, `finite`, `excluded`, `kde_status` |
| `histogram` | `feature`, `left`, `right`, `count`, `density` |
| `kde` | `feature`, `x`, `density` |

Statuses are `ok`, `no_finite_values`, `insufficient_spread` (one observation or
constant data), and `singular_kde`. The last three omit the KDE curve. There is
one Plotly figure per selected feature. Finite-input histogram density integrates
to one across its bins; the displayed finite KDE window need not.

## CorrelationAnalysis

```text
CorrelationAnalysis(*, type, partition, features=None, features_from=None,
                    targets=None, methods=('pearson', 'spearman'),
                    percentile_ranks=True, categorical_features=(),
                    categorical_targets=(), feature_thresholds=<new dict>,
                    target_thresholds=<new dict>)
CorrelationAnalysis.run(context) -> StudyResult
supported_types = ('per_fold', 'overall')
```

| Setting | Contract |
| --- | --- |
| `features`, `targets` | `None` selects all available columns; string or ordered nonempty sequence of distinct known names selects a subset. |
| `features_from` | Explicit prior selector composition. |
| `methods` | One name or distinct nonempty sequence from `pearson`, `spearman`, `kendall`, `mcc`. |
| `percentile_ranks` | Default true adds average-tie feature rank percentages based on absolute coefficient within each target/metric. |
| `categorical_features`, `categorical_targets` | Explicit collections of categorical column names for MCC. No automatic categorical detection. |
| `feature_thresholds`, `target_thresholds` | Name to finite numeric threshold; strict `value > threshold` produces classes zero/one. Takes precedence over a categorical declaration for that column. |

Numeric coefficients use pairwise finite values. Spearman uses average tied
ranks; Kendall is tau-b. MCC drops missing/infinite categorical pairs, uses a
shared label vocabulary and requires explicit treatment of both columns.
Undeclared MCC cells have `n=0` and status
`requires_categorical_declaration_or_threshold`. Other statuses are `ok`,
`insufficient_pairs` and `zero_spread`. Undefined estimates stay missing.

`coefficients` columns are `feature`, `target`, `metric`, `value`, `n`, `status`
and optional `percentile`. There is one row per requested combination.
`leaderboard` begins with `feature`, `pooled_magnitude`, `n_coefficients`, then
`<target> · <metric>` and optional `<target> · <metric> · percentile` columns in
requested target/method order. Rows sort by descending sum of absolute defined
coefficients, stably by input feature order for ties; entirely missing scores
come last. This sum is not normalized by coverage. One leaderboard artifact
references the coefficient/rank column names for signed bars and rank shading.

## Common FeatureSelector and voting

```text
FeatureSelector(*, type, partition, features_from=None, across_folds=False)
FeatureSelector.run(context) -> StudyResult
FeatureSelector.select(context) -> StudyResult                 # implement
FeatureSelector.combine(context, fold_results) -> StudyResult  # implement for consensus
FeatureSelector.for_fold() -> FeatureSelector
vote_selections(fold_results, *, feature_order, k, score_column=None) -> FeatureSelection
```

`FeatureSelector` is a reusable reporter base with supported types `per_fold` and
`overall`. `features_from` composes a prior selection. Without `across_folds`,
`run` validates the result of one `select`. With `across_folds=True`, execution
must be `overall` with selected folds. The base creates one isolated child
context per fold and calls `for_fold().run(child)`, then `combine(context, results)`.
The base's `select` and `combine` raise `NotImplementedError` until implemented.
Default `for_fold` makes a dataclass replacement with `type='per_fold'` and
`across_folds=False`; custom local settings may override it. Its
`_selection_result` helper requires `StudyResult` plus `FeatureSelection`, with
distinct selected names drawn from that context's input.

| Voting parameter | Contract |
| --- | --- |
| `fold_results` | Nonempty fold-ID mapping to study results carrying selections. |
| `feature_order` | Ordered distinct universe of feature names. Every nomination must belong to it. |
| `k` | Positive integer maximum, or `None` for every nominee. Booleans are rejected. |
| `score_column` | `None` ranks by votes then feature order. Otherwise each selection ranking needs unique `feature` plus this numeric column; nonfinite scores are excluded from the mean. |

The helper counts each selected name once per fold, ranks by decreasing votes,
optional decreasing mean finite score, then explicit input order. It selects
only names with positive votes. Missing scores do not erase nominations under
this generic policy. Custom selectors decide whether such nominations are valid.
The result ranking contains `feature`, `votes`, optional `mean_score` and
`scored_folds`, `vote_fraction`, and `selected`. Scores must be comparable across
folds when used. The helper does not inspect raw feature or target values.

## TopKCorrelationSelector

```text
TopKCorrelationSelector(*, type, partition, features_from=None, across_folds=False,
                        k, method='spearman', source=None, features=None,
                        targets=None, fold_k=None)
TopKCorrelationSelector.select(context) -> StudyResult
TopKCorrelationSelector.for_fold() -> TopKCorrelationSelector
TopKCorrelationSelector.combine(context, fold_results) -> StudyResult
```

| Setting | Contract |
| --- | --- |
| `k` | Positive integer final number requested; fewer are returned if fewer defined/nominated features are eligible. |
| `method` | One of the four correlation methods; default Spearman. |
| `source` | Earlier matching correlation study, or `None` to calculate locally. Missing requested feature/target/metric pairs, duplicates or scope mismatches raise. |
| `features`, `targets` | Ordered subsets or `None` for all context columns. |
| `features_from` | Restrict inputs using an explicitly named earlier selector. |
| `across_folds` | False for local/pooled selection; true uses the shared consensus routing. |
| `fold_k` | Positive integer local nomination limit, or `None` to use `k`. Invalid outside consensus. |

Local ranking sums absolute finite values for the selected method across targets.
All-missing features are ineligible. Ties follow selected/input feature order.
Tables are `ranking` (`feature`, `pooled_magnitude`, `n_targets`, `selected`) and
the filtered `coefficients`. `FeatureSelection.ranking` is an independent copy.
For configured MCC, reuse a correlation source with declarations/thresholds;
direct internal MCC has no such declarations and therefore no eligible scores.

Consensus calls the generic voting helper with `pooled_magnitude` as score and
renames its mean to `mean_magnitude`. It retains `ranking`, `fold_rankings` and
`coefficients`; the latter two have a leading `fold_id`. `scored_folds` counts
finite fold scores; means include non-nominating folds with defined scores.
The final ranking sorts by votes, mean magnitude and input order. Fold overlap
does not create independent votes. A separate outer test is needed for evaluating
a consensus chosen using these folds.

Zero input columns with `features=None` return an empty selection/ranking and a
note. Constant/all-missing scores also yield fewer or zero choices. Explicit
empty feature/target lists are configuration errors, not the no-input shortcut.

## FeatureTimeline

```text
FeatureTimeline(*, type, partition, features=None, features_from=None,
                entity='team', teams=None, team_column=None,
                league_aggregation=None, max_points=None)
FeatureTimeline.run(context) -> StudyResult
supported_types = ('per_fold', 'overall', 'timeline')
```

| Setting | Contract |
| --- | --- |
| `features`, `features_from` | Feature subset and explicit selection composition as above. |
| `entity` | `team` (default) or `league`. |
| `teams` | Optional iterable of exact team IDs, filtering team plots only. Invalid with league mode. An empty/no-match filter shows no-observations output. |
| `team_column` | Explicit metadata identity column; otherwise team-match uses `team_id`, match `home::`/`away::` features use `home_id`/`away_id`. Unprefixed match features require an explicit choice. |
| `league_aggregation` | `None` requires agreeing finite values per competition/kickoff. `mean` or `median` explicitly aggregates selected rows at that coordinate. |
| `max_points` | `None` displays all points; integer at least two evenly samples each entity's series, including endpoints. Extra missing breakpoints can exceed the limit. |

Kickoff, competition and used team identities must be present. Values are
numerically converted; nonfinite values become missing gaps. Points always sort
chronologically with stable ties. Team grouping is competition/team, preserving
identity across seasons. No imputation, smoothing or state reconstruction occurs.

`points` contains `feature`, `row_position`, `kickoff_at`, `competition_id`,
`value`, `team_id` in team mode. League mode omits `row_position` and `team_id`
because one coordinate can combine multiple observations. `plotted_points`
retains the display subset with the same schema. Full points remain downloadable.
At a league coordinate, agreement considers finite values only; all-missing
coordinates remain missing. Different LOO/team values require explicit aggregation.
Multiple entity traces receive a dropdown; lines never connect across missing points.

## Rendering and notebook display

```text
AnalysisReport.selections -> {name: {fold_id_or_None: FeatureSelection}}
AnalysisReport.to_html(path=None, *, renderers=None) -> str
AnalysisReport.to_notebook(*, height=800, renderers=None) -> IPython.display.IFrame
AnalysisReport.show(*, height=800, renderers=None) -> None
AnalysisReport._repr_html_() -> str
AnalysisReport._notebook_html(*, height=800, renderers=None) -> str
```

`selections` exposes recorded results with their fold identities; it does not
choose one global set. `to_html` optionally creates parent directories and writes
UTF-8 HTML, returning the document either way. `renderers` maps kinds to callables
receiving one artifact and returning an HTML fragment; a custom key can override
a built-in renderer. Unknown kinds fail explicitly. Rendering never runs reporters.

`height` is a positive integer pixel height, not a boolean. `to_notebook` returns
a native IFrame with `about:blank` and escaped self-contained `srcdoc`; `show`
displays it once and returns `None`. `_repr_html_` supports a final notebook cell
value through `_notebook_html`. IPython is loaded only for notebook display.
Trust/HTML/JavaScript restrictions belong to the frontend. Standalone HTML is an
alternative when inline interaction is suppressed.

| Artifact kind | Display/data requirements |
| --- | --- |
| `text` | Escaped preformatted text. |
| `table` | DataFrame; sortable preview of up to 200 rows plus full CSV. |
| `plotly` | Valid Plotly figure; JSON escaped against script closing, lazy SVG-capable plot initialization. |
| `leaderboard` | DataFrame, all rows shown. `options['coefficient_columns']` enables signed bars/absolute sorting; `options['percentile_columns']` enables rank shading. Built-in studies also expose full data-table downloads. |
| `html` | Trusted raw local HTML. |
| custom kind | Supplied trusted renderer decides its layout. |

Study navigation/search, expand/collapse, fold controls and entity menus operate
on computed views. Plotly JS is embedded once when needed. Plots inside closed
`details` or hidden fold scopes wait until opened. Tables sort numeric values,
including exact integer strings beyond JavaScript Number precision, with missing
values last in either direction. Coefficient sorting uses absolute magnitude;
other numeric sorting retains sign. Text sorting is lexical.

CSV is UTF-8 with BOM and includes every row, without the DataFrame index. Each
run also provides its original-position scope CSV. Plotly's toolbar exports SVG.
Ordinary titles, labels, values and notes are escaped; trusted HTML/custom
renderers intentionally bypass that escaping. The viewer is a local artifact,
not a sandbox for untrusted renderer code.

## Internal helpers and error boundaries

| Helper | Responsibility |
| --- | --- |
| `studies._plotting` | Lazy Plotly import with optional-dependency guidance. |
| `studies._scipy` | Lazy SciPy statistics import with dependency guidance. |
| `studies._columns` | Resolve all/string/sequence selections; reject empty, duplicate or unknown columns. |
| `studies._numeric` | Convert numeric-like values to floating arrays, coercing unusable values to NaN. |
| `studies._style` | Common chart template, labels, colors and dimensions. |
| `studies._categorical` | Explicit declaration or finite strict threshold encoding for MCC. |
| `studies._mcc` | Shared-vocabulary multiclass coefficient; zero denominator stays missing. |
| `FeatureSelector._selection_result` | Enforce selection result type and distinct input names. |
| `viewer._e` | HTML escaping with quoted attributes. |
| `viewer._csv_link` | Full DataFrame CSV encoded as a local download link. |
| `viewer._table` | Preview cells, signed/rank bars and sortable exact value attributes. |
| `viewer.render_report` | Assemble arbitrary study/artifact layouts and optionally persist the HTML. |
| `viewer.CSS`, `viewer.JS` | Responsive/print presentation, deferred charts, navigation, scope switches and sorting. |

Malformed dataset alignment, row positions, names, scope/partition choices,
dependencies, selected names or numeric settings raise instead of guessing.
Missing columns generally raise `KeyError`; wrong result/table types raise
`TypeError`; unsupported configurations generally raise `ValueError`. Deliberate
missing estimates and empty populations remain explicit numerical outputs/notes.
The dataclass containers themselves do not comprehensively validate manual
construction. The [guide](reporting.md#limits-before-a-predictive-experiment)
states the statistical and future state-preparation boundaries.
