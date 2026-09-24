# Training API and output reference

All eight public names are exported by `xdiyo_analytics.training`. The package
itself requires no scikit-learn import; custom adapters can use other frameworks.
The optional `training` extra declares `scikit-learn>=1.5,<2` for estimator examples.
No package installation was needed for verification.

See the [practical guide](training.md) and [coverage/evidence](training_documentation_checklist.md).

## TrainingRunner configuration and execution

```text
TrainingRunner(model_factory, feature_columns=None, target_columns=None)
TrainingRunner.run(dataset, split_plan, *, fold_ids=None,
                   analysis_report=None, features_from=None) -> TrainingResult
```

| Setting | Contract |
| --- | --- |
| `model_factory` | Zero-argument callable returning a fresh object with callable `fit` and `predict`. The factory owns adapter and nested estimator/preprocessor construction. |
| `feature_columns` | None uses all X columns; a string or ordered iterable selects distinct, nonempty, existing names. |
| `target_columns` | Same selection rules for y; selected order is preserved. |
| `dataset` | `ModelDataset` with aligned X/y/metadata indexes and unique frame column names. |
| `split_plan` | `SplitPlan` with `n_rows` equal to dataset length. Same-population/order binding remains the caller's responsibility. |
| `fold_ids` | None processes every fold; otherwise distinct valid zero-based integers in requested order. Booleans are rejected. An empty selection returns an empty result without constructing a model. |
| `analysis_report` | Optional existing report containing the explicitly named selection; not recomputed. |
| `features_from` | Name of a stored selector study; requires the report and cannot accompany fixed `feature_columns`. |

There is one initial fit per processed fold. No full-data final fit, search,
metric calculation, scheduled refitting, callback API, file output or post-training
reporter is implicit. Errors stop the run; a partial `TrainingResult` is not
returned. Previously constructed custom models may already have side effects.

Within one run, returning the same adapter twice raises. For `EstimatorAdapter`,
returning a previously wrapped estimator also raises. This is identity-based
detection, not cloning or a check of every nested object/global state.

### Consuming a recorded feature selection

The runner chooses the named same-fold `StudyRun` first, then a named overall run
with `fold_id=None` if no same-fold run exists. There must be exactly one candidate
and it must carry `result.selection`. An ambiguous or non-selection same-fold
candidate does not fall back to an overall result.

The recorded layout must equal the dataset layout, and every recorded calculation
position must belong to that fold's train array. The selected columns then follow
normal feature-name validation. The consumed run is deep-copied into `FoldResult`.
Scope containment uses positions; no source hash, target identity, partition-label
validation or recalculation proves how the original selection was obtained.
Changing the underlying dataset while retaining compatible positions is a caller
error that the interface cannot detect.

## fit_predict and fold validation

```text
fit_predict(dataset, fold, model_factory, *, fold_id=0,
            feature_columns=None, target_columns=None) -> FoldResult
```

| Parameter | Meaning |
| --- | --- |
| `dataset` | Same aligned dataset contract as the runner. |
| `fold` | `Fold` containing original train/test/score integer positions. |
| `model_factory` | Called once after data/target validation, returning a fresh in-place adapter. |
| `fold_id` | Provenance value passed to contexts/results, default zero. Standalone use does not validate it against a plan; the runner supplies the original valid fold number. |
| `feature_columns`, `target_columns` | Ordered name selection as above; no report consumption at this lower level. |

Each position array is one-dimensional, contains unique integers in range, and
keeps the supplied order. Train and test are nonempty and disjoint; score can be
empty and must be a test subset. Each of the three scopes must include every row
of any match it contains, using exact `dataset.groups` identities.

Selected training targets containing missing values raise before the factory is
called. No rows/targets are dropped or imputed. Test targets may be missing and
remain in `y_true`. Feature NaNs pass through. Nonmissing infinities and additional
model-specific constraints are left to the adapter/estimator.

Before fitting, separate train and prediction contexts are built. Original integer
positions replace their frame indexes, named `row_position`. Training receives X,
y and metadata; prediction receives X and metadata only. `model.fit(training)`
updates the adapter in place and its return value is ignored. An exception from
fit/predict retains its type/object and receives a fold-identifying note. Factory,
scope and output-validation errors propagate directly.

Outputs must form a nonempty dict with nonempty string keys and DataFrame values.
Each frame must have exactly the test index/order and unique nonempty columns.
No finiteness, calibration, class-probability normalization or target-column
semantics are imposed on generic custom outputs. DataFrames are copied into the
result. Standalone calls do not detect a factory reusing a model across calls.

## PredictionContext and FitContext

```text
PredictionContext(X, metadata, layout, match_columns, fold_id,
                  fold_metadata=<new dict>, definitions=<new dict>)
FitContext(X, metadata, layout, match_columns, fold_id,
           fold_metadata=<new dict>, definitions=<new dict>, y=<new DataFrame>)
context.groups -> Series of exact match tuples
```

| Field/property | Meaning |
| --- | --- |
| `X` | Copied selected feature rows; original position index and explicit feature order. |
| `metadata` | Copied corresponding metadata rows, retaining exact identifier dtypes. Never automatically appended to X. |
| `layout` | Dataset declaration, normally match or team_match; no inferred reshaping. |
| `match_columns` | Tuple of metadata names defining whole-match groups. |
| `fold_id` | Original selected fold number, or standalone caller's value. |
| `fold_metadata` | Separate deep copy of the fold's metadata for each context. |
| `definitions` | Separate deep copy of dataset definitions for each context. |
| `y` | FitContext only: copied selected training targets as a DataFrame, including the single-target case. |
| `groups` | Computed Series named match_group, indexed like X, whose elements are exact match tuples. These are not ranker group sizes. |

Frame copying uses pandas `copy(deep=True)`, which isolates ordinary numeric/string
table data; it does not recursively clone arbitrary Python objects stored inside
object-valued cells. Nested definitions and fold metadata use `deepcopy`.
Custom adapters own safe use of metadata, grouped/ranking conversion, framework
configuration, output restoration and any external state.

## ModelAdapter protocol

```text
ModelAdapter.fit(context: FitContext) -> None
ModelAdapter.predict(context: PredictionContext) -> dict[str, DataFrame]
```

This is structural: subclassing/registration is unnecessary. `fit` updates the
new adapter in place. `predict` returns named frames with exact input index/order.
Custom outputs may represent labels, probabilities, embeddings or other values;
consumers must request the intended name. Training targets are absent from the
prediction context. Metadata can contain caller-supplied fields, so that boundary
does not sanitize leaking values inside metadata or X.

## EstimatorAdapter

```text
EstimatorAdapter(estimator, prediction_methods=('predict',))
EstimatorAdapter.fit(context) -> None
EstimatorAdapter.predict(context) -> dict[str, DataFrame]
```

| Setting/state | Contract |
| --- | --- |
| `estimator` | Fresh estimator/pipeline exposing fit(X, y) and the requested prediction methods. |
| `prediction_methods` | Nonempty distinct sequence choosing predict and/or predict_proba; order retained. Use a tuple/list, not a bare string. |
| `target_columns_` | Selected target names recorded by fit; available on the fitted adapter. |
| `estimator.classes_` | Required for probability output; one vocabulary for a single target, list/tuple of vocabularies for multiple targets. |

`fit` validates requested method names/availability, records target names, and
passes a Series for one target or a DataFrame for multiple targets to the estimator.
No clone, imputation, scaling, fit-parameter forwarding or label encoding occurs
inside this adapter. Place learned preprocessing inside the fresh pipeline.

### Output shapes and identities

| Output | Accepted estimator result | Returned frame |
| --- | --- | --- |
| `predict`, one target | Array-like shape `(n,)` or `(n,1)`, or aligned pandas values. | `(n,1)` with selected target name. |
| `predict`, multiple targets | Array-like shape `(n,k)`; a DataFrame must have selected target columns in exact order. | `(n,k)` with selected target names. |
| `predict_proba`, one target | Array-like shape `(n,c)` with fitted class vocabulary of length c. A DataFrame must match class labels/order. | `(n,c)` with MultiIndex columns `(target, class)`. |
| `predict_proba`, multiple targets | One probability array and one class vocabulary per target, supplied as lists/tuples. Each target's shape is `(n,c_t)`. | Concatenated class columns, preserving selected target order and fitted class order. |

Any pandas prediction result must retain the input row index/order. Plain arrays
are treated as positional in that order. A `predict` DataFrame must also retain
selected target names/order. Probability DataFrames must retain fitted class
names/order; the final runner validation rejects duplicate/empty output columns.
Unsupported shapes or missing fitted class identities raise. The wrapper does not
verify numerical probability validity or infer absent classes.

## FoldResult

```text
FoldResult(fold_id, model, train_positions, test_positions, score_positions,
           feature_columns, target_columns, predictions, y_true, metadata,
           fold_metadata=<new dict>, selection=None)
```

| Field | Meaning |
| --- | --- |
| `fold_id` | Original fold number, retained even for a subset/reordered run. |
| `model` | Fitted adapter, retained by reference for inspection/reuse. |
| `train_positions` | Copied original training positions; training tables are not duplicated here. |
| `test_positions` | Copied original held-out positions in supplied order. |
| `score_positions` | Copied scoring subset in supplied order; may be empty. |
| `feature_columns` | Tuple of selected feature names, including consumed selections. |
| `target_columns` | Tuple of selected target names. |
| `predictions` | Name to copied DataFrame mapping, covering every test row. |
| `y_true` | Copied selected test targets, indexed by row_position; missing values retained. |
| `metadata` | Copied metadata for every test row; exact identity types retained. |
| `fold_metadata` | Deep copy of original fold metadata. |
| `selection` | Deep-copied consumed StudyRun for runner report integration, otherwise None. |

Records are mutable Python containers, not immutable audit records. If an adapter
retains its own contexts internally, the runner does not remove them from `model`.

## TrainingResult and prediction accessors

```text
TrainingResult(folds, layout, identity_columns, match_columns,
               target_perspective, definitions=<new dict>)
TrainingResult.predictions_by_fold(output='predict', *, scored_only=False) -> dict
TrainingResult.prediction_frame(output='predict', *, scored_only=False) -> DataFrame
```

| Field/argument | Meaning |
| --- | --- |
| `folds` | Ordered list of FoldResult records in requested processing order. |
| `layout` | Dataset layout. |
| `identity_columns` | Tuple defining row identities. |
| `match_columns` | Tuple defining whole-match identities. |
| `target_perspective` | Dataset target perspective retained without reinterpretation. |
| `definitions` | Deep copy of dataset definitions. |
| `output` | Exact prediction mapping key; defaults to predict. Missing keys raise KeyError for nonempty results. |
| `scored_only` | False retains all test rows; True selects each frame by score_positions in that order. |

The dictionary is keyed by original fold number and holds copied frames indexed by
`row_position`. `prediction_frame` concatenates them with index levels `fold_id`
and `row_position`. Overlapping held-out observations remain separate occurrences.
Pandas aligns differing columns across folds; absent class columns stay missing.
An empty run returns `{}` and a DataFrame with an empty two-level named index.

Numeric, common-column, complete-fold outputs can feed `reconstruct_paths`; use
all test rows. The path helper's own contract still applies, including identical
ordered columns and no reserved metadata column names. Varying probability
vocabularies and generic nonnumeric outputs require explicit preparation.

## Internal helpers and validation boundaries

| Helper | Responsibility |
| --- | --- |
| `runner._columns` | Resolve None/string/iterable into ordered distinct nonempty existing names. |
| `runner._positions` | Validate one-dimensional integer positions, bounds, duplicates and allowed emptiness; return an integer copy. |
| `runner._frame` | Copy positional rows and apply the original row_position index. |
| `TrainingRunner.run.fresh_model` | Nested factory wrapper that detects reused adapter/directly wrapped estimator identities within one run. |

No dataclass fully validates arbitrary manual construction. Scope validation is
performed when fitting; selecting no folds performs no fit-time frame validation.
Same-length reordered/replaced datasets, feature timing, leaking custom metadata,
nested shared estimator state and tampered selection provenance remain caller
responsibilities. The [guide](training.md#keep-future-selection-and-refitting-separate)
separates implemented fixed fitting from future optional search, nested evaluation,
scheduled refits and post-training reporters.
