# Fit one configuration and retain held-out predictions

The training layer connects an assembled `ModelDataset` and prepared `SplitPlan`
to a fresh model for each selected fold. It retains predictions and identities for
later evaluation. Every example below uses small **synthetic data**; these are API
examples, not a football experiment or a model recommendation.

See the [complete API/helper reference](training_reference.md),
[verification coverage](training_documentation_checklist.md) and
[minimal ninth notebook](../../notebooks/09_training_quickstart.ipynb).

## Start from aligned data and prepared folds

Ordinary project usage obtains data from [dataset assembly](datasets.md) and folds
from [the split library](splits.md). This self-contained example constructs both
directly so the training behavior is visible without loading football data.

```python
import numpy as np
import pandas as pd
from xdiyo_analytics.datasets import ModelDataset
from xdiyo_analytics.splits import Fold, SplitPlan
from xdiyo_analytics.training import EstimatorAdapter, TrainingRunner, fit_predict

positions = np.arange(12)
X = pd.DataFrame({
    'recent': [2., 3., np.nan, 4., 2., 5., 3., 6., 4., 7., 5., 8.],
    'venue': positions % 2,
    'trend': positions / 10,
})
y = pd.DataFrame({'quantity': 3. + positions / 2 + positions % 3})
metadata = pd.DataFrame({
    'competition_id': 17, 'season_id': 2024,
    'event_id': pd.Series([2**63 + 101 + int(i) for i in positions], dtype='uint64[pyarrow]'),
    'kickoff_at': pd.date_range('2024-01-01', periods=12, freq='2D', tz='UTC'),
})
keys = ('competition_id', 'season_id', 'event_id')
dataset = ModelDataset(X, y, metadata, 'match', keys, keys, 'total',
                       {'features': {'kind': 'synthetic'}, 'label': {'kind': 'synthetic'}})
plan = SplitPlan([
    Fold(np.arange(6), np.arange(6, 9), np.array([7, 8]), {'stage': 'first'}),
    Fold(np.arange(9), np.arange(9, 12), np.array([], dtype=int), {'stage': 'second'}),
], n_rows=12, row_order=positions)
```

Positions are integer offsets into the original dataset, independent of its index
labels. Training, test and scoring scopes each retain whole matches; with a
team-match layout this keeps both team observations together. For fold \(f\),

\[
T_f\cap V_f=\varnothing,\qquad S_f\subseteq V_f.
\]

\(T_f\) is the training scope, \(V_f\) the held-out test scope and \(S_f\) the
scoring subset. Both training and test must be nonempty. An empty scoring subset
is valid and does not suppress prediction. The runner preserves each supplied
array's order and original fold numbers.

## Put fitted preprocessing inside a fresh model

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

def model_factory():
    return EstimatorAdapter(make_pipeline(
        SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=1.0),
    ))

result = TrainingRunner(model_factory).run(dataset, plan)
result.prediction_frame()
```

Each factory call constructs the adapter, preprocessing and estimator afresh.
There is one `fit` and one adapter `predict` call per selected fold, with no final
full-data fit. Reusing an adapter or the same directly wrapped estimator within
one run raises. Custom factories remain responsible for avoiding shared nested
objects or global fitted state; the runner does not clone arbitrary models.

The pipeline learns its median, scaling and coefficients from training rows only.
For the imputed training column \(x_j\), standardization uses

\[
\mu_{fj}=\frac{1}{|T_f|}\sum_{i\in T_f}x_{ij},\qquad
s_{fj}^{2}=\frac{1}{|T_f|}\sum_{i\in T_f}(x_{ij}-\mu_{fj})^2,
\qquad z_{ij}=\frac{x_{ij}-\mu_{fj}}{s_{fj}}.
\]

The same fitted quantities transform held-out values; zero-variance columns use
scale one. This is the estimator's behavior, not a transformation performed by
the runner. [StandardScaler reference](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

For one target, the example's Ridge estimator minimizes

\[
\sum_{i\in T_f}(y_i-b-z_i^{\mathsf T}\beta)^2
+\alpha\lVert\beta\rVert_2^2.
\]

The intercept is unpenalized. With centered training matrix \(Z_f\), centered
target \(y_f^c\), positive \(\alpha\), and standardized test matrix \(Z_f^*\),
the independent verification uses

\[
\widehat\beta_f=(Z_f^{\mathsf T}Z_f+\alpha I)^{-1}Z_f^{\mathsf T}y_f^c,
\qquad \widehat y_f=\bar y_f\mathbf 1+Z_f^*\widehat\beta_f.
\]

The test computes this with a linear solve, not explicit inversion. Multiple
targets use separate target columns in the same expression.
[Ridge objective](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

Fit imputers, scalers and other learned transformations inside the supplied
pipeline. Fitting them on the full dataset would leak held-out information.
[Pipeline and leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

## Distinguish prediction from scoring

```python
all_test = result.predictions_by_fold()
scoring_rows = result.predictions_by_fold(scored_only=True)
scoring_frame = result.prediction_frame(scored_only=True)
pd.DataFrame([
    {'fold': fold.fold_id, 'train_rows': len(fold.train_positions),
     'test_rows': len(fold.test_positions), 'score_rows': len(fold.score_positions),
     'features': fold.feature_columns}
    for fold in result.folds
])
```

`all_test` retains three predictions per fold in this example. `scoring_rows`
retains two from fold zero and an empty frame from fold one. Each dictionary value
uses the original `row_position` index. The combined frame uses the pair
`(fold_id, row_position)`. Repeated held-out observations remain separate fold
occurrences, with no averaging.

`FoldResult` also retains the fitted adapter, selected feature/target names,
training/test/scoring positions, copied test metadata, `y_true`, fold metadata
and any consumed selection. Returned prediction helper frames are copies.
Missing test targets stay missing; selected missing training targets raise before
model construction. No targets are dropped or imputed. Features, including NaNs,
reach the adapter unchanged. Other estimator constraints, such as rejecting
infinities, remain estimator responsibilities.

Metadata is available to custom adapters but is never appended automatically to
`X`. The prediction context has no `y` field. If caller-supplied metadata itself
contains outcomes, a custom adapter must avoid using them as predictors.

## Choose fixed columns or consume a training selection

Fixed feature and target names are ordered and explicit; omitted options use all
columns. A single string selects one column. Fold subsets retain their original
numbers and the requested processing order.

```python
fixed_subset = TrainingRunner(
    model_factory, feature_columns=['trend', 'recent'], target_columns='quantity',
).run(dataset, plan, fold_ids=[1])

one_fold = fit_predict(
    dataset, plan.folds[0], model_factory, fold_id=0,
    feature_columns=['recent', 'trend'], target_columns='quantity',
)
```

Standalone `fit_predict` is the one-fold building block. It calls its factory once
and cannot compare that returned object with earlier separate calls. Use a fresh
factory there as well.

Existing pre-training selectors can supply names explicitly:

```python
from xdiyo_analytics.analysis import PreTrainingAnalysis
from xdiyo_analytics.reporting import TopKCorrelationSelector

analysis = PreTrainingAnalysis({
    'training choice': TopKCorrelationSelector(
        type='per_fold', partition='train', k=2, method='spearman',
    ),
}).run(dataset, split_plan=plan)

selected_result = TrainingRunner(model_factory).run(
    dataset, plan, analysis_report=analysis, features_from='training choice',
)
pd.DataFrame([{'fold': f.fold_id, 'selected': f.feature_columns}
              for f in selected_result.folds])
```

The runner consumes the stored selection without recomputing it. A matching
fold result takes precedence; otherwise one named overall result may be used.
Its calculation rows \(A_f\) must satisfy

\[
A_f\subseteq T_f,
\]

and its layout must match the dataset. A pooled/consensus selection influenced by
this fold's held-out rows is rejected. An overall selection can be valid for an
independent holdout when all of its calculation rows belong to that holdout's
training scope. An empty selection is rejected because training requires at
least one feature.

This checks recorded row membership and layout, not data hashes, target provenance
or how a custom selector reached its decision. The caller must bind the report,
split plan and current dataset to the same unchanged row population. Matching row
counts do not detect reordered or replaced data. Do not combine `features_from`
with the runner's fixed `feature_columns`; supplying a report also requires an
explicit `features_from` name. The consumed `StudyRun` is copied into the result.

## Retain labels and class probabilities

An estimator adapter can request `predict`, `predict_proba`, or both. Probabilities
use fitted class labels in `(target, class)` columns; a class missing from one
fold's fitted vocabulary stays missing when fold frames are concatenated.

```python
from copy import deepcopy
from sklearn.dummy import DummyClassifier

classification = deepcopy(dataset)
classification.y = pd.DataFrame({'category': positions % 3})
classifier = TrainingRunner(lambda: EstimatorAdapter(
    DummyClassifier(strategy='prior'), ('predict', 'predict_proba'),
))
classified = classifier.run(classification, plan)
classified.prediction_frame('predict_proba')
```

This synthetic frequency classifier illustrates output identities. It makes no
claim about predictive quality. The adapter checks shape, class-column identities
and pandas row order; it does not assess calibration or validate probability sums.

For multi-target classifiers, each target needs its own probability array and
class vocabulary. A compatible estimator can supply those as lists:

```python
from sklearn.multioutput import MultiOutputClassifier

multi = deepcopy(dataset)
multi.y = pd.DataFrame({'binary': positions % 2, 'three': positions % 3})
multi_result = TrainingRunner(lambda: EstimatorAdapter(
    MultiOutputClassifier(DummyClassifier(strategy='prior')),
    ('predict', 'predict_proba'),
), target_columns=['three', 'binary']).run(multi, plan)
multi_result.prediction_frame('predict_proba')
```

The declared target order is retained.
[MultiOutputClassifier probability contract](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html#sklearn.multioutput.MultiOutputClassifier.predict_proba).

## Implement a custom adapter

Custom frameworks can implement the structural `fit(context)` / `predict(context)`
contract without importing scikit-learn or subclassing `ModelAdapter`.

```python
class MeanModel:
    def fit(self, context):
        self.means = context.y.mean()

    def predict(self, context):
        values = np.tile(self.means.to_numpy(), (len(context.X), 1))
        return {'predict': pd.DataFrame(values, index=context.X.index,
                                        columns=self.means.index)}

custom_result = TrainingRunner(MeanModel).run(dataset, plan)
custom_result.prediction_frame()
```

`fit` mutates the fresh adapter in place; its return value is ignored. `predict`
returns a nonempty mapping of output names to DataFrames. Custom named outputs may
contain embeddings or other model-specific columns. Every output must preserve
the exact input index and row order and have unique, nonempty columns.

Contexts expose exact match tuples through `groups`, not ranking group sizes.
A custom ranker adapter may form contiguous groups and reorder internally, but
must restore output row order. No ranker-specific training is built in. The adapter
also owns framework/device handling, custom fit arguments and reproducibility.

## Reuse numeric predictions in CPCV paths

```python
from xdiyo_analytics.splits import CPCV, create_split_plan, reconstruct_paths

cpcv_plan = create_split_plan(dataset, CPCV(n_blocks=4, n_test_blocks=2))
cpcv_result = TrainingRunner(MeanModel).run(dataset, cpcv_plan)
paths = reconstruct_paths(cpcv_plan, cpcv_result.predictions_by_fold())
paths.head()
```

Supply all folds and all test rows. Reconstruction needs numeric frames with
identical ordered columns; it cannot directly combine varying class vocabularies
or generic string outputs. The result retains `path_id`, `row_position`, `fold_id`
and prediction columns. Reconstructed paths share observations/models and are not
independent evidence. This example uses synthetic point-interval folds and makes
no temporal-validity claim about a real experiment.

## Keep future selection and refitting separate

| Workflow | Current status | Responsibility |
| --- | --- | --- |
| One fixed configuration per supplied fold | Implemented | `TrainingRunner` or standalone `fit_predict`. |
| Optional stored feature selection | Implemented | Explicit consumption of a selection whose calculation scope is inside training. |
| Search followed by an untouched holdout | Future | Select candidates using development data, then evaluate once on the holdout. |
| Nested evaluation | Future | Repeat candidate/target-informed selection within each outer training scope. |
| Scheduled refitting | Future | A separate policy would decide later fits; none occurs here. |
| Post-training reporters and performance analysis | Future | Consume predictions and fitting provenance from structured results. |

There is no compulsory inner loop and no search/refit placeholder argument in the
current API. A future runner may compose the one-fold function, but needs explicit
scope rules. Fitted feature selection must be repeated within inner training when
used during model search; a selection learned on all outer training rows can leak
into its inner validation.

Histories, ratings, information cutoffs and fold-aware state reconstruction belong
upstream. This runner consumes precomputed values and cannot repair future
information already present in them. It does not enforce past-only training on a
manually supplied or retrospective split. No experiment is selected or run here.
