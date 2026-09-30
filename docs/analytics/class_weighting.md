# Class balancing for classifiers

`ClassWeightReporter` describes class balance and produces observation weights
that a classifier can consume. Adding a report alone does not change training:
select it explicitly with `Candidate.weights_from`. With no weighting selected,
the existing fitting behavior is unchanged.

## In the experiment builder

1. Select a classifier under **Model & training**.
2. Under **Pre-training analysis → Fitted preparation**, add **Class Weight
   Reporter**, give it a name such as `balance`, and use **Per fold**, **Train**.
3. Choose its balancing mode. **Adjustable balancing**, power `0.5`, applies
   partial balancing; **Balanced** is equivalent to power `1`.
4. Under **Model & training → Candidate, feature selection and training controls**,
   enable **Weights from** and select `balance`.

For custom weights, first use **Discover feature columns**, select **Custom class
weights**, enable **Class weights**, then add discovered classes and their numeric
weights. These are original label values (for example, 5 total corners), not an
estimator's encoded class indexes. Supply every class present in fitting rows;
extra entries for classes absent from a particular fit are harmless.

Only the parameters of the chosen mode are displayed. Registered custom
calculations are selectable callbacks; the UI does not ask users to write code.
Registration is described below. Recipe saving, Python/notebook export, inner
search, refitting and saved-run reuse retain these settings. Changing the fitted
weighting configuration invalidates training reuse; changing a post-training
report still reuses the fitted model and predictions.

## Modes and equations

For a fitting population of size \(N\), let \(K\) be the number of observed classes
and \(n_c\) the number of observations in class \(c\). Adjustable balancing uses

\[
a_c=\left(\frac{N}{K n_c}\right)^\gamma,\qquad
w_i=\frac{a_{y_i}}{N^{-1}\sum_{j=1}^{N}a_{y_j}}.
\]

Thus the average observation weight is one. This controls the overall weight
scale while changing the relative influence of classes.

| Mode | Behavior |
| --- | --- |
| `none` | Every observation has weight 1 |
| `balanced` | Uses \(\gamma=1\); every observed class has equal total weight |
| `power` | Configurable nonnegative \(\gamma\): 0 gives uniform weights; 0.5 partially balances; values above 1 emphasize rare classes more strongly |
| `custom` | Maps original class labels to finite nonnegative weights, then normalizes to mean 1 |
| `callable` | Calls `calculator(context)` and normalizes its row-aligned weights to mean 1 |

For example, with 800 common-class and 200 rare-class training observations,
balanced weights are 0.625 and 2.5. Each class then has total weight 500.

The report displays class, count, frequency, mean observation weight and weighted
proportion. For a custom callback, weights can vary within a class; its displayed
weight is their mean. Its effective weighted proportion is

\[
p_c^{(w)}=\frac{\sum_{i:y_i=c} w_i}{\sum_i w_i}.
\]

## Which rows determine weights?

Each actual fit follows this order:

1. Select that fold's training population.
2. Reserve any calibration rows and early-stopping validation rows.
3. Run fitted preparation, including class weighting, on the remaining rows.
4. Fit the classifier with those identity-aligned observation weights.

Inner search folds, selected outer fits and final deployment refits recompute
weights independently. Test, calibration and early-stopping rows do not determine
class frequencies and receive no class weights from this feature. Monitoring and
evaluation metrics remain unweighted. Preprocessing still fits only fitting rows;
the observation weights are routed to the classifier, not to its scaler.

`type="overall"` is useful for a descriptive pooled view. Its output can be
consumed only if its row population exactly matches the consuming fit. It cannot
silently provide global weights to several different training folds. The report's
row-indexed weights and class summary are retained with fitted studies and saved
results; final refits retain their class summary in the fitted model.

The first implementation supports one selected classification target per fit.
An explicit report `target` selects it when the dataset has several labels.
Balancing does not bin labels, create synthetic examples or manufacture absent
classes. The classifier adapter determines its fitted classes from training
labels and preserves original class identities in probability outputs.

## Python configuration

```python
from sklearn.linear_model import LogisticRegression
from xdiyo_analytics.analysis import PreTrainingAnalysis
from xdiyo_analytics.reporting import ClassWeightReporter
from xdiyo_analytics.selection import Candidate
from xdiyo_analytics.training import EstimatorAdapter

candidate = Candidate(
    "Partially balanced classifier",
    lambda: EstimatorAdapter(LogisticRegression(max_iter=1000),
                             ("predict", "predict_proba")),
    pre_analysis=PreTrainingAnalysis({
        "balance": ClassWeightReporter(type="per_fold", partition="train",
                                       mode="power", power=0.5),
    }),
    weights_from="balance",
)
```

For direct fitting without report display, use the same calculation:

```python
from xdiyo_analytics.training import ClassWeightPolicy, TrainingRunner

runner = TrainingRunner(
    candidate.model_factory,
    weighting=ClassWeightPolicy(mode="power", power=0.5),
)
training = runner.run(dataset, split_plan)
```

`fit_predict()` and `refit_model()` also accept `weighting`. For explicit report
consumption, prepare it using `runner.selection_plan(dataset, split_plan)`, then
pass `analysis_report=report, weights_from="balance"` to `runner.run`. This keeps
report rows aligned when calibration or early stopping is enabled.

## Custom calculations and adapters

A callback receives the scoped `context.X`, `context.y`, and `context.metadata`.
Return a pandas Series indexed by **exactly** those original row positions;
reordered Series are aligned by identity, while missing or extra rows are rejected.
Weights must be finite, nonnegative and have positive total weight.

```python
def my_weight_calculation(context):
    # Example: a custom partial-balancing rule using this fit's labels only.
    y = context.y["corners"]
    counts = y.value_counts()
    return y.map((len(y) / (len(counts) * counts)) ** 0.5)

reporter = ClassWeightReporter(mode="callable", calculator=my_weight_calculation)
```

Register the callback in the same catalog supplied to `launch_ui`:

```python
from xdiyo_analytics.ui import catalog_for_ui, launch_ui

catalog = catalog_for_ui()
catalog.register("custom.corner_weights", my_weight_calculation,
                 category="callback", title="Custom corner weights")
builder = launch_ui(catalog=catalog)
```

An exported recipe preserves the registered callback's identifier, not its Python
implementation. In a separate script or notebook, register that callback again
and pass `catalog=catalog` to both `prepare_recipe` and `run_recipe`. Built-in
balancing modes need no extension registration.

Built-in routing supports sklearn classifiers accepting `sample_weight`, their
ordinary preprocessing pipelines, XGBoost/LightGBM classifiers, and supported
`PartialFitBackend` classifiers. Native `class_weight`, nondefault
`scale_pos_weight`, LightGBM `is_unbalance`, or another supplied observation-weight
source cannot be combined silently with the common weights. Disable the native
option or use that option alone. Unsupported estimators raise a clear error.

A custom adapter declares `supports_sample_weight = True` and consumes
`FitContext.sample_weight`; an iterative custom backend declares the same
capability and consumes it in `initialize`. This explicit declaration prevents a
generic adapter from silently ignoring requested balancing.

Class weighting changes the training objective. Its probabilities can therefore
need [probability calibration](probability_calibration.md) before interpreting
them as observed-population event probabilities. The optional calibrator uses
its separate, unweighted calibration rows and remains reusable downstream.
