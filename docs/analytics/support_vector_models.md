# Support vector models

The experiment builder's **Model & training → Model** selector includes:

- **SVR — Support vector regression** for one numeric target.
- **SVC — Support vector classification** for one discrete target. Each fitting fold determines its own classes; no class-count setting is needed.

These use scikit-learn's native estimators through the existing `EstimatorAdapter`.
JSON recipes, Python/notebook exports, grid search, persisted models and future
predictions use the same paths as other sklearn models. Existing recipes and
model defaults are unchanged.

## Configuration

Keep an imputer and a scaler under **Preprocessing** when features contain missing
values or have different scales. Both are fitted only on the model's fitting rows.
SVR supports the existing label scaler and inverse transformation. SVC requires
discrete class labels; leave label scaling disabled.

The main controls are `C` (inverse regularization), `kernel`, `gamma`, and SVR's
`epsilon` (error-insensitive tube width in the model's target units). Gamma offers
`scale`, `auto`, or a numeric value. Polynomial degree and kernel offset appear
for the kernels that use them. The builder offers linear, polynomial, RBF and
sigmoid kernels; a precomputed kernel needs a separate Gram-matrix workflow.
Large kernel models can be costly to fit, especially during nested searches.

Grid keys refer to model parameters, for example:

```python
grid = {"C": [0.5, 1.0, 2.0], "gamma": ["scale", 0.1]}
# For SVR, epsilon may also be searched.
```

## Class probabilities and balancing

For binary SVC with fully chronological calibration, use the new
[decision-margin calibration route](temporal_svc_calibration.md) with
`probability=False`. The following describes the existing internal-probability route.

For probability-based reporters or the existing `ProbabilityCalibrator`, enable
SVC's **Enable class probabilities** before fitting. The recipe automatically
retains `predict_proba` with the original class labels. The default is disabled.
SVC's internal probability estimation adds five-fold fitting and is not a
chronological split. The separate calibration layer uses its reserved chronological
training tail; neither path uses the outer test labels. SVC's `predict` can differ
from the largest probability's class.

Compatibility note: scikit-learn 1.9 still supports `probability=True` but emits a
deprecation warning announcing removal in 1.11. This integration uses the current
native option; it does not silently substitute a different calibration algorithm.

For balancing, either enable native **Class weight → balanced**, or configure a
**ClassWeightReporter** and select it under **Weights from** for power/custom
weights. Weights are computed from fitting rows and routed to SVC. The pipeline
rejects combining native class weights with reporter weights to avoid applying
the adjustment twice.

## Recipe model blocks

```python
from xdiyo_analytics.ui.recipe import node

recipe["model"] = node("sklearn.svm.SVR", C=1.0, kernel="rbf", gamma="scale", epsilon=0.1)

# Alternatively, with discrete labels:
recipe["model"] = node("sklearn.svm.SVC", C=1.0, kernel="rbf", gamma="scale",
                       probability=True, random_state=17)
```

Both models support one target per fit. They do not expose native incremental
training or early stopping; `max_iter=-1` means an unlimited solver iteration count.
