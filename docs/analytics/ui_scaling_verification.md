# Scaling and prediction units

## What the pipeline currently does

**Feature scaling and target scaling are separate operations.** A scaler in the recipe's preprocessing list transforms the input features (X). It does not scale the target (y), and predictions do not need an inverse feature transformation. If the model was fitted against corner counts, its numeric prediction remains in corner-count units even when the input features were standardized.

For a standard feature scaler, the fitted mean and standard deviation come only from the fitting rows:

\[
z_j=\frac{x_j-\mu_{j,\mathrm{fit}}}{s_{j,\mathrm{fit}}}.
\]

Validation, test, and future rows use those same saved values. They do not recompute the mean or scale from their own population. If validation was reserved from a fold's training population, it is excluded from the scaler's fitting rows as well.

The existing paths are:

- `ModelFactory` places ordinary preprocessing steps inside the estimator pipeline. `TrainingRunner` passes that pipeline only the fitting rows.
- `BoostingAdapter` fits its preprocessing pipeline on the fitting rows, transforms its separately supplied validation inputs, and uses the same preprocessing state for held-out and future predictions.
- `PartialFitBackend` fits its preprocessor once when initializing an attempt and transforms validation and prediction inputs with the retained state. The existing iterative-control tests independently check its updates and losses.
- Explicit saved-model artifacts retain the fitted preprocessing together with the fitted estimator. `load_model(...).predict(...)` applies this saved state without fitting again.

The general adapter interface also permits custom implementations. A custom adapter is responsible for respecting these rules; the interface cannot make an arbitrary implementation use its inputs correctly.

## When an inverse transformation is needed

An inverse transformation is needed only when the **target itself** was transformed before fitting. For example, if target standardization is

\[
u=\frac{y-\mu_{y,\mathrm{fit}}}{s_{y,\mathrm{fit}}},
\]

then the model first predicts in transformed-target units:

\[
\widehat u=f(z),
\]

and the final prediction is restored afterward:

\[
\widehat y=\mu_{y,\mathrm{fit}}+s_{y,\mathrm{fit}}\widehat u.
\]

That inverse target transformation belongs between the estimator's internal prediction and the retained public prediction used by reporters and metrics. There is no inverse transformation of (X) in this sequence.

**Model & training → Scale the target** exposes StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer and PowerTransformer. Their catalog IDs use the `targets.` prefix. The recipe field is `target_transformer`, with `None` as the default. The model factory wraps the regression adapter in `TargetTransformAdapter`; each fresh fitting operation fits its own target transformer and public predictions are inverted exactly once. Validation labels use the fitting transformer. Original labels remain unchanged in the assembled dataset and retained results. The same contract applies inside grid/nested evaluation, final refitting, and saved-model prediction. These families also appear independently in input preprocessing.

In Python, wrap a fresh regression adapter with `TargetTransformAdapter(EstimatorAdapter(model), StandardScaler())`. An explicitly configured sklearn `TransformedTargetRegressor` also works through the ordinary estimator adapter; do not add another target wrapper unless that additional transformation is intentional.

For \(n\) fitting rows, the scaler uses

\[
\mu_{y,\mathrm{fit}}=\frac1n\sum_{i\in\mathrm{fit}}y_i,
\qquad
s_{y,\mathrm{fit}}^2=\frac1n\sum_{i\in\mathrm{fit}}(y_i-\mu_{y,\mathrm{fit}})^2.
\]

A constant target uses scale one, following StandardScaler. `with_mean=False` omits centering; `with_std=False` omits division by the scale. For multiple regression targets these quantities are fitted separately by column.

Supported initial adapters are ordinary sklearn regression through `EstimatorAdapter` and native LightGBM/XGBoost regression through `BoostingAdapter`, using numeric `predict`. Classifiers, rankers, other custom/iterative adapters and checkpoint resumption are not covered by this wrapper. Native objective constraints still apply: centered targets can be negative, so do not combine centering with objectives requiring nonnegative/positive labels, such as Poisson, Gamma or Tweedie. Disable target scaling, or use an appropriate non-centering transformation compatible with the objective.

Native training/validation losses are labelled **[transformed target]**. Evaluation metrics, plots and saved predictions use original units. Affine target-transform coefficient reports restore slopes and intercepts to original target units; input-feature units remain those of the fitted X pipeline. QuantileTransformer and PowerTransformer coefficients remain in transformed-target units, explicitly identified by the report: their inverse transformation is nonlinear and cannot be represented by a single original-unit slope.

Scaling y changes the meaning of a regularization parameter. For Lasso's squared-loss objective, with fixed X and corresponding intercept handling, \(\alpha_{\mathrm{original}}=s_y\alpha_{\mathrm{scaled}}\). Retune alpha when changing target scaling; an unchanged numeric alpha is not the same model. Keep `fit_intercept=True` ordinarily: disabling it with centered X and uncentered y forces the prediction baseline to zero.

Manually replacing dataset labels with transformed values without retaining a target transformer cannot be automatically reversed later. Such labels would be the model's declared target units. Likewise, a custom adapter returning transformed predictions must perform its own inverse target transformation before returning original-unit outputs.

XGBoost classification's restoration of original class labels is a separate categorical encoding operation. It is not numeric target scaling.

## Available transformations

All fitted extrema, quantiles, centers and power parameters below use fitting labels only, separately for each target column.

| Transformer | Main options | Interpretation |
|---|---|---|
| StandardScaler | `with_mean`, `with_std` | Mean and population-standard-deviation scaling. |
| MinMaxScaler | `feature_range`, `clip` | Maps fitting extrema to the requested interval; clipping is optional. |
| MaxAbsScaler | `copy` | Divides by the largest fitting absolute value, without centering. |
| RobustScaler | `with_centering`, `with_scaling`, `quantile_range`, `unit_variance` | Median and selected interquantile range; less sensitive to extremes. |
| QuantileTransformer | `n_quantiles`, `output_distribution`, `subsample`, `random_state` | Empirical rank mapping to uniform or normal output. |
| PowerTransformer | `method`, `standardize` | Fitted Yeo–Johnson or Box–Cox power, optionally followed by standardization. |

For an affine transform \(u=(y-a)/b\), inversion and coefficient conversion are

\[
\widehat y=a+b\widehat u,
\qquad \beta_j^{(y)}=b\beta_j^{(u)},
\qquad \beta_0^{(y)}=a+b\beta_0^{(u)}.
\]

For StandardScaler, \(a\) and \(b\) are the fitted mean and standard deviation when enabled. MaxAbsScaler uses \(a=0\), \(b=\max_{\mathrm{fit}}|y|\). RobustScaler uses the fitting median as \(a\) when centering is enabled, and \(b=Q_{q_h}-Q_{q_l}\) when scaling is enabled. Its `unit_variance=True` divides this range by the corresponding standard-normal quantile difference. Disabled scaling uses \(b=1\). sklearn's constant-column handling avoids division by zero.

MinMaxScaler with interval \([L,H]\) uses

\[
u=L+(H-L)\frac{y-y_{\min}}{y_{\max}-y_{\min}},
\qquad
y=y_{\min}+\frac{u-L}{H-L}(y_{\max}-y_{\min}).
\]

These formulas describe nonconstant columns; fitted sklearn scale/offset values also handle constant columns. `clip=True` clips transformed validation labels to the selected interval, losing information outside the fitting range. It does not constrain arbitrary model predictions to that interval. Leave clipping disabled when that loss is unwanted.

For QuantileTransformer, write \(\widehat F_{\mathrm{fit}}\) for the interpolated empirical distribution estimated from fitting labels. The two output choices are approximately

\[
u=\widehat F_{\mathrm{fit}}(y)
\quad\text{(uniform)},
\qquad
u=\Phi^{-1}\!\left(\widehat F_{\mathrm{fit}}(y)\right)
\quad\text{(normal)}.
\]

Inversion applies the fitted empirical quantile function, using \(u\) or \(\Phi(u)\), respectively. sklearn clips distribution tails to finite bounds and maps out-of-range values to fitted endpoints. **Quantile inversion cannot extrapolate beyond the fitted target range.** Validation labels beyond that range lose their distance in transformed-space losses, although original-unit evaluation retains their actual values. `n_quantiles` is limited by the fitting sample count; `subsample` and `random_state` control estimation when subsampling is used. QuantileTransformer changes the target representation; it **does not implement quantile regression** or produce prediction intervals by itself.

Box–Cox requires strictly positive labels and fits \(\lambda\):

\[
g_\lambda(y)=
\begin{cases}(y^\lambda-1)/\lambda,&\lambda\ne0,\\\log y,&\lambda=0.\end{cases}
\]

Yeo–Johnson also accepts zero and negative labels:

\[
g_\lambda(y)=
\begin{cases}
((y+1)^\lambda-1)/\lambda,&y\ge0,\ \lambda\ne0,\\
\log(y+1),&y\ge0,\ \lambda=0,\\
-((1-y)^{2-\lambda}-1)/(2-\lambda),&y<0,\ \lambda\ne2,\\
-\log(1-y),&y<0,\ \lambda=2.
\end{cases}
\]

With `standardize=True`, PowerTransformer additionally centers/scales these transformed fitting values. Its inverse reverses standardization and the fitted power. Box–Cox cannot be used directly on corner targets containing zero; no hidden offset is added. Power inverses also have domain limits for some fitted powers, so extreme transformed predictions may be invalid. The adapter raises a clear error when finite internal predictions invert to nonfinite values. A constant label may not identify a usable power; native sklearn errors are retained rather than silently selecting another transform.

Finally, nonlinear transformation changes the estimation problem. Squared-error training approximates the conditional mean in transformed space, and generally

\[
g^{-1}\!\left(\mathbb E[g(Y)\mid X]\right)
\ne \mathbb E[Y\mid X].
\]

The inverse restores units, not an automatic original-scale mean correction. Compare the resulting models using the retained original-unit predictions and metrics.

## Verification evidence

The bounded synthetic checks use fitting values near zero and deliberately large validation/test values. This makes accidental fitting on held-out rows detectable instead of relying on equal row counts alone. No production football data or GPU was used.

New tests in `tests/analytics/test_ui_scaling.py`: **4 passed in 1.74 seconds**.

1. The UI model factory's ordinary estimator pipeline fits its input scaler only on training rows, predicts using the retained transform, never calls inverse transformation on (X), and retains prediction parity after saving/loading.
2. LightGBM and XGBoost CPU cases capture the actual transformed arrays passed to native fitting and validation. Their fitted means/scales match only the fitting rows; validation/test inputs use those values; numeric target values remain unchanged. Saved/reloaded predictions match.
3. An explicit transformed-target regressor fits the target scaler only on training labels. Its internal standardized predictions are distinct from the public outputs; the public outputs match the independently calculated inverse transformation and original-unit truth. Saving/loading preserves those outputs.

Existing independent numerical checks: **5 passed, 43 deselected in 1.34 seconds**. These cover Ridge predictions against normal-equation calculations across layouts/target selections and iterative SGD updates/losses against hand calculations with fitting-only scaling.

Commands used with the project's Python environment:

```text
python -m pytest tests/analytics/test_ui_scaling.py -q --tb=short -p no:cacheprovider
python -m pytest tests/analytics/test_training_estimators.py tests/analytics/test_training_controls_scope.py -q --tb=short -p no:cacheprovider -k "pipeline_preprocessing_and_ridge or actual_sgd_updates_and_losses"
```

Those original X-scaling paths required no repair. The optional target-scaling extension has focused tests in `test_target_scaling.py`: independent sklearn transformed-target comparison; fitting-only label statistics; native validation transforms including XGBoost quantiles; original-unit coefficients; nested selection, final refit and saved/recovered predictions; fresh-state enforcement; and explicit unsupported contracts. `test_lasso_prediction_units.py` independently verifies the no-intercept zero-baseline behavior and explicit target-transform recovery. No production experiment was retrained for these checks.

The combined target-scaling/inverse-unit suite passed **16 tests in 3.39 seconds**. Existing estimator, model-selection freshness and UI workflow compatibility suites passed **55 tests in 14.61 seconds**. These checks establish the supported paths above, not arbitrary custom adapters or every model objective.

After adding all six transformer families, the focused suite passed **42 tests in 4.32 seconds**. `test_target_transformers.py` adds multi-target independent sklearn comparisons, per-target fitting-scope checks, affine coefficient conversion, nonlinear report units, saved-model parity, constant labels, quantile endpoints, Box–Cox positivity, maintained UI choices, `copy=False` label preservation and power inverse-domain errors. These were bounded synthetic checks; production models and notebooks were unchanged.

After the final inventory refresh, existing UI workflow and training/model-diagnostic suites passed **48 tests in 14.35 seconds**. Browser checks exercised all six choices and their parameter controls; a separate 23-row synthetic Ridge/quantile-target experiment completed with 16 fitting and 7 test rows, displaying original-unit corner predictions. No browser console warnings or errors were observed.
