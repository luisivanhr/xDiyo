# Negative binomial count regression

In **Model & training → Estimator**, choose **Negative Binomial regression** (`training.NegativeBinomialRegressor`). It models a count target such as total corners whose variability exceeds the Poisson mean–variance relation. The default prediction is the expected count. An optional **mode** output returns the most probable integer count from the same fitted NB2 distribution.

```python
from xdiyo_analytics.training import EstimatorAdapter, NegativeBinomialRegressor

def model_factory():
    return EstimatorAdapter(NegativeBinomialRegressor(
        dispersion=1.0, fit_intercept=True, max_iter=100, tol=1e-8,
    ))
```

The same estimator works inside a sklearn Pipeline; imputation/scaling of **X** is independent of the response distribution. Leave **Scale the target** disabled: this component requires labels in their original count units and rejects the target-transform wrapper. Zeros are allowed, negative/missing labels are not. The numerical GLM also accepts nonnegative fractional labels, although literal negative binomial observations are counts. Only one target column is supported initially. An all-zero fitting target is rejected because the intercept-only maximum has no finite log-mean intercept.

## Model and parameters

The initial implementation is a statsmodels NB2 generalized linear model with a log link:

\[
\eta_i=\beta_0+x_i^\top\beta,
\qquad \mu_i=\exp(\eta_i),
\qquad \mathbb E[Y_i\mid x_i]=\mu_i,
\qquad \operatorname{Var}(Y_i\mid x_i)=\mu_i+\kappa\mu_i^2.
\]

Here \(\kappa>0\) is `dispersion`, a **fixed, user-selected** value during each fit. It is not estimated automatically and is separate from the regularization strength `alpha`. You can compare dispersion values through the existing training-only model-selection grid. As dispersion tends to zero, the variance relation approaches Poisson's.

For nonnegative integer \(y\), the NB2 mass function is

\[
P(Y=y\mid\mu,\kappa)=
\frac{\Gamma(y+\kappa^{-1})}{\Gamma(\kappa^{-1})\Gamma(y+1)}
\left(\frac1{1+\kappa\mu}\right)^{\kappa^{-1}}
\left(\frac{\kappa\mu}{1+\kappa\mu}\right)^y.
\]

The coefficients are fitted at the specified dispersion, with optional L1, L2 or Elastic Net regularization described below. This implementation does not add exposure/offset inputs, zero inflation, automatic dispersion estimation or full predictive-distribution outputs. The existing adapter exposes `predict` using the selected mean/mode output. Betting probabilities or predictive intervals are not inferred automatically from that output.

| Parameter | Default | Meaning |
|---|---:|---|
| `dispersion` | 1.0 | Positive NB2 dispersion \(\kappa\). |
| `prediction` | `"mean"` | Return the expected count or the most probable count (`"mode"`). |
| `penalty` | `"none"` | `"none"`, `"l1"`, `"l2"` or `"elasticnet"`. |
| `alpha` | 0.1 | Nonnegative penalty strength, ignored with `penalty="none"`. |
| `l1_ratio` | 0.5 | L1 share for Elastic Net, between zero and one. |
| `fit_intercept` | True | Include \(\beta_0\), the baseline log mean. |
| `max_iter` | 100 | Maximum fitting iterations. |
| `tol` | 1e-8 | Solver convergence tolerance. |

The fitted estimator exposes `coef_`, `intercept_`, `dispersion_`, `prediction_`, `n_iter_`, `converged_`, `n_features_in_` and, for named DataFrames, `feature_names_in_`. A nonconverged fit emits sklearn's `ConvergenceWarning`; inspect the inputs or increase the iteration budget rather than assuming convergence. Its ordinary sklearn score is R²; choose an explicit metric such as MSE/MAE in model selection when that is the intended comparison. Save/load through the existing model helpers retains input preprocessing and the fitted output choice.

## Mean or mode predictions

Set `prediction="mode"` for an integer-valued modal count. This changes the returned summary of the distribution, **not the likelihood, fitted mean parameters, dispersion or penalty**. The public `predict_mean(X)` method always returns \(\mu=\exp(\beta_0+X\beta)\), regardless of the selected `predict(X)` output.

Using the fitted dispersion \(\kappa\), the returned mode is

\[
\widehat y_{\mathrm{mode}}=
\begin{cases}
\left\lfloor\mu(1-\kappa)\right\rfloor,&0<\kappa<1,\\
0,&\kappa\ge1.
\end{cases}
\]

When \(\mu(1-\kappa)=m\) is an exact positive integer, both \(m-1\) and \(m\) maximize the probability mass; this implementation chooses the upper mode \(m\). For example, mean 10 and dispersion 0.2 have modes 7 and 8, and the selected output is 8. A degenerate/numerically zero mean returns zero. Outputs remain floating arrays for compatibility with regression frames but contain integer-valued counts.

**With the default dispersion of 1, mode predictions are all zero.** This is the NB2 distribution's modal behavior, not an inverse-scaling bug. Selecting the mode does not guarantee a wider spread of predictions, better calibration or a lower error. It may narrow or collapse predictions: dispersion describes conditional outcome variability, while a single returned point prediction summarizes that variability. Evaluate the chosen output against the metric that matters for the experiment. R² assesses squared-error performance and is not an accuracy guarantee for a modal estimator.

`prediction_` and `dispersion_` freeze the output behavior of the fitted model; changing constructor parameters with `set_params` takes effect on a subsequent fit. Previously saved models without `prediction_` retain mean predictions. The training summary records the selected output, and coefficients remain **log-mean coefficients even when the report displays modal predictions**.

## Optional regularization

The default remains `penalty="none"`: older recipes retain their original unpenalized fit. Setting `alpha=0` also uses that exact fitting path. Regularization is an opt-in for a new fit; adding these options does not change a running experiment or a saved fitted model.

Let \(\ell_i(\beta_0,\beta;\kappa)\) be observation \(i\)'s NB2 log likelihood. The penalized objective is

\[
\min_{\beta_0,\beta}
-\frac1n\sum_{i=1}^n\ell_i(\beta_0,\beta;\kappa)
+\alpha\left[
\rho\sum_{j=1}^p|\beta_j|
+\frac{1-\rho}{2}\sum_{j=1}^p\beta_j^2
\right].
\]

The intercept \(\beta_0\) is **not penalized**. For `l1`, \(\rho=1\); for `l2`, \(\rho=0\); for `elasticnet`, \(\rho=\texttt{l1\_ratio}\). L1 can set slopes exactly to zero; L2 shrinks slopes without selecting exact zeros in general. `dispersion` controls the conditional variance and likelihood, whereas `alpha` controls coefficient shrinkage: these parameters serve different purposes and can be searched independently.

L1 and mixed Elastic Net use statsmodels' regularized GLM coordinate solver. Pure L2, including Elastic Net with zero L1 share, uses scipy BFGS on the same average negative log likelihood plus quadratic penalty, with the analytic gradient. The unpenalized path retains the original GLM solver. `max_iter` and `tol` apply to the selected solver. Regularized fits are not followed by an unpenalized refit on selected features. For the native L1/mixed solver, `n_iter_` is `None` because its result does not expose an iteration count; no count is fabricated.

Example for a future fit:

```python
model = NegativeBinomialRegressor(
    dispersion=0.5,
    penalty="elasticnet",
    alpha=0.1,
    l1_ratio=0.5,
    fit_intercept=True,
    max_iter=500,
    tol=1e-8,
)
```

Input feature scaling changes the penalty's meaning because it changes coefficient units. A fitted StandardScaler in the X pipeline often makes the shrinkage comparison easier across differently scaled predictors. Fit that preprocessor only on training inputs and tune `alpha` using training-only selection. Target scaling remains unsupported for this count model.

## Interpreting coefficients

The coefficient reporter labels these values **log-mean units**, rather than reporting them as additive corner-count effects. Holding the other fitted input features fixed, an increase of one fitted-feature unit multiplies the expected count by

\[
\frac{\mu(x+e_j)}{\mu(x)}=\exp(\beta_j).
\]

If inputs were standardized, that unit refers to a fitted input standard deviation. The intercept is also on the log scale. The reporter's “surviving coefficients” means coefficients exceeding its display tolerance. L1 and mixed Elastic Net can produce exact zero slopes; the default unpenalized fit does not perform sparse feature selection.

## Dependencies and verification

The training extra declares sklearn 1.6 or newer and statsmodels 0.14.5 or newer, within the package's upper bounds. Current verification uses sklearn 1.9.0 and statsmodels 0.14.5; the minimum-version boundary was not independently installed/tested in this session.

Focused tests compare coefficients and predictions with independently constructed statsmodels GLMs for multiple dispersion/intercept settings, check cloning/parameter search and input validation, and exercise an actual synthetic UI recipe, coefficient report and saved-model prediction. Public sklearn estimator checks are maintained alongside these mathematical checks. No production football run or notebook is retrained by this verification.

On 19 September 2026, the new estimator plus affected target-scaling and model-diagnostic suites passed **100 tests in 4.48 seconds**, with one standard sklearn array-API skip because `SCIPY_ARRAY_API` was unset. A separate public `check_estimator` call reported **51 passed / 1 skipped**, without expected-failure overrides. One upstream sklearn/pytest generator-parametrization deprecation warning remains; it is unrelated to model numerics. Numerical checks also confirm that the wrapper preserves native solver nonconvergence status and reports it explicitly.

Reference: [statsmodels NB2 family documentation](https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.family.NegativeBinomial.html).

Browser verification also selected the model from the Estimator dropdown, changed dispersion to 0.2, explicitly removed a previously enabled target scaler through Use raw count labels, and completed a synthetic 16-training/7-test match run. Saved predictions were finite expected corner counts (6.479–8.141), with the selected dispersion and raw-target setting retained in the run configuration. The browser console had no errors or warnings.

The later regularization extension passed **86 tests in 3.49 seconds**, including the original estimator/common checks, with the same one array-API skip. New checks compare L1, mixed Elastic Net and pure ridge against independent native statsmodels fits, verify likelihood-gradient/subgradient optimality, preserve an unpenalized intercept under strong shrinkage, assert exact unpenalized behavior when alpha is zero, check ridge iteration limits and exercise a tiny model-selection grid plus saved UI model. The tiny grid emitted a nonconvergence warning for one candidate at its configured tolerance; the status was retained rather than hidden. This testing used separate synthetic data/artifacts and did not alter or restart the user's active experiment.

An isolated browser check verified conditional penalty controls and numeric grid entries. A synthetic Elastic Net run with alpha 0.1 and dispersion 1 completed on 16 fitting/7 test rows, retained the penalty settings, and predicted 7.375 for every test fixture: all slopes were removed, leaving the valid unpenalized baseline. The browser console had no warnings/errors; the active user session was unchanged.

The mean/mode extension passed **151 tests across the three NB suites**, with two standard array-API skips. The new mode suite independently verifies maxima of scipy's NB probability mass, dispersion below/equal/above one, upper tie selection, numerical-zero means, unchanged fitted parameters for every penalty, frozen fitted choices, sklearn composition/checks, log-mean reporting and saved UI predictions. After strengthening the old-pickle test to remove both new attributes, the mode suite rerun passed **65 tests / 1 standard skip in 2.86 seconds**. Existing serialized models without a constructor or fitted prediction field continue to return means. No production experiment was retrained.

Browser verification selected Mode at dispersion 0.2 and completed an isolated 16-fitting/7-test run. The saved configuration retained `prediction="mode"`; its predictions were `[5, 5, 5, 5, 5, 5, 6]`. The UI explains zero modes at dispersion >=1 and the absence of a spread guarantee. No browser console warnings/errors were observed.
