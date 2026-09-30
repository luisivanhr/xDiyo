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

The model uses an NB2 distribution with a log link:

\[
\eta_i=\beta_0+x_i^\top\beta,
\qquad \mu_i=\exp(\eta_i),
\qquad \mathbb E[Y_i\mid x_i]=\mu_i,
\qquad \operatorname{Var}(Y_i\mid x_i)=\mu_i+\kappa\mu_i^2.
\]

Here \(\kappa>0\) is dispersion. It is **fixed by default**, or **estimated jointly with the coefficients** when `learn_dispersion=True`. It is separate from the regularization strength `alpha`. You can compare fixed values through the training-only model-selection grid. As dispersion tends to zero, the variance relation approaches Poisson's.

For nonnegative integer \(y\), the NB2 mass function is

\[
P(Y=y\mid\mu,\kappa)=
\frac{\Gamma(y+\kappa^{-1})}{\Gamma(\kappa^{-1})\Gamma(y+1)}
\left(\frac1{1+\kappa\mu}\right)^{\kappa^{-1}}
\left(\frac{\kappa\mu}{1+\kappa\mu}\right)^y.
\]

The coefficients support optional L1, L2 or Elastic Net regularization described below. Exposure/offset inputs and zero inflation are not implemented. The adapter retains `predict` using the selected mean/mode output and `count_distribution` containing the fitted mean and dispersion. [BetOutcomeReporter](bet_outcomes.md) uses those parameters for exact tail probabilities, including when point predictions use the mode.

| Parameter | Default | Meaning |
|---|---:|---|
| `dispersion` | 1.0 | Fixed positive NB2 dispersion, or starting value when learning it. |
| `learn_dispersion` | False | Jointly estimate one dispersion per fitted model from its training rows. |
| `prediction` | `"mean"` | Return the expected count or the most probable count (`"mode"`). |
| `penalty` | `"none"` | `"none"`, `"l1"`, `"l2"` or `"elasticnet"`. |
| `alpha` | 0.1 | Nonnegative penalty strength, ignored with `penalty="none"`. |
| `l1_ratio` | 0.5 | L1 share for Elastic Net, between zero and one. |
| `fit_intercept` | True | Include \(\beta_0\), the baseline log mean. |
| `max_iter` | 100 | Maximum fitting iterations. |
| `tol` | 1e-8 | Solver convergence tolerance. |

The fitted estimator exposes `coef_`, `intercept_`, `dispersion_`, `prediction_`, `n_iter_`, `converged_`, `n_features_in_` and, for named DataFrames, `feature_names_in_`. A nonconverged fit emits sklearn's `ConvergenceWarning`; inspect the inputs or increase the iteration budget rather than assuming convergence. Its ordinary sklearn score is R²; choose an explicit metric such as MSE/MAE in model selection when that is the intended comparison. Save/load through the existing model helpers retains input preprocessing and the fitted output choice.

## Learn dispersion

In **Model & training → Estimator → Negative Binomial regression**, enable
**Learn dispersion**. Leave it disabled to preserve fixed-dispersion behavior.
The **Dispersion** field becomes the starting value. Save the recipe and restart
the notebook kernel/relaunch the builder after updating the library.

```python
model = NegativeBinomialRegressor(
    learn_dispersion=True, dispersion=0.5,
    penalty="l2", alpha=0.1, max_iter=500,
)
model.fit(X_train, y_train)
print(model.dispersion_)  # fitted value; model.dispersion remains 0.5
```

The joint fitting objective is

\[
\min_{\beta_0,\beta,\,\kappa>0}
-\frac1n\sum_{i=1}^{n}\log P(Y_i=y_i\mid\mu_i,\kappa)
+\alpha\left[\rho\sum_j|\beta_j|+
\frac{1-\rho}{2}\sum_j\beta_j^2\right].
\]

Neither the intercept nor dispersion is penalized. With no penalty or zero
strength, this is maximum likelihood. Each fit learns a **single shared dispersion**,
not a feature-dependent dispersion for each match. Inner folds, outer folds and
final refits learn their own values using only the rows passed to their fit.
Held-out outcomes never enter this estimate.

The joint solver uses L-BFGS-B in coefficients and log dispersion. L1 terms use
nonnegative positive/negative slope parts, preserving exact zeros instead of
approximating the absolute-value penalty. For numerical stability, learned
dispersion is bounded to \([10^{-6},10^6]\); starting values outside this interval
are clipped. These numerical bounds do not restrict fixed-dispersion fits.
The lower boundary approximates a Poisson fit and can occur for data with little
extra count variability. Boundary estimates and convergence remain inspectable.

`dispersion_estimated_` records whether learning was enabled, and
`dispersion_at_boundary_` records a numerical-boundary solution. The training
summary retains both flags and `dispersion_`. Mean/mode predictions, retained
distributions and saved-model betting probabilities all use the fitted value.
Constructor parameters remain unchanged, preserving sklearn cloning and grids.

When learning dispersion, search regularization parameters rather than treating
different dispersion starting values as different fixed assumptions. Estimated
dispersion describes conditional outcome uncertainty; learning it does not force
the point predictions to have the same spread as observed counts. Joint optimization
can cost more than a fixed-dispersion fit and can converge to a local solution.

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

With fixed dispersion, L1 and mixed Elastic Net use statsmodels' regularized GLM coordinate solver. Pure L2, including Elastic Net with zero L1 share, uses scipy BFGS on the same average negative log likelihood plus quadratic penalty, with the analytic gradient. The unpenalized fixed path retains the original GLM solver. Learned dispersion uses the joint solver described above for all penalties. `max_iter` and `tol` apply to the selected solver. Regularized fits are not followed by an unpenalized refit on selected features. For the native fixed-dispersion L1/mixed solver, `n_iter_` is `None` because its result does not expose an iteration count; no count is fabricated.

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

The learned-dispersion extension passed **321 tests, with 3 optional Array API
environment skips**, across the learned/fixed NB, mode, penalties, betting and UI
suites. Numerical checks compare against independent statsmodels discrete NB2
MLE and likelihood/subgradient calculations. They also verify unchanged fits when
outer test outcomes change, sklearn cloning/search/checks, saved-model parity and
native betting parameters. The browser's Learn dispersion checkbox produced a
valid synthetic run and retained the fitted value, without console errors.

The training extra declares sklearn 1.6 or newer and statsmodels 0.14.5 or newer, within the package's upper bounds. Current verification uses sklearn 1.9.0 and statsmodels 0.14.5; the minimum-version boundary was not independently installed/tested in this session.

Focused tests compare coefficients and predictions with independently constructed statsmodels GLMs for multiple dispersion/intercept settings, check cloning/parameter search and input validation, and exercise an actual synthetic UI recipe, coefficient report and saved-model prediction. Public sklearn estimator checks are maintained alongside these mathematical checks. No production football run or notebook is retrained by this verification.

On 19 September 2026, the new estimator plus affected target-scaling and model-diagnostic suites passed **100 tests in 4.48 seconds**, with one standard sklearn array-API skip because `SCIPY_ARRAY_API` was unset. A separate public `check_estimator` call reported **51 passed / 1 skipped**, without expected-failure overrides. One upstream sklearn/pytest generator-parametrization deprecation warning remains; it is unrelated to model numerics. Numerical checks also confirm that the wrapper preserves native solver nonconvergence status and reports it explicitly.

Reference: [statsmodels NB2 family documentation](https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.family.NegativeBinomial.html).

Independent learned-dispersion likelihood reference:
[statsmodels discrete NB2 model](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.NegativeBinomial.html).

Browser verification also selected the model from the Estimator dropdown, changed dispersion to 0.2, explicitly removed a previously enabled target scaler through Use raw count labels, and completed a synthetic 16-training/7-test match run. Saved predictions were finite expected corner counts (6.479–8.141), with the selected dispersion and raw-target setting retained in the run configuration. The browser console had no errors or warnings.

The later regularization extension passed **86 tests in 3.49 seconds**, including the original estimator/common checks, with the same one array-API skip. New checks compare L1, mixed Elastic Net and pure ridge against independent native statsmodels fits, verify likelihood-gradient/subgradient optimality, preserve an unpenalized intercept under strong shrinkage, assert exact unpenalized behavior when alpha is zero, check ridge iteration limits and exercise a tiny model-selection grid plus saved UI model. The tiny grid emitted a nonconvergence warning for one candidate at its configured tolerance; the status was retained rather than hidden. This testing used separate synthetic data/artifacts and did not alter or restart the user's active experiment.

An isolated browser check verified conditional penalty controls and numeric grid entries. A synthetic Elastic Net run with alpha 0.1 and dispersion 1 completed on 16 fitting/7 test rows, retained the penalty settings, and predicted 7.375 for every test fixture: all slopes were removed, leaving the valid unpenalized baseline. The browser console had no warnings/errors; the active user session was unchanged.

The mean/mode extension passed **151 tests across the three NB suites**, with two standard array-API skips. The new mode suite independently verifies maxima of scipy's NB probability mass, dispersion below/equal/above one, upper tie selection, numerical-zero means, unchanged fitted parameters for every penalty, frozen fitted choices, sklearn composition/checks, log-mean reporting and saved UI predictions. After strengthening the old-pickle test to remove both new attributes, the mode suite rerun passed **65 tests / 1 standard skip in 2.86 seconds**. Existing serialized models without a constructor or fitted prediction field continue to return means. No production experiment was retrained.

Browser verification selected Mode at dispersion 0.2 and completed an isolated 16-fitting/7-test run. The saved configuration retained `prediction="mode"`; its predictions were `[5, 5, 5, 5, 5, 5, 6]`. The UI explains zero modes at dispersion >=1 and the absence of a spread guarantee. No browser console warnings/errors were observed.
