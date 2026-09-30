# Probability calibration and exact bet probabilities

Calibration is optional fitted preparation after the base model, shared by all
downstream consumers. It is not fitted by a reporter. The first implementation
calibrates class-probability distributions, including classifiers of exact corner
counts. Point predictions alone and parametric negative-binomial distributions
are not converted into calibrated probabilities implicitly.

For UI-driven O/U selection and shared profit accounting, see
[BetOutcomeReporter](bet_outcomes.md). It consumes upstream probabilities;
`CalibrationReporter` continues to assess individual classes.

## Enable it in the builder

Open **Model & training → Candidate, feature selection and training controls**.
Enable **Calibration**, choose **Probability Calibrator**, and configure:

| Setting | Meaning |
| --- | --- |
| `method="temperature"` | Temperature, sigmoid, or isotonic calibration |
| `fraction=0.2` | Latest 20% of distinct training kickoff batches, rounded up, reserved for calibration |
| `time_column="kickoff_at"` | Timestamp defining the chronological tail |
| `update_predict=True` | Produce point class predictions from the calibrated argmax; disable to preserve the model's original point output |

Leave Calibration disabled to retain the existing workflow. The builder enables
`predict_proba` automatically for ordinary estimator/boosting adapters when
calibration is selected. The model must support class probabilities; selecting it
for Lasso, a quantile regressor or NegativeBinomialRegressor produces an explicit
unsupported-output error. A custom adapter must implement the same named output.
Save your recipe and restart/relaunch the backend after installing these changes.

## Python configuration

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xdiyo_analytics.training import EstimatorAdapter, ProbabilityCalibrator
from xdiyo_analytics.selection import Candidate

def classifier():
    return EstimatorAdapter(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        prediction_methods=("predict", "predict_proba"),
    )

candidate = Candidate(
    "Calibrated corner counts",
    classifier,
    calibration=ProbabilityCalibrator(method="temperature", fraction=0.2),
)
# result = experiment.run(prepared, model=candidate, post_analysis=reporters)
```

`TrainingRunner(..., calibration=...)`, `fit_predict(..., calibration=...)` and
`refit_model(..., calibration=...)` expose the same option. A `RefitPolicy` uses
its candidate's calibration configuration and fits a fresh calibrator on its own
reserved population. Each inner/outer fold gets fresh model and calibration state.
Grid search can vary `candidate.calibration` or its nested parameters through the
existing candidate configuration paths. Calibration is part of the fitting cache
key; changing it requires a new fitted result, whereas changing post reporters does not.

## Which rows are used?

For each supplied training population:

1. Reserve its latest calibration fraction by distinct kickoff batches.
2. Reserve any early-stopping validation from the remaining population.
3. Fit feature selection, input/target preprocessing and the base model only on
   the remaining fitting rows.
4. Predict the calibration rows without supplying their labels to the base model.
5. Fit the calibrator against those reserved observed labels.
6. Apply the frozen model and calibrator to the outer test rows or future fixtures.

Whole matches and equal kickoff times remain together. Explicit monitoring rows
must not overlap calibration rows. Fitting and calibration must both be nonempty.
Calibration labels are unweighted, so a class-weighted classifier can be adjusted
using the observed class frequencies in the reserved population. The base model is
not subsequently refitted on calibration rows, which would change the predictions
on which the calibrator was trained.

`fold.calibration_positions` and `refit.calibration_positions` expose provenance;
`fit_positions` and `validation_positions` retain their distinct meanings. Model
selection maps these positions back to original dataset positions. Training
summaries retain calibration method, row count, fraction and time column.

For independently constructed held-out or out-of-fold probabilities, the same
component is usable directly:

```python
calibrator = ProbabilityCalibrator(method="sigmoid")
calibrator.fit(oof_probabilities, oof_labels)
future_probabilities = calibrator.transform(raw_future_probabilities)
```

Both inputs are aligned DataFrames; probability columns are a `(target, class)`
MultiIndex. In this direct API the caller supplies the population; `fraction` does
not split it again. Automatic OOF model fitting is not implemented by this first
pipeline integration, which uses the dedicated chronological holdout described above.

## Outputs and persistence

The fitted `CalibratedAdapter` wraps the base model and fitted calibrator together.
It exposes:

- `predict_proba`: calibrated probabilities, used by downstream probability metrics.
- `predict_proba_raw`: the original model probabilities for comparison.
- `predict`: calibrated argmax classes when enabled, otherwise the original output.

Target/class order and row identities are preserved. Temperature scaling preserves
class ranking (aside from exact ties), so it primarily changes confidence, not the
winning class. Sigmoid and isotonic can change the winning class. None of these
methods guarantees better accuracy or future calibration.

Saved fold models and deployment refits retain the calibrator and fitted scalers.
`FootballExperiment.load` and `load_model` restore them without fitting again.
Report-only refresh uses retained calibrated predictions. `load_models=False`
still supports numerical-only inspection. Optimizer checkpoints resume the base
fit; calibration is fitted after that base fit completes, not mid-optimizer step.

Calibration labels must be represented in the model's probability classes; an
unseen label raises a clear error instead of dropping the row or inventing class
support. Missing/nonfinite probabilities, non-unit sums, missing labels and altered
class-column order are rejected. Very rare counts may require a broader base-fit
population or a deliberately chosen classification target.

## Calibration equations

Let \(p_{ik}\) denote raw probability of class \(k\) for observation \(i\), and
\(y_i\) its observed class. Logarithms use probabilities clipped to
\([10^{-12},1-10^{-12}]\) for numerical stability.

Temperature scaling fits one positive temperature per target:

\[
q_{ik}(T)=\frac{\exp(\log p_{ik}/T)}{\sum_j\exp(\log p_{ij}/T)},
\qquad
T^*=\arg\min_{T\in[e^{-5},e^5]}
-\frac{1}{n}\sum_i\log q_{i,y_i}(T).
\]

Sigmoid calibration fits a separate logistic map for each class against its
one-vs-rest binary labels:

\[
z_{ik}=\log\frac{p_{ik}}{1-p_{ik}},\qquad
s_{ik}=\sigma(a_k z_{ik}+b_k),\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
\]

The parameters minimize binary log loss, with each coefficient bounded to
\([-100,100]\) for numerical stability. When a class is uniformly absent/present
in calibration labels, its map is the smoothed constant \((n_k+1)/(n+2)\).

Isotonic calibration fits a nondecreasing map \(g_k\) by squared error against
the one-vs-rest labels, giving \(s_{ik}=g_k(p_{ik})\); out-of-range inputs use
the fitted endpoint value. Sigmoid and isotonic outputs are normalized jointly:

\[
q_{ik}=\frac{s_{ik}}{\sum_j s_{ij}}.
\]

If all isotonic maps return zero on a new row, that row retains its raw normalized
distribution. Both methods therefore produce one coherent class distribution;
nested count tails formed from it remain monotone. See the official
[scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html)
for the held-out fitting principle and method tradeoffs. Our component operates on
retained probability frames and is independent of a particular estimator framework.

## Exact bet boundaries: no midpoint averaging

For integer counts \(Y\) and any literal line \(\ell\):

\[
P(\text{Under }\ell)=P(Y<\ell)=F(\lceil\ell\rceil-1),
\qquad
P(\text{Over }\ell)=P(Y>\ell)=1-F(\lfloor\ell\rfloor).
\]

In particular:

\[
P(\text{Under }7.5)=\sum_{k=0}^{7}p_k,
\qquad
P(\text{Over }7.5)=\sum_{k=8}^{\infty}p_k.
\]

No probability is averaged between counts 7 and 8. Under 7 wins at 0–6; count 7
is a push under `on_equal="push"`, or a loss under `on_equal="loss"`.

```python
from xdiyo_analytics.features import Stat
from xdiyo_analytics.labels import MatchTotal, BetOption
from xdiyo_analytics.evaluation import bet_probabilities

corners = MatchTotal(Stat("ALL", "Match overview", "cornerKicks"))
under = BetOption(corners, selection="under", line=7.5)
resolved = bet_probabilities(calibrated_probabilities, under, target="corners")
# Columns: p_win, p_push, p_loss. Original row identities are retained.
```

The target must describe `option.source`; callers make that association explicitly.
Numeric count columns must denote exact nonnegative integers, not interval bins,
midpoints or an overflow category. Sum over all classes supplied by the classifier;
do not truncate/renormalize a subset before resolving a bet. Outcome sources also
support win/draw/loss and draw-no-bet; Above sources support yes/no. Voids require
actual status-based settlement later and are not assigned an invented probability.
Quarter lines retain the current BetOption literal-threshold semantics; Asian
split-stake settlement and parlays remain separate future work.

### Negative binomial predictions

`negative_binomial_bet_probabilities(mean, dispersion, option)` uses the full NB2
distribution. Pass an indexed mean Series, and a scalar or identically indexed
dispersion Series. Obtain the conditional mean with `predict_mean` even when the
regressor's point prediction is configured to return the mode.

\[
\operatorname{Var}(Y)=\mu+\alpha\mu^2,
\qquad r=\alpha^{-1},\qquad p=(1+\alpha\mu)^{-1}.
\]

Use \(F\) for the lower tail, survival function for the upper tail, and PMF for
integer-line equality. Dispersion zero uses the Poisson limit. This includes the
entire upper tail without enumerating a finite maximum count. See
[SciPy's NB distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html).
This helper computes native probabilities; parametric NB recalibration is not
implemented by the class-probability calibrator. No odds or betting decision is
implied: selection policies and the planned BetOutcomeReporter consume these
probabilities in the next step.
