# Chronological binary SVC calibration

This optional route trains **SVC with internal probabilities disabled** on earlier
rows, then learns one sigmoid from its decision margins on a later training tail.
The complete earlier pipeline and sigmoid are frozen for evaluation and future
predictions. No libsvm probability cross-validation or out-of-fold ensemble is used.
It is a stricter fitting design, not a claim of improved predictions or profits.

Existing `SVC(probability=True)` and probability-to-probability calibration keep
their behavior when the new response method is absent. Old serialized calibrators
without `response_method` resolve to `predict_proba`.

## Builder and recipe

In **Model & training**, select **SVC**, leave **Enable class probabilities** off,
and enable **Probability Calibrator** in the training configuration:

- Response method: `decision_function`.
- Method: `sigmoid` (required in margin mode).
- Fraction: latest fraction of distinct training kickoff batches.
- Availability: choose an explicit timestamp column **or** a declared delay proxy.
- Prediction groups: the metadata keys defining a single decision time.
- Prediction lead: duration before the group's earliest kickoff, unless an
  explicit cutoff column is supplied. Default: `1h`.

For a study assuming outcomes are available three hours after kickoff and making
all round predictions one hour before that round's first kickoff:

```python
from xdiyo_analytics.ui.recipe import node

recipe['model'] = node('sklearn.svm.SVC', probability=False, C=1.0, gamma='scale')
recipe['preprocessors'] = [
    node('sklearn.impute.SimpleImputer', strategy='median', keep_empty_features=True),
    node('sklearn.preprocessing.StandardScaler'),
]
recipe['candidate']['calibration'] = node(
    'training.ProbabilityCalibrator',
    method='sigmoid', response_method='decision_function', fraction=0.2,
    availability_delay='3h', prediction_lead='1h',
    prediction_group_by=['competition_id', 'season_id', 'tournament_id', 'round'],
    min_calibration_rows=10, min_calibration_per_class=2,
)
```

Use the **actual metadata columns**: if tournament identity is named `stage`, use
that key instead of `tournament_id`. Missing grouping keys fail; none are silently
dropped. Empty `prediction_group_by` means one group per match. The three-hour
delay is an **assumption**, never presented as a verified publication timestamp.
An actual availability column replaces the delay; do not specify both.
These calibration settings control label eligibility and fitting boundaries.
Historical feature construction remains upstream: configure its prediction
cutoffs consistently with the study's round-level decision times. Calibration
does not recompute already prepared features or certify their source timing.

The base label must be binary. The model determines class order, including string
or nonconsecutive numeric labels. Leave label scaling disabled. Calibration needs
both fitted classes; defaults require at least 10 calibration **rows**, with at
least 2 rows per class. Team-layout rows count as rows, not distinct matches.

## Population boundaries

1. Select the latest kickoff batches within the supplied training population.
   Matches and simultaneous kickoff batches cannot straddle that boundary.
2. Purge incomplete prediction groups and groups crossing that boundary, rather
   than expanding the calibration tail.
3. Purge calibration groups whose labels are missing/unavailable at the model
   issue time. Issue time is no later than the earliest outer prediction cutoff
   and any supplied outer `fit_at` or explicit `issue_at`.
4. Purge earlier groups whose labels are missing/unavailable at the **earliest
   calibration prediction cutoff**, not its kickoff. This includes delayed results.
5. Reserve any supported monitoring population from those earlier eligible rows.
   SVC still does not support early stopping; unsupported controls are rejected.
6. Fit selectors, weights, imputation, scaling and SVC only on the remaining base
   fitting rows. Calibrate on the held-out tail, without class weights or refitting
   preprocessing. Preserve train-only empty-feature behavior and feature order.

Invalid identities/timestamps, empty populations, missing availability policy,
unknown classes, insufficient per-class counts and incompatible settings raise
clear errors. There is no random-split fallback, probability=True retry, tail
expansion or uncalibrated fallback. The fold's calibration summary records both
boundaries, availability assumptions, every purged row's position and match
identity, and the reason for removal. Nested selection remaps these positions to
the original dataset.

For an explicit final refit without an outer fold, supply `issue_at` in the
calibrator. The same temporal procedure repeats inside the declared development
population. It does not fit the SVC on the calibration rows afterward.

## Exact sigmoid contract

For the fitted class order `[c0, c1]`, a positive margin favors `c1`:

```text
P(c1 | s) = expit(a*s + b)
P(c0 | s) = 1 - P(c1 | s)
```

The target is `(n1+1)/(n1+2)` for class c1 and `1/(n0+2)` for class c0. Minimize
the **mean binary log loss** against these Platt-smoothed targets, with no
regularizer or calibration weights. Implementation uses stable `logaddexp`, an
analytic gradient, and L-BFGS-B with `maxiter=1000`, `ftol=1e-12`, `gtol=1e-10`,
starting at zero slope/intercept. Margins are divided by their maximum absolute
value for numerical conditioning; the recorded original-unit `a,b` describe the
same sigmoid. Coefficients, scale, objective version, loss, convergence, class
counts and optimizer settings are retained. Exactly constant/zero margins are
rejected; optimizer failure never returns a fitted calibrator.

## Outputs and recovery

- `predict_proba`: calibrated `(target, class)` probabilities, directly consumable
  by probability metrics, CalibrationReporter and BetOutcomeReporter.
- `decision_function`: finite raw margins, with fitted class/positive-margin
  metadata. It is not a probability output.
- `predict`: calibrated argmax with `update_predict=True`; otherwise the base SVC
  class decision.
- There is **no `predict_proba_raw` in margin mode**. Explicitly requesting that
  unavailable output in a reporter raises an error. Probability-input calibration
  continues to retain raw probabilities.

Saved models retain the complete earlier preprocessing/SVC pipeline, fitted
sigmoid, response schema and temporal diagnostics. Recipe/export/recovery identity
includes the response method, availability/grouping policy, minimum sizes and
other calibration settings, model/preprocessing settings and relevant code/data.
The shared recovery source fingerprint covers the new margin and split modules;
no manual hash override is used.

This implementation does not restart any discarded research comparison or modify
frozen experiment artifacts.
