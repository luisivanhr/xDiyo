# Count Over/Under Brier score

`count_ou_brier` is an opt-in probability metric, minimized during selection.
It accepts categorical count probabilities, negative-binomial parameters or
Poisson means. `distribution="categorical"` remains the default.
Existing objectives, binary `brier_score`, and multiclass `log_loss` are unchanged.

## Calculation

Default lines: 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5. For match
\(i\), retained count probabilities \(p_i(c)\), and line \(\ell\):

\[
p_{i,\ell}^{O}=\sum_{c>\ell}p_i(c),\qquad
p_{i,\ell}^{U}=\sum_{c<\ell}p_i(c).
\]

The observed events are \(z_{i,\ell}^{O}=\mathbf{1}\{y_i>\ell\}\) and
\(z_{i,\ell}^{U}=\mathbf{1}\{y_i<\ell\}\). With \(L\) lines and \(N\) eligible rows:

\[
B_i=\frac{1}{2L}\sum_{\ell}\left[
(p_{i,\ell}^{O}-z_{i,\ell}^{O})^2+(p_{i,\ell}^{U}-z_{i,\ell}^{U})^2\right],
\qquad B=\frac{1}{N}\sum_{i=1}^{N}B_i.
\]

Both sides and all lines have equal weight. The complementary sides give the
same score as the one-side average for normalized probabilities; the 16 events
are not independent observations. This is a proper score for these events,
not a profit estimate or a strictly identifying score for the entire distribution.

Probabilities 0.2/0.3/0.5 over counts 0/10/20 with observations 10/5/25 yield
losses 0.145/0.445/0.145 and overall 0.245. Observed 25 remains included even
without class 25. No tail mass is invented.

## UI and native recipe

In **Model selection → Selection rule and numerical evidence**, add
`count_ou_brier` to Metrics, edit **Over/under lines** and **Count distribution**, and choose that metric
(or its configured key) in Metric Selection with direction minimize. It is also
available in the post-training `PerformanceReporter`. Restart a running notebook
kernel/builder to load the new library/catalog. No existing objective changes.

Merge this fragment into `search.options`, preserving the grid and other settings:

```json
{
  "metrics": ["mae", {
    "component": "evaluation.Metric",
    "params": {
      "name": "count_ou_brier",
      "target": "total_corners",
      "output": "predict_proba",
      "parameters": {"lines": [6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]},
      "key": "ou_brier",
      "direction": "minimize"
    }
  }],
  "decision": {
    "component": "selection.MetricSelection",
    "params": {
      "metric": "ou_brier",
      "direction": "minimize",
      "selector": {
        "study": "selection_metrics", "type": "overall", "partition": "score",
        "target": "total_corners", "output": "predict_proba"
      }
    }
  }
}
```

Python uses the same contract:

```python
Metric("count_ou_brier", target="total_corners", output="predict_proba",
       parameters={"lines": [7.5, 9.5, 11.5]}, key="ou_brier")
```

## Inputs and pooling

### Categorical inputs (default)

- Columns use `(target, class)` identities. Classes must be distinct numeric
  nonnegative integers; targets must be finite numeric nonnegative integers.
  One-class distributions are allowed; boolean/string counts are rejected.
- Lines must be a nonempty unique list of positive finite half-integers.
  Integer/quarter lines, booleans, duplicates and malformed values are rejected.
- Every selected probability vector must be complete, finite, within \([0,1]\),
  and sum to one within absolute tolerance \(10^{-6}\). Invalid vectors/targets
  raise rather than dropping observations. Other metrics keep existing masking.
- Exact sums use `bet_probabilities`: no clipping, renormalization, smoothing,
  recalibration, class weighting, odds, EV filters, or stakes.
  `predict_proba` uses calibration when retained in that output; `Metric.output`
  can explicitly select another retained output.
- Overall scores pool selected rows, preserving `n`, `n_total`, `n_missing`,
  status, sample hashes, partitions and identities. One row with loss 0.145 and
  two rows averaging 0.295 produce 0.245 overall, not 0.22.
- `PerformanceReporter` retrieves selected rows from their source folds before
  aligning class supports. Only structurally absent classes receive zero mass.
  A missing value in a class the source fold exposed is invalid. Stored
  predictions and generic pooling are unchanged.
- `occurrences`, `first` and `last` retain the existing repeated-row policies.
  Repeats still require an explicit policy. **`mean` pooling is unsupported** for
  this metric because it loses the required source-fold distinction.
- Direct `evaluate_metrics` accepts single-fold distributions normally. For
  concatenated differing supports, pass `source_folds` (fold ID to FoldResult)
  with occurrence-indexed rows, or use the reporter. Never blanket-fill NaNs.

### Negative-binomial and Poisson inputs

Choose `distribution="negative_binomial"` or `distribution="poisson"` in
`Metric.parameters`. When `Metric.output` is omitted, these representations use
`count_distribution`; categorical scoring continues to use `predict_proba`.
An explicitly named output is honored and must have the required parameter
shape. Point predictions are never used as a fallback.

| Representation | Columns for the selected target | Validation |
|---|---|---|
| `categorical` | `(target, count)` | Normalized discrete probabilities |
| `negative_binomial` | `(target, "mean")`, `(target, "dispersion")` | Exactly both; finite and nonnegative |
| `poisson` | `(target, "mean")`; optional `(target, "dispersion")` | Mean finite and nonnegative; any supplied dispersion must be exactly zero |

Columns and rows must have distinct parameter labels and exact identity alignment.
Unknown, duplicate or mixed parameter columns are errors. Missing parameters or
nonfinite values are never dropped or zero-filled. A mean-only Poisson explicitly
defines zero dispersion; that is distinct from a missing NB dispersion parameter.
Other targets may coexist in the same two-level output frame.

For NB2 mean \(\mu\) and dispersion \(\alpha>0\), the native parameterization is

\[
r=1/\alpha,\qquad p=\frac{1}{1+\alpha\mu},\qquad
\operatorname{Var}(Y)=\mu+\alpha\mu^2.
\]

At \(\alpha=0\), the model is Poisson with rate \(\mu\) and variance \(\mu\).
Mean zero is permitted in either case and puts all mass at zero. For each
positive half-line \(\ell\), the existing native betting helper computes

\[
P(Y<\ell)=F(\lfloor\ell\rfloor),\qquad
P(Y>\ell)=S(\lfloor\ell\rfloor),
\]

using the exact NB/Poisson CDF and survival function. The Brier average is the
same as above. There is no finite PMF truncation, normalization or fitted-class
membership test, so observed counts such as 25 or 26 remain eligible.

The native `EstimatorAdapter` retains `count_distribution` for
`NegativeBinomialRegressor` and sklearn `PoissonRegressor`, directly or as the
last estimator of a preprocessing pipeline. NB stores its conditional mean even
when `prediction="mode"`, plus fitted dispersion (including learned dispersion).
Poisson stores its conditional mean and dispersion zero. Keep targets unscaled;
feature imputation/scaling remains fitted inside each training fold. Ordinary
`prediction_methods=["predict"]` is sufficient to retain these parameters.

In the metric editor, changing the representation updates the normal output
between `predict_proba` and `count_distribution`. An explicitly customized
alternate output name is preserved. Existing categorical recipes are unchanged.
For example, place this metric in the existing search's `options.metrics`:

```json
{
  "component": "evaluation.Metric",
  "params": {
    "name": "count_ou_brier",
    "target": "total_corners",
    "output": "count_distribution",
    "key": "ou_brier",
    "parameters": {
      "distribution": "negative_binomial",
      "lines": [6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]
    }
  }
}
```

For Poisson, change only `distribution` to `"poisson"` and use a Poisson model.
The selection rule can remain `MetricSelection("ou_brier", direction="minimize")`.
If its selector explicitly specifies `output`, set it to `count_distribution`.

Both parameter representations retain source-fold identities for occurrences,
first and last pooling, and weight pooled rows equally. `mean` pooling is still
unsupported: averaging distribution parameters is not a mixture of event
probabilities. Empty inputs retain `no_valid_observations`; valid inputs retain
all rows with `n_missing=0`. Invalid inputs raise without masking. Lines, observed
count validation and sample hashing follow the same rules as categorical scoring.

## Reuse and selection

The metric supplies standard scalar `PerformanceReporter` evidence. Lower is
better and exact ties preserve candidate order. No external registration or
custom evidence reporter is required. With the same experiment store and
`resume=True`, compatible retained inner predictions can be rescored without
refitting candidates: metrics and decision remain outside fitting identity.
A changed winner may still require normal final evaluation/refitting.

No research experiment was run as part of this feature. A specific retained
research run's expected row count must be verified when that run is rescored.
