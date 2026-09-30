# XGBoost quantiles for total match corners

[Notebook 19](../../notebooks/19_xgboost_quantile_total_corners.ipynb) expands the
corner experiment to a large library of historical features and predicts lower,
median and upper conditional corner totals. It leaves the Lasso notebook intact.

The notebook's settings remain visible Python. Shared preparation and repeated
quantile fitting live in `notebooks/helpers/quantile_corners.py`; that helper calls
the existing public feature, assembly, split, training and reporting APIs.

## Population and evaluation

The experiment loads every available league for `22_23`, `23_24` and `24_25`
from `data/xDiyo_data`. Awarded matches are excluded. The target is total observed
corners, home plus away, with one model row per match. Matches with missing
targets are removed at assembly; missing feature values remain missing for
XGBoost's native missing-value handling.

One pooled chronological outer fold trains on all loaded seasons preceding the
latest and tests that latest season. With this notebook's original three-season
configuration, this remains two training seasons and one test season. Expanding
`SEASONS` automatically expands the training span. Its calendar uses `source_season`, not provider season IDs
that differ across competitions. Training fixtures after the earliest test
prediction cutoff are ineligible. Hyperparameters are fixed before this run. The latest 15% of distinct training
kickoff batches form a chronological validation tail for native early stopping;
whole matches and equal-time batches stay together. The test population is not
used for stopping or tuning.

Historical features may use earlier completed matches in the test season when
predicting later fixtures. Model weights remain fixed for the entire held-out
season. The default availability assumption is earlier finished kickoffs;
the notebook exposes the same configurable prediction lead time as the other
feature workflows.

## Feature families

The builder discovers available members of a 22-statistic shortlist in the
development seasons only. New test-season fields cannot select predictors.
The shortlist and six half-period statistic keys are displayed in the notebook.
The configurable period choices include `ALL`, `1ST` and `2ND`; a first-half
feature describes historical first halves, not the first half of the fixture
being predicted. Statistics without observations are not fabricated.

| Family | Information represented |
|---|---|
| Team statistics | Own and conceded observations for discovered raw statistic identities |
| Lags | Selected preceding eligible match values |
| Rolling means | Recent level over configurable match windows |
| Rolling standard deviations | Variability across individual historical observations |
| Rolling Z-scores | Latest eligible observation relative to its selected history window |
| EMA | A level emphasizing more recent eligible observations |
| League/LOO corners | League baseline and a baseline excluding the focal team's contributions |
| H2H corners | History specific to the ordered team/opponent pairing |
| Ratings | Evolving match-result and corner-comparison Glicko rating/RD |
| Context | Eligible rest days, normalized standings, calendar and training-discovered league indicators |
| Match combinations | Home/away feature alignment and explicitly generated paired summaries |

The notebook displays and saves the generated feature manifest. This is the
authority for the actual source statistics, expression names and column count
in each execution. The full-match mean windows, lags, EMA spans, period filters and rating switch
are visible settings. The fixed supplementary definitions are std 5/10, Z-score
5, half-period lag 1 and means 5/10, H2H means 3/5 and completed-round league
mean/std plus LOO mean 3/5. Selected mean/rating/context columns receive home-away
sums/differences; short-versus-long trends use mean 3 minus mean 20.

For/against statistics remain distinct. The raw current-match statistic is
never substituted for a historical feature. Partial eligible windows use the
available data; early missing observations are not padded with fabricated games.
The large feature set is an experimental starting point, not a claim that every
column improves prediction.

## What a quantile prediction means

For total corners \(Y\) and match features \(x\), the \(\tau\)-quantile is

\[
Q_{\tau}(Y\mid x)
= \inf\{y : \Pr(Y\leq y\mid x)\geq\tau\}.
\]

The default levels are \(0.1\), \(0.5\) and \(0.9\). The middle prediction is
a conditional **median**, not the conditional mean estimated by a squared-error
model. The lower and upper predictions form a nominal 80% interval.

Each quantile uses an independent XGBoost model with
`objective="reg:quantileerror"` and its own `quantile_alpha`. The existing
`BoostingAdapter` therefore retains an ordinary one-target prediction contract.
The notebook aligns their predictions by retained match identity and evaluates
the combined quantile output afterward.

Training minimizes pinball loss. For residual \(u=y-\widehat q_{\tau}\),

\[
\rho_{\tau}(u)
= \begin{cases}
\tau u, & u\geq0,\\
(\tau-1)u, & u<0.
\end{cases}
\]

The reported test pinball loss averages this over the evaluated matches:

\[
\operatorname{Pinball}_{\tau}
= \frac{1}{n}\sum_{i=1}^{n}
\rho_{\tau}(y_i-\widehat q_{i,\tau}).
\]

Lower is better. At \(\tau=0.5\), pinball loss is half the MAE. MAE and MSE of
the median are useful point-prediction summaries; they do not turn that median
into a mean or define the model's quantile training objective.

## Baseline and interval diagnostics

The comparison baseline uses the empirical corner quantile calculated on the
**optimization labels only**, excluding both the training validation tail and
held-out test labels, then predicts that fixed value for every test match. It answers whether feature-conditioned predictions improve upon a
simple unconditional training distribution.

For lower/upper levels \(a<b\), interval coverage and mean width are

\[
\operatorname{Coverage}_{a,b}
= \frac{1}{n}\sum_{i=1}^{n}
\mathbf 1\{\widehat q_{i,a}\leq y_i\leq\widehat q_{i,b}\},
\qquad
\operatorname{Width}_{a,b}
= \frac{1}{n}\sum_{i=1}^{n}
(\widehat q_{i,b}-\widehat q_{i,a}).
\]

The nominal coverage is \(b-a\). Counts are discrete, so equality at a predicted
quantile affects empirical coverage. A single finite sample need not hit the
nominal level exactly, and matching aggregate coverage does not establish
calibration for every league or match type.

Independently fitted quantiles can cross. For ordered levels
\(\tau_1<\cdots<\tau_K\), the crossing rate is

\[
\frac{1}{n}\sum_{i=1}^{n}
\mathbf 1\!\left\{\exists k<K:
\widehat q_{i,\tau_k}>\widehat q_{i,\tau_{k+1}}\right\}.
\]

The notebook retains raw predictions and reports crossings. It does not silently
sort quantiles, clip negative values, round corner counts or repair intervals.
These predictions are not calibrated over/under bet probabilities by themselves.
No odds, stakes or betting profit are invented.

## Execution and artifacts

The notebook exposes CPU/CUDA selection, native training threads and ordinary
XGBoost model parameters. The three quantile models run through the experiment
orchestrator and retain their completed results. Reuse skips an exactly matching
completed fit; no claim is made that a native XGBoost fit resumes mid-tree from
these numerical artifacts.

Gain is extracted only from the trees used for prediction: through
`best_iteration + 1` when early stopping is active. Patience trees are excluded
from this report. With `early_stopping_rounds=None`, all fitted trees are used;
best-iteration/best-validation fields remain null rather than inventing a selected
iteration. The configured training validation tail is still reserved when
stopping is disabled.

An explicit CUDA request must be supported by the installed native build and
device. Device failures remain visible; this experiment does not silently claim
GPU execution after falling back to CPU.

The notebook shows a readable run summary, feature manifest, quantile metrics,
baseline comparison, interval diagnostics and representative match predictions.
Its stored artifacts make those results available outside the current kernel.

## Verification and observed results

Notebook structure and all code cells pass validation. Seven focused tests pass:
independent sklearn pinball agreement, unchanged crossing predictions, shape/order
rejections, mismatched retained match-row rejection, and a tiny native CPU
XGBoost run verifying optimization-only baseline quantiles, prediction identity,
saved models, prediction-tree-only gain, and operation with early stopping disabled. The source is
`tests/analytics/test_quantile_corners_notebook.py`.

The configured real-data notebook completed top to bottom on CUDA. A second
full Run All also completed: all three quantile results reported `reused=True`,
retained the same artifact run IDs and preserved their prediction metrics. It
regenerated gain using only the saved models' prediction trees. The delivered
notebook retains that executed output. Neither execution required package
installation or a library API change.

### Real-data preparation

The default preparation completed with **601 historical expressions**,
**1,589 assembled inputs**, and **13,971 labelled matches**. The outer population
contains **9,371 training matches** and **4,600 held-out matches**. This is one
pooled model per quantile across the 13 leagues, not a separate model per league.
Preparation took approximately 136 seconds in the first recorded execution.


### Held-out results

All three native model configurations report **`cuda:0`**. The 9,371-row training
population was divided into **7,969 optimization matches** and **1,402 validation
matches**. The validation-selected best iterations were 102, 160 and 127 for the
10th, 50th and 90th quantiles respectively; these are zero-based iteration numbers.
The optimization-only empirical baseline quantiles were 6, 10 and 14 corners.

| Quantile | Model pinball loss | Baseline pinball loss | Relative improvement |
|---|---:|---:|---:|
| 0.10 | 0.535074 | 0.536261 | +0.22% |
| 0.50 | 1.324238 | 1.335217 | +0.82% |
| 0.90 | 0.629699 | 0.628957 | −0.12% |

These are **small improvements at the lower and median quantiles**, with the
upper quantile slightly worse than the constant baseline. The feature count by
itself did not produce a large gain. Median-prediction MAE was **2.64848 corners**,
compared with **2.67043** for the constant training median: about 0.022 corners
less absolute error per held-out match in this trial.

The raw 10th–90th interval covered **80.8913%** of the 4,600 held-out matches,
with mean width **8.61993 corners**. No quantile crossings or negative quantile
predictions occurred in this evaluation. Those finite-sample observations do not
impose a noncrossing guarantee on later predictions. Aggregate coverage was close
to its nominal 80%; the per-league table remains useful for inspecting variation.

The following historical numerical evidence remains in local, ignored experiment
outputs and is not distributed with the example inputs. Running the notebook from
the published season tables creates new results. The original evidence is in the
[run summary](../../experiments/total_corners_xgboost_quantiles/total-corners-xgboost-quantiles-broad-features--cf41ff29/quantile_summaries/ca3aef95-a50b-4cf7-9070-63925d29dca2/summary.json),
[pinball table](../../experiments/total_corners_xgboost_quantiles/total-corners-xgboost-quantiles-broad-features--cf41ff29/quantile_summaries/ca3aef95-a50b-4cf7-9070-63925d29dca2/pinball_metrics.csv)
and [interval diagnostics](../../experiments/total_corners_xgboost_quantiles/total-corners-xgboost-quantiles-broad-features--cf41ff29/quantile_summaries/ca3aef95-a50b-4cf7-9070-63925d29dca2/interval_diagnostics.csv).
The notebook retains the executed tables, figures and detailed median report.

## Related contracts

- [Historical features and their equations](features.md)
- [League windows, leave-one-out and H2H semantics](league_warmup.md)
- [Glicko state and snapshot construction](ratings.md)
- [CPU/CUDA execution policy](execution_policy.md)
