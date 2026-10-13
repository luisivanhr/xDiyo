# Optional GAT boundary memory and match-statistics forecasts

Branch: `codex/gat-boundary-memory-stat-forecast`.
Base: `de32880eb0ebbf6892d88276737acef118d68462` (main).
These are opt-in **different models**, not equivalent full-history optimizations.
No predictive/financial benefit is established by implementation or timing tests.

## Native Python and UI

```python
from xdiyo_analytics.graphs import (
    TemporalGraphModel, TemporalGraphConfig, BoundaryMemory,
    StatisticsForecast, StatisticTarget,
)
from xdiyo_analytics.training import TrainingControl
from xdiyo_analytics.reporting import GATLossReporter, GraphStatisticsReporter

model = TemporalGraphModel(
    config=TemporalGraphConfig(
        timezone='UTC', deterministic=True,
        # Omit for the original full-history model.
        boundary_memory=BoundaryMemory(
            policy='fixed_days', days=180,
            anchor='1970-01-01T00:00:00Z', retention=1.,
        ),
        # Omit for the original outcome head and objective.
        statistics_forecast=StatisticsForecast(
            route='combined', weight=.2, point_loss='huber',
            targets=(
                StatisticTarget('cornerKicks', side='home', period='ALL', family='negative_binomial'),
                StatisticTarget('cornerKicks', side='away', period='ALL', family='negative_binomial'),
                StatisticTarget('ballPossession', side='home', period='ALL', family='inflated_beta'),
            ),
        ),
    ),
    control=TrainingControl(max_steps=100, seed=17),
)
reporters = {
    'loss': GATLossReporter(type='overall', metrics=(
        'train_loss', 'validation_loss', 'train_statistics_loss',
        'validation_statistics_loss',
    )),
    'statistics': GraphStatisticsReporter(type='overall', partition='score'),
}
```

Use the existing native dataset/recipe preparation and training runner. These
constructors are registered in the same builder catalogue and JSON/Python
exports; no research wrapper is required. In **Model → Temporal team GAT →
Config**, enable **Boundary memory** and/or **Statistics forecast**. Both start
disabled. The ordered Targets list exposes statistic, side, period, family and
relative weight. Route choices are Auxiliary, Combined and Bottleneck. The
boundary Days/Anchor fields appear for Fixed days only. Add **Graph Statistics
Reporter** under post-training reports. GAT Loss Reporter exposes the new metrics.

The default early-stopping monitor remains native validation loss when a
validation population exists, otherwise post-update training loss. Explicit
`TrainingControl(monitor='train_statistics_loss')`, `validation_statistics_loss`,
`train_objective` or `validation_objective` selects the corresponding metric.
Training-monitor metrics are recomputed under the checkpoint's updated weights.
Ordinary unmonitored training curves retain the native pre-update timing;
validation curves are evaluated after every update. Nothing removes an existing
loss evaluation or changes patience/selection cadence.

## Boundary semantics

Let E(T) be the existing eligible directed edges at local-midnight cutoff T:
finished paired matches with kickoff **strictly before T**, and result availability
**at or before T**, respecting reviewed completion dates and the existing
By Round/One Day Ahead rules. For a segment starting at B, explicit edges at T
are **E(T) minus E(B)**. This partitions *eligibility*, not kickoff dates alone.
An old postponed result released after B enters the new segment. At a tie,
a kickoff at B is not in E(B); an earlier kickoff released exactly at B is.

Two policies are provided:

- `season`: global chronological segments at the **union of all league-season
  membership-entry dates**, floored to local midnight. This is one shared graph,
  not a separate per-league reset. Another league's entry can create a global
  segment boundary. Membership identities themselves still use their actual
  effective entry timestamps. The membership/boundary calendar is fixed and
  known before the experiment; midnight is the start of that local calendar day.
  Morning and evening issuances use that same boundary and previous-day result
  cutoff. Later same-day results do not become available early. Changing the
  supplied calendar can legitimately change segmentation; dynamic late-known or
  retroactively inserted calendars are not supported. No real affected-fixture
  count is claimed. There is no inference of result availability from a season label.
- `fixed_days`: nonoverlapping blocks of `days` local calendar days aligned to
  `anchor`. Anchor must be timezone-aware and local midnight in the model's
  timezone; e.g. use an appropriate `+01:00` date for Europe/Berlin. Calendar days
  preserve local-midnight boundaries through DST. This is **not a sliding window**.

An entirely empty history produces no boundary carry: predictions use the current
native prior. The original schedule and memberships are retained, and explicitly
attaching new evidence uses its actual eligibility cutoff. Statistics supervision
still requires observed training labels; an empty history does not invent them.

The first boundary summarizes eligible history from the beginning of the source
store. Subsequent boundary states summarize only the newly eligible segment
edges, starting from the previous carry. At each boundary the GAT result is
`detach()`ed. For an existing team it replaces the ordinary initial prior; it is
not added to that prior. The original residual layers then process the current
segment. An in-segment fixture never changes the fixed boundary state, so repeated
queries do not accumulate the same match. End-of-segment edges are removed from
the next explicit graph. All team IDs with a carry remain in the node mapping.

At each boundary, optional retention applies once:

`new_carry = retention * encoded_state + (1 - retention) * current_native_prior`.

The default retention is 1: teams with no new edges keep their numerical state,
including through observation gaps. A smaller retention intentionally moves them
towards the current global/league/entry prior, including after movement to a new
league. A team without any carry starts from its current trainable native prior.
Promotion/relegation and competition changes use the existing membership rules.

Explicit edges retain native age/season weighting and support gates. Once an
edge has been summarized, its contribution is carried numerically; it is not
continuously re-decayed as a raw edge. There is no second implicit seasonal decay
on the carried vector. **Rest-day and support fixture inputs in this optional
mode describe the active segment**, with missing rest when it contains no team
match. This, detached temporal gradients, and changed history compression are
part of the new model, not claims of full-history equivalence.

## Passes, gradients and storage

One optimizer step still follows an entire chronological pass, using the native
full-population outcome-loss denominator. Weights are fixed during that pass.
Boundary construction runs without autograd; current-segment message passing and
the heads retain gradients. Every training, validation, post-update monitored
training-loss, prediction, embedding and attention call creates a fresh replay.
Its boundary state advances once through the prefix; it is not reconstructed
from scratch per fixture. No learned boundary cache survives an operation or an
optimizer update. Existing deterministic source-row caching remains bounded and
checks store/config/scaler identity.

Completed validation/test matches may enter later graph history when the native
cutoff makes them eligible. This is the existing sequential forecasting policy,
not an assumption that all held-out match results stay forever hidden.

Inference artifacts persist model parameters, configuration, target transforms,
feature ordering and the fingerprinted store (or require that exact store on
load). This is enough to reconstruct carries under the loaded weights; there is
no separately editable state cache to accidentally reuse. `model_version` includes
enabled configurations and statistics transforms. Store/team mapping and source
identity checks remain mandatory. Updated source observations must use the native
explicit attachment contract. Saved artifacts **are not resumable optimizers**;
cross-process training resume for these modes is unsupported/unverified.

## Future-statistic supervision

The explicit target list currently supports:

| Statistic | Native group | Allowed period | Units/type |
|---|---|---|---|
| goals | canonical `goals_for` | ALL only | count |
| cornerKicks | Match overview | ALL, 1ST, 2ND | count |
| totalShotsOnGoal | Match overview | ALL, 1ST, 2ND | count |
| shotsOnGoal | Shots | ALL, 1ST, 2ND | count |
| accuratePasses | Passes | ALL, 1ST, 2ND | count |
| ballPossession | Match overview | ALL, 1ST, 2ND | native percentage |

Each non-goal target resolves the exact native
`team::<period>::<group>::<statistic>::value` column on the requested home/away row.
There is no fuzzy aliasing, period substitution, all-statistic expansion, or
inference that a sparse half table exists. Missing columns produce a named error;
a target with no observed, positive-weight training coverage is rejected.
Half-goal targets are explicitly unsupported in this first version. Goal targets
never toggle `include_goal_features`; half-only edge predictors stay half-only.

Actual upcoming-fixture statistics are separate supervised arrays, never entries
in the forward tensor dictionary or a teacher-forcing input. They are looked up
only for fit/validation labels and, separately, held-out diagnostics. A row with
no statistics still trains the outcome loss. Missing/unreleased labels are masked
before arithmetic; all-missing groups have a differentiable zero statistics loss.

Training-only transformations are log1p for counts and native percentage/100 for
possession. Means and standard deviations are fitted on observed training labels;
constant columns use scale 1. Their order, transforms, valid counts, packed
training-validity masks, center and scale are saved. `StatisticsScale.inverse`
returns counts or native percentages. Neural inference produces native-scale
constrained forecasts, and does not require target values.

### Routes

| Route | Outcome head receives | Statistics supervision |
|---|---|---|
| Disabled (`None`) | Existing embedding/context concatenation | None |
| auxiliary | Existing embedding/context concatenation | Enabled |
| combined | Existing concatenation + forecast summaries | Enabled |
| bottleneck | Forecast summaries only | Enabled |

The statistics predictor receives the original fixture representation, including
configured fixture/Bayesian context. The bottleneck has **no direct context or
embedding bypass**. Home/away identity is encoded by the ordered outputs.
Combined/bottleneck outcome gradients flow through the predictions into their
head and the GAT; auxiliary-only outcome loss has no path through that head.
A strictly positive supervision multiplier is required: zero is rejected rather
than silently turning interpretable forecasts into an unconstrained latent layer.
The optional historical heatmap CNN and its independent reconstruction loss can
coexist with both additions.

### Point and probabilistic targets

- Point counts: softplus mean; possession: sigmoid fraction. Huber (default) or
  MSE is evaluated in **training-standardized transformed units**.
- Poisson counts: positive mean, variance equal to mean; exact count NLL.
- Negative-binomial counts: positive mean and dispersion; variance
  `mean + mean**2 / dispersion`; exact count NLL.
- Possession: endpoint-inflated Beta. Three softmax masses distinguish exactly
  zero, exactly one, and the interior. Positive Beta shape parameters model
  interior fractions. Exact endpoints never enter a Beta logarithm. The marginal
  mean/variance include both endpoint masses.

For target t, its loss is the sum of observed-row weighted losses divided by the
**full current population's observed weight for that target**. A convex average
using the relative target weights forms `statistics_loss`. Its configured weight
is added to the original outcome objective. Coverage therefore does not simply
multiply a target's contribution. Point scaling addresses differing units;
proper NLLs remain in their respective distribution units and are not inherently
comparable across different target families. Missing validation coverage is
reported and contributes zero for that target; it is not evidence of a good fit.
CNN reconstruction and the existing prior penalty retain their separate roles.

Point-route inputs are `(transform(mean)-training_center)/training_scale` in
configured target order. Probabilistic targets supply mean then standard
deviation: the transformed mean as above and `log1p(std)/training_scale` for counts,
`std_fraction/training_scale` for possession. The MLP consumes these marginal
summaries directly. It does not sample, integrate an outcome probability over a
joint distribution, or model cross-target dependence.

Native outcome `predict` / `(target,class)` `predict_proba` outputs are unchanged.
When enabled an additional `statistics` DataFrame stores ordered means, standard
deviations and supported distribution parameters. `adapter.predict_statistics`
returns that output. Ordinary prediction computes both heads in the same pass.
`GraphStatisticsReporter` uses these retained values for observed-vs-forecast
plots, MAE, proper marginal NLL and central 80% empirical interval coverage.
Missing diagnostics are explicit. Uncertainty is **not automatically calibrated**.
Existing losses, tSNE/PCA, team badges, attention, bet/ticket and financial/P&L
reporters remain native consumers.

## Quantile / pinball forecasts (opt-in)

Select **Quantile** in a statistic target's Family field. Its dependent **Quantiles**
list defaults to `[0.25, 0.50, 0.75]`; the UI previews target-channel and learned-output
counts. The list must be nonempty, finite, unique and strictly inside (0,1). Duplicate
levels are rejected, and accepted levels are persisted in increasing order.

```python
StatisticsForecast(
    route='bottleneck', weight=.2,
    targets=tuple(
        StatisticTarget('cornerKicks', side=side, period='ALL',
                        family='quantile', quantiles=(.25, .5, .75))
        for side in ('home', 'away')
    ),
)
```

Each target/side/period learns K independent logits. For S targets sharing K levels,
the outputs are logically `[fixtures, S, K]`, flattened in configured target order,
then ascending level. Four statistics for each team and three levels give **24**
learned values. K=1 is supported; no median is required. Individual targets may
also use different levels or families, in which case output widths sum over targets.
Columns are named, for example, `home::ALL::cornerKicks::quantile::0.25`.

The parameterization sorts the K logits **inside the trained forward pass**, then
applies softplus for counts or sigmoid for possession fractions. The same ordered
values are used for loss, outcome inputs, prediction and reporting. This enforces
nondecreasing forecasts while permitting ties; it introduces no required spacing.
Gradients pass through the ordered values. Count forecasts are neither rounded nor
sampled. Possession is exported in native 0–100 percentage units. The current
catalogue still supports the six native statistics listed above; arbitrary continuous
columns, nominal labels and half-goal targets are not added by this option.

For each observed y and level tau the pinball is
`max(tau * (y-q), (tau-1) * (y-q))`. Training uses the saved **training-only**
transform and scale: log1p(count), or possession/100. The residual is
`(transform(y)-transform(q))/training_scale`. Centering cancels from that residual;
route inputs use `(transform(q)-training_center)/training_scale`. This log-count
objective is not equivalent to original-count-unit pinball loss. The reporter
separately computes original-unit pinball scores.

Loss first averages across the requested quantiles, then sums over observed rows
with the native sample weights and full-population per-target observed-weight
denominator. Existing relative target weights and the overall statistics multiplier
apply afterward. Increasing K does not multiply the auxiliary weight. Missing values
are excluded before arithmetic; zero counts are observed values; all-missing groups
supply a differentiable zero auxiliary loss while outcome training continues.

Auxiliary-only keeps the original outcome inputs. Combined appends **every ordered
quantile**; bottleneck uses only those quantiles. No direct context bypass or detach
is introduced. No mean column is manufactured. The reporter includes a median only
when 0.5 is configured, never labels it as an expected value, and never interpolates
unrequested levels. It plots all configured forecasts and reports per-level pinball,
empirical `P(Y <= q_tau)`, transformed-space scores and the equal-observed-row held-out
quantile objective component (including configured target/auxiliary weights). Native
GAT loss curves retain the actual sample-weighted training/validation objectives.

The outer requested levels define the displayed quantile interval. Default 25th–75th
is a **nominal central 50% interval**, not a 95% interval or confidence interval for
a mean. For discrete counts, ties can make empirical coverage differ from tau.
These are marginal quantiles, not a full joint distribution or an automatic
calibration guarantee. Levels, order, support, parameterization and transforms are
bound to the saved schema; changed schemas are rejected before loading weights.
Unused quantile defaults are omitted from legacy model fingerprints.

## Review repairs and statistics performance

- Empty history uses prior-only inference in both boundary policies.
- Inflated-Beta predictions now retain `log_p_zero`, `log_p_one` and
  `log_p_interior`. The reporter uses those values directly for proper NLL,
  preserving finite native scores even when displayed probability masses underflow
  to zero. Legacy retained predictions without logs remain readable, with an explicit
  limitation when an already-underflowed mass cannot recover its original score.
  No epsilon clipping substitutes for the likelihood.
- Fixed label selections, local integer indices and per-target weight denominators
  are cached once for each fit's training/validation population. No forecasts or
  learned states are cached. This removes repeated missing-mask `nonzero` operations
  (and their CUDA synchronization) in point/count/quantile loss paths. Beta still
  separates endpoint/interior observations for its likelihood. Cache size is O(N*S),
  not O(history edges * issuances), and fresh initialization clears it. Quantile
  levels move with the module as configuration-derived, nonpersistent buffers.

See `data/gat_statistics_repair_20261014/` for current-source checks and isolated
loss-path benchmark receipts. The pre-repair measurements below remain historical.

Current-source isolated statistics-head forward/backward measurements, milliseconds
per 256-row pass (8 groups, 6 targets, 25% missing labels, six alternating paired
repeats; medians):

| Family | CPU uncached | CPU cached | Ratio | GPU uncached | GPU cached | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| point | 10.11 | 7.82 | 1.29x | 48.20 | 30.53 | 1.58x |
| poisson | 10.39 | 8.16 | 1.27x | 47.61 | 32.36 | 1.47x |
| negative_binomial | 17.24 | 14.40 | 1.20x | 86.07 | 65.40 | 1.32x |
| inflated_beta | 30.92 | 28.08 | 1.10x | 190.73 | 166.40 | 1.15x |
| quantile | 12.61 | 10.39 | 1.21x | 61.03 | 41.56 | 1.47x |

These are loss-path measurements with synthetic fixed labels and native heads, not
whole-GAT epoch speedups. The one-time cache-building cost is recorded separately.
Every benchmark requires exact loss/gradient agreement. CPU and deterministic CUDA
regressions additionally compare two complete native updates, post-update checkpoint
metrics, validation metrics, all gradients, parameters and full AdamW state against
the uncached reference. Existing families use the frozen f224a48 head/loss code;
quantiles compare cached and uncached execution of the new mathematical objective.
No worker was interrupted, and no large study or accuracy experiment was run.

Repair verification: **255 Python tests and 29 JavaScript UI checks passed**,
including bounded RTX 4070 SUPER checks. Exact commands, environment, source hashes
and timing scope are recorded in `data/gat_statistics_repair_20261014/verification.json`.
The reference constructor, default path and old artifacts remain covered.


## Validation and benchmark receipts

See `data/gat_extensions_20261014/*.json` for exact source/input hashes, commands,
versions, population sizes, wall times and process peak RSS. The disposable
benchmark reads the isolated saved expanded-totals fold 02 and verifies its model
commitment and input hashes before and after. It makes one fresh optimizer update;
it neither resumes the study nor writes to its inputs. The input store has
55,976 directed team rows. The bounded population has 236 training fixtures over
20 issuances, 143 validation fixtures and 92 predictions. Original 32-dimensional,
two-layer/two-head architecture and 72 scalar input columns are retained.

The final regression run passed **164 tests** across extensions, core graph,
speed/preparation, goal features, orchestration, deterministic execution and CNN
integration. The native UI renderer/export suite passed **5 tests**. The new
CUDA tests cover all three routes with boundary memory, repeat-fit losses,
gradients, parameters and full AdamW state, distribution gradients, GPU save/load
and CPU-loaded inference. Deterministic GPU repeat/save-load comparisons were
exact; cross-device inference used `atol=1e-6, rtol=1e-5`. Cross-process optimizer
resume remains unverified and unsupported.

The eight single-run timing measurements below used the same saved population.
They ran while the research GPU workers were absent. Timings used the original
nondeterministic setting; exact correctness tests used deterministic execution.
Seconds are wall time; memory is MiB. CUDA memory is peak allocated tensor memory;
reserved CUDA memory is recorded separately in each receipt. RSS is whole-process
peak including source loading. CPU VRAM entries are zero/not applicable.

| Device | Mode | Cold preparation | Update | Post-update train loss | Validation | Prediction | Peak RSS | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | default | 0.181 | 3.944 | 1.967 | 1.022 | 1.129 | 1029.7 | 0.0 |
| cpu | memory | 0.309 | 0.265 | 0.207 | 0.205 | 0.276 | 859.8 | 0.0 |
| cpu | statistics | 0.187 | 4.049 | 1.875 | 0.957 | 1.071 | 1012.4 | 0.0 |
| cpu | combined | 0.322 | 0.306 | 0.258 | 0.204 | 0.290 | 860.0 | 0.0 |
| cuda:0 | default | 0.216 | 0.800 | 0.320 | 0.162 | 0.220 | 1640.5 | 159.9 |
| cuda:0 | memory | 0.454 | 0.517 | 0.264 | 0.155 | 0.207 | 1594.3 | 28.1 |
| cuda:0 | statistics | 0.231 | 0.846 | 0.374 | 0.198 | 0.260 | 1749.2 | 160.0 |
| cuda:0 | combined | 0.446 | 0.588 | 0.276 | 0.174 | 0.240 | 1693.5 | 28.3 |

On this bounded population, memory-only update plus post-update training-loss
work was about **12.5x faster on CPU and 1.43x on GPU** than full history.
Boundary construction costs more in cold preparation. This is a different model:
no forecast equivalence, accuracy improvement or full-study speedup is claimed.

**Receipt qualification:** timing receipts precede the final label-lookup
refinement in `graphs/statistics.py`, which selects only key/target columns and
builds the lookup once instead of copying unrelated history columns. All other
graph source hashes match the tested final source. This refinement passed the
final 164-test suite but was not rebenchmarked because a new research worker had
started. The default and memory-only paths do not call this label code; statistics
mode timings are measurements of the immediately preceding implementation.
`final_source_verification.json` records final hashes and the exact mismatch per
receipt. Existing receipts were not rewritten to imply final-source timings.

To rerun the bounded CUDA tests, wait for research workers to be idle and use a
fresh process:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:XDIYO_RUN_GPU_AUDIT='1'
$env:PYTHONPATH='src'
& 'C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe' -m pytest tests/analytics/test_graph_extensions.py -q
node --test tests/analytics/ui_cnn.mjs
```

No large experiment, hyperparameter sweep or held-out financial-performance
study was launched. Neither the original frozen model nor study inputs were
modified. The benchmark commands and input hashes are preserved in the receipts.

## Migration and review scope

No migration is required for old recipes or artifacts. Absent new options mean
`None`, no new network parameters are allocated, and disabled options are omitted
from the legacy model fingerprint. Frozen baseline network tests also compare
initialization, losses, gradients and full AdamW state with the actual base
network source, including the optional CNN.

Changes are confined to native graph configuration/replay/supervision/network,
serializer, catalogue/UI metadata, native diagnostics, tests, documentation and
disposable benchmark code. No datasets, frozen experiments or active-worker
sources were changed. This branch is for review; it has not been merged to main.
