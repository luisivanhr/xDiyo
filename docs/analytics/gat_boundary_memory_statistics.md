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
  effective entry timestamps. There is no inference of result availability from
  a season label.
- `fixed_days`: nonoverlapping blocks of `days` local calendar days aligned to
  `anchor`. Anchor must be timezone-aware and local midnight in the model's
  timezone; e.g. use an appropriate `+01:00` date for Europe/Berlin. Calendar days
  preserve local-midnight boundaries through DST. This is **not a sliding window**.

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
