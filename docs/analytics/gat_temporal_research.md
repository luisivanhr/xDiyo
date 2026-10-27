# Time-indexed GAT research log

## Scope and reference

This document tracks the implementation on `codex/gat-temporal-research`, based
on audited commit `0cb13cac57255f5e2f8e0eeb692e2b6f22344381` and the
[GAT Architecture Audit conversation](https://chatgpt.com/c/6adc47d6-4fa8-83e8-9745-eed115a38703).
The requested change is explicit timing and historical opponent representations.
The existing snapshot GAT remains the default and comparison baseline.

The construction follows the conversation's time-indexed attention proposal,
related to [TGAT](https://arxiv.org/abs/2002.07962). This implementation retains
xDiyo's edge-conditioned attention, residual support gate and outcome head;
it is not a reproduction of the TGAT paper's time encoding or experiments.

## Implementation checklist

- [x] Establish the unchanged default and three explicit comparison modes.
- [x] Add elapsed-time features in attention scores and message values.
- [x] Add deduplicated historical queries and differentiable layer batches.
- [x] Preserve availability cutoffs, historical membership priors and exact evidence.
- [x] Enforce target identity exclusion and conservative label availability.
- [x] Verify baseline compatibility, invariance, gradients and persistence.
- [x] Record verification and stop at the first experiment checkpoint.

## Model contract

`TemporalGraphConfig.temporal_mode` selects one of:

| Mode | Historical message context | Explicit time inputs |
| --- | --- | --- |
| `snapshot` (default) | Opponent at forecast issuance | None, preserving the baseline |
| `time_features` | Opponent at forecast issuance | Age and both historical observed-match gaps |
| `historical` | Opponent at the encounter's pre-match information boundary | Same time inputs |

The new modes require boundary memory, statistics supervision and heatmaps to be
disabled for this first controlled comparison. The H/D/A head, class order,
cross-entropy, fixture features, decay and support gate retain their definitions.
Scheduled-rest/horizon improvements, binary objectives, retrospective mixtures
and recurrent memory remain separate experiments.

### Three clocks and historical queries

For an outer query at issuance `T`, retain the existing previous-local-day cutoff:
kickoff is strictly before local midnight and result availability is at or before
that midnight. For an eligible encounter at `tau`, query the opponent at `tau`
using that encounter's own previous-local-day cutoff. This deliberately excludes
same-local-day observations even if they were already released before kickoff.
Age and membership priors use the full query timestamp. An effective membership
entry at that timestamp follows the existing inclusive membership convention.

A later-round match can have more pre-match context than the original round's
forecast had. That context enters only when interpreting the completed encounter
in a later forecast. It never revises the original issued forecast.

With fixed model weights and preprocessing, evidence unavailable at a query's
cutoff cannot affect that query. Historical embeddings are reconstructed with
current fitted weights; they are not frozen embeddings from an older fit.

### Explicit time inputs

New temporal configurations use `time_encoding="rational_decay_v1"`. For a known
duration `d` in days, the first coordinate is the bounded fixed-scale function
`r(d; tau) = tau / (tau + d) = 1 / (1 + d/tau)`. The separate defaults are
`age_scale_days=180.0` and `gap_scale_days=7.0`. They are engineering choices,
not optimized values, and never depend on archive length, maxima or batch
statistics. Existing `2**(-d/s)` channels retain their configured half-lives in
`time_scales_days` (default 7, 30, 180). The three blocks describe:

- the historical match's age at the query time;
- the receiving team's gap before that historical encounter;
- the opponent's gap before that historical encounter.

The default feature order is:

| Block | Coordinates |
| --- | --- |
| Age `a` | `[r(a;180), 2**(-a/7), 2**(-a/30), 2**(-a/180)]` |
| Known receiver gap `g` | `[r(g;7), 2**(-g/7), 2**(-g/30), 2**(-g/180), 0]` |
| Known opponent gap `g` | Same five-coordinate layout as the receiver gap |
| Missing gap | `[0, 0, 0, 0, 1]` |

Thus a known zero gap is `[1,1,1,1,0]`, distinct from missing. The default still
adds 14 coordinates and preserves fractional days. Known durations must be
finite and nonnegative; all scales must be finite and strictly positive.
Invalid inputs raise errors. There is no clipping, learned scale, added
normalization or fit-dependent scaling. Root and recursive historical queries
call the same vectorized feature builder.

Each gap is measured from the latest match whose result was eligible at that
encounter's own previous-day cutoff. This **observed-history gap is not necessarily
physiological rest** when other competitions or periods are absent. The rational
age block tends to zero as an encounter becomes ancient; its two gap blocks
describe conditions at that encounter and do **not** vanish simply because the
encounter ages. Existing attention recency weights, seasonal retention and support
gates still apply separately.

`time_encoding="log1p_v1"` retains the original three `log1p(days)` coordinates
for reproduction; the rational age/gap scales are inactive under this option.
Every completed experiment recorded below used the logarithmic representation.
**Predictive benefit of the rational representation remains untested.** Raw match
statistics retain fitting-only preprocessing. Both the edge attention projection
and message value projection receive the selected coordinates. Optimizer
`weight_decay` remains a separate mechanism.

### Computation and compatibility

`graphs/temporal.py` indexes results by the time they become eligible, separately
for each team. Every historical query reads that team's permitted evidence and
its membership context at the query timestamp. Queries are deduplicated within
each layer and evaluated in batches from layer zero upward. Full query timestamps
are retained in keys because age and membership can differ within one calendar
day even when the observation cutoff is identical.

The receiver still reads its own preceding-layer state at the query time. Every
historical edge reads its sender at the encounter time. Layer zero uses the
shared global, league and entry priors. There is no historical-state detachment
or persistent learned embedding cache. Reusing prepared input tensors across
optimizer steps is safe because learned states are recomputed each forward pass.

Native training, validation, prediction, team exports and attention inspection
use the same path. Historical exports traverse the actual temporal dependencies
when reporting evidence. Attention tables retain the root edge order and add
the historical sender's query time and information cutoff. Saved models include
the mode, encoding identifier, age and gap scales, half-lives and temporal feature
order. Legacy artifacts missing all three encoding fields are explicitly migrated
to `log1p_v1`, never to the new constructor default. A legacy temporal artifact
must have the recognized logarithmic feature order; missing or inconsistent
orders and partial encoding metadata are rejected. Older baseline artifacts
default to `snapshot`; their inactive encoding choice does not affect predictions.

Rational model fingerprints include the encoding and its scales. Logarithmic
model fingerprints retain the original canonical configuration so previously
saved versions still verify; snapshot fingerprints also remain compatible.
Native configuration/experiment identities serialize the new settings. Prepared
input keys include mode, encoding, rational scales and half-lives, and direct
reuse of a snapshot carrying a different encoding is rejected. The immutable
resident-cache generation and ordinary source-row transforms retain their existing
behavior. Native training resume and frozen experiment runners still reject
source/configuration mismatches; reproduce old experiments with their frozen
sources instead of changing their manifests.

The full dependency graph can grow substantially with history and depth. There
is no hidden sampling or history cap. Raw per-team indexes and source transforms
can be reused; query tensors and differentiable states belong to the current
issuance computation. The reports below measure the original logarithmic models;
real-data runtime, memory and predictive performance of the rational version
have not been measured.

### Shared causal contract repairs

All three modes reject a forecast whose target identity is already present in
the full eligible history. This also covers evidence older than a boundary-memory
segment. Legitimate schedule revisions with their original issuance anchors
remain permitted. Fitting and validation-selection boundaries use the later of
the target metadata's release bound and the corresponding historical event's
release bound. These repairs address inconsistent declarations without changing
the baseline's encoder or loss on valid inputs.

## Usage

```python
from dataclasses import replace
from xdiyo_analytics.graphs import TemporalGraphConfig, TemporalGraphAdapter

baseline = TemporalGraphConfig("Europe/Berlin")
timing_only = replace(baseline, temporal_mode="time_features")
historical = replace(baseline, temporal_mode="historical")
logarithmic = replace(historical, time_encoding="log1p_v1")

# Select one mode for the existing native training/validation pipeline.
model = TemporalGraphAdapter(historical, event_store, control=training_control)
```

Keep configuration, populations and fitting controls fixed across the comparison
except for the declared intervention. Custom time scales are a separate hyperparameter
choice. The native experiment builder exposes the mode, encoding and scales in graph
configuration; optional extensions must remain disabled for the two new modes.

## Change map

| Files | Change |
| --- | --- |
| `graphs/config.py` | Three modes, validated scales, explicit extension compatibility |
| `graphs/temporal.py` | Release index, time features, historical query plan and evidence traversal |
| `graphs/preprocessing.py` | Dynamic temporal inputs beside the existing source-row cache |
| `graphs/network.py` | Per-edge historical sender states and differentiable layer evaluation |
| `graphs/store.py`, `graphs/adapter.py` | Target exclusion, reconciled release bounds and native integration |
| `graphs/persistence.py`, `graphs/inspection.py` | Versioned temporal feature order and historical diagnostics |
| `ui/graph_inventory.py`, `ui/inventory.json` | Native configuration controls |
| `tests/analytics/test_graph_temporal_attention.py` | Synthetic temporal, causal, gradient and compatibility checks |
| `tests/analytics/test_graph_integration.py` | Recipe preparation, training, Python export and saved prediction for the new modes |

Existing frozen-reference test helpers omit the new, inactive configuration
fields when comparing baseline fingerprints with versions predating them.
The broader checks also exposed an existing pandas 3 incompatibility in optional
statistics labels: a Boolean array returned by pandas was modified in place.
`graphs/statistics.py` now requests its own copy before masking. The same
read-only-array operation was verified in the audited baseline commit; this
compatibility repair changes ownership, not the selected labels or objective.
The frozen-source parity harness applies that same one-line repair only to its
temporary extracted statistics source, retaining all exact numerical assertions.

## Experiment checkpoints

1. **Pause after implementation verification, before a real-data fit.** Compare
   `snapshot`, `time_features`, and `historical` on identical fixtures, permitted
   information, chronological fitting/validation/test windows, seeds, and a
   comparable training budget. Choose the target and primary validation metric
   before looking at test results. Start with one/two layers and profile query
   count, runtime and peak memory on a bounded population before scaling up.
   Include a rolling-feature logistic baseline and a per-team history encoder
   without opponent messages when assessing whether graph propagation adds value.
2. **Pause after that controlled comparison.** Review paired outcome log loss,
   Brier score, class-specific calibration and time-blocked uncertainty. A timing
   ablation isolates explicit temporal inputs; the next ablation isolates
   historical sender states. Implementation correctness alone establishes no
   predictive advantage.
3. **Only after reviewing the evidence**, consider a separately fitted current
   relational-context predictor and a validation-selected probability mixture
   including zero retrospective weight. A genuine two-time smoother and
   recurrent GRU updates require separate design and verification.

## Verification record

Verified on 2026-10-25 with PyTorch `2.14.0+cpu` and pandas `3.0.1`.

- **32 dedicated temporal tests passed.** These cover historical-state and
  unavailable-evidence invariance with fixed weights/preprocessing, each target's
  exclusion from its own forecast, delayed release boundaries, DST, time-feature
  values and missingness, membership changes, one-to-three-layer agreement with
  an independent recursive definition, historical gradient flow, repeated
  backpropagation, cold starts, persistence and inspection.
- **The snapshot network's outputs and gradients match the frozen baseline
  exactly.** All five pinned-source integration comparisons also pass, including
  two optimizer updates and legacy reload. The temporary pandas compatibility
  adjustment is described above.
- **All four recipe workflow cases passed**, covering the baseline and both new
  modes through preparation, training, Python export and saved-model prediction.
- The complete graph scope (`test_graph*.py` plus `test_temporal_graph.py`) was
  executed in chunks: **291 passed, 56 GPU cases skipped, 3 failed**, across 350
  collected cases. The browser reporter module was additionally skipped during
  collection because Playwright is unavailable. The dedicated tests and recipe
  cases above are included in those totals. `git diff --check` passed.

The three remaining failures are the `standard`, `robust` and `none` variants of
`test_training_only_scale_inverse_and_constant` with `signed_log1p`. The optional
statistics transform produces a gradient of zero at zero in this installed
PyTorch runtime, whereas those tests require one. Executing the function directly
from audited commit `0cb13cac` reproduces the same failure. This inherited issue
remains separate from temporal attention; both new modes require statistics
supervision to be disabled. CUDA behavior and browser rendering were not verified.

To rerun the focused contract checks from the repository root:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/analytics/test_graph_temporal_attention.py -q
```

## First controlled comparison

The user authorized checkpoint 1 profiling and the three-mode comparison on
2026-10-25, then cancelled the CPU run and supplied a CUDA environment. The active
protocol is frozen in `experiments/gat_temporal_cuda_allocator_20261025/protocol.json`.
The reusable runner is `examples/compare_gat_temporal.py`. All 43 targeted checks
passed in the supplied environment, including CPU/CUDA agreement of historical
states and gradients, native CUDA profile/fit, exact update budgets, metric
order, causal rolling features and input-integrity enforcement. The three added
cache checks require exact losses, parameter updates and gradients between
cached and uncached CUDA execution for every mode.

The cancelled CPU artifacts remain in `experiments/gat_temporal_20261025/`, with
a `CANCELLED.json` receipt and the original runner source. Only the snapshot and
time-features fits for seed 17 completed there; they are excluded from the active
comparison. CPU profiling measured approximately 0.93, 1.22 and 66.2 seconds per
update including validation. The CUDA run uses a different Python/PyTorch
environment, so these observations are not a controlled CPU/GPU speed benchmark.
An initial CUDA profile in `experiments/gat_temporal_cuda_20261025/` stopped at
the conservative 2.5 GiB host-RAM guard before scored fitting. Its `ABORTED.json`
records the resource-only revision: a 4 GiB host bound and the existing immutable
input cache, capped identically at 3 GiB for all modes. Models, data, seeds,
selection and update budgets were unchanged. No test scores informed this change.
The next profile retained no historical inputs: inactive CUDA reservations grew
to nearly the device capacity. The final execution protocol therefore sets the
native allocator's `garbage_collection_threshold:0.6` and per-process memory
fraction `0.8` for every mode. Expandable segments were tested and reported
unsupported on this Windows build. The supported controls are described in the
[PyTorch 2.9 CUDA notes](https://docs.pytorch.org/docs/2.9/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf);
the [2.9 allocator implementation](https://github.com/pytorch/pytorch/blob/v2.9.0/c10/cuda/CUDACachingAllocator.cpp)
requires the memory fraction to be set for this reclamation branch. These are
execution settings, without changes to model equations or numerical precision.

| Choice | Frozen value |
| --- | --- |
| Historical archive | 2019/20 through 2024/25, Bundesliga and Bundesliga_2; 3,670 fixtures |
| Target | Bundesliga home/draw/away, finished fixtures with observed scores |
| Fitting | 2022/23, 306 fixtures, 34 round issuances |
| Validation | 2023/24, 306 fixtures, 34 round issuances |
| Test | 2024/25, 305 fixtures, 34 round issuances |
| Encoder | Two layers, 32 dimensions, two heads; other baseline defaults |
| Seeds | 17, 42, 73, separately fitted for every mode |
| Budget | 20 full-population optimizer updates per fit, no early stopping or scheduler |
| Selection | Lowest validation log loss within each fixed-budget run |
| Execution | CUDA on RTX 4070 SUPER; one host Torch thread; fresh sequential worker processes |
| Environment | User-supplied `.misc314/Scripts/python.exe`, Python 3.14, PyTorch 2.9.0+cu130 |
| Profiling | Two disposable updates including validation, reported separately; synchronized CUDA timing, query sizes, RSS and peak VRAM |
| Input residency | Existing immutable-input cache, maximum 3 GiB per process, with 1 GiB free-VRAM headroom; no learned state is cached |
| CUDA allocator | Native backend, garbage-collection threshold 0.6, per-process memory fraction 0.8 |
| Memory bound | 4 GiB worker RSS; abort rather than silently alter a model's history |
| Uncertainty | Paired circular bootstrap of four consecutive test rounds, 2,000 resamples, seed 20261025 |

All modes use the same hash-verified prepared tables, split ordering, cutoff
policy and permitted history. Source hashes include the uncommitted encoder and
runner; each worker verifies them before execution. Fitting labels are known by
2023-05-27 22:00 UTC, and validation labels by 2024-05-18 22:00 UTC, before the
next population's first information cutoff. Inputs are read-only. This bounded
archive is shared by all three models and differs from the older 15-season pilot.

Equal updates do not imply equal wall time. Paired seed numbers use the same
initialization algorithm; the extra time-input dimensions alter RNG consumption.
The retrospective schedule/membership assumptions remain explicit, and these
benchmark seasons have been explored previously. They are not a pristine final
holdout. A fixed rolling-goals logistic comparator provides context; a
no-opponent history encoder is still needed before making claims about graph
propagation itself.

The final profile completed. Snapshot updates took 1.05/0.39 s, explicit-time
updates 1.48/0.38 s, and historical updates 49.15/40.55 s. The second historical
update is only partially cached: the profile recorded nine cache hits and 127
misses across its two updates. Peak historical working set was 1.72 GiB, peak
allocated VRAM 2.90 GiB, and peak reserved VRAM 9.58 GiB. These are two measured
updates, not a steady-state timing distribution. The three-mode scored run uses
exactly this common execution policy; longer-run reuse is recorded separately.

**Status: completed; paused at checkpoint 2.** All nine scored CUDA fits completed
their 20-update budgets and passed the recorded input, selection and checkpoint
checks. Mean test log losses were 1.060626 (snapshot), 1.095034 (time features),
and 1.095437 (historical). Both new modes were worse in all three seeds;
historical-versus-time-features uncertainty includes zero. Historical fitting
averaged 125.18 s versus 11.05 s for snapshot, and prediction over all 305 test
fixtures averaged 39.48 s versus 0.25 s.

The [full comparison report](gat_temporal_experiment_20261025.md) records paired
uncertainty, calibration, cold/warm costs, memory, caveats and reproduction.
Snapshot remains the research reference. Inspect the shared time-input and
optimization path before a further experiment; do not automatically add another
architecture or tune against the test results.

### Authorized follow-up: 200-epoch cap and patience 15 (2026-10-25)

The user requested extending training or adding patience of 15 epochs. The
follow-up uses both: a common maximum of **200 full-training-population updates**
and early stopping after **15 consecutive updates without a strictly lower
validation log loss** (`min_delta=0`). Every run restores its lowest-validation-loss
checkpoint. There is no scheduler, restart, architecture change or hyperparameter
search. Realized epoch counts may differ under this shared stopping rule.

The original prediction exports contain model weights and preprocessing, without
AdamW optimizer state. All nine fits therefore start again from the original
seeds. Their first 20 training/validation measurements must exactly reproduce the
original trajectories. The original experiment remains preserved. The new run is
`experiments/gat_temporal_cuda_patience15_20261025/`; its protocol records the
parent protocol and artifact hashes, copies the prepared Parquet inputs byte for
byte, and checks unchanged model/shared-library source. Seeds, populations,
learning rate, loss, input residency and CUDA execution settings remain fixed.
The disposable two-update resource profiles are explicitly inherited from the
original run; complete fitting and prediction costs are measured again.

**Status: completed; paused at checkpoint 2 again.** All nine runs stopped through
patience between epochs 41 and 64; none reached 200. Snapshot retained epochs
36/26/28, time_features 42/36/49, and historical 42/37/49 for seeds 17/42/73.
Every first-20-epoch training/validation trajectory matched the original exactly.
All 54 targeted experiment and native-control checks passed, along with the
post-run input, stopping, score and checkpoint audit. The report renderer supports
unequal trajectory lengths, showing the mean only where all seeds are observed.

Mean test log losses are 1.057438 (snapshot), 1.060439 (time_features), and
1.059513 (historical). The earlier gaps versus snapshot, +0.034408 and +0.034811,
shrank to +0.003001 and +0.002075. All paired exploratory 95% intervals include
zero. Longer training removed most of the short-budget deficit; predictive
superiority of either extension remains unestablished. Historical complete fits
averaged 152.53 s and complete-test prediction 35.23 s, versus 20.68 s and 0.23 s
for snapshot. The [follow-up report](gat_temporal_patience15_20261025.md) records
the full results, uncertainty, resource costs, provenance and reproduction.

Snapshot remains the default and cost reference. This remains an exploratory
comparison on an already inspected test season. Do not automatically increase
the cap, change the model or begin another architecture experiment.

### Authorized diagnostics of the patience-15 experiment (2026-10-25)

The user supplied a critique of the original 20-update experiment and requested
its tests on the updated patience-15 results. This authorizes the following
diagnostics in `experiments/gat_temporal_diagnostics_20261025/`, preserving both
previous experiments and the production model implementation:

- Re-score all nine saved test predictions for draw/not-draw log loss and Brier;
  verify the exact draw/winner decomposition; compare against saved logistic,
  training-frequency and uniform baselines with the existing paired block method.
- Inspect all saved learning curves and probability distributions, including
  per-fixture cancellation between the temporal models.
- On seed 17, inspect initial and retained models: full train/validation scores
  and gradients; root-query projection magnitudes, tanh inputs and native attention
  at first/middle/last train and validation issuances. Root time-feature summaries
  cover all training and validation groups; historical dependency-feature moments
  cover the six selected groups. Quantiles for large activation/dependency arrays
  use explicit bounded deterministic samples; scalar moments are exact.
- Hold each retained temporal model's parameters fixed and change only current
  versus historical sender lookup on all validation fixtures; compare sender
  states and resulting probabilities.
- Run two small training/validation arms from the same snapshot initial function,
  using the expanded `time_features` architecture in both: time channels zeroed
  versus real time channels with learnable connections. Seed 17, 200-epoch maximum,
  patience 15, `min_delta=0`, unchanged optimizer and selection. New training does
  not evaluate test fixtures. Copy common parameters and zero appended E/U weights;
  verify initial logits within `atol=1e-6, rtol=1e-5`, preprocessing and input identity.

**Status: completed; paused at checkpoint 2 again.** Sixteen targeted tests and
the independent score/stopping/checkpoint audit passed. The exact draw/winner
decomposition holds within 1.67e-16. Draw log losses are 0.566338 / 0.566114 /
0.565516 for snapshot/time_features/historical, versus 0.565166 for the training
frequency constant and 0.563534 for logistic. Draw AUC is approximately 0.50–0.51;
the small temporal gains over snapshot do not establish useful draw discrimination.
Paired comparisons against logistic remain inconclusive.

The measured time projections are large relative to other edge projections,
but severe tanh saturation affects less than 0.02% of measured root-message
coordinates, and time-connection gradients are nonzero. Historical sender states
differ measurably; switching only lookup mode at fixed weights changes validation
class probabilities by approximately 0.2–0.3 percentage points on average.

The matched arms start with bit-identical probabilities across all training and
validation fixtures and match snapshot within 5.96e-8. Zero-time stops at epoch 51,
selecting 36; learn-time stops at 49, selecting 34. Learn-time has slightly lower
retained training loss but worse validation loss, 0.979803 versus 0.974998. No new
test evaluation occurred. Both exports reload with bit-identical validation
probabilities, and both previous experiments remain intact.

The [diagnostic report](gat_temporal_diagnostics_20261025.md) records the complete
requested checks, exact/sampled measurement scope, initialization control,
uncertainty and limitations. Keep snapshot as the default and cost reference.
No GRU, retrospective mixture, additional layer, binary refit or no-opponent
encoder was added. Further fitting requires a separately declared experiment.

### Bounded temporal representation (2026-10-25)

Implemented the focused representation change requested after the diagnostics.
The three logarithmic coordinates are replaced by `r(d;tau)` for new temporal
configurations, with fixed age/gap defaults of 180/7 days. The half-life channels,
missingness indicators and 14-dimensional default layout are preserved. The
current feature and compatibility contracts above describe this version.

Changes in this step:

| Files | Purpose |
| --- | --- |
| `src/xdiyo_analytics/graphs/config.py`, `temporal.py` | Encoding selector, separate validated scales, stable vectorized rational evaluation and versioned feature names |
| `src/xdiyo_analytics/graphs/preprocessing.py`, `streaming.py` | Encoding-aware snapshot reuse and immutable prepared-input keys |
| `src/xdiyo_analytics/graphs/adapter.py`, `persistence.py` | Rational fingerprints, explicit metadata, recognized legacy logarithmic loading and compatible old fingerprints |
| `src/xdiyo_analytics/ui/graph_inventory.py`, `inventory.json` | Native encoding/scale controls; preserve snapshot as the model-mode default |
| `tests/analytics/test_graph_temporal_attention.py`, `test_graph_integration.py` | Focused numerical, query, gradient, cache, persistence and UI coverage |
| `tests/analytics/test_graph_goal_features.py`, `test_graph_speed.py`, `test_graph_preparation_memory.py` | Exclude inactive new settings from frozen pre-option baseline reference fingerprints |
| `docs/analytics/gat_temporal_research.md`, `temporal_team_gat.md` | Current representation contract, reproduction guidance and verification record |

The network, causal query rules, ordinary edge transforms, recency/seasonal
weights, support gates, head, loss, optimizer and stopping policy were not changed
in this step. No real-data fit or representation comparison was launched.

Validation used `C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe`
(Python 3.14.0, Torch 2.9.0+cu130), including CPU and RTX 4070 SUPER CUDA checks:

```powershell
& 'C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe' -m pytest `
  tests/analytics/test_graph_temporal_attention.py `
  tests/analytics/test_graph_integration.py::test_catalog_fields_and_defaults `
  tests/analytics/test_graph_integration.py::test_recipe_prepare_fit_reuse_export_and_saved_prediction `
  tests/analytics/test_ui_inventory.py::test_saved_inventory_has_widgets_help_and_packaged_file `
  tests/analytics/test_graph_goal_features.py::test_default_matches_legacy_preparation_gradients_update_and_fingerprint `
  tests/analytics/test_graph_speed.py::test_full_update_gradient_optimizer_rng_and_continuation `
  tests/analytics/test_graph_preparation_memory.py::test_streaming_bit_exact_training_and_exports `
  -q --basetemp=.pytest_tmp/rational_final
```

**Result: 90 passed in 39.98 seconds, with no skips.** These checks cover zero and
half-scale values, monotonicity, finite bounded outputs, invalid inputs, fractional
days, missing versus zero gaps, unchanged half-life coordinates, root/dependency
feature equality, causal invariance and target exclusion. Both temporal modes
have finite nonzero E/U temporal gradients. CPU/CUDA agreement uses the existing
value tolerances (`rtol=1e-5`, `atol=1e-6`) and gradient tolerances (`rtol=2e-5`,
`atol=2e-6`). Same-device export/reload, resident-cache updates and frozen snapshot
baseline checks passed exact comparisons. Both encodings and custom scales survive
reload, while ambiguous metadata and incompatible snapshot reuse are rejected.

Separately, all nine unmodified patience-15 model exports loaded successfully
with their original model fingerprints. The six temporal exports explicitly
resolved to logarithmic features. SHA-256 checks passed for all 80 files covered
by the three earlier completion manifests (11 original, 10 patience-15 and 59
diagnostic files), including frozen source archives. `git diff --check` passed.

The concrete compatibility constraint is equal input width: old weights cannot
identify their intended encoding. The loader now handles this through explicit
metadata and recognized legacy feature orders. Existing source/configuration
checks deliberately prevent resuming an old frozen experiment against the changed
working tree; its saved prediction models remain loadable. No unresolved
integration failure was found in the targeted checks.

**Paused before the next experiment.** Predictive benefit and real-data resource
cost of the rational representation remain untested. A comparison of zero-time,
logarithmic and rational representations requires the separately specified
controlled protocol requested by the user; do not start it automatically.

### Parallel auxiliary goal supervision (2026-10-26)

The user separately authorized an auxiliary-supervision experiment on the current
two-layer **snapshot** reference. This does not evaluate the new temporal encoding.
The question is whether goal supervision improves outcome forecasting; improved
goal forecasts alone do not count as success.

`goal_auxiliary=None` remains the ordinary default. `GoalAuxiliary(beta=.1)` adds
one parallel linear projection from `[home, away, home-away, home*away, fixture]`
to two outputs, with rates `softplus(z) + rate_floor` (`rate_floor=1e-6`). The
outcome head retains its original inputs and probabilities. Rates are neither
concatenated into that head nor converted to Poisson outcome probabilities.
`detach_encoder=True` is the diagnostic route. The new module is initialized
after the existing modules, preserving their random-number consumption.

Training adds `beta * mean_i(sum_side(lambda - y*log(lambda)))` to outcome loss
and the existing prior-embedding penalty. AdamW's decoupled weight decay remains
an unchanged optimizer operation. Native global fixture weights normalize every
issuance group's contribution; output coordinates are summed, not averaged.
Separate outcome, goal, prior-penalty and total-objective columns are recorded.
The reported full marginal Poisson NLL adds exact `lgamma(y+1)` constants. No joint
goal distribution or independence claim follows from these marginal objectives.

The availability-aware statistics-label lookup is reused for the two canonical
goal columns. The existing statistics MLP and its average-over-targets loss are
incompatible with this contract and remain disabled. Goal labels are strict:
both sides must exist, be available, match fixture/team identities and agree with
the outcome label. A target fixture contributes once, independent of its two
directed history rows. No imputation, target transformation or population removal
is performed. Public `predict_goals(context)` returns a separate DataFrame and
enforces the same prediction boundary as outcome inference.

The frozen matrix is:

| Arm | Beta | Goal input | Seeds |
| --- | --- | --- | --- |
| Control | 0 | Shared fixture representation; goal branch excluded from backward | 17, 42, 73 |
| Attached | 0.1 | Shared fixture representation, with encoder gradients | 17, 42, 73 |
| Detached diagnostic | 0.1 | Detached fixture representation | 17 |

All seven fits copy the same saved encoder, outcome-head and goal-head tensors
within seed. Initial outcome logits are checked against the original snapshot
model, and paired initial probabilities/rates and prepared inputs must match
exactly across all fitting/validation fixtures. The reference populations contain
306 fitting and 306 validation targets, with 34 issuance groups each. Maximum 200
full-population updates, patience 15, `min_delta=0`, native AdamW/regularization,
and restoration/selection on **validation outcome log loss only** are retained.
No test prediction, test-driven selection or additional arm is authorized.

Full fitting-population gradient diagnostics run at initialization and each
restored selected checkpoint. They exclude both exclusive heads and the
regularizer, report `beta * ||g_goal|| / ||g_outcome||` and cosine with explicit
zero-norm handling, and verify that parameters, optimizer state and existing
gradient buffers remain unchanged. There is no global clipping or shared loss
scaling in this training path. Detached shared updates must match control within
the existing numerical policy across the entire observed trajectory.

Implementation files are `graphs/extensions.py`, `config.py`, `network.py`,
`goals.py`, `adapter.py`, `persistence.py` and `__init__.py`, with native cleanup
in `training/native_resume.py` and construction controls in `ui/recipe.py`,
`graph_inventory.py` and `inventory.json`. Disabled-head fingerprints retain
legacy canonicalization. The dedicated tests are `test_graph_goal_auxiliary.py`
and `test_gat_goal_experiment.py`; existing baseline helpers omit the new inactive
configuration field. The runner is `examples/compare_gat_goals.py`; verification,
paired tables, static figures and reporting are in `examples/render_gat_goals.py`.

Preflight: **108 targeted checks passed in 48.69 seconds** using the CUDA Python
environment, including CPU/CUDA loss and gradient contracts, unequal issuance
groups, short reference/control/detached trajectories, legacy loading, public
boundaries and three disposable synthetic worker runs. All nine existing
patience-15 exports also loaded with no goal head and their original fingerprints.
The initial combined preflight exposed two inherited temporal GPU reload fixtures
requesting bit-exact comparisons with determinism disabled (differences of order
1e-8). Those fixtures now explicitly enable the existing deterministic option;
the production numerical path and tolerances were not changed.

The protocol and 217 source files were frozen in
`experiments/gat_goal_auxiliary_cuda_20261026/` before the seven real-data fits.
The protocol SHA-256 is
`a31f988b43648ba324dd934c6d13f88dcc9e13c18110d81c290878caaeafe3d7`;
the frozen source archive SHA-256 is
`572aea657459d88f58f13f989703d5e322c9f7739b5baade2acb0153c3613f1f`.
The preservation manifest records 229 prior artifact files. Label checks confirm
306 complete, uniquely aligned home/away pairs in each supervised partition.
For all three seeds, copied initial outcome logits match the reference snapshot
model exactly. Saved parameter, preprocessing and complete prepared-input digests
are checked independently in every arm.

The CUDA resource policy remains the reference policy: one inner thread, 3 GiB
input-cache budget, 1 GiB headroom, 4 GiB worker RSS limit, 80% CUDA allocation
fraction and allocator garbage-collection threshold 0.6. No mixed precision,
new clipping or learning-rate schedule was introduced.

**Status: all seven fits completed; paused for review.** The mean validation
outcome log loss is 0.976712 for control and 0.976091 for attached, a difference
of -0.000622. Seeds 17 and 42 improve (-0.001793 and -0.001061); seed 73 worsens
(+0.000990). This passes the declared preliminary screen, with a small effect on
already inspected validation data. The optional feature remains disabled by
default.

The gain comes from mean draw log loss (-0.002218); conditional winner log loss
worsens (+0.002171), in all three seeds. Attached full marginal goal NLL is
3.129441 versus 3.214171 for fitting-target constant rates. Home MAE slightly
worsens versus that comparator; away MAE improves. Goal benefit is therefore
distinct from the small outcome gain.

The detached arm has identical full outcome curves, retained probabilities and
all 21 common selected parameter tensors to control. Its goal NLL of 3.320045
does not beat the constant comparator, although away MAE improves. Its goal
gradient contribution to shared parameters is zero. Attached selected-checkpoint
gradient ratios are 1.1321 / 1.7602 / 0.3498 for seeds 17/42/73, with cosines
-0.4601 / -0.0969 / -0.5180. These local alignment diagnostics are not proofs of
generalization or harmful transfer.

Control completed/selected updates are 51/36, 41/26 and 43/28; attached are 51/36,
45/30 and 49/34. Detached is 51/36. Attached outcome optimization is not uniformly
faster: it has lower validation loss at only 18/51, 15/41 and 5/43 common observed
updates. Its slightly lower observed minima in two seeds coexist with delayed or
worse trajectories elsewhere. No stopped suffix was extrapolated.

Mean measured fit time is 29.571 s control and 31.336 s attached (about 6% more),
including initialization diagnostics and post-update training/validation curves.
The different completed budgets also contribute to this ratio. All arms have
28,383 parameters, including the 284-parameter linear goal head.

Independent stopping, retained scoring, goal-label and loss-decomposition audits
passed. Both public prediction boundaries remain enforced; all seven exports
reload exactly. The controls reproduce the prior snapshot curves and selected
weights bit for bit. All 229 preserved artifact hashes and 94 new completion
manifest hashes verify. No goal-experiment workers remain and no new test
predictions were generated.

The [research interpretation](gat_goal_auxiliary_20261026.md) answers all eight
questions, including goal calibration, detached-readout limits, common-epoch
curves and cost. The [complete generated report](../../experiments/gat_goal_auxiliary_cuda_20261026/report.html)
contains per-seed/arm metrics, rates, gradients and plots; the run directory holds
per-fixture and per-round differences and the frozen protocol/source. Stop here;
any stronger chronological evaluation requires a separately declared experiment.

### Prespecified four-study suite (2026-10-26)

The user authorized implementation and execution of the full 108-fit handoff, superseding the prior pilot stop-for-review boundary for this new suite. The immutable handoff, 108-row matrix and durable status are under `experiments/gat_four_study_suite_20261026_01/`. Earlier experiment artifacts remain preserved.

The suite adds structured attack/defence Poisson, independent NB and integrated shared-Gamma auxiliary objectives, two fixed chronological folds, fold-common recent-window preprocessing, copied canonical initial states, inner draw-intercept calibration, practical baselines and paired circular-block intervals. All families remain optional; the original classifier supplies outcome probabilities. Source/configuration freeze and preflight precede any main outer scoring. Main fits have not started at this entry; see the suite ledger for current status.

The source and configuration were frozen at 2026-10-25 18:24 UTC after 25 new
correctness/lifecycle/report checks and 108 targeted regression checks passed.
Both fold baseline sets completed and main CUDA fits started. The largest
disposable historical profile used 3.68 GiB of allocated CUDA memory and about
2.04 seconds per warmed full-population update; these profiles do not contribute
model-comparison results. The suite keeps the 200-update cap, patience 15 and raw
inner outcome-loss selection for every declared fit. Current execution and
per-study completion remain in the suite ledger.

Study 1 completed all 18 snapshot fits and its saved-prediction audit. Expanding
the supervised window improved raw outer log loss for all six fold/seed pairs;
the fixture-weighted mean difference is -0.010570, with a descriptive four-round
95% interval [-0.022048, 0.000280]. Adding flexible goal supervision to the
expanded control worsened the mean by 0.004227 [0.002134, 0.006348]. Calibration
does not remove that worsening (difference 0.004338). These are already-inspected
development seasons, and the comparisons do not change the remaining matrix.
See the [Study 1 report](../../experiments/gat_four_study_suite_20261026_01/01_supervised_window/report.html).

A reporting-only source revision fixes one-ULP CSV parsing in the exact stopping
audit by requesting round-trip float parsing. The native in-memory audit and
checkpoint choice were correct. Original source/manifest evidence and the patch
are preserved under `revisions/001_report_float_roundtrip/`; the active source
manifest records that patch. No training source, input, prediction or selected
checkpoint changed. All 18 baseline models (54 prediction exports) also passed
independent categorical/count-score and fixture-alignment checks.

Study 2 completed 36 new temporal/historical fits and independently audited all
54 fitted models represented in its report. For the expanded time-feature
encoder, flexible auxiliary supervision improves combined raw outer log loss by
0.006763 (C-B difference -0.006763, descriptive interval [-0.009480, -0.004140]);
the calibrated difference is -0.006511 [-0.009204, -0.003819]. Its effect differs
from the snapshot result, so the snapshot pilot does not determine the temporal
conclusion. Historical minus time-feature differences are small: +0.000039 for B
and +0.000212 for C in raw log loss, both with intervals spanning zero. Historical
outer replay averages about 30 seconds versus 0.46 seconds for time features;
allocated CUDA peaks average about 4.14 GiB versus 0.13 GiB for expanded arms.
The exact fold/seed and draw-component evidence is in the
[Study 2 report](../../experiments/gat_four_study_suite_20261026_01/02_temporal_historical/report.html).
Study 3 proceeds with every prespecified encoder, regardless of these results.

Study 3 completed all 18 structured Poisson fits and audited the 72 represented
models. Structured minus expanded outcome-only raw log-loss differences are
+0.001121 for snapshot, +0.001556 for time features, and +0.000492 for historical;
none has a lower combined mean. Against the flexible head, structured time
features worsen outcome loss by 0.008319 [0.005531, 0.011141], while the
corresponding aggregate goal score improves slightly. This is a concrete example
of improved goal forecasting failing to improve the actual outcome objective.
Selected structured-head weighted gradient ratios span about 0.157–0.387. Their
cosines are positive in fold 1 and negative in fold 2; those local measurements
do not establish generalization or causal harm. The
[Study 3 report](../../experiments/gat_four_study_suite_20261026_01/03_structured_poisson/report.html)
contains all fold/seed scores and centered effects. All 36 declared NB/shared-Gamma
fits proceed without changing their initialization or dispersion learning rate.

Study 4 completed all 36 declared count-family fits. Independent NB minus
structured Poisson improves the combined time-feature outcome loss by 0.001008
(difference -0.001008, descriptive four-round interval [-0.001487, -0.000571]),
but this is concentrated in fold 1, seed 42. The fold-2 difference is only
-0.000038 [-0.000158, 0.000105]. Shared Gamma loses that gain: its time-feature
G-N difference is +0.001020 [0.000573, 0.001507]. Other-mode N-S and G-N
differences are tiny with intervals spanning zero. Shared Gamma has worse
combined native joint goal NLL than independent NB in every encoder. Selected
dispersion is 0.0178–0.0523, below its fixed 0.1 initialization; these short,
outcome-selected fits do not establish exhaustive goal-likelihood optimization.
The [Study 4 report](../../experiments/gat_four_study_suite_20261026_01/04_shared_random_effects/report.html)
contains the complete paired comparisons and dispersion diagnostics.

The full suite is complete: 108 new fits, zero reused, failed, blocked or
invalidated fitting jobs, and no workers running. There were 4,626 full-population
updates, 31–69 completed updates per fit and selected checkpoints at 16–54.
Measured fitting time sums to 2.052 hours; per-model end-to-end time sums to
3.257 hours. Freeze through the first complete automated audit took 3.495 hours.
Maximum allocated/reserved CUDA memory was 4.694/9.033 GiB, and maximum sampled
host RSS was 1.996 GiB. The 25 new and 108 regression checks passed before
fitting. All 108 selected models and all 18 baselines were audited, and all 373
protected earlier artifact hashes remain unchanged.

Final report review independently reproduced all 62 primary combined point
comparisons from separately scored seed exports. It preserves the generated
originals and 2,765 unchanged evidence files. Derived gradient tables now label
the beta-weighted cosine explicitly: beta-zero controls have zero auxiliary
contribution and an undefined weighted cosine, even when the counterfactual
unweighted goal gradient exists. No gradient measurement, training source,
selected checkpoint or prediction changed. This report-only clarification is
recorded under `revisions/002_report_interpretation/` in the suite archive.

The most promising candidate for a separately declared evaluation is expanded
supervision with bounded time features and the flexible Poisson head. Its mean
outcome gain survives calibration and appears in seeds 17 and 42 in both folds;
seed 73 worsens in both. The gain mainly concerns conditional winner prediction:
the combined draw-loss interval spans zero and fold 2 has a small deterioration.
The expanded rolling logistic baseline remains competitive, with better mean
draw scores; the candidate's three-class advantage over logistic has an interval
spanning zero. Historical states add little outcome benefit at much greater
measured replay and memory cost. Structured goal supervision improves goal
scores without improving the actual outcome objective, and the count-family
extensions do not displace the flexible time-feature candidate.

All conclusions remain exploratory because these chronological periods were
already inspected in development. The [reviewed suite summary](../../experiments/gat_four_study_suite_20261026_01/summary.html)
answers the eight handoff questions and links every report, frozen protocol and
audit receipt. Optional goal features remain disabled by default. Changes remain
uncommitted on `codex/gat-temporal-research`. Stop here for review; no additional
fits, tuning, deployment or newly selected evaluation period have been started.

### Cross-league supervised expansion (2026-10-26)

The user subsequently authorized all 36 fresh fits in
`xdiyo_cross_league_expansion_handoff.md`. The new archive is
`experiments/gat_cross_league_expansion_20261026_01/`; the completed four-study
archive and all earlier protected evidence remain unchanged. The matrix is
snapshot B, time B and time C, local/pooled supervision, two folds and seeds
17/42/73. Checkpoint selection remains raw Bundesliga inner CE for every model.

The authoritative competition registry contains 13 European leagues, including
2. Bundesliga. Pooled fitting populations contain 9,157 and 13,954 fixtures;
the exact parent Bundesliga fitting populations remain 611 and 917, with outer
populations 306 and 305. Foreign targets follow parent issuance/availability
windows, so eligible seasons are sometimes partial. Both arms share the same
all-league history, schedule, category registry and copied ordinary scalers.

Semantic expansion of untrained canonical weights reproduces old-context
outputs within 5.96e-8. Added indicator/temporal columns start at zero, and new
league priors use a separately derived RNG stream. Full versus complete-component
computation passed 24 CPU/CUDA output and all-parameter gradient cases, with
exact support, rest and temporal-gap inputs. This restriction reduces the
largest-fold warmed profile from roughly 19 to 14 seconds per update and peak
allocated CUDA memory from 4.22 to 1.31 GiB. It retains all edges in every
component connected to target teams and leaves the global regularizer unchanged.

Foreign entry boundaries use reviewed movement evidence at the first original
team-round issuance of the season, rather than mechanically applying July 1.
This is a conservative retrospective-schedule assumption; earlier forecasts do
not receive a backfilled future movement status. The German assumptions and
source rows remain as archived. Full inventory, effective boundaries, feature
coverage, source versions and preflight checks are recorded in the new archive.
Main fitting starts only after its local protocol freeze and relevant checks.
See its durable status ledger for execution progress.

The protocol/source freeze completed at 2026-10-26 05:58 UTC. All 11 focused
experiment tests and 85 relevant regression tests passed. The two initially
gated CUDA continuation tests also passed when run explicitly; the interrupted
broader regression and successful isolated persistence check remain documented.
All 54 outcome baselines and 26 fitting-mean goal references completed. An
independent read-only audit reproduced their saved-model probabilities, raw
losses, exact decomposition, inner calibration, and full Poisson scores across
162 outcome and 78 goal prediction tables. No solver extension or class fallback
was required. The fixed 36-fit CUDA queue is running from fresh canonical states.

During pooled time-B export for fold 1 / seed 17, the first completed trajectory
(170 updates, selected update 155) hit a 5.24e-12 diagnostic decomposition error
from subtracting a conditional probability near one. A recorded equivalent
formula now uses the two non-draw probabilities directly, preserving the 1e-13
assertion. Two focused numerical tests pass, including bitwise preservation of
primary CE on 10,000 fixed examples. Torch fitting, monitor, protocol, original
frozen files, probabilities and matrix remain unchanged. Export recovery used
the original completed state with optimizer updates forbidden and reproduced
the saved weights/probabilities exactly. Two automatic retries and the abandoned
14-update partial retry are preserved; they do not select a new trajectory.
The failed process's original peak-memory telemetry is unavailable and clearly
separated from recovery memory. See `SCORING_REPAIR.json` and the archive README
for the recorded recovery entry point. The remaining declared fits continue.

All 36 declared cross-league fits subsequently completed, together with the 54
outcome baselines and 26 constant-goal references. The primary pooled-minus-local
temporal-C Bundesliga CE difference is +0.002894, with four-round descriptive
95% interval [-0.003113, +0.009367]. The raw scores are 1.006698 pooled and
1.003803 local. Three of six fold/seed comparisons improve; fold 1 worsens by
+0.006524 and fold 2 improves slightly by -0.000748. Calibration does not reverse
the finding. The small draw-CE gain (-0.001437, interval spanning zero) is
outweighed by the winner contribution (+0.004331).

Snapshot B benefits more consistently from pooling: -0.011145
[-0.016544, -0.005551], with all six matched comparisons improving. Pooled time C
is worse than pooled snapshot by +0.003793 [+0.000733, +0.006959]. Its auxiliary
gain against pooled time B is -0.002358 [-0.005809, +0.001278] and is concentrated
in seed 42. The population-by-auxiliary interaction is +0.004396
[+0.000651, +0.008603], so the observed auxiliary gain is smaller after pooling.

Across all 13 leagues, pooled C improves on time B in the macro/micro means but
does not beat pooled snapshot. Raw macro CE is 1.013834 snapshot, 1.017837 time B,
and 1.015554 time C; micro CE is 1.017356, 1.021345 and 1.019095 respectively.
C beats each league's logistic in all 13 point comparisons and pooled logistic
in 12, with Serie B the exception. Bundesliga comparisons with both logistic
references have intervals spanning zero; the local logistic has better mean
draw scores. C's Bundesliga marginal goal NLL also worsens after pooling
(3.163048 versus 3.126720 locally), although both beat the constant 3.228404
reference. Goal forecasting alone does not determine the research decision.

The retained trajectories total 1,974 full-population updates and approximately
3.95 fitting hours. Pooled C costs about 9.8 times its local control's fitting
time. Original peak telemetry exists for 35/36 fits; the largest recorded
allocated/reserved CUDA peaks are 1.719/1.752 GiB and sampled host RSS 1.788 GiB.
The one timestamp-derived duration, unavailable original memory counter,
37.1-second export recovery, and 194.1 seconds of active retry overhead are
explicit in the report. An abandoned 14-update retry is preserved and excluded
from the retained trajectory totals.

The independent final arithmetic review passed 270 outcome tables, 36 trained
goal-head tables, 1,980 scored league rows, 2,340 primary point rows, 1,560 paired
seed rows, 234 primary CE intervals and 1,080 secondary CE intervals. Calendar
week groups were reconstructed independently from original issuance dates. The
first report pass's Arrow-versus-NumPy integer identity comparison was repaired
without changing any ordered IDs, labels or predictions. All 36 common
Bundesliga identity/orientation checks pass. The final review also preserved
1,133 fitting/baseline evidence files and all 3,187 protected earlier artifacts.

The [reviewed cross-league report](../../experiments/gat_cross_league_expansion_20261026_01/summary.html)
answers all ten handoff questions, retains every comparison, and links coverage,
class proportions, missingness, per-fixture/round diagnostics, goal scores,
learning curves and resource receipts. The result does not support promoting
pooled temporal C. The simpler pooled snapshot B is the narrower candidate for
any separately declared future evaluation, with logistic references retained.
These periods remain exploratory. Optional features remain disabled by default;
no additional fits, tuning or deployment have been started. Stop for review.

## 2026-10-26: corrected stage clocks and all-league selection

The user authorized the all-league validation handoff and added ranked
probability score (RPS) reporting. The new run is
`experiments/gat_all_league_validation_20261026_01/`, with 18 fresh pooled fits:
snapshot B, bounded-time B and bounded-time C, two chronological folds and
seeds 17/42/73. There is no local-GAT rerun or separate Bundesliga-monitor
repair experiment. The existing source/configuration defaults remain unchanged.

The full archived schedule yields 128 corrected Belgian playoff clocks and
zero changed non-Belgian clocks or membership boundaries. Counts match the
handoff: 9,177/4,593/4,315 and 13,974/4,315/4,275 fitting/inner/outer targets.
Stage identity is grouping metadata only; original scalers and untrained
canonical states are reused. Real examples regain 612 and 480 eligible
directed history edges respectively, without admitting target/future results.

The trainer now monitors `all13_inner_raw_hda_ce_micro` over the actual full
inner population, with one validation pass per update. Macro and per-league
CE are derived from that pass. The original 200-update cap, patience 15,
min_delta zero and earliest-best restoration remain. RPS uses home/draw/away
order and half the sum of squared cumulative-probability errors; it is
secondary and never selects checkpoints. Fold-local configuration choices
use all three seed scores and are sealed before new GAT outer scoring.

The largest-fold disposable CUDA profile used 13,974 fitting and 4,315 inner
fixtures, approximately 16.3 seconds per warm update, 1.19 GiB allocated CUDA
memory and 1.71 GiB sampled host RSS. The new runner saves original training
telemetry before export and keeps last-update optimizer/RNG continuation
separate from restored best weights. Eight affected baseline models/references
will be updated; 48 outcome and 24 goal references will be reused after
invariance checks. Earlier experiment artifacts are hash-protected. The
declared stopping point is after the 18-fit report and final verification.


## 2026-10-27: all-league run completed and independently reviewed

All 18 declared CUDA fits completed on their first attempt, with 1,484 full
fitting-population updates and 5.587 fitting hours. The six affected outcome
models and two goal references were updated; 48 outcome and 24 goal references
were reused after invariance checks. The 128 Belgian clock repairs, unchanged
non-Belgian clocks/memberships, and all six expected population counts passed.
All source, numeric-scaler and canonical-initialization identities stayed frozen.

Fold 1 selected time_B (inner CE 1.013497), only 0.0000647 ahead of snapshot.
Fold 2 selected time_C (1.004268), only 0.0000559 ahead of time_B. Fold 2's
auxiliary improvement is driven by seed 42; seeds 17 and 73 worsen inner CE.
The choices were sealed before new outer scoring. Following the declared rule,
the latest-fold recommendation is time_C despite the weak validation separation.
The lower fixed outer CE of snapshot does not replace that choice after scoring.

The selected procedure scores the two folds with their respective configurations,
averaging separately scored seeds. On 8,590 outer fixtures it achieves raw CE
1.012123, summed Brier 0.605885 and normalized home/draw/away RPS 0.205873.
Pooled logistic scores CE 1.033829 / RPS 0.213054, and own-league logistic
1.034669 / 0.212876. The selected-minus-reference CE differences are -0.021707
[-0.025910, -0.017935] and -0.022547 [-0.026950, -0.018282], using the prescribed
four-week paired calendar bootstrap. Macro differences are -0.022453 and
-0.023691, also with intervals below zero. CE and RPS improve against both
logistic references in all 13 leagues. All three combined seeds improve CE;
selected CE ranges from 1.011515 to 1.012441.

Most of the CE gain comes from conditional-winner prediction. Draw CE improves
by 0.001511 versus pooled logistic and 0.003184 versus own-league logistic;
draw gains occur in 12/13 and 11/13 leagues respectively. Per-league inner
draw calibration worsens the selected outer CE to 1.012986 and RPS to
0.205942, while preserving its advantages over the references. Raw evaluation
remains primary. RPS is secondary, normalized by two, and never selects a model.

The fixed additions do not improve combined outer outcome CE. Time_B minus
snapshot is +0.000275 [-0.000300, +0.000852], although its RPS improves by a
small 0.000033. Time_C minus time_B is +0.000805 [+0.000152, +0.001403] and
RPS worsens by 0.000271. All three combined C-minus-B seed differences are
positive. The auxiliary deterioration is concentrated in fold 1 (+0.001597);
fold 2 is essentially tied (+0.000006). Its draw-CE difference is nearly zero
(-0.000003), with the deterioration in the winner contribution (+0.000809).
Two/eight-week sensitivities retain positive C-minus-B intervals. Paired curves
show no uniform auxiliary optimization speedup; the small late inner gain is
seed-dependent. No stopped trajectories are extrapolated.

C's full marginal goal NLL improves to 2.915960 from the fitting-mean reference's
2.998486, with lower home/away MAE, but this does not establish an outcome
benefit. Predicted means remain below observed means. Full rates and per-league
calibration are retained, including the very low home-rate extremes. The goal
curves omit factorial constants, while reported full Poisson NLL includes them.

The maximum recorded CUDA allocated/reserved peaks are 1.192/1.229 GiB and
sampled process RSS 1.723 GiB. Average fit times are 16.54, 19.62 and 19.72
minutes for snapshot, time_B and time_C. All checkpoints reproduce probabilities
exactly after reload, and public prediction boundaries remain enforced.

All 12 focused tests and 34 relevant regressions passed. The independent review
recomputed 4,050 score rows, 1,890 seed-averaged summaries, 96 Poisson tables,
144 unchanged outcome and 72 unchanged goal prediction tables, and 36 raw
primary-contrast intervals from the frozen 20,000 resamples. The built-in
audit reconciled all 396 CE contrast points, checked continuations and the
selection gate, and preserved 4,538 earlier files. Source/configuration and
all numerical outputs are unchanged by the final editorial interpretation.

The [completed report](../../experiments/gat_all_league_validation_20261026_01/summary.html)
and three configuration reports include RPS, all league/seed comparisons,
curves, goal diagnostics and resource receipts. These were previously inspected
development periods, and timing plus checkpoint selection changed jointly;
earlier-versus-new differences do not isolate either change. Optional features
remain disabled by default. No extra fits, tuning or deployment were started.
The authorized experiment is complete. Stop for review.

### Bounded observed-gap and opponent-propagation protocol (2026-10-27)

The user authorized the new handoff and permits pushing completed research
changes to `codex/gat-temporal-research`, without merging. The new artifact root
is `experiments/gat_rest_propagation_20261027_01/`. Earlier artifacts remain
protected by hashes; no previous experiments are rerun.

The main contrast is corrected full two-layer time B minus an otherwise
identical model whose sender K/V inputs use the trainable layer-zero prior.
Both use the bounded observed-gap contract documented in
`temporal_team_gat.md`. A secondary contrast against archived parent time B
describes the complete input repair and checkpoint reselection; it does not
isolate bounding alone. The 830.270833-day Reggiana gap is an archive-coverage
example, not a true physiological rest measurement or a tuning criterion.

No matching corrected controls existed. The declared queue therefore contains
12 fresh fits: corrected full first, then prior-context, each on two unchanged
13-league folds and seeds 17/42/73. Both arms share semantically mapped parent
untrained states. Unchanged columns and equivalent missingness coefficients
are copied; new continuous and 30-day flag coefficients start at zero.
The fixture input has 26 coordinates, the joint decoder input 154, and both
models have 35,993 parameters. Goal beta remains zero.

Training retains the native full-population updates, AdamW, regularization,
200-update ceiling, patience 15, min_delta zero and earliest-best restoration.
Only `all13_inner_raw_hda_ce_micro` selects checkpoints. Mean selected own-fold
inner CE across seeds chooses the configuration, with exact ties preferring
prior-context. Both choices must be sealed before corrected outer predictions.
The latest-fold inner winner is the working recommendation, separate from the
scientific assessment of the fixed outer contrasts.

CE and normalized RPS receive paired uncertainty for all six declared
contrasts; draw CE receives an interval for the main architectural contrast.
The exact parent 20,000 common calendar-week multiplicities are reused after
fixture/week/mixture checks: four-week primary, two/eight-week sensitivity,
seed 20261027. Seeds are separately scored and then averaged. Unchanged
logistic and frequency references are reused without fitting. No extra arms,
architecture changes or feature family are authorized by this protocol.

### Bounded-gap and propagation experiment completed (2026-10-27)

All 12 fresh CUDA fits completed on their first attempt: zero reused corrected
controls, resumptions or failed attempts. The exact parent populations, clocks,
unrelated scalers and edge-time features were preserved. The new native bounded
gap default and persisted `updated`/`layer_zero` sender modes passed 35 focused
tests, 213 relevant regressions and 91 additional checks, with seven existing
opt-in GPU checks skipped. Two legacy-fixture assumptions were corrected and
rechecked; their original failures and resolution remain documented. The focused
K/V intervention, trainable-prior gradient and persistence checks ran on CPU/CUDA.

The fold-local inner rule selected corrected full time B in both folds, by
0.00013708 and 0.00011103 mean CE. Full wins two of three seeds in each fold;
these are weak separations. Both choices were sealed before corrected outer
evaluation. The latest-fold choice supplies the working architecture for the
next separately declared feature experiment. The earlier time-C selection is
preserved as a historical decision.

On 8,590 outer fixtures, full scores raw micro CE 1.011839 / RPS 0.205801,
and prior-context scores 1.011812 / 0.205792. Full minus prior is +0.0000263 CE
[-0.0002306, +0.0002934] and +0.0000099 RPS [-0.0000673, +0.0000865] under
the predeclared four-week paired bootstrap. One of three combined seeds favors
full on either metric; full has lower CE and RPS in 6/13 leagues. Fold 1 favors
prior on CE and full on RPS, while fold 2 reverses those directions. Macro and
two/eight-week sensitivities also leave the incremental effect unresolved.
An interval spanning zero does not establish equivalence. The draw-CE difference
is +0.00000047 [-0.00010721, +0.00010775], with no resolved draw benefit.

Corrected full minus archived time B is -0.0002810 CE
[-0.0013616, +0.0004306] and -0.0000583 RPS [-0.0002833, +0.0001027].
All three combined seeds improve CE and two improve RPS; fold 1 improves both
metrics and fold 2 worsens both. This describes the complete repair and
reselection rather than bounding alone. The repaired input semantics are
verified, while a persistent predictive gain remains uncertain. The named
Reggiana fixture's corrected probabilities are retained as a case study only.

Both corrected arms outperform both rolling-goal logistic references on CE and
RPS in all 13 leagues, with negative combined micro/macro four-week intervals.
Full minus pooled logistic is -0.021991 CE / -0.007253 RPS; full minus own-league
logistic is -0.022831 / -0.007074. That advantage is also present in the
prior-context control and cannot be attributed to updated-opponent propagation.
Inner-only draw calibration worsens outer CE and RPS for both corrected arms.

Full uses 554 updates and 2.092 fitting hours; prior-context uses 577 and 2.108.
Both have 35,993 parameters. Mean warm updates are 12.486 versus 11.968 seconds,
with peak CUDA allocations 1.192/1.187 GiB and sampled process RSS 1.723/1.725 GiB.
Full reaches its selected checkpoint earlier in all six pairs, but paired curves
change sign across updates and seeds. Its modest inner advantages do not resolve
an outer benefit. Timing is descriptive and total fitting costs are close.

The independent review recomputed all 4,050 score rows and all 468 interval
point/endpoints. Continuations, matched initialization, raw selection before
outer evaluation and 5,626 protected earlier files passed the final audit.
The [tracked result note](gat_rest_propagation_20261027.md) and
[complete local report](../../experiments/gat_rest_propagation_20261027_01/summary.html)
include CE/RPS uncertainty, phase/seed/league metrics, case probabilities,
learning curves and measured costs. These inspected development periods leave
generalization and incremental propagation benefit unresolved. No extra arms,
new feature experiment or deployment was started. Stop for review.
