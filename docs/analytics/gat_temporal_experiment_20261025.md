# First controlled temporal-attention experiment

**The snapshot baseline performed best in this bounded CUDA comparison.**
Adding explicit time features increased test log loss in all three seeds.
Historical opponent states added substantial preparation and inference cost,
with no clear improvement over the time-features model.

These findings apply to the declared 20-update, single-league benchmark. Most
validation curves were still improving at the budget boundary, so this is not a
comparison of converged models. The experiment stops at research checkpoint 2.

The subsequent [patience-15 follow-up](gat_temporal_patience15_20261025.md)
substantially narrowed the temporal-model gaps. This report retains the original
20-update results; use the follow-up for the longer-training comparison.

## Protocol and provenance

- Branch: `codex/gat-temporal-research`, based on `0cb13cac`, with the encoder and
  experiment source frozen by file hashes before scored CUDA fitting.
- Read-only source publications: Bundesliga and Bundesliga_2, 2019/20 through
  2024/25; 3,670 matches / 7,340 directed historical perspectives. All modes use
  this same bounded archive, including cross-division opponents.
- Bundesliga targets: 2022/23 fitting (306), 2023/24 validation (306), 2024/25
  test (305). Each population has 34 round issuances. The normal loader excludes
  awarded matches; one such fixture is excluded from the test season.
- Outcome: home/draw/away, class order `(1, 0, -1)`. By Round forecasts use the
  previous-local-day cutoff in `Europe/Berlin`, with explicit availability and
  target-exclusion checks. Validation begins after all fitting labels are
  available; testing begins after all validation labels are available.
- Two layers, 32-dimensional states, two heads, 64-wide outcome head, decay
  half-life 180 days, seasonal retention 0.5, support constant 5. AdamW learning
  rate 0.001, weight decay 0.001, prior penalty 0.001. Optional heatmaps, memory
  and statistics supervision are disabled. The time bases use 7/30/180 days.
- Seeds 17, 42 and 73; **20 complete training-population updates per fit**;
  no scheduler, restarts or early stopping. Each fit retains its lowest
  validation-log-loss checkpoint. There was no test-guided tuning or refitting.
- Execution: the supplied `.misc314/Scripts/python.exe`, Python 3.14.0,
  PyTorch 2.9.0+cu130, RTX 4070 SUPER, one host Torch thread, sequential fresh
  worker processes. CPU: Ryzen 7 7800X3D; system RAM approximately 31.1 GiB.
- Common execution settings: 3 GiB immutable-input residency budget, 1 GiB free
  VRAM headroom, 4 GiB sampled host-RSS stop, native allocator collection threshold
  0.6 and memory fraction 0.8. No embeddings or autograd states are cached.

The initial CPU run was cancelled at the user's request. Its two completed fits
are excluded. CUDA resource settings were resolved through disposable profiling
before scored CUDA fitting; all neural/data/selection choices stayed fixed.
The [research log](gat_temporal_research.md) records those resource observations.

The schedule and July 1 membership assumptions are retrospective. These seasons
have been explored in earlier project work, so this is not a pristine
confirmatory holdout. Equal seed numbers use the same initialization procedure;
the extra input dimensions change RNG consumption relative to snapshot. The
two temporal modes have the same parameter shapes and parameter count.

## Predictive results

Values are averages of **separately scored seed models**, without probability
ensembling. Log loss, Brier score and ECE are lower-is-better. Brier is the sum
over the three classes, averaged over fixtures. ECE uses ten confidence bins.

| Mode | Parameters | Validation log loss | Test log loss | SD across seeds | Test Brier | Confidence ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| snapshot | 28,099 | **1.003408** | **1.060626** | 0.010762 | **0.640206** | **0.057726** |
| time_features | 31,683 | 1.051025 | 1.095034 | 0.003231 | 0.665885 | 0.089753 |
| historical | 31,683 | 1.050992 | 1.095437 | 0.002783 | 0.666168 | 0.090639 |

| Seed | Snapshot test log loss | Time features | Historical |
| --- | ---: | ---: | ---: |
| 17 | 1.070565 | 1.091519 | 1.092383 |
| 42 | 1.049196 | 1.095708 | 1.096100 |
| 73 | 1.062117 | 1.097875 | 1.097829 |

The snapshot runs all retained update 20. Both temporal modes retained updates
20, 20 and 19 for seeds 17, 42 and 73 respectively. Every run nevertheless
completed all 20 updates. Mean validation loss continued to decline over the
last five updates in all three modes; a longer-budget study could change these
comparisons and has not been performed here.

### Paired uncertainty

Differences are candidate minus reference; positive values favor the reference.
For each fixture, losses were averaged across the three seeds, then paired
circular blocks of four consecutive rounds were resampled 2,000 times using seed
20261025. The same fixture/round draws underlie each paired contrast.

| Contrast | Mean log-loss difference | Exploratory 95% interval |
| --- | ---: | ---: |
| time_features − snapshot | +0.034408 | [+0.020325, +0.046001] |
| historical − snapshot | +0.034811 | [+0.020911, +0.046263] |
| historical − time_features | +0.000404 | [−0.000209, +0.000992] |

These intervals are conditional on these fitted seeds and this season. They do
not quantify all seed uncertainty, multi-season generalization, or account for
repeated use of this benchmark. The historical-versus-time-features interval
includes zero; this comparison establishes no benefit from historical sender
states at this budget.

### Calibration and contextual comparator

Class-specific ECE is mixed, even though snapshot has the lowest overall
confidence ECE. These small-sample binned values are descriptive.

| Mode | Home ECE | Draw ECE | Away ECE |
| --- | ---: | ---: | ---: |
| snapshot | 0.092991 | 0.012965 | 0.078460 |
| time_features | 0.091566 | 0.009604 | 0.081088 |
| historical | 0.092484 | 0.008815 | 0.082797 |

A fixed ten-match rolling-goals logistic comparator scored **1.071337 log loss**,
**0.647421 Brier**, and **0.083206 confidence ECE**. It uses goals-for/against
means for both teams and their differences, under the same history cutoffs;
imputation/scaling fit only on training rows. `C=1` and `max_iter=500` are fixed.
Its solver budget differs from the neural updates, so it is contextual rather
than part of the equal-update three-mode ablation. A no-opponent history encoder
is still absent; this experiment does not establish the value of graph
propagation over strong non-graph alternatives.

## Runtime and memory

Full-fit times include initialization, input construction, all 20 updates,
validation and best-state restoration. Prediction times cover **all 305 test
fixtures**, issued in 34 round groups. They are not per-fixture times.

| Mode | Mean full fit, s | Mean test prediction, s | Median later update, s |
| --- | ---: | ---: | ---: |
| snapshot | 11.05 | 0.25 | 0.414 |
| time_features | 11.36 | 0.42 | 0.406 |
| historical | 125.18 | 39.48 | 0.933 |

Later updates means updates 3–20 across all seeds, including validation and
native training bookkeeping. Historical preparation dominates the initial
updates: the separate disposable profile measured 49.15 s and 40.55 s, with
only nine cache hits over those two updates. In the scored fits, sustained reuse
began around update 3. This is why reporting only warm updates would understate
the end-to-end cost. Historical full fitting was approximately **11.3 times**
snapshot and whole-test prediction approximately **158 times** snapshot here.

Memory below is the maximum across scored fits and test prediction. Host RSS is
the Windows lifetime peak working set. Allocated VRAM measures live Torch
allocations; reserved VRAM also includes allocator-held blocks.

| Mode | Peak host RSS, GiB | Peak allocated VRAM, GiB | Peak reserved VRAM, GiB |
| --- | ---: | ---: | ---: |
| snapshot | 1.490 | 0.098 | 0.117 |
| time_features | 1.495 | 0.115 | 0.135 |
| historical | 1.812 | 3.101 | 9.572 |

At the last test issuance, historical preparation has 45 outer team queries and
7,349 lower-layer queries. Its lower layer contains 677,183 message edges,
compared with 7,304 root edges. Prepared tensors for that issuance occupy about
74.2 MiB. This measured dependency expansion explains the additional work; no
hidden neighbor sampling or historical cap was used within the common archive.

Timings come from one desktop system, with two disposable profile updates and
three scored fits per mode. They are descriptive measurements, not a broad
hardware benchmark. CPU and CUDA preliminary timings should not be used as a
controlled device-speed comparison because their environments differ.

## Verification and artifacts

- 43 distinct targeted contracts passed in the CUDA environment; the runner
  checks were rerun after the final allocator options were added.
- Cached/uncached CUDA losses, weights and gradients matched exactly for all
  three modes on the controlled fixture. CPU/CUDA historical outputs and
  gradients agreed within the test's floating-point tolerances.
- All nine results match the frozen source, prepared-data and fixture-order
  hashes. Update counts, selected validation checkpoints, and independently
  recomputed log loss/Brier agree with the saved receipts.
- Restoring seed 17 for every architecture reproduced the complete first/last
  test-issuance groups, 18 fixtures each, with exactly equal probabilities.
- Prepared data and model source hashes remained unchanged; original source
  manifest hashes also matched after execution.

The complete local evidence is in
`experiments/gat_temporal_cuda_allocator_20261025/`: `protocol.json`,
`environment.json`, `frozen_source.zip`, `results.json`, `verification.json`,
`comparison.csv`, `paired_log_loss.csv`, `class_calibration.csv`, individual
training histories, predictions and model checkpoints. `report.html` contains
interactive learning curves and paired-result plots. Generated evidence remains
outside tracked source.

Reproduction uses a new output directory and the supplied CUDA interpreter:

```powershell
$python = 'C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe'
$run = 'experiments/gat_temporal_cuda_reproduction'
& $python examples/compare_gat_temporal.py prepare --run $run --device cuda:0 --rss-limit-gib 4 --input-cache-gib 3 --cuda-allocator-config garbage_collection_threshold:0.6 --cuda-memory-fraction 0.8
& $python examples/compare_gat_temporal.py profile --run $run
# Review resource results before the scored fits.
& $python examples/compare_gat_temporal.py fit --run $run
& $python examples/compare_gat_temporal.py report --run $run
& $python examples/render_gat_temporal_report.py $run
```

## Checkpoint 2: pause and review

Keep snapshot as the comparison reference. The immediate finding is a shared
performance deficit in both explicit-time models, with little additional change
when switching to historical opponent states. This motivates inspecting time
input scales, saturation and initialization before adding architectural
complexity; it does not identify the cause of that deficit. Any follow-up should
use training/validation diagnostics and a declared optimization budget, with a
fresh evaluation design for stronger claims. No additional model, longer-budget
fit, retrospective mixture or GRU experiment was started.
