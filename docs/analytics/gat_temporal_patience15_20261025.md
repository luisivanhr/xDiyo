# Temporal attention: longer training with validation patience

**All nine CUDA fits completed with patience 15. Longer training substantially
reduced the temporal models' deficit from the 20-update comparison.** Snapshot
still has the lowest mean test log loss, but all three paired exploratory
intervals now include zero. These results establish no predictive advantage for
either extension on this benchmark.

The experiment is complete and paused at research checkpoint 2. This report
extends the [original controlled comparison](gat_temporal_experiment_20261025.md)
and is tracked in the [research log](gat_temporal_research.md).

The subsequent [diagnostic report](gat_temporal_diagnostics_20261025.md) adds
draw/winner decomposition, paired logistic comparisons, internal measurements,
fixed-parameter sender checks and the matched-initial-function experiment.

## Frozen follow-up protocol

The user requested a longer run or patience of 15 epochs. The declared follow-up
uses a **maximum of 200 full-training-population optimizer updates**, with
**15 consecutive epochs without a strictly lower validation log loss** triggering
early stopping. `min_delta=0`; the lowest-validation-loss checkpoint is restored.
One update processes the complete fitting population and is called an epoch here.
No scheduler, restarts, hyperparameter search or architecture change was added.

Every run stopped through patience. None reached the maximum of 200. This is a
common stopping-rule comparison; realized update counts and computation differ.
It does not establish convergence to a global optimum.

| Component | Preserved setting |
| --- | --- |
| Historical archive | Bundesliga and Bundesliga_2, 2019/20–2024/25; 3,670 fixtures |
| Fitting targets | Bundesliga 2022/23, 306 fixtures |
| Validation targets | Bundesliga 2023/24, 306 fixtures |
| Test targets | Bundesliga 2024/25, 305 fixtures |
| Issuance | 34 round groups per population; previous-local-day information cutoffs |
| Models | `snapshot`, `time_features`, `historical`; unchanged head, loss and preprocessing |
| Encoder | Two layers, 32 dimensions, two attention heads; optional extensions disabled |
| Optimization | AdamW, learning rate 0.001, weight decay 0.001; prior penalty 0.001 |
| Seeds | 17, 42, 73 for every mode |
| Runtime | Supplied `.misc314/Scripts/python.exe`; Python 3.14.0, PyTorch 2.9.0+cu130, RTX 4070 SUPER |
| Execution | One host Torch thread; sequential fresh processes; identical native CUDA determinism policy |
| Resources | 3 GiB immutable-input cache, 1 GiB VRAM headroom, 4 GiB sampled host-RSS guard |
| Allocator | Native garbage-collection threshold 0.6, per-process memory fraction 0.8 |

The original exports contain prediction weights and preprocessing, without
AdamW optimizer state. The follow-up therefore reran every model from its
original seed. **All nine runs reproduced all 20 original training and validation
loss measurements exactly.** The prepared Parquet inputs are byte-identical;
model and shared-library source hashes are unchanged. The only changed source
in the training manifest is the experiment runner, which passes the existing
native early-stopping policy through and validates its stopping point.

The parent protocol, predictions and models remain intact. The new protocol
records their hashes and the origin of the reused disposable resource profiles.
Fitting and prediction timings below were measured again for the new fits.

## Stopping and retained checkpoints

Each cell shows **completed epoch → retained best epoch**.

| Mode | Seed 17 | Seed 42 | Seed 73 |
| --- | --- | --- | --- |
| snapshot | 51 → 36 | 41 → 26 | 43 → 28 |
| time_features | 57 → 42 | 51 → 36 | 64 → 49 |
| historical | 57 → 42 | 52 → 37 | 64 → 49 |

The temporal modes selected later checkpoints than snapshot for every seed.
Selection used validation loss only; test scores were not used for stopping or
checkpoint selection. The learning-curve visualization shows individual runs and
a mean only over epochs where all three seeds still have observations. It does
not extrapolate stopped runs or average only the surviving seeds.

## Prediction results

These are means of three separately scored models, without probability
ensembling. Lower is better. Brier is the summed three-class Brier score;
confidence ECE uses ten bins.

| Mode | Mean validation log loss | Mean test log loss | Test log-loss seed SD | Test Brier | Confidence ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| snapshot | **0.976712** | **1.057438** | 0.006497 | **0.634891** | 0.081595 |
| time_features | 0.985740 | 1.060439 | 0.003540 | 0.636849 | 0.084424 |
| historical | 0.984559 | 1.059513 | 0.004312 | 0.636055 | **0.075257** |

| Seed | Snapshot test log loss | Time-features test log loss | Historical test log loss |
| --- | ---: | ---: | ---: |
| 17 | 1.058904 | 1.064218 | 1.063683 |
| 42 | 1.050333 | 1.057199 | 1.055072 |
| 73 | 1.063077 | 1.059899 | 1.059783 |

Both temporal modes outperform snapshot for seed 73, whereas snapshot performs
best for seeds 17 and 42. Historical has slightly lower test log loss than
time_features for each seed, but the paired interval below includes zero.

### Change from the original 20-update experiment

| Mode | Original test log loss | Follow-up test log loss | Follow-up minus original |
| --- | ---: | ---: | ---: |
| snapshot | 1.060626 | 1.057438 | −0.003188 |
| time_features | 1.095034 | 1.060439 | −0.034595 |
| historical | 1.095437 | 1.059513 | −0.035925 |

The time-features gap versus snapshot decreased from +0.034408 to +0.003001;
the historical gap decreased from +0.034811 to +0.002075. The much larger deficit
in the first comparison was largely associated with its short optimization
budget. The fixed model inputs and exact initial loss trajectories support this
interpretation within these runs. They do not establish an advantage from the
temporal architecture after longer training.

### Paired uncertainty and calibration

The unchanged exploratory procedure averages each fixture's loss over the three
seeds and resamples paired circular blocks of four consecutive rounds, with
2,000 resamples and bootstrap seed 20261025. Positive differences favor the
reference model.

| Comparison | Mean test log-loss difference | Exploratory 95% interval |
| --- | ---: | --- |
| time_features − snapshot | +0.003001 | [−0.003217, +0.008602] |
| historical − snapshot | +0.002075 | [−0.004199, +0.007770] |
| historical − time_features | −0.000926 | [−0.002087, +0.000354] |

All intervals include zero. They are conditional on the fitted seeds and this
single test season, without adjustment for repeated comparisons. They establish
neither superiority nor equivalence. Historical has the lowest confidence ECE
in this follow-up, while the classwise calibration results are mixed:

| Mode | Home ECE | Draw ECE | Away ECE |
| --- | ---: | ---: | ---: |
| snapshot | 0.100649 | 0.017098 | **0.086371** |
| time_features | 0.096311 | 0.019608 | 0.091435 |
| historical | **0.095170** | **0.016646** | 0.092435 |

The unchanged rolling-goals logistic comparator again scores 1.071337 test log
loss. Its solver budget differs from the neural models. A no-opponent history
encoder remains absent, so this comparison does not establish the value of
graph propagation itself.

## Runtime and memory

Full-fit time includes initialization, input construction, optimization,
validation and best-state restoration. Prediction time covers all **305 test
fixtures in 34 issuance groups**, not one fixture. Later-epoch medians cover
updates 3 onward and include validation and native bookkeeping.

| Mode | Mean complete fit, seconds | Mean complete test prediction, seconds | Median later epoch, seconds |
| --- | ---: | ---: | ---: |
| snapshot | 20.68 | 0.231 | 0.396 |
| time_features | 25.86 | 0.400 | 0.398 |
| historical | 152.53 | 35.227 | 0.922 |

| Mode | Maximum host working set, GiB | Peak allocated VRAM, GiB | Peak reserved VRAM, GiB |
| --- | ---: | ---: | ---: |
| snapshot | 1.490 | 0.098 | 0.117 |
| time_features | 1.495 | 0.115 | 0.135 |
| historical | 1.815 | 3.101 | 9.572 |

These peaks cover the new scored fits and their test predictions. The separate
two-update profiles displayed in the interactive report are explicitly inherited
from the original unchanged model/execution configuration. They are not new
profile measurements. Historical preparation and inference remain costly, even
though repeated training epochs reuse immutable input tensors. No learned states
are cached. Desktop timings are descriptive and are not a hardware-isolated
benchmark; full fitting costs also reflect the different stopping epochs.

## Verification, artifacts and reproduction

- All 54 targeted experiment/native-control checks passed, including CPU and CUDA
  execution of an exact patience-15 boundary, reset/tie/cap checks, source-drift
  rejection and preservation of prepared parent inputs.
- All nine fit receipts passed independent stopping, best-checkpoint, fixture
  identity, model-weight hash and metric-recomputation checks.
- Every initial 20-epoch training/validation trajectory matched the original
  exactly. All source publications and frozen prepared inputs remained unchanged.
- Reloading seed 17 for each architecture reproduced the complete first and last
  test issuance groups, 18 fixtures per model, with bit-identical probabilities.

Generated evidence is in `experiments/gat_temporal_cuda_patience15_20261025/`:
`protocol.json`, `environment.json`, `frozen_source.zip`, per-fit histories,
models and predictions, `results.json`, `verification.json`, CSV summaries,
`report.html`, `report.md` and `COMPLETE.json`. The directory is separate from
the preserved original experiment and excluded from tracked source.

Using the supplied CUDA interpreter, from the repository root:

```text
python examples/compare_gat_temporal.py prepare --run experiments/NEW_RUN --from-run experiments/gat_temporal_cuda_allocator_20261025 --max-steps 200 --patience 15 --min-delta 0
python examples/compare_gat_temporal.py fit --run experiments/NEW_RUN
python examples/compare_gat_temporal.py report --run experiments/NEW_RUN
python examples/verify_gat_temporal_experiment.py experiments/NEW_RUN
python examples/render_gat_temporal_report.py experiments/NEW_RUN
```

The preparation command rejects a changed model/shared-library source or reused
output directory. The original prepared run must remain available for its
lineage and attributed profile receipts.

## Research decision

**Pause at checkpoint 2 again.** Keep snapshot as the default and cost reference.
Revise the interpretation of the first experiment: its large temporal-model
deficit mostly disappeared with longer training. The remaining predictive
differences are small and unresolved by this exploratory comparison; historical
states still carry a substantial computational cost.

This is a follow-up on an already inspected test season, under the same
retrospective schedule and July 1 membership assumptions. The patience rule
controls validation selection but does not create a fresh confirmatory test.
Do not automatically expand the cap, tune the model against these test scores,
or add retrospective mixtures or recurrent memory. A future experiment should
declare its comparison and evaluation design before further fitting.
