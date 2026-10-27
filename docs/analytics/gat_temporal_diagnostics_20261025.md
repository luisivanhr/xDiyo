# Diagnostics of the patience-15 temporal GAT experiment

The supplied recommendations were applied to the **updated patience-15
experiment**, not the original 20-update predictions. All requested diagnostic
groups completed, including two new CUDA training/validation fits from matched
initial functions. Both previous experiments and the production model source
remain unchanged.

The main findings are:

- The temporal models have slightly better draw scores than snapshot by point
  estimate, but worse conditional winner scores. None establishes useful draw
  discrimination or an improvement over the training-frequency draw baseline.
- Snapshot's point advantage over logistic has a paired interval containing zero.
- The time connections receive gradients. The measured message computations do
  not exhibit widespread severe tanh saturation. Historical sender states differ,
  but switching their lookup at fixed weights produces small probability changes.
- Matching initialization does not remove the validation deficit in seed 17.
  Learning time connections slightly improves training loss while worsening the
  best validation loss relative to the zero-time control.

**Status: complete; pause at research checkpoint 2 again.** The earlier large
20-update deficit was substantially reduced by longer training. These additional
tests help explain the remaining behavior without establishing a unique cause or
a predictive advantage for the extensions.

## Scope and provenance

Parent: `experiments/gat_temporal_cuda_patience15_20261025/`.
Diagnostics: `experiments/gat_temporal_diagnostics_20261025/`.
The copied user recommendations, their hash, source hashes, parent prediction and
model hashes, seed, stopping policy, probes and tolerances were recorded before
execution in the diagnostic `protocol.json`.

Saved test analysis covers all nine retained models, seeds 17/42/73. Internal
measurements and new matched-function training use **seed 17**. The archive still
contains 3,670 historical fixtures; **306 fixtures provide supervised fitting
targets**, with 306 validation and 305 test fixtures. Historical input matches are
not additional supervised targets. Data, cutoffs, preprocessing, model settings,
optimizer and CUDA resource policy remain those of the parent experiment.

Only two new fits were run, both using training/validation and no test evaluation.
Other new neural passes evaluate fixed weights or compute gradients without an
optimizer update. Public prediction boundaries remain intact; re-evaluation of
training/validation data is explicitly a diagnostic operation, not an as-issued
forecast claim.

## 1. Draw/not-draw scores and exact loss decomposition

For each saved model, use `q=p_draw`, `d=1{draw}`, and compute binary log loss and
binary Brier. Conditional winner probability on non-draws is
`r=p_home/(p_home+p_away)`. There are 77 draws and 228 non-draws in the test set:

`three-way log loss = draw/not-draw log loss + (228/305) × conditional winner log loss`.

The identity was independently verified for every seed; the maximum error in
the mean decomposition was **1.67e-16**. Float64 row normalization removes the
original float32 probability-sum error, at most 1.19e-7. This changes displayed
six-decimal scores negligibly. All results average separately scored seeds,
without probability ensembling.

| Model | Three-way log loss | Draw log loss | Binary draw Brier | Conditional winner log loss | Draw AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| snapshot | **1.057438** | 0.566338 | 0.189322 | **0.656954** | 0.501557 |
| time_features | 1.060439 | 0.566114 | 0.189183 | 0.661268 | 0.503057 |
| historical | 1.059513 | 0.565516 | 0.188986 | 0.660829 | 0.508753 |
| rolling logistic | 1.071337 | **0.563534** | **0.188195** | 0.679297 | 0.526828 |
| training-frequency constant | 1.101686 | 0.565166 | 0.188778 | 0.717713 | 0.500000 |
| uniform H/D/A | 1.098612 | 0.580456 | 0.195264 | 0.693147 | 0.500000 |

The training-frequency baseline uses only fitting labels: H/D/A frequencies
47.386% / 24.510% / 28.105%. The observed test frequencies are
38.689% / 25.246% / 36.066%. The constant draw forecast therefore starts close to
test draw prevalence, although it has no fixture discrimination. No test-frequency
constant was fitted or used as a baseline.

The temporal models' small draw-score gains over snapshot are offset by the
non-draw winner component:

| Difference versus snapshot | Draw component | Weighted winner component | Total |
| --- | ---: | ---: | ---: |
| time_features | −0.000224 | +0.003225 | +0.003001 |
| historical | −0.000822 | +0.002897 | +0.002075 |

Draw AUCs near 0.5 and draw Brier/log-loss point estimates worse than the
training-frequency constant do not demonstrate useful draw prediction. Logistic
has the lowest draw proper scores here, but its apparent advantage also remains
uncertain. Lower draw calibration error alone would not establish discrimination.

The unchanged paired circular four-round bootstrap uses 2,000 resamples, seed
20261025, applied to fixture losses averaged over fitted seeds. For draw log loss:

| Contrast | Difference | Exploratory 95% interval |
| --- | ---: | --- |
| time_features − snapshot | −0.000224 | [−0.003090, +0.002659] |
| historical − snapshot | −0.000822 | [−0.003773, +0.002223] |
| historical − time_features | −0.000597 | [−0.001224, +0.000002] |
| historical − training-frequency constant | +0.000350 | [−0.007695, +0.007149] |

All include zero. Full proper-score contrasts, including binary Brier and the
weighted winner component, are in `paired_diagnostics.csv`.

## 2. Paired comparisons against logistic

These use the already-saved logistic predictions and their original declared
fitting policy. No logistic retuning was performed.

| GAT minus logistic | Three-way log-loss difference | Exploratory 95% interval | Brier difference | Exploratory 95% interval |
| --- | ---: | --- | ---: | --- |
| snapshot | −0.013899 | [−0.047344, +0.024065] | −0.012531 | [−0.031779, +0.006720] |
| time_features | −0.010898 | [−0.047365, +0.029780] | −0.010573 | [−0.031293, +0.010615] |
| historical | −0.011824 | [−0.047979, +0.028509] | −0.011366 | [−0.032329, +0.009460] |

The point advantages do not establish superiority. These intervals are
conditional on one season and the fitted models, and do not cover all training,
seed, season or repeated-comparison uncertainty. Different model families can
legitimately use different declared optimizers; the missing no-opponent encoder
and limited evaluation evidence still prevent a claim about graph value itself.

## 3. Probabilities, fixture-level differences and learning curves

The updated models are not outputting uniform probabilities. Across seeds,
their mean maximum class probability is approximately 0.54, and mean entropy is
0.973–0.975 nats, versus `log(3)=1.098612` for uniform predictions. Mean draw
probabilities and within-seed variation are:

| Model | Mean P(draw) | Mean within-seed SD of P(draw) |
| --- | ---: | ---: |
| snapshot | 0.238493 | 0.032241 |
| time_features | 0.243837 | 0.029411 |
| historical | 0.244458 | 0.028293 |

Probability ranges and all requested quantiles are in
`probability_distributions.csv`. Variation in draw probabilities is present, but
the AUC measurements show little useful draw ranking in this test season.

Historical and time-feature predictions are similar, with some fixture gains and
losses cancelling. Total variation is half the sum of absolute H/D/A changes:

| Seed | Mean historical/time total variation | Largest class-probability change | Fixtures improving / worsening log loss | Changed argmax |
| --- | ---: | ---: | --- | ---: |
| 17 | 0.003903 | 0.017410 | 164 / 141 | 0 / 305 |
| 42 | 0.005549 | 0.021303 | 164 / 141 | 0 / 305 |
| 73 | 0.007025 | 0.026129 | 129 / 176 | 2 / 305 |

Their average total variation is 0.005492, about 0.55 percentage points of
probability mass per fixture. These are not exactly identical models, and
counting fixtures won does not replace averaging loss magnitudes.
`fixture_differences.parquet` records every paired fixture comparison.

All nine original training curves continue downward near their eventual stopping
points, while their last-five-epoch validation slopes are positive. Each stopped
after 15 epochs without a new validation minimum, not at the 200-epoch cap.
`learning_curve_summary.csv` records each trace. Saved training loss is pre-update;
the following seed-17 values re-evaluate each retained checkpoint after selection:

| Model | Initial training loss | Initial validation loss | Retained training loss | Retained validation loss |
| --- | ---: | ---: | ---: | ---: |
| snapshot | 1.114561 | 1.111986 | 0.986392 | 0.974998 |
| time_features | 1.104322 | 1.105805 | 0.987514 | 0.982338 |
| historical | 1.104298 | 1.105809 | 0.987366 | 0.980785 |

Mean validation probability change from initialization, measured by total
variation, is 0.2475 / 0.2144 / 0.2140 respectively. The models made substantial
learning progress. Similar retained training losses with different validation
losses are consistent with a generalization issue; they do not identify its
unique cause. The original temporal models also did not start with higher mean
loss in this seed, although initialization can affect later optimization.

## 4. Feature scales, message computations, attention and gradients

The 14 time inputs are `log1p(days)` and half-life bases `2**(-days/s)` for
`s=7,30,180`, for age and each team's observed pre-match history gap. Missing gaps
use zeros plus a separate missingness indicator. No fitted standardization is
applied to these time channels. Ordinary goal channels retain fitting-only
standardization and missingness masks.

Exact root-feature summaries cover **145,570 fitting** and **187,176 validation**
edge occurrences across all issuance groups, including repeated observations of
the same historical match at different queries. Every temporal channel's range,
mean, SD and quantiles is in `time_feature_ranges.csv`; the interactive report
also displays the complete table. Representative validation measurements:

- Age `log1p(days)`: range 0.629–7.472, median 6.717, 95th percentile 7.357.
- Each history-gap `log1p(days)`: range 0–7.027, median 2.079, 95th percentile 3.253.
- The half-life channels lie in [0,1]; the seven-day age basis underflows to zero
  for sufficiently old edges in float32. Missingness remains separately encoded.
- Gap missingness is 0.981% of fitting and 0.781% of validation edge occurrences.
  Receiver/opponent pooled marginals agree because historical matches have both
  directed perspectives; their individual edge values need not agree.

Native projection outputs and attention were captured at the first, middle and
last training and validation issuances, at initialization and the retained state.
Each layer/state contains **1,879,424 root-message coordinates**. Lower historical
dependency queries are excluded from activation summaries; their temporal feature
moments and sampled quantiles are separately recorded in
`historical_dependency_feature_ranges.csv`.

The captured quantities include `Qh_receiver`, `Kh_sender`, `E_base*x_base`,
`E_time*x_time`, their pre-LeakyReLU sum, `Vh_sender`, both `U` contributions and
the actual pre-tanh sum. Moments and saturation fractions are exact over the
declared root probes. Large-array quantiles use deterministic samples of at most
32,768 scalar coordinates per issuance; they are descriptive samples, not exact
full-array quantiles. Splitting E/U into components differs from their native
combined float32 projections by at most 9.54e-7.

At the retained seed-17 checkpoints:

| Model / layer | RMS other-edge U term | RMS time U term | RMS sender V term | Mean tanh derivative | Fraction with derivative < 0.01 |
| --- | ---: | ---: | ---: | ---: | ---: |
| snapshot / 1 | 0.384752 | 0 | 0.015633 | 0.885935 | 0% |
| snapshot / 2 | 0.387769 | 0 | 0.105344 | 0.879836 | 0% |
| time_features / 1 | 0.251446 | 0.881828 | 0.020377 | 0.622516 | 0.010908% |
| time_features / 2 | 0.229141 | 0.794494 | 0.188022 | 0.669629 | 0.000372% |
| historical / 1 | 0.251418 | 0.881640 | 0.022396 | 0.624089 | 0.012025% |
| historical / 2 | 0.228644 | 0.796764 | 0.183316 | 0.668945 | 0.000213% |

Time contributions have larger raw RMS and reduce average local tanh sensitivity,
but these measurements do not support widespread severe saturation. RMS values
are unweighted over root-edge coordinates, including old edges with small
attention, and are not output feature-importance measures.

Attention also does not collapse onto a single match. Mean effective attended
match counts `exp(entropy)` are 40.09/39.57 for snapshot's two layers,
41.41/36.99 for time_features, and 41.76/37.41 for historical. These correspond to
approximately 38–42% of each query's incoming degree. They include the model's
existing age/season log weights.

All added E/U connections have finite, nonzero full-training-loss gradient RMS
at initialization and selection. For time_features, attention/time gradient RMS
ranges from 4.43e-5–4.82e-5 initially to 2.13e-4–2.85e-4 at selection; value/time
gradient RMS changes from 1.69e-3–1.75e-3 to 2.25e-4–2.56e-4. Historical values
are similar. Full measurements are in `projection_measurements.csv` and
`gradient_measurements.csv`.

## 5. Historical lookup at fixed parameters

For each retained temporal parameter state, keep weights, preprocessing, fixture
inputs, gates and root edges fixed and switch only the network's current versus
historical sender-query mode. Evaluate all **306 validation fixtures**, without
training or a new test evaluation.

| Fixed parameter state | Mean class-probability change | Largest class-probability change | Changed argmax | Historical minus current validation loss |
| --- | ---: | ---: | ---: | ---: |
| Retained time_features | 0.002788 | 0.014499 | 1 / 306 | −0.000964 |
| Retained historical | 0.002029 | 0.009617 | 1 / 306 | −0.000528 |

For three classes, the maximum absolute class change per fixture equals total
variation. At the final layer, mean sender-state L2 distance is approximately
0.397 in both parameter states, versus mean current sender norms 1.65–1.68;
mean cosine similarity is 0.972–0.974. First-layer prior-state differences are
also nonzero, reflecting historical membership/entry context.

The sender states are not identical. Their effect on probabilities is small and
partly cancels across fixtures: the historical-parameter switch improves loss on
148 validation fixtures and worsens it on 158, with a small negative mean change.
This supports a functioning but weakly influential path in these fits. It does
not show that the mechanism is universally useless or that all effects are
caused by saturation. A contrasting toy also produces unequal predictions when
time connections are zero and only sender lookup changes.

## 6. Matched-initial-function training diagnostic

The installed PyTorch 2.9 `Linear.reset_parameters` source was saved as
`linear_initialization.txt`; its default scale depends on input dimension.
Instead of relying on equal seeds to align different architectures, both arms
use the expanded architecture, copy every common snapshot parameter and set
the appended E/U columns to zero.

- **zero_time:** time inputs are held at zero for both E and U. Prepared input
  caches are not mutated; new weights receive zero gradients and remain zero.
- **learn_time:** real time inputs are supplied and all connections can learn.

All 612 initial training/validation probabilities match **bit for bit between
the two arms**. The maximum difference from the narrow snapshot model is
5.96e-8 in probabilities and 2.98e-8 in logits, within the prespecified
`atol=1e-6, rtol=1e-5`. Common preprocessing and category maps also match. The
zero-time control's fitted predictions are numerically consistent with the
snapshot baseline; equality of initial functions alone does not generally imply
identical optimizer trajectories across different matrix shapes.

Both arms use seed 17, the unchanged optimizer, max 200, patience 15 and
`min_delta=0`. Neither reaches the cap. No test predictions were generated.

| Arm | Completed epochs | Selected epoch | Re-evaluated selected training loss | Best validation loss |
| --- | ---: | ---: | ---: | ---: |
| zero_time | 51 | 36 | 0.986392 | **0.974998** |
| learn_time | 49 | 34 | **0.985692** | 0.979803 |

Allowing time connections reduces selected training loss by 0.000700 and raises
selected validation loss by 0.004805. It helps early optimization at epoch 20
(validation 1.014745 versus 1.021612), but its best validation score is worse.
The four newly added projection blocks have positive gradient RMS at the matched
start, 0.000175/0.003238/0.000199/0.002701 for E1/U1/E2/U2; their zero-time control
gradients are exactly zero. New weights therefore can and do learn.

Matching initialization improves on the original time_features seed-17 validation
score of 0.982338, but does not remove its deficit against the zero-time control.
This single-seed pattern is consistent with limited generalization of the added
inputs. It rules out a changed starting prediction function as the sole
explanation for this diagnostic's deficit; it does not establish a universal
overfitting diagnosis or a preferred architecture on unseen data.

## 7. Costs and decision

Use the updated experiment's measurements: complete fits averaged 20.68 s for
snapshot and 152.53 s for historical; prediction over the full 305-fixture test
population averaged 0.231 s and 35.227 s. Later epoch medians were 0.396 s and
0.922 s. These imply approximately 7.4× complete fitting cost, 153× complete test
prediction cost and 2.3× later-epoch cost, with different patience-driven epoch
counts. They are desktop measurements, not universal per-fixture latency ratios.
This diagnostic did not rerun the resource benchmark.

Keep snapshot as the default and cost reference. The new evidence makes the
remaining questions more specific: draw discrimination is weak, historical sender
lookup changes outputs modestly, and learning time connections did not improve
the matched diagnostic's best validation score. Widespread tanh saturation,
zero gradients and an inactive historical code path are not supported by these
measurements. A new architecture or an automatic increase in training budget is
not justified by these diagnostics alone.

**Pause here for research review.** Any subsequent experiment should declare its
target, initialization, seeds and chronological evaluation design in advance.
The current test and validation seasons have both been inspected repeatedly;
these are exploratory findings, not fresh confirmatory evidence. No no-opponent
encoder, binary-target refit, GRU or retrospective mixture was added.

## Verification and reproduction

- Sixteen targeted tests passed, including CPU/CUDA initialization equivalence,
  exact native-logit preservation under instrumentation, immutable control inputs,
  loss decomposition, attention bounds, a contrasting historical toy and native
  matched-arm early stopping.
- Independent post-run checks recomputed all nine binary scores/decompositions,
  validated both stopping points, and verified identical arm initialization.
- Both new saved models reproduced all 306 retained validation probabilities
  bit for bit after reload. Their public historical prediction boundaries still
  reject prediction before the fitted/selection boundary.
- All frozen parent evidence hashes and the 21 completion-file hashes across
  both original experiments remain unchanged.

The diagnostic directory contains `scores.json`, per-fixture differences,
complete probe results, initial/selected train-validation probabilities, two new
models and their learning curves, detailed CSV tables, `verification.json`,
the copied recommendations, frozen source/environment, `report.html`, this
report as `report.md`, and `COMPLETE.json`.

From the repository root with the supplied CUDA interpreter:

```text
python examples/gat_temporal_diagnostics.py prepare --run experiments/NEW_DIAGNOSTICS --base experiments/gat_temporal_cuda_patience15_20261025 --recommendations PATH_TO_PASTED_TEXT
python examples/gat_temporal_diagnostics.py scores --run experiments/NEW_DIAGNOSTICS
python examples/gat_temporal_diagnostics.py probe --run experiments/NEW_DIAGNOSTICS
python examples/gat_temporal_diagnostics.py matched --run experiments/NEW_DIAGNOSTICS
python examples/verify_gat_temporal_diagnostics.py experiments/NEW_DIAGNOSTICS
python examples/render_gat_temporal_diagnostics.py experiments/NEW_DIAGNOSTICS
```
