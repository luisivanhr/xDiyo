# Auxiliary goal supervision: seven-fit snapshot experiment

The attached head passes the declared **preliminary** outcome screen: mean
validation log loss decreases from **0.976712 to 0.976091**, a change of
**-0.000622**, with improvements in **two of three** matched seeds. This is a
small exploratory gain on an already inspected validation season. Keep the
feature disabled by default and pause for review.

The complete [HTML report](../../experiments/gat_goal_auxiliary_cuda_20261026/report.html)
and [Markdown report](../../experiments/gat_goal_auxiliary_cuda_20261026/report.md)
contain every retained training/validation score, rate distribution, gradient
measurement, resource receipt and curve. The [research log](gat_temporal_research.md#parallel-auxiliary-goal-supervision-2026-10-26)
records the implementation and frozen protocol.

## Controlled comparison

All arms used the two-layer snapshot model, identical 306 fitting and 306
validation fixtures, raw goals, original preprocessing, inputs, AdamW and prior
regularization. Both main arms contain the same linear goal head. Its two outputs
become `softplus(z)+1e-6` rates. The existing outcome head retains its inputs and
three-class decoder. The goal head adds 284 parameters; all arms have 28,383
parameters, including 18,816 shared encoder parameters.

Within each seed, encoder, outcome-head and goal-head parameters were explicitly
copied from saved initial states. All 612 initial probabilities and rates matched
exactly between arms, and initial logits matched the original snapshot model.
No test predictions were generated. The protocol allowed exactly seven fits:
control beta=0 and attached beta=0.1 at seeds 17/42/73, plus detached beta=0.1 at
seed 17. Each used a 200-update maximum, patience 15, `min_delta=0`, and restoration
of the checkpoint with lowest validation **outcome** log loss.

| Seed | Control completed / selected | Attached completed / selected | Control validation LL | Attached validation LL | Attached - control |
| --- | --- | --- | --- | --- | --- |
| 17 | 51 / 36 | 51 / 36 | 0.974998 | 0.973204 | -0.001793 |
| 42 | 41 / 26 | 45 / 30 | 0.986002 | 0.984941 | -0.001061 |
| 73 | 43 / 28 | 49 / 34 | 0.969137 | 0.970127 | +0.000990 |

The detached diagnostic completed 51 updates and selected update 36. No arm
reached the maximum. Separate seed scores were averaged; no ensemble was scored.

## What improved, and what did not?

The small outcome gain comes from the draw component. Conditional winner
prediction deteriorates in all three seeds. Draw log loss improves in seeds 17
and 42 and slightly worsens in seed 73; draw Brier improves in all three.

| Validation metric, averaged over seeds | Control | Attached | Difference |
| --- | --- | --- | --- |
| Three-class log loss, primary | 0.976712 | 0.976091 | -0.000622 |
| Summed three-class Brier | 0.583680 | 0.583362 | -0.000319 |
| Draw/not-draw log loss | 0.576890 | 0.574672 | -0.002218 |
| Binary draw Brier | 0.194330 | 0.193472 | -0.000858 |
| Conditional home/away log loss among non-draws | 0.543759 | 0.545929 | +0.002171 |

There are 225 non-draws among the 306 validation fixtures. Thus the exact
decomposition is `three-class LL = draw LL + (225/306) * conditional winner LL`.
The mean draw improvement of -0.002218 is partly offset by a +0.001596 change in
the weighted winner contribution. Maximum per-fixture decomposition error across
retained predictions is below 9e-16.

The attached head also beats the fitting-only constant-rate comparator on full
marginal Poisson NLL in every seed. That is a separate result from outcome success.

| Validation goal metric | Fitting-only constants | Attached, mean of three seeds | Detached, seed 17 |
| --- | --- | --- | --- |
| Full summed marginal Poisson NLL | 3.214171 | 3.129441 | 3.320045 |
| Home-goal MAE | 1.121898 | 1.126043 | 1.122696 |
| Away-goal MAE | 0.974476 | 0.912292 | 0.886527 |
| Mean home rate | 1.856209 | 2.026188 | 1.132927 |
| Mean away rate | 1.316993 | 1.401169 | 1.030705 |

Observed validation means are 1.807190 home and 1.411765 away. Attached home rates
therefore overpredict on average, and home MAE is slightly worse than the constant
comparator. Attached rates vary by fixture: across seeds, validation home rates
range from 1.112833 to 3.413003 and away rates from 0.647485 to 2.162657. The full
report includes per-arm/seed quantiles. There is no observed collapse to a single
rate or numerical extreme. The beta=0 goal head is untrained and is not used as a
meaningful goal-forecasting baseline.

## What does the detached control establish?

Its full observed training/validation outcome-loss trajectory, regularization,
selected probabilities and all 21 selected common parameter tensors match the
seed-17 control **bit for bit**. Its goal gradients on shared encoder parameters
are zero. Native AdamW has per-parameter moments; there is no global gradient
clipping or shared scaling that could couple the goal-only update into the
encoder. Beta=0 omits the auxiliary graph from backward entirely, so unused head
parameters receive neither optimizer state nor decoupled decay.

The detached head learns and improves away MAE over constant rates, but does not
beat the constant marginal NLL at the outcome-selected checkpoint. Its mean rates
underpredict both sides. This experiment therefore does not establish strong
marginal Poisson forecasting from that readout under the declared training budget.
It also does not establish that goal information is absent from the baseline
representation. The head's goal-loss curve was still declining when outcome-based
selection and stopping took effect. No goal-selected checkpoint was substituted.

## Gradient scale and optimization

The diagnostics use the full fitting population and the 18,816 shared encoder
parameters, excluding both heads and regularization. Parameters, optimizer state
and pre-existing gradient buffers were verified unchanged after each measurement.

| Seed | Initial beta-weighted norm ratio | Initial cosine | Selected norm ratio | Selected cosine |
| --- | --- | --- | --- | --- |
| 17 | 0.849467 | -0.217377 | 1.132059 | -0.460097 |
| 42 | 1.288516 | -0.023050 | 1.760237 | -0.096903 |
| 73 | 1.501030 | +0.300959 | 0.349781 | -0.517970 |

Beta=0.1 does not make the auxiliary encoder gradient negligible: its weighted
norm is comparable to, and sometimes larger than, the outcome gradient. All three
selected cosines are negative. These local measurements neither prove harmful
transfer nor establish generalization; AdamW further transforms the gradients.
Detached norms are zero, and their cosines are reported as unavailable.

The attached model was **not consistently faster to optimize outcome loss**.
Across the complete common-epoch intersections, attached validation loss is lower
at 18/51, 15/41 and 5/43 updates for seeds 17, 42 and 73 respectively. Its mean
common-epoch differences are +0.001960, +0.000864 and +0.004994. The primary
comparison still uses each arm's own selected checkpoint: seeds 17 and 42 achieve
slightly better observed minima, while seed 73 achieves a worse minimum despite
selecting later. After their selected minima, validation deterioration triggers
the declared patience rule. No stopped suffix is extrapolated and no favorable
common epoch is chosen retrospectively.

![Matched outcome curves](../../experiments/gat_goal_auxiliary_cuda_20261026/outcome_curves.png)

## Cost, verification and stopping point

Mean measured fit time is 29.571 seconds for control and 31.336 seconds for
attached, about **6.0% more**. These times include initialization verification,
initial gradient diagnostics, training, validation and post-update training
curves. They include different completed update counts, so they are not isolated
estimates of the head's per-update cost. Selected-checkpoint diagnostics and
end-to-end timings are retained separately. Sampled worker RSS stays around
1.55 GiB; peak CUDA allocated memory is about 0.10 GiB (0.11 GiB for detached),
within the preserved resource policy.

The result is a candidate for a separately declared chronological evaluation;
its very small, seed-dependent outcome gain does not justify enabling it by
default. In particular, the improvement trades better mean draw scoring against
worse conditional winner scoring.

Verification completed:

- 108 targeted tests passed, including CPU/CUDA and synthetic worker checks.
- All seven declared fits completed, restored their independently verified best
  outcome-validation checkpoints and reloaded with identical probabilities/rates.
- All three controls reproduce the previous snapshot validation curves and
  selected parameters bit for bit.
- Independent retained-score, label-alignment, loss-decomposition and stopping
  audits passed. Public prediction boundaries still reject in-sample requests.
- All 229 previous artifact files retained their recorded hashes. The new
  completion manifest's 94 evidence files passed SHA-256 verification.
- No goal-experiment workers remain active; there are no new test predictions.

An inherited exact-CUDA-reload test fixture had determinism disabled. Its roughly
1e-8 discrepancies were resolved by explicitly selecting the existing deterministic
option in the fixture, without changing production arithmetic or tolerances. No
unresolved targeted-check failure remains.

Detailed artifacts include `summary.csv`, `paired_metrics.csv`,
`paired_fixture_losses.parquet`, `paired_round_losses.csv`,
`paired_common_epoch_curves.csv`, `gradients.csv`, `goal_rate_distributions.csv`,
the separate outcome/goal plots, frozen source/protocol, and seven model exports
under `experiments/gat_goal_auxiliary_cuda_20261026/`.

**Completed and paused for review.** The optional feature remains disabled by
default. No additional weight, architecture, population, training-budget or
decoder experiment was started.
