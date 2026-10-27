# Bounded observed gaps and opponent propagation

Completed 27 October 2026 on `codex/gat-temporal-research`.
The local evidence is in
[`experiments/gat_rest_propagation_20261027_01/summary.html`](../../experiments/gat_rest_propagation_20261027_01/summary.html).
Large experiment artifacts remain local and are intentionally excluded from Git.

The declared inner rule selects **corrected full time B in both folds**, with
small margins. The outer comparison does **not resolve an advantage from updated
opponent states** over trainable layer-zero opponent context. Both models retain
clear advantages over the declared rolling-goal logistic references. These are
previously inspected development periods, so this is a working research decision.

## What changed and what stayed fixed

The new native default, `bounded_observed_gap_v1`, measures days from forecast
issuance to the latest eligible archived match kickoff. For each fixture side,
known finite nonnegative duration `d` becomes `(d/(d+7), 0, 1[d>30])`; no eligible
observation becomes `(0, 1, 1)`. The coordinates are named `observed_gap_r7`,
`no_eligible_history`, and `no_eligible_observation_30d`. Exactly 30 days sets
the last flag to zero. All three coordinates bypass standardization, and no
duplicate masks or raw/standardized duration aliases enter the neural tensor.
Raw durations, timestamps and known-history flags remain diagnostic metadata.
This is an observed match gap, not a measure of physiological rest or inactivity.

The fixture width changes from 24 to 26 and joint decoder width from 152 to 154
for this 13-league configuration. Unrelated scaler values, the 14 bounded
edge-time coordinates, target IDs and orientations, issuance/cutoff clocks,
historical eligibility and target exclusion are unchanged. Legacy artifacts
without the new metadata explicitly retain their old transformation and
fingerprint; representative archived inner predictions reproduce exactly.

`sender_state_mode='updated'` retains the original/default propagation.
`'layer_zero'` sends the trainable prior from the current forward pass through
both attention-key K and message-value V at every layer. Receiver updates,
queries, edge/time projections, residuals, support gates, attention normalization,
own-match history and complete-component computation remain the same. The prior
is neither detached nor cached across optimizer updates. Unsupported historical
recursion or boundary-memory combinations reject clearly. Mode and schema persist
and participate in cache, fingerprint and continuation identities.

The two arms start from bit-identical untrained states within every fold/seed.
Semantic column mapping preserves unchanged encoder/decoder parameters and
equivalent missing-history columns; the four new duration/30-day columns start
at zero in both heads. Both arms have **35,993 parameters**. Goal beta is zero,
and dormant goal rates are not treated as trained forecasts.

## Execution and verification

All **12 fresh CUDA fits** completed on their first attempt: six corrected full
controls followed by six layer-zero controls. There were zero reused corrected
controls, resumed fits or failed attempts. Existing logistic/frequency models
and archived time-B predictions were reused after population/input checks.
No snapshot/C refits or additional arms were run.

The exact parent fitting/inner/outer populations were retained:
9,177/4,593/4,315 fixtures in fold 1 and 13,974/4,315/4,275 in fold 2.
Seeds are 17, 42 and 73. Training preserves two layers, the time-features model,
AdamW and native regularization, one full-population optimizer update, maximum
200 updates, patience 15, min_delta zero and earliest-best restoration.
Only raw all-13-league inner micro CE selects checkpoints. Both fold choices
were sealed before any corrected outer evaluation. Best inference weights and
last-update optimizer/RNG continuations remain separate.

Verification passed **339 distinct tests**: 35 focused checks, 213 relevant
regressions and 91 additional checks. Seven existing opt-in GPU checks were
skipped; focused sender-route and persistence checks ran on CPU and CUDA.
Two legacy-fixture assumptions in the additional batch were corrected and
rechecked, with the original failures and resolution retained. There are no
unresolved failures in these checked batches.

The tests establish one-layer output/gradient parity, identical two-layer first
states/attentions, and a deliberately nondegenerate second-layer difference.
Changing only J-K edge observations leaves A-B logits invariant in the
layer-zero control and changes the full model through J's updated state. Both
K/V routes, trainable priors, own-history and edge/time gradients are checked.
Real-data provenance, input invariance, target exclusion, mode-specific resume
rejection and corrected/legacy persistence checks pass.

An independent review recomputed **4,050 score rows and all 468 interval
point/endpoints** from saved predictions and frozen bootstrap weights. The
built-in audit checked continuations, initialization and the selection gate,
and verified **5,626 protected earlier files**. No model updates occurred during
the diagnostic review.

## CE and RPS results

Outer scores below cover 8,590 fixtures. Each seed is scored separately before
averaging; there is no probability ensemble. CE is the primary metric and RPS
is a substantive predeclared secondary metric.

| Model | Micro CE | Micro RPS | Macro CE | Macro RPS |
| --- | ---: | ---: | ---: | ---: |
| Corrected full time B | 1.011839 | 0.205801 | 1.008162 | 0.204924 |
| Corrected prior-context time B | 1.011812 | 0.205792 | 1.008097 | 0.204897 |
| Archived time B | 1.012120 | 0.205860 | 1.008356 | 0.204962 |
| Pooled logistic | 1.033829 | 0.213054 | 1.030772 | 0.212352 |
| Own-league logistic | 1.034669 | 0.212876 | 1.032010 | 0.212297 |

Micro combines folds by original fixture counts. Macro combines folds by their
original counts within each league, then equally weights all 13 leagues.
Normalized RPS uses H/D/A order and half the sum of the two squared cumulative
probability errors. Summed three-class Brier remains a separate metric.

The primary four-week paired intervals use the parent's exact 20,000 common
Monday-start Europe/Berlin resamples, seed 20261027, separate fold sampling and
fixed original league/fold mixtures. Matched seed losses are averaged before
resampling. Intervals condition on the fitted models and observed seed set.

| Combined raw contrast (first minus second) | CE difference [95% interval] | RPS difference [95% interval] |
| --- | ---: | ---: |
| Full - prior-context | +0.0000263 [-0.0002306, +0.0002934] | +0.0000099 [-0.0000673, +0.0000865] |
| Full - archived time B | -0.0002810 [-0.0013616, +0.0004306] | -0.0000583 [-0.0002833, +0.0001027] |
| Full - pooled logistic | -0.021991 [-0.026214, -0.018192] | -0.007253 [-0.008682, -0.005985] |
| Full - own-league logistic | -0.022831 [-0.027259, -0.018575] | -0.007074 [-0.008547, -0.005669] |
| Prior-context - pooled logistic | -0.022017 [-0.026194, -0.018220] | -0.007263 [-0.008679, -0.006008] |
| Prior-context - own-league logistic | -0.022857 [-0.027233, -0.018665] | -0.007084 [-0.008540, -0.005703] |

The main combined point estimates favor prior-context. Full propagation improves
CE and RPS in one of three combined seeds and 6/13 leagues. Fold 1 full-minus-
prior is +0.000083 CE / -0.000013 RPS; fold 2 is -0.000031 / +0.000033.
All these main four-week intervals span zero. Combined macro differences also
favor prior-context (+0.000065 CE / +0.000027 RPS), with intervals spanning zero.
The two/eight-week sensitivities retain that uncertainty. These findings do not
establish equivalence or rule out small beneficial effects.

The combined draw-CE difference is +0.00000047
[-0.00010721, +0.00010775]; the weighted winner component contributes the
remaining +0.00002581. There is no resolved draw or conditional-winner benefit.
The full model's raw CE decomposes into 0.575892 draw CE plus 0.435947 weighted
winner CE. Conditional winner CE among non-draws is 0.595071, summed Brier is
0.605777, draw Brier is 0.194175, draw AUC is 0.556853 and AP is 0.304374.
The complete report retains every fold/seed/league and raw/calibrated metric,
per-fixture/round differences and the established AUC/AP aggregation convention.

Both corrected models beat both logistic references on CE and RPS in every
league. Their combined micro/macro primary intervals are below zero. That
advantage is shared by prior-context and cannot be attributed to updated
opponent propagation.

The repair comparison favors corrected full on combined CE in all three seeds
and RPS in two. Fold 1 improves by 0.000792 CE / 0.000131 RPS, while fold 2
worsens by 0.000235 / 0.000015. Average draw CE worsens by 0.00010660 while
the winner component improves by 0.00038759. The intervals span zero. This
describes the complete input repair, initialization mapping and reselection;
it does not isolate bounding or establish a persistent predictive gain.

Inner-only draw calibration worsens outer CE/RPS for both corrected arms.
Full calibrated CE/RPS are 1.012762/0.205869; prior-context scores are
1.012696/0.205855. Raw evaluation remains primary.

## Selection, learning dynamics and cost

| Fold | Full mean selected inner CE | Prior-context mean selected inner CE | Declared choice |
| --- | ---: | ---: | --- |
| 1 | 1.01256303 | 1.01270011 | Full |
| 2 | 1.00371134 | 1.00382237 | Full |

Full wins two of three matched seeds in each fold. The margins of 0.00013708
and 0.00011103 are weak. The latest-fold rule supplies **corrected full time B**
as the working architecture for the next separately declared feature experiment.
The control's slightly lower outer point estimates do not change this sealed
choice retrospectively. The previous study's time-C selection remains historical.

| Arm / fold | Selected updates (17, 42, 73) | Completed updates (17, 42, 73) |
| --- | --- | --- |
| Full / 1 | 47, 41, 42 | 62, 56, 57 |
| Full / 2 | 132, 115, 87 | 147, 130, 102 |
| Prior-context / 1 | 48, 43, 43 | 63, 58, 58 |
| Prior-context / 2 | 145, 118, 90 | 160, 133, 105 |

Paired curves use only updates observed in both runs. Their difference changes
sign across time and seeds. Full reaches its selected checkpoint earlier in
all six pairs and has lower selected inner CE in four, but these modest
optimization differences do not resolve an outer benefit. No favorable common
epoch was selected afterward, and stopped trajectories are not extrapolated.

| Measurement | Full | Prior-context |
| --- | ---: | ---: |
| Completed full-population updates | 554 | 577 |
| Fitting hours | 2.092 | 2.108 |
| Mean warm update seconds | 12.486 | 11.968 |
| Maximum CUDA allocation, GiB | 1.192 | 1.187 |
| Maximum CUDA reservation, GiB | 1.229 | 1.207 |
| Maximum sampled process RSS, GiB | 1.723 | 1.725 |

Total fitting was 4.200 hours over 1,131 updates. Timing is descriptive because
execution order was fixed. The control's small measured per-update saving is
offset by its longer trajectories; total fitting costs are close. The added
propagation route has no demonstrated outer outcome benefit in this study.

## Named fixture and remaining uncertainty

For Cittadella-Reggiana, event 11425870, issuance is 2023-08-18 18:30 UTC,
cutoff 2023-08-17 22:00 UTC and the latest eligible Reggiana observation is
2021-05-10 12:00 UTC. The raw gap remains 830.270833 days, encoded as
approximately `(0.9916395, 0, 1)`; target kickoff remains 2023-08-20 18:30 UTC.
Across seeds, corrected full home probabilities are 0.383-0.422 and prior-context
0.375-0.421, versus archived time B's 0.004-0.057. The full report records all
H/D/A probabilities. This predeclared case verifies provenance and illustrates
changed forecasts; its outcome was not used to tune or judge the repair.

Retain the corrected native input semantics. Follow the declared inner choice
for the next research step while retaining prior-context as a competitive
control. The experiment leaves the incremental benefit of updated opponent
states unresolved; it does not test removing attention, own-history aggregation
or all graph structure. No deployment, extra arms or further feature experiment
is included. The authorized experiment is complete. Stop for review.
