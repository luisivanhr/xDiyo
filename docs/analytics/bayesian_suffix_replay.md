# Completion-aware Bayesian replay: suffix checkpoints

The longer-calibration preflight at 8207e29 identified a runtime blocker:
46,774 eligible matches generated 16,030 knowledge boundaries, and the global
prefix strategy repeated 379,446,500 match assimilations per objective.
This repair preserves the knowledge-time versions and chronological equations.

## Replay strategy

The fixed plan still validates the paired events, availability, movement records
and anchors once per fit. Each version now identifies its earliest changed
chronological leaf. Numerical replay keeps a current state and sparse rollback
checkpoints, one per 128 chronological positions. If the new observations append
in order, replay continues from the current state. Otherwise it restores the last
checkpoint strictly before the earliest changed position, drops stale later
checkpoints, and replays the complete affected suffix.

The suffix is global: it preserves shared home-advantage updates, cross-league
bridges and simultaneous season-entry dependencies. It never assumes that only
the two teams in a delayed match are affected. Equal-boundary pre-kickoff and
post-kickoff versions remain distinct. Previously published forecasts are retained.

State copies contain immutable Gamma/TeamState values with copied dictionaries,
counters and sets. They are local to a single parameter evaluation. A new trial
always starts empty; no fitted state or optimizer state is reused. The reusable
plan is not mutated. Public native runs use the same suffix mechanism and retain
snapshot ordering, timestamp precision, diagnostics and checkpoint serialization.
Checkpoint storage trades memory for replay work; adversarial continual late
insertion can still require long suffixes. This is not a universal linear-time
claim or a persisted optimizer-resume facility.

The VB/EM kernels, simultaneous-match equations, convergence checks, probability
summation, priors, nine-coordinate L-BFGS-B setup, bounds, maxiter=100, ftol=1e-8
and regularization=0.01 are unchanged. The final native objective verification
remains in `train_bayesian`.

## Measured supplied ten-season population

These are local fixed-parameter measurements at the original pre-fit parameters,
not a calibration run or optimizer result:

| Operation | Time |
| --- | ---: |
| Reusable plan preparation | 75.89 s |
| One objective evaluation | 12.14 s |
| Complete native run, including preparation and prediction output | 178.75 s |
| Independent final chronological state replay | 5.02 s |

The objective and native run each assimilated **97,055 matches**, versus the
original strategy's **379,446,500**: approximately **3,910 times less numerical
replay work**. This is an operation-count ratio, not a measured wall-time speedup.
The old complete ten-season objective was not run. The preflight's ten-hour
estimate was an extrapolation on another runtime, not a directly comparable
measurement. Preparation still performs the existing pandas validation and is
paid once per fit, rather than once per trial.

The objective was 1.0156578991331813. Every forecast, class probability and
score-log probability matched the complete native run exactly. Final team and
league posteriors matched a fresh full chronological replay without suffix
checkpoints. Input hashes, parameters, configuration and precise timings are in
`data/bayesian_suffix_benchmark_20261008.json`.

Reproduce without launching optimization:

```powershell
$env:PYTHONPATH = 'src;.'
python examples/benchmark_bayesian_suffix_replay.py `
  'C:/Users/luisi/Downloads/Longer calibration preflight evidence 8207e29.zip' `
  'benchmark.json'
```

The script reads parquet and JSON members only; it does not extract or execute
archive scripts. Keep the supplied archive to reproduce the full measurement.

## Regression evidence and scope

The frozen test-only global-prefix publisher from 6f9c2fa supplies the reference.
Exact comparisons cover snapshots, league posteriors, predictions, native payloads
and diagnostic totals, mirrored/bridge transitions, multiple nondefault parameter
bundles, both signs of all nine finite-difference directions, simultaneous matches,
late releases, checkpoint strides 1 and 128, multi-block rollbacks, repeated trials
and native save/load. An 86-match source-hashed fixture covers both actual resumed
2011 matches in La Liga and Serie B. Existing final-fit objective verification,
checkpoint continuation and causal-cutoff tests remain in the regression suite.

Final verification: **150 tests passed on pandas 2.2** across the Bayesian/history
suite; **23 suffix-equivalence tests passed on pandas 3** after preserving the
original timestamp precision.

No research calibration, XGB refit, feature-bank rebuild, betting run, dataset
rewrite or change to a frozen experimental specification was performed. The
research owner can now rerun preflight on the published revision before launching
the separately coordinated optimizer.
