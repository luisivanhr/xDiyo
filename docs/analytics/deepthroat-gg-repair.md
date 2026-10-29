# Deepthroat-gg review repairs

Review target: `104c8c9b469550c13639a9f56e6dc301c5988d6a`.
Scope: `codex/deepthroat-gg-research`; no main merge or real-data experiment.

## Corrections

1. Fixture identities now require one identical UTC prediction cutoff across
   both perspectives, before evaluating features. A regression recreates the
   January 7 away-mean difference (2.5 versus 3.5) and rejects mixed cutoffs.
   Other tests cover a one-nanosecond mismatch, validation before evaluation,
   and equivalent instants expressed in different timezones.
2. Neighbor tuning now fits each leave-one-out population, in original row
   order, and predicts at every actual candidate k. Each population's two
   KD trees are reused across candidates. Independent sklearn estimators are
   refitted for every (population, k) in the test oracle. Every candidate's
   RMSE and RPS, selected k and objective ties agree exactly. Cases include
   duplicate vectors, equal nonzero distances, larger trees with duplicates,
   untied inputs, missing classes and reversed input order. The final models
   use the same sklearn KD-tree, Euclidean, uniform-weight policy; native
   persistence and fixed H/D/A probability columns are unchanged. New study
   manifests explicitly identify the repaired neighbor policy.
3. Outcomes are calculated only for finite complete score pairs. Missing
   nullable `Int64` and `Float64` pairs, including partially missing pairs,
   produce the same features, targets and metadata as float-NaN inputs.
   Incomplete results enter neither team's history. Inputs remain unchanged.

## Verification

All 18 added cases pass in both installed environments; the research module
has 32 passing cases in each. The same broader 185-test research/model/UI/
standings/label suite ran in both environments:

| Environment | Result |
| --- | --- |
| Python 3.12.14, pandas 3.0.1, sklearn 1.9.1 | 183 passed; 2 inherited MatchScore failures |
| Python 3.14.0, pandas 2.3.3, sklearn 1.9.0 | 184 passed; 1 inherited MatchScore failure |

These are the two existing local environments, not the reviewer's exact
pandas 2.2.3 / sklearn 1.8.0 installations. The frame-shape assertion predates
additional half-score columns and fails in both environments. The other
failure assigns a nanosecond timestamp into a pandas 3 microsecond fixture
column. Both were previously reproduced on the exact base `d38ae319`;
the pandas 2 frame-shape failure was additionally reproduced during this
repair. No assertion was skipped or weakened. Original receipts remain
historical evidence; fresh `deepthroat-gg-repair-*` receipts record these checks.
Generic KNN controls and exports were covered by the UI regression tests;
the browser itself was not rechecked for these numerical-only repairs.

## Synthetic cost

The unchanged bounded runner exercised 144 synthetic fixtures, all 92 n
candidates and all 348 k candidates (infeasible k are recorded, not evaluated),
87 development fixtures and 38 held-out fixtures. No production data was used.

- Preparation: 2.97 s.
- Corrected selection, fitting and holdout scoring: 4.81 s.
- Persistence: 0.18 s; total: 7.95 s.
- Sampled peak RSS: 183.9 MiB, 13.7 MiB above imported-runtime baseline.
- Selected n=30, goal k=72, outcome k=51 on this synthetic dataset.

The historical invalid prefix implementation took 0.16 s for selection/fitting/
scoring and 3.53 s overall. Correct selected-k evaluation is more expensive;
the original speed claim does not apply to the repair. Selected parameters
happen to remain unchanged on this synthetic dataset, but tied inputs can
change selection. This benchmark is not evidence of predictive performance
or a real-population runtime bound. Memory is sampled every 10 ms and may miss
instantaneous peaks. Exact commands, dependency versions, source hashes and
test/benchmark receipt hashes are in `deepthroat-gg-repair-verification.json`.
