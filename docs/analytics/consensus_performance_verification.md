# Native consensus performance repair

Date: 7 October 2026. Branch: `codex/research-quote-availability`.
Baseline: `c31f215` (includes the quote-contract correctness repairs following
the handoff's original `b2a9927` review target).

## Implementation

- Keep decision, allocation and candidate audits in a local output-metadata
  dictionary during settlement, group summaries, metrics and disclosure updates.
  Attach the complete records to returned ticket attrs only afterward. Direct
  `finalize_tickets` callers still receive audits; allocated singles also defer
  publication until metrics are ready.
- Index membership once by exact ticket ID and reuse each ordered leg frame in
  ticket batch construction, both model valuations and settlement. No leg rows
  are deduplicated or aggregated. Conflicting quote rows still reach validation.
- Keep the candidate universe, policy, arithmetic, quote guards and model
  provenance checks unchanged. Returned schemas, attrs and report exports remain
  the same. No recipe/UI configuration change is needed.

## Bounded synthetic comparison

`examples/benchmark_quote_consensus.py` constructs 3 and 6 decision/round groups,
each containing 8 distinct fixtures. Each leg has decimal odds 4; XGB supplies
draw/non-draw probabilities .30/.70 and LGBM .32/.68. Native doubles use fixed
stake 1 and the existing strict positive-EV AND gate after the fixture OR gate.
Observed quote timestamps remain missing; assumed timestamps and both models'
original quote references are explicit. Settlement is missing in both profiles.

Only the consensus call is profiled, excluding construction/imports, baseline
comparison and serialization. The final before/after pair below uses the exact
literal class probabilities in the handoff and baseline source extracted from
`git archive c31f215 src`. An earlier diagnostic pair also passed exact parity.
Both pairs used Python 3.12.6, pandas 2.2.3, NumPy 2.5.3 on Windows.

| Measurement | 84 tickets before | 84 tickets after | 168 tickets before | 168 tickets after |
| --- | ---: | ---: | ---: | ---: |
| Profiled elapsed seconds | 5.0551 | 3.8013 | 11.5215 | 7.7031 |
| deepcopy cumulative seconds | 0.9965 | 0.0113 | 3.4455 | 0.0241 |
| deepcopy calls, including recursive | 1,044,252 | 8,043 | 3,684,933 | 16,077 |
| Quote validation calls | 360 | 360 | 714 | 714 |
| Quote validation cumulative seconds | 2.0014 | 1.9789 | 4.0510 | 4.0796 |

These samples include profiler overhead and are not a real-study profile, memory
benchmark or completion estimate. The remaining quote validation cost is retained.

All output frames match the baseline exactly (`assert_frame_equal` with
`check_exact=True`, plus recursive complete attrs comparison). This covers full
candidate and selected identities, ordered memberships, both models' votes/EVs,
odds, stakes, settlement/payout/profit, metrics, fixture audits and ticket audit
payloads. Seven saved cases cover both benchmark sizes, mixed synthetic
win/loss/push/void settlements, all-OR-rejected input, zero rows, one-row
undersized input and model/fixture/class permutations.

Local before/after profiles, outputs and comparison results are in
`%TEMP%/xdiyo-consensus-performance/{literal-before,literal-after}`. The checked-in harness makes
the bounded comparison reproducible; its pickle comparison inputs must be
trusted files produced locally by that harness.

## Regression checks

- **393 passed**, 7 dependency warnings, 31.74s: all `test_composition_*`, quote,
  consensus performance, bet ticket, ticket EV, stake/bankroll and composition UI
  checks. Includes the isolated installed-wheel example and saved-model
  restoration tests. Normal environment: Python 3.14.0 / pandas 2.3.3.
- **112 passed** on pandas 2.2.3 (6.72s, 137 dependency deprecation warnings).
- **112 passed** on pandas 3.0.6 (8.53s).
- New structural tests forbid publishing large audits during internal pandas
  finalization, check one membership-index build across 28 doubles without
  per-ticket equality scans, preserve all duplicate/conflicting leg rows and
  their order, and verify direct finalization retains its audit contract.
- Existing strict observed/research timing, missing streams, per-leg identity,
  future/contradictory evidence, outcome mutation, ordering invariance, empty
  schemas, pre-group timing rejection and Parquet/report reuse tests pass.

Focused-suite log: `%TEMP%/xdiyo-consensus-performance/focused.log`.
No full-library suite or new browser session was run for this bounded repair.

## Study verification still outstanding

The handoff references 24 saved-model/legacy-control study replays without
providing their artifacts or a local path. Those exact replays have **not** been
rerun here; the repository's saved-model tests are not a substitute for them.
This patch therefore does not certify readiness to resume the full study.

No cached data/models, datasets, frozen experiments or study outputs were
replaced, and no real-data study or model refit was launched. The parent
composition branch and main are unchanged. Publication is on the separate
quote-availability branch under the user's explicit push instruction; no merge.
