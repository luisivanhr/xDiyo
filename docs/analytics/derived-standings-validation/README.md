# Bounded derived-standings validation

Source baseline: `0cb13cac57255f5e2f8e0eeb692e2b6f22344381`; exact implementation hashes and commands are in `validation.json` and `benchmark.json`. No fitting, protected 2026/27 reads, dataset mutations or experiment reruns.

## Arithmetic and causality

Independent from-scratch oracle: all rank, points, GF, GA, GD, played/W/D/L values agree exactly for all 10,660 saved team/window/cutoff rows across three complete seasons. A second pass delays the first match by 30 days and also agrees exactly. Synthetic tests cover further delays, independent phases, ties, large IDs, source immutability, arbitrary windows and native recipe/export preparation.

Workbook arithmetic: all 380 matches / 20 teams agree with cached totals; no macros executed. The spreadsheet name/fractional ranking key is not used.

## Provider cross-check

Release assumption: kickoff + 3 hours, raised by reviewed result bounds. Native default round issuance stays at the earliest kickoff for the original assigned round. “Boundary aligned” below means no additional result becomes eligible between issuance and the target kickoff under that assumption; actual provider observation timestamps are unavailable. These are not certified historical point-in-time snapshots.

| League / season | Provider rows | Exact | MAE | Max error | Aligned rows | Aligned exact | Aligned MAE | Aligned max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Premier_League 22_23 | 740 | 74.73% | 0.3946 | 8 | 316 | 92.41% | 0.1013 | 3 |
| Premier_League 23_24 | 740 | 63.24% | 0.5730 | 7 | 306 | 75.82% | 0.3595 | 7 |
| Bundesliga 22_23 | 594 | 73.23% | 0.3653 | 6 | 104 | 98.08% | 0.0192 | 1 |

Full comparisons and representative mismatches are retained in the CSV/JSON receipts. Of the mismatches, 518 have intervening result availability, 10 are aligned shared ties, and 90 remain unassigned official-rule/source discrepancies. No ranking parameter was tuned to improve these rates. Cold-start rows have no supplied provider comparisons in these cohorts. Awarded/revised and unsupported phase cases are guarded by dedicated tests rather than silently converted. Deductions cannot be identified reliably from a rank discrepancy alone. EPL 2023/24 did include official deductions: the [Premier League handbook](https://resources.premierleague.pulselive.com/premierleague/document/2024/12/11/e9aa1b9e-a7d5-4788-8afe-6e07b8a5f5fc/TM1603-PL_Handbook-and-Collateral-2024-25_11.12_DIGITAL.pdf) records six plus two points for Everton and four for Nottingham Forest. This is a documented season-level source of difference, not a verified per-row attribution; no adjustment is applied to the feature.

Bundesliga 2021/22 originally had 305 of 306 fixtures. The reviewed Bochum–Mönchengladbach 0–2 supplement now restores the full result schedule, with an abandoned/awarded distinction and a conservative 2022-03-24 23:00 UTC release bound. All 305 original rows and unrelated tables are preserved. Native and nested outputs use ZSTD with real updated hashes. At target kickoff (a diagnostic comparison, not a change to production round cutoffs), all 128 subsequent provider positions match; skipping the result produces 23 mismatches. Rolling-form totals/ranks also match the independent oracle. See missing-fixture-repair.json, missing-fixture-validation.json and Bundesliga_21_22_repaired_positions.csv. The other three validation cohorts remain separate and unchanged.

## Time and memory

On 380 actual EPL 2022/23 fixtures, 38 round cutoffs and five form windows [1,3,6,12,30], the native replay after canonical validation took 0.250 s versus 2.167 s for grouped pandas reconstruction (8.68x). With a delayed release: 0.293 s versus 2.155 s (7.36x). Canonical preparation was separately 1.05–1.06 s and is performed once. The grouped candidate shares that validated source. Both agree exactly on every rank. These are bounded microbenchmarks under tracemalloc, not an end-to-end training speedup.

The native replay retains totals and provenance: peak Python allocations about 4.93–4.99 MB versus 1.09–1.16 MB for the rank-only grouped candidate. This is a speed/storage tradeoff, not a claim of lower memory; it is not process RSS or GPU memory. Three repeated native lookups took 0.117–0.219 s across the bounded cohorts. Saved rows scale with requested cutoffs and windows, not every possible event snapshot.

## Verification boundaries

Final relevant suite: **289 passed, 1 skipped**, with seven existing dependency/test deprecation warnings. This includes 40 original standings tests and five real repaired-season tests. After a final help-text correction, 21 inventory/catalogue tests passed again. See `regression.xml` and `verification.json` for exact command and final source hashes.

The native CLI example prepares raw-only plus [3,6] features and a six-column match-layout recipe successfully. JSON/Python round trips, assembly and constructor/inventory compatibility are tested. The browser-dependent existing saved-rating test is skipped if Playwright is unavailable; the built-in Browser was separately used to select Derived Standings, set [3,6], enable cumulative output/normalization and verify persistence after reload. No browser console errors or warnings were observed. The Python Playwright skip is an existing saved-Bayesian-run test, not the standings interaction check. No physical-device matrix is claimed. No official head-to-head ranking, deductions, points halving, cup overtime/shootout rules, arbitrary incomplete schedules or incremental checkpoint continuation are claimed.


## Follow-up repair of the abc5b29 review

Approved standings overrides now require both scores on both paired team rows.
Partial pairs, release timestamps without approved scores, and parsed missing
ruling dates raise instead of falling back to ordinary results. All-null sparse
override fields retain ordinary scoring; valid zero scores remain valid.

The Bundesliga repair stages and validates all four payloads, flushes them, and
publishes a pending journal before replacing any live file. On interruption,
rerun the same command with `--apply`: recovery validates every live/pending hash
and completes known before-to-after transitions **before** reading the possibly
mixed manifests. Dry runs stop with recovery instructions and never mutate the
dataset. External edits, damaged/missing pending payloads, or changed evidence
stop recovery without additional replacements. Keep the journal and remaining
`.standings-tmp` payloads until recovery finishes. An OS-owned lock prevents two
instances of this repair from publishing concurrently and releases on exit.

This is recoverable roll-forward publication, not a simultaneous four-file
atomic switch. Keep readers and other writers off these files while publishing
or recovering. Process-interruption recovery is tested; universal hardware or
power-loss durability is not claimed.

Validation: **142 passed**, including all 14 incomplete score masks, fresh-process
retry after termination at each of five publication boundaries, real-file copy
interruptions, external-change guards, locking, published source hash validation,
128 subsequent provider positions, recipe/export integration, countdown and
fixture assembly regressions. The live dataset was not republished or modified.
Exact source hashes and command: [repair receipt](review-abc5b29-repairs.json).
