# Temporal-provenance verification

The candidate is based on main `d38ae3192067f73df26f1c2f5baa8f1d148d215a`.
Verification runs on an isolated copy of that tracked source plus the owned patch.
Concurrent attention-reporter edits are excluded and remain untouched.

## Final receipts

Results are recorded in the adjacent `*-receipt.json`, `.log`, and `.xml` files.
Each JSON includes the exact command, Python and package versions, base commit,
owned source/test hashes, complete isolated runtime source hashes, exit status,
source-stability check, and log/XML digests. Source hashes identify the tested
patch without circularly embedding this evidence commit's own hash.
These are raw tested Windows file hashes (including their line endings).
`publication-checks.json` additionally records staged Git-blob hashes and verifies
that any source-byte differences are solely CRLF/LF checkout normalization.
The evidence directory is exempt from Git text conversion so receipt and script
bytes survive publication exactly.

| Suite | Python / pandas | Passed | Elapsed |
| --- | --- | ---: | ---: |
| `broader-1-py312` | 3.12.14 / 3.0.1 | 497 | 289.10 s |
| `broader-2-py312` | 3.12.14 / 3.0.1 | 403 | 133.27 s |
| `broader-3-py312` | 3.12.14 / 3.0.1 | 353 | 277.14 s |
| `focused-py314` | 3.14.0 / 2.3.3 | 242 | 43.06 s |
| `browser-py314` | 3.14.0 / 2.3.3 | 4 | 25.50 s |

The three disjoint broader partitions passed **1,253 tests** in total; no test
module was omitted. The second environment passed **242 tests** and the browser
suite passed **4 tests**. All five runs had zero failures, errors or skips, and
identical source/runtime hashes. Receipt log/XML digests and the final working
source were independently checked after the runs. Existing warnings concern a
rank-deficient synthetic statsmodels fit and sklearn `n_jobs` deprecation.

The broader run covers betting decisions, singles/parlays/systems, flat/Kelly
allocation, guarded bulk/reference parity, narrow numeric and composite identity
handling, quote contracts, report rendering, execution/report recovery, and
experiment ranking. The focused run repeats the new temporal cases and nearby
betting/reporting/performance regressions in a second environment. Browser tests
exercise synthetic recipe save/export/round trips and explicitly prohibit
training, preparation or prediction jobs.

## Contract coverage

- Discarded candidates and shared decision groups are validated before selection;
  later information cannot influence an earlier grouping/placement decision.
- Postponed and resumed first starts, strict kickoff inequality, availability
  equality, missing timestamps, UTC offsets, and naive legacy limitations.
- Exact forecast/feature/source/artifact identities, delayed training inputs,
  train-only fitted stages, unsupported content, and physical versus simulated
  creation/availability.
- Conditional retrospective recomputation, bounded quote assumptions, and
  missing evidence distinguished from contradictory evidence.
- Row-order invariance, historical-source perturbation/truncation, scalar/bulk
  rejection parity, template/membership substitution and later file mutation.
- Native persistence and report export revalidation; edited certificates,
  dependencies, metric values, placement amounts and ledger events are rejected.
- Conservative latest-leg settlement, deterministic simultaneous events and
  cash plus reserves equals wealth. Unknown settlement evidence cannot become
  verified through reload.
- Audit evidence is not exposed to staking/selector callbacks, and a probability
  appended after certification cannot become an input to native Kelly staking.

## Compatibility and limitations

Payoff multiplication, existing settlement arithmetic, reserve accounting,
strategy grouping and SeasonCountdown are unchanged. There was no retraining of
real-data models, experiment rerun, dataset modification, merge, deployment, or
collector intervention. Tiny synthetic fits inside existing integration tests
remain part of their original test fixtures.

Two existing tests were aligned with the new mandatory contract: placement must
precede kickoff, and a retained ticket must be consumed with its original model
policy. Unique quote evidence now receives mandatory preflight checks in addition
to the existing single bulk membership check. Existing numerical parity checks
retain their exact comparisons.

This is a bounded regression suite, not the entire repository suite or a proof
of a vendor's historical publication times. Certificates check content and
consistency; producers must supply complete truthful dependency inventories.
Unknown model availability, first starts, schedule revisions, quote publication,
and unsupported learned-policy lineage remain unverified. Explicit bounded
assumptions remain conditional research. No existing experiment is declared
leaky solely because its legs are months apart.

See [README.md](README.md) for the producer contract and legacy migration.
