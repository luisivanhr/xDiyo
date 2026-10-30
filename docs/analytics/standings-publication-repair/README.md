# DerivedStandings publication repair

The single current policy is `independent_3_1_0_available_results_v2`.
Every requested cutoff publishes the actual eligible sporting totals. A postponed
or permanently unresolved opening fixture cannot suppress the rest of the table.
Idle teams retain zero totals but participate in normal competition ranking.
Initial raw ranks are 1, normalized form is 1, and normalized raw position is 1/N.
No new switch, threshold, rank formula or rolling operator is introduced.

Eligibility remains source kickoff < cutoff and result availability <= cutoff.
Unknown result publication times are not inferred from kickoff by default. Known
result release times and explicit approved standings release bounds are supported;
explicit caller-provided retrospective proxies remain assumptions, not observations.
Venue filtering precedes each independent last-w window. Late releases enter
kickoff/event order only at eligible queries; earlier snapshots are unchanged.

## Four-season audit

Input root: `experiments/gat_phase_form_rank_20261027_01` in the user's primary
checkout. The audit only reads saved native sources, exact feature-bank cutoffs,
snapshot sets, fold memberships and model graph manifests. It does not load model
weights, retrain, overwrite artifacts or access the active collector.

The frozen baseline is commit `4266575f62c899f3c6b3c7ca95a76fd633f93ece`.
All four original snapshot tables and source hashes were reproduced exactly.
Rows below are team x cutoff x window, with cumulative standings and windows
1, 5, 10 and 20. Rank and total change counts overlap; they must not be added.

| Season | Snapshot rows | Changed rank rows | Changed total rows | Changed cutoffs |
| --- | ---: | ---: | ---: | ---: |
| La Liga 2020/21 | 3,800 | 2,179 | 2,250 | 23 |
| Premier League 2020/21 | 3,800 | 1,701 | 1,780 | 18 |
| Eredivisie 2020/21 | 3,060 | 1,098 | 1,160 | 13 |
| Serie B 2023/24 | 3,800 | 847 | 860 | 9 |

These isolate the gate repair using the saved experiment's explicitly supplied
legacy release proxy. They are not a claim of verified historical publication
times. A separate explicitly assumed kickoff+3h replay, respecting reviewed lower
bounds, changes another 2, 21, 0 and 0 rank rows respectively. This sensitivity is
availability uncertainty, separate from the suppression defect. Neither proxy is
installed as the new default.

An independent brute-force per-team selector, totals calculator and count-greater
rank oracle checked **28,920 rows**, covering both replay assumptions. It does not
call the production replay or ranking helper. The sample CSV contains full tables
at each season's first and last changed cutoffs, cumulative and window 5.
For example, at La Liga's 2020-09-18 22:00 UTC cutoff seven results are eligible
under the saved assumption, while three original openers remain outstanding.
Osasuna's 2-0 win gives three points and rank 2; Real Betis's 1-0 win gives rank 4;
unplayed Barcelona has zero points and rank 11; Athletic Club's 0-2 loss gives
rank 19. The old gate assigned all of them rank 1 and zero totals.

Evidence:

- `season-summary.csv`: aggregate differences and original-source reproduction.
- `cutoff-evidence.csv`: exact cutoff, eligible event identities, outstanding old
  openers, latest source kickoff and assumed release for every publication.
- `changed-snapshots.csv` and `independently-checked-sample-tables.csv`: arithmetic
  and ranks; `changed-bank-coordinates.csv`: 4,657 changed saved rank coordinates.
- `audit-receipt.json`: commands, source/input/output hashes and confirmation that
  all opened experiment files remained unchanged.

## Affected artifacts and migration

The manifest inventory inspects **156 models**. **120 have actual rank inputs**:
CD_DW (24), D1 (24), D10 (12), D20 (12), D5 (12), DE1 (12), E1 (24).
The other **36 are controls without direct rank inputs**, N (24) and CD (12).
Classification uses recorded fixture/edge columns, not route-name guessing.
Preserve controls with their original bank/source binding; do not claim their
weights need retraining solely because of this repair.

The 68 legacy standings sets, shared `phase_context.json`, `phase_context.parquet`
and `rank_diagnostics.parquet` require rebuilding before current-policy use.
Regenerate dependent per-model feature copies and preprocessing with a new source
identity, then retrain the 120 rank-dependent models before corrected comparisons.
Recompute downstream calibration, window selection, predictions and reports that
depend on those fits. **None of this retraining was performed by the repair.**

For the four cited seasons, changed fixtures intersect fold_1_fit and fold_2_fit
in 518 fixtures for every audited window. Fold_1_outer and fold_2_inner each have
88 changed fixtures for window 1 and 89 for windows 5/10/20. The cited seasons
produce zero intersections in fold_1_inner and fold_2_outer. These are lower bounds
for the full experiment: other seasons' numerical changes were not enumerated.
See `model-inventory.csv`, `feature-artifact-inventory.csv`, and `fold-impact.csv`.

New saves use artifact version 3 and the new policy in source/state identity.
Publication and feature audits use `publication_rule` and eligible-result counts,
without `gate_open` or pending-first-round semantics. Native execution identity
also incorporates the numerical standings source. Legacy version 1/2 files load
for inspection and can be copied with their original policy and suppressed totals,
but current feature lookup rejects them with a rebuild instruction. Merely
relabeling legacy snapshots as version 3 is rejected. Do not patch old totals or
hashes in place. Source data, ticket timing and SeasonCountdown were not changed.

## Verification and reproduction

- Python 3.14 / pandas 2.3.3: **323 passed**, zero skips/failures, broader regression
  suite including feature assembly, UI/recipe compatibility, recovery and shared
  temporal-provenance tests.
- Python 3.12 / pandas 3.0.1: **140 passed**, zero skips/failures, focused standings,
  publication, serialization, missing-fixture and venue-history regressions.
- Coverage includes delayed/unresolved opening results, empty seasons, idle teams,
  ties, arbitrary windows and short history, all venue/role modes, season resets,
  equal cutoffs, resumed games, late releases, future perturbation/truncation,
  target exclusion, exact save/reload and explicit legacy rejection.

`verify.py` runs an isolated committed-source tree plus the scoped repair. It
excludes unrelated local reporter edits, including from the shared UI inventory.
Its receipts record tested source/runtime hashes, package versions, commands and
XML/log hashes. No source changed during either test run.

From the repository root in PowerShell:

```powershell
& 'C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe' docs/analytics/standings-publication-repair/verify.py broader
& 'C:/Users/luisi/Documents/Programming/Python/xDiyo/.venv/Scripts/python.exe' docs/analytics/standings-publication-repair/verify.py focused
$env:PYTHONPATH = 'src'
& 'C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe' docs/analytics/standings-publication-repair/audit.py --experiment-root 'C:/Users/luisi/Documents/Programming/Python/xDiyo/experiments/gat_phase_form_rank_20261027_01' --output docs/analytics/standings-publication-repair
```

The frozen baseline source is obtained by its exact commit in the audit script.
Audit and verification scripts/evidence preserve their byte representation in Git;
source files may have Git's normal CRLF/LF conversion. The publication receipt
records staged source hashes and confirms equality after line-ending normalization.
