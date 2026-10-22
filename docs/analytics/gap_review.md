# GAP critical review

Reviewed on 2026-10-23 against `gap-implementation-prompt.md` and the supplied
Wheatcroft (2021) PDF. The reviewed implementation was
`2216363619a002a17d2e92784a203cc18741518e`, on `codex/wheatcroft-gap`, based on
`0cb13cac57255f5e2f8e0eeb692e2b6f22344381`. Fixes remain in the isolated worktree
`C:\Users\luisi\.codex\worktrees\wheatcroft-gap\xDiyo`.

## Confirmed findings and repairs

The first seven adversarial regression tests failed against the original
implementation before their corresponding fixes. All now pass.

| Priority | Finding | Repair and regression evidence |
| --- | --- | --- |
| P1 | Omitting feature cutoffs silently selected each kickoff. A later fixture in a frozen production round could consume an earlier fixture's result. | Explicit cutoffs are now mandatory through direct features, fixture features, diagnostics, `Rating`, `GAPFixture`, and the fitting objective. The synthetic same-batch case remains frozen. |
| P1 | Fixture perspectives could select different information sets; declared production batches were ignored. | Require identical cutoff/kickoff and reciprocal teams for paired rows. If `prediction_batch_id` is supplied, both perspectives must name the same actual scheduler batch and the batch has one cutoff. Half-missing paired cutoffs fail; wholly missing cutoffs yield unavailable features. |
| P1 | A fit date and arbitrary next-season date did not prove that training seasons had ended or identify the held-out seasons. A January-only sample from an ongoing season could reach optimization. | `tune_gap` requires explicit closed training seasons and disjoint `target_seasons`. `target_start` must equal the earliest target entry. Unknown training season identities, observations contradicting closure, post-fit releases, and held-out observations even behind a false objective mask fail before optimization. Target predecessor/donor evidence is validated too. |
| P2 | A postponed fixture from the previous season was rejected after the vintage's calendar expiry, despite the season requiring its original parameters. | Select each vintage by season entry and retain it for the whole season. Continue producing that season's snapshots through the declared data cutoff. A postponed fixture after the next season starts retains its original numerical prediction, including after native reload. |
| P2 | Future season plans could require unavailable future predecessor metadata when constructing past states. | Exclude season entries beyond `data_cutoff` and unselected statistic plans from the artifact. A future plan with an unknown predecessor no longer blocks an earlier build. |
| P2 | The transition-context adapter inferred a retained predecessor despite explicitly unknown predecessor evidence. | Reject missing predecessor identities and inconsistent retained-team provenance instead of replacing them with inferred values. |

Additional validation rejects inconsistent season entries across statistics,
numeric timestamp inputs, empty required donor cohorts, and parameter vintages
that have no target season. The latter prevents unused mid-season vintages from
suggesting that parameters may switch within a season.

Changes are concentrated in `ratings/gap_data.py` (cutoffs and season evidence),
`ratings/gap.py` (season-based vintage selection and persistence), and
`ratings/gap_training.py` (training/held-out fold validation). Existing GAP tests
now state their intentional kickoff-based diagnostic cutoffs explicitly.
`tests/analytics/test_gap_audit.py` contains the new adversarial regressions.

## Paper fidelity

Printed p.82 / PDF p.6, equations (6)-(9), was checked visually: predicted counts,
all eight simultaneous old-state updates, asymmetric phi roles, nonnegative
clipping, and paired absolute-error sum divided by N matches remain correct.
No numerical GAP update or existing Glicko/Bayesian mathematics was changed.
Printed pp.83-85 support Nelder-Mead statistic fitting, signed predicted
home-minus-away differences, annual parameter estimation from preceding seasons,
league-isolated states, retained-team carryover, and the opposite outgoing
promotion/relegation donor means. The season fixes enforce that interpretation.

The prompt's production scheduler contract is stricter than a generic
match-by-match replay. Explicit cutoffs are therefore required; nominal round
labels are not used to invent batch identities for postponed games.

## API and artifact migration

- Supply `cutoffs` explicitly for every GAP feature query. Production callers
  should also supply actual `prediction_batch_id` values so batch consistency
  can be checked. `cutoffs='kickoff_at'` is an explicit diagnostic choice.
- Fitting requires `GAPSeason.closed_at` for every training season and explicit
  `target_seasons`. Closure is externally verified evidence, not inferred from
  the last observed row. `target_start` is the earliest declared target entry.
- `GAPVintage.valid_from/valid_until` now delimit target season entries; they do
  not expire the parameters used by an already-selected season.
- Native artifacts now use schema 2 and record `vintage_selection='season_entry'`.
  Schema-1 artifacts must be rebuilt: their snapshots may end at the former
  calendar expiry, so reinterpreting their metadata would be unsafe. Reloaded
  schema-2 artifacts preserve the strict cutoff and vintage checks.

See [the usage guide](gap_ratings.md) for the runnable synthetic example and
full replication choices.

## Validation and limits

Final focused run: **210 passed in 15.56 seconds**, including **22 new audit
cases**. It covers GAP core/replay/features/training/audit tests, existing
ratings and transitions, Bayesian features/replay/kickoff/transition tests,
and fixture assembly. The documented four-row synthetic example also passed:
both fixtures in its frozen batch receive zero pre-batch GAP differences.
`git diff --check` passed. Test stdout is retained in the ignored local file
`scratch/gap-review-tests.txt`.

Only the existing bounded two-match synthetic optimizer smoke test was run
(`maxiter=3`, `maxfev=8`). Invalid training folds were tested with the optimizer
patched to fail if invoked. The existing environment was used read-only, with
bytecode disabled and numerical libraries limited to one CPU thread.

Remaining replication choices and evidence obligations before real fitting:

- Replay under each new parameter vintage is the supported explicit state
  policy. The paper does not uniquely resolve replay versus continuing state
  accumulated under older parameters.
- Burn-in is implemented literally as home pre-match game count greater than
  six. The objective mask remains explicit because its fitting use is ambiguous.
- Complete-pair omission for missing statistics and freezing inherited states
  at season entry are documented adapter policies. A late prior-season release
  does not retroactively rewrite a newer season's initialization.
- The caller must establish source completeness, trustworthy availability,
  closure and membership evidence, and correct provider statistic definitions.
  Without actual batch IDs the caller must enforce a common cutoff throughout
  each production batch. These facts cannot be proved from arbitrary input rows.
- No live provider mapping, real-data throughput or predictive performance was
  tested. This review does not claim an accuracy improvement.

No real-data fitting, backtest, sweep, feature rebuild, GPU operation, environment
change, merge or push was performed. Main was verified clean at the base commit.
Existing CUDA seasonal refits were not stopped, restarted or reconfigured.
