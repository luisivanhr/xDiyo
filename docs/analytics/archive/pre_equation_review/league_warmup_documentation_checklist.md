# Documentation coverage: league features and warm starts

Status: complete. All 22 coverage rows link to source-reviewed explanations.
The batch passed 275 tests (97 new), bounded two-season reconciliation, the fresh
9-cell notebook, 18 executable guide examples and isolated wheel import. The final
source/preservation/link audit passed; evidence is recorded in
[league_warmup_check.json](league_warmup_check.json).

## Required reference coverage

For every public API below, document its import, purpose, full signature,
parameter meanings and defaults, supported inputs/compositions, output shape,
missing-data behavior and a practical example. Explain restrictions where they
affect use. Public helpers can have a separate advanced reference so the main
notebook stays short.

| API/change | Source | Required explanation | Completed guide section |
| --- | --- | --- | --- |
| `League` | `features/league.py` | Team contributions versus one match total; schedule, window unit and round keys | [Reference](league_warmup_reference.md#league-and-leaveoneout) |
| `LeaveOneOut` | `features/league.py` | Own contributions versus whole fixtures; window before exclusion; no refill; distinct from LOCO | [Reference](league_warmup_reference.md#league-and-leaveoneout) |
| League rolling mean/std/Z-score | `features/league.py` | Pooled observations, observation counts, partial windows, ddof and explicit historical Z reference | [Reference](league_warmup.md#loo-and-pooled-moments) |
| `RollingZScore.reference` | `features/expressions.py`, `features/evaluation.py` | Default team reference, explicit reference, column pairing and exclusion of observed target values | [Reference](league_warmup_reference.md#rolling-reducers-and-reference) |
| `WarmStart` | `features/warmup.py` | Explicit opt-in per expression, policy=None, wrapper placement, unchanged unwrapped features | [Reference](league_warmup_reference.md#warmstart-and-seededema) |
| `SeededEMA` | `features/warmup.py` | Alpha, league weight, prior selection, supported operators, round keys, updates and fallback behavior | [Reference](league_warmup_reference.md#warmstart-and-seededema) |
| `Hard` | `features/warmup.py` | Completed-round boundary, one-round enabled default, zero-round no-op and possible discontinuity | [Reference](league_warmup_reference.md#handoff-classes-and-moment-helper) |
| `LinearFade` | `features/warmup.py` | Start/duration, clipped weight, exact endpoints and example weights | [Reference](league_warmup_reference.md#handoff-classes-and-moment-helper) |
| `ObservationCount` | `features/warmup.py` | Strength, valid new observations, missing values and asymptotic handoff | [Reference](league_warmup_reference.md#handoff-classes-and-moment-helper) |
| `blend_moments` | `features/warmup.py` | First/second moments, between-means term, endpoint weights and population variance | [Reference](league_warmup_reference.md#handoff-classes-and-moment-helper) |
| `build_team_seasons` | `features/transitions.py` | Competition tier/system metadata, season years, complete memberships, overrides, evidence and exact IDs | [Reference](league_warmup_reference.md#movement-adapter) |
| `LeaguePopulation` | `features/league.py` | Reusable positional row plans, round completion/anchors, shared work and public methods | [Reference](league_warmup_reference.md#leaguepopulation-helper) |
| `TransitionContext` | `features/transitions.py` | Reusable boundary/membership/prior context and public methods; helpers do not select a rating or policy | [Reference](league_warmup_reference.md#transitioncontext-helper) |
| `evaluate_features` additions | `features/evaluation.py` | team_seasons and season_starts formats, cutoff interaction, separate policy caches and aligned outputs | [Reference](league_warmup_reference.md#context-inputs) |
| `GlickoTransition` | `ratings/transitions.py` | Every multiplier/cohort/rank/aggregation parameter, defaults, full-state transformation and sigma behavior | [Reference](league_warmup_reference.md#glickotransition) |
| `RatingTransition` | `ratings/transitions.py` | Custom adapter contract, context/cohort contents, state fields, immutability and a small custom example | [Reference](league_warmup_reference.md#ratingtransition-custom-protocol) |
| `build_ratings` additions | `ratings/replay.py` | Transition opt-in, context inputs, direct replay boundary versus evaluator boundary, supported scopes | [Reference](league_warmup_reference.md#build_ratings-and-persisted-transitions) |
| Transition snapshots and lookup | `ratings/replay.py`, `ratings/snapshots.py` | snapshot_kind/order, once-only updates, equal-time order, training features, save/load and metadata | [Reference](league_warmup_reference.md#build_ratings-and-persisted-transitions) |

| `league_values` / `reduce_league` | `features/league.py` | Module helper contracts, callback/results, source restrictions and missing values | [Module support](league_warmup_reference.md#module-support-functions) |
| `seeded_moments` | `features/warmup.py` | Finite observations, exact recurrence, alpha=1 and missing spread | [Module support](league_warmup_reference.md#module-support-functions), [equations](league_warmup.md#ema-and-mixture-equations) |
| `evaluate_warm_start` | `features/warmup.py` | Internal evaluator integration, context/callbacks and return scope | [Module support](league_warmup_reference.md#module-support-functions) |
| `TransitionReplay` | `ratings/transitions.py` | Constructor, key/apply methods, scope collisions, frozen entry batching and intentional state mutation | [Module support](league_warmup_reference.md#module-support-functions) |

## Behavior and equations

- [x] Compare completed-round and kickoff schedules with a small timeline. Explain
  the minimum target-round cutoff, postponed/partial rounds, stage keys, completion
  relative to supplied fixtures, and crossing seasons within a competition.
- [x] Explain match/day/round windows, how missing measurements affect window
  membership, and why three-round std pools observations rather than round means.
- [x] Give a numerical LOO example including its effect on count, mean and variance;
  explain the indivisibility of match totals and the role of ForAgainst.
- [x] Give exact cutoff examples, including None and two days before kickoff;
  distinguish optional availability from the earlier-finished-kickoff assumption.
- [x] Explain mean priors for retained teams, known movers and missing prior data;
  league spread uses the matching statistic/period/unit, without rating cohorts.
- [x] State the seeded EMA equations for mean and variance, including missing
  observations and alpha=1. Explain which observations enter after transition.
- [x] State all three handoff weight equations with short numeric examples.
- [x] State the agreed blended mean/variance equations and the between-means term.
  Explain population moments during blending versus the configured rolling ddof
  after handoff, partial windows, absent rolling components and missing spread.
- [x] Explain Z-score reference eligibility, latest-value inclusion, zero variance,
  and how it combines with warmed moments.
- [x] Provide exact small team_seasons and season_starts input examples. Distinguish
  observed membership from evidence of promotion/relegation; newly seen is unknown.
- [x] Explain retained-season phi inflation and mover mu shrinkage, how both
  multipliers combine, public rating/RD versus internal mu/phi, and unchanged sigma.
- [x] Explain rating-only top/bottom cohorts, standings versus rating ranking,
  mean versus median, missing rankings/cohorts, frozen same-stream information,
  exclusion of incoming teams, and treatment of departing prior-season teams.
- [x] State that available standings are eligible pregame positions, not a
  reconstructed final table. State the limits of cross-league rating calibration.
- [x] Explain separate result/statistic/period/warmed/unwarmed streams and applying
  transitions once to the full state before selecting fields or perspectives.
- [x] Explain custom state adapters without assuming Glicko coordinates, and why a
  saved Rating(name) must be warmed while building its run rather than afterward.
- [x] List supported compositions/scopes and deferred work, including James-Stein,
  arbitrary model-state dynamics, and orchestration/pilot work still outstanding.

## Examples, navigation and final check

- [x] Practical examples cover ordinary versus league/LOO features, all handoff
  choices, rating transitions and prepared snapshot reuse, with complete imports.
- [x] Keep the notebook minimal; link the full reference instead of placing the
  entire design explanation in notebook cells. Preserve the existing notebooks.
- [x] Link the new guides from src/README.md, the feature guide and rating guide.
- [x] Update current summaries in IMPLEMENTATION_PROGRESS.md and the working notes;
  supersede stale discussion-only statements while retaining archived history.
- [x] Check every public export and changed signature against this matrix; add any
  missed API or behavior and replace every Pending cell with a guide section link.
- [x] Verify documented examples against the final source and distinguish passing
  checks, pending checks and known limitations. Do not claim completion from dispatch.

## Coverage evidence and limits

Public signatures/defaults were checked against the four new modules and the changed
expression/evaluator/replay/snapshot interfaces. Support functions not re-exported
from package roots are documented separately. Historical planning settings are not
presented as current defaults. All three handoffs are implemented and opt-in.

The practical guide and API reference contain 18 executable Python blocks, all run
against the pinned local two-season history. The nine-cell notebook contains five
code cells executed in order in a fresh `misc314_py314` kernel. The evidence file
records source/input hashes and test/example/notebook/wheel results. No unexplained
API omission has been identified. Explicit limits remain: retrospective kickoff
availability proxy, incomplete origin evidence for new teams, pregame rather than
final standings, no automatic cross-league calibration, and no fitting/orchestrator.
