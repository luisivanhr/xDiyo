# Documentation coverage: league features and warm starts

Status: documentation and independent verification are in progress. This checklist
is an acceptance checklist, not evidence that the guides or tests are complete.
The documentation task should link each item to the finished guide section and
mark it complete only after checking the explanation against the source.

## Required reference coverage

For every public API below, document its import, purpose, full signature,
parameter meanings and defaults, supported inputs/compositions, output shape,
missing-data behavior and a practical example. Explain restrictions where they
affect use. Public helpers can have a separate advanced reference so the main
notebook stays short.

| API/change | Source | Required explanation | Completed guide section |
| --- | --- | --- | --- |
| `League` | `features/league.py` | Team contributions versus one match total; schedule, window unit and round keys | Pending |
| `LeaveOneOut` | `features/league.py` | Own contributions versus whole fixtures; window before exclusion; no refill; distinct from LOCO | Pending |
| League rolling mean/std/Z-score | `features/league.py` | Pooled observations, observation counts, partial windows, ddof and explicit historical Z reference | Pending |
| `RollingZScore.reference` | `features/expressions.py`, `features/evaluation.py` | Default team reference, explicit reference, column pairing and exclusion of observed target values | Pending |
| `WarmStart` | `features/warmup.py` | Explicit opt-in per expression, policy=None, wrapper placement, unchanged unwrapped features | Pending |
| `SeededEMA` | `features/warmup.py` | Alpha, league weight, prior selection, supported operators, round keys, updates and fallback behavior | Pending |
| `Hard` | `features/warmup.py` | Completed-round boundary, one-round enabled default, zero-round no-op and possible discontinuity | Pending |
| `LinearFade` | `features/warmup.py` | Start/duration, clipped weight, exact endpoints and example weights | Pending |
| `ObservationCount` | `features/warmup.py` | Strength, valid new observations, missing values and asymptotic handoff | Pending |
| `blend_moments` | `features/warmup.py` | First/second moments, between-means term, endpoint weights and population variance | Pending |
| `build_team_seasons` | `features/transitions.py` | Competition tier/system metadata, season years, complete memberships, overrides, evidence and exact IDs | Pending |
| `LeaguePopulation` | `features/league.py` | Reusable positional row plans, round completion/anchors, shared work and public methods | Pending |
| `TransitionContext` | `features/transitions.py` | Reusable boundary/membership/prior context and public methods; helpers do not select a rating or policy | Pending |
| `evaluate_features` additions | `features/evaluation.py` | team_seasons and season_starts formats, cutoff interaction, separate policy caches and aligned outputs | Pending |
| `GlickoTransition` | `ratings/transitions.py` | Every multiplier/cohort/rank/aggregation parameter, defaults, full-state transformation and sigma behavior | Pending |
| `RatingTransition` | `ratings/transitions.py` | Custom adapter contract, context/cohort contents, state fields, immutability and a small custom example | Pending |
| `build_ratings` additions | `ratings/replay.py` | Transition opt-in, context inputs, direct replay boundary versus evaluator boundary, supported scopes | Pending |
| Transition snapshots and lookup | `ratings/replay.py`, `ratings/snapshots.py` | snapshot_kind/order, once-only updates, equal-time order, training features, save/load and metadata | Pending |

## Behavior and equations

- [ ] Compare completed-round and kickoff schedules with a small timeline. Explain
  the minimum target-round cutoff, postponed/partial rounds, stage keys, completion
  relative to supplied fixtures, and crossing seasons within a competition.
- [ ] Explain match/day/round windows, how missing measurements affect window
  membership, and why three-round std pools observations rather than round means.
- [ ] Give a numerical LOO example including its effect on count, mean and variance;
  explain the indivisibility of match totals and the role of ForAgainst.
- [ ] Give exact cutoff examples, including None and two days before kickoff;
  distinguish optional availability from the earlier-finished-kickoff assumption.
- [ ] Explain mean priors for retained teams, known movers and missing prior data;
  league spread uses the matching statistic/period/unit, without rating cohorts.
- [ ] State the seeded EMA equations for mean and variance, including missing
  observations and alpha=1. Explain which observations enter after transition.
- [ ] State all three handoff weight equations with short numeric examples.
- [ ] State the agreed blended mean/variance equations and the between-means term.
  Explain population moments during blending versus the configured rolling ddof
  after handoff, partial windows, absent rolling components and missing spread.
- [ ] Explain Z-score reference eligibility, latest-value inclusion, zero variance,
  and how it combines with warmed moments.
- [ ] Provide exact small team_seasons and season_starts input examples. Distinguish
  observed membership from evidence of promotion/relegation; newly seen is unknown.
- [ ] Explain retained-season phi inflation and mover mu shrinkage, how both
  multipliers combine, public rating/RD versus internal mu/phi, and unchanged sigma.
- [ ] Explain rating-only top/bottom cohorts, standings versus rating ranking,
  mean versus median, missing rankings/cohorts, frozen same-stream information,
  exclusion of incoming teams, and treatment of departing prior-season teams.
- [ ] State that available standings are eligible pregame positions, not a
  reconstructed final table. State the limits of cross-league rating calibration.
- [ ] Explain separate result/statistic/period/warmed/unwarmed streams and applying
  transitions once to the full state before selecting fields or perspectives.
- [ ] Explain custom state adapters without assuming Glicko coordinates, and why a
  saved Rating(name) must be warmed while building its run rather than afterward.
- [ ] List supported compositions/scopes and deferred work, including James-Stein,
  arbitrary model-state dynamics, and orchestration/pilot work still outstanding.

## Examples, navigation and final check

- [ ] Practical examples cover ordinary versus league/LOO features, all handoff
  choices, rating transitions and prepared snapshot reuse, with complete imports.
- [ ] Keep the notebook minimal; link the full reference instead of placing the
  entire design explanation in notebook cells. Preserve the existing notebooks.
- [ ] Link the new guides from src/README.md, the feature guide and rating guide.
- [ ] Update current summaries in IMPLEMENTATION_PROGRESS.md and the working notes;
  supersede stale discussion-only statements while retaining archived history.
- [ ] Check every public export and changed signature against this matrix; add any
  missed API or behavior and replace every Pending cell with a guide section link.
- [ ] Verify documented examples against the final source and distinguish passing
  checks, pending checks and known limitations. Do not claim completion from dispatch.
