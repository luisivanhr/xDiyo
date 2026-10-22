# Paper GAP ratings and fixture differences

This opt-in implementation follows Edward Wheatcroft (2021), *Forecasting
football matches by predicting match statistics*, Journal of Sports Analytics
7, 77-97, DOI **10.3233/JSA-200462**. Equations (6)-(8), printed p.82 / PDF p.6,
provide the four team states, paired updates and predicted counts. Equation (9)
and p.83 specify statistic MAE and Nelder-Mead; p.84 uses signed predicted
home-minus-away differences; p.85 specifies seasonal parameter fitting,
initialization, opposite outgoing promotion/relegation donors and burn-in.
The supplied PDF was read and equations/season rules were checked visually.

## API and feature schema

- `GAPState(HA, HD, AA, AD)` and `WheatcroftGAP.predict/update` implement the
  immutable, simultaneous paired-count calculation. Each update uses both old
  states and both old residuals, then clips each component at zero.
- `GAPParameters(learning_rate, phi1, phi2)` has **no fitted defaults**.
  `learning_rate` is the paper's lambda. Each statistic's triple is shared over
  leagues; states remain separate by league, season and statistic.
- `GAPStatistic` maps explicitly defined prepared-history team/opponent columns.
  `GAPStatistic.from_stat(history, name, Stat(...), definition=...)` resolves an
  exact existing statistic identity, including period and numeric field.
- `GAPSeason` records dated, verified membership and outgoing donor cohorts for
  one statistic/league/season. `gap_season_from_context` accepts the existing
  `TransitionContext`/`build_team_seasons` movement evidence. It does not use
  Glicko transition mathematics. First-stat coverage must be declared; unknown
  entrants, missing predecessors and empty donors fail clearly.
- `GAPVintage` records the parameter triples, fit cutoff, half-open validity
  interval and provenance. `build_gap_ratings` takes chronological vintages
  supplied by the caller. It never fits them. Queries cannot use later-season
  fits for earlier rows or switch parameters within the same target season.
- `GAPRatingRun.features` supplies `Rating('name', fields=('HA', ...))` through
  the existing numeric `RatingRun` lookup contract. `fixture_features` returns
  signed differences by default; explicit `fields=('home','away','difference')`
  enables count inspection. `diagnostics` returns initialization availability,
  pre-cutoff seasonal game counts, complete-stat update counts and burn-in
  flags, independently of predictors.
- `gap_feature_set('name', statistics)` returns an ordinary named expression
  mapping. `GAPFixture` consumes a named run through `evaluate_features`.

The four selectable outcome predictors are `gap_goals_diff`,
`gap_shots_on_target_diff`, `gap_shots_off_target_diff`, and `gap_corners_diff`.
No feature bank or incumbent is changed. Direct `GAPFixture` expressions follow
the evaluator's usual one-column naming/multiple-column suffix convention.
The helper uses the exact names above. Both perspective rows carry identical
home-frame fixture values; `keyed=True` retains native identities. Match-layout
assembly exports `fixture::gap_<stat>_diff` once, rather than duplicating it into
home/away feature blocks.

The existing GAT `TemporalGraphConfig.fixture_features` accepts selected
external `X` columns. Its adapter also requires `feature_cutoff_at` metadata to
match the graph's history cutoff. The GAP adapter can supply those columns;
no GAT message passing, heads, losses or training settings were changed.
XGBoost consumes the same assembled numeric columns. The prompt's particular
incumbent40, frozen-11 and covariance research configurations are not named
public presets in this committed baseline, so this patch does not edit or
redefine them.

## Small synthetic usage

This example is deliberately **synthetic**. Its parameters are not published or
recommended production values. It does not call an optimizer or load data.

```python
import pandas as pd
from xdiyo_analytics.ratings import (
    GAPParameters, GAPStatistic, GAPSeason, GAPVintage, build_gap_ratings,
)
from xdiyo_analytics.features import evaluate_features, gap_feature_set, Rating

kick = pd.to_datetime(['2020-01-02', '2020-01-02',
                       '2020-01-04', '2020-01-04'], utc=True)
history = pd.DataFrame({
    'competition_id': [10] * 4, 'season_id': [2020] * 4,
    'event_id': [1, 1, 2, 2], 'team_id': [1, 2, 1, 2],
    'opponent_id': [2, 1, 2, 1], 'side': ['home', 'away'] * 2,
    'status': ['finished'] * 4, 'kickoff_at': kick,
    'result_available_at': kick + pd.Timedelta(hours=2),
    'goals_for': [4., 2., 3., 1.], 'goals_against': [2., 4., 1., 3.],
})
stat = GAPStatistic('goals', 'goals_for', 'goals_against',
                    'Synthetic full-time goals; no extra time or penalties')
season = GAPSeason(10, 2020, 'goals', '2020-01-01', '2020-01-01',
                   ((1, 'initial'), (2, 'initial')), 'Synthetic verified membership')
vintage = GAPVintage(
    (('goals', GAPParameters(.2, .75, .6)),),
    '2019-12-31', '2020-01-01', '2021-01-01',
    'Synthetic example only; not fitted or published parameters',
)
run = build_gap_ratings(
    history, statistics=(stat,), seasons=(season,), vintages=(vintage,),
    data_cutoff='2020-01-05', state_policy='replay',
)
# This synthetic round has two fixtures for the same team: freeze both before
# the first kickoff. In production supply the scheduler's actual cutoff Series.
cutoffs = pd.Series(pd.Timestamp('2020-01-01', tz='UTC'), index=history.index)
X = evaluate_features(history, gap_feature_set('gap', ('goals',)),
                      ratings={'gap': run}, cutoffs=cutoffs, keyed=True)
inspection = evaluate_features(history, {'state': Rating('gap', fields=('HA','HD'))},
                               ratings={'gap': run}, cutoffs=cutoffs)
diagnostics = run.diagnostics(history, cutoffs=cutoffs)
```

For a later real season, obtain parameter vintages only from its preceding
training seasons and set `fit_cutoff <= valid_from`. Do not build historical
training features from a fit on the full dataset. The paper's first available
season is parameter training only: a subsequently fitted vintage deliberately
cannot generate its earlier training-season features. Synthetic parameters can
produce diagnostics in that season, but those are not paper forecast evaluation.
No automatic season refit command is provided or executed.

## Timing and replication choices

1. **Parameter vintage reconstruction is explicit.** `state_policy='replay'`
   rebuilds old history under each supplied vintage. The paper does not resolve
   replay versus continuing the old accumulated state. This implementation
   supports the former only when explicitly selected; it rejects `continue`.
   Fitting is separate, and the pure state/update API remains suitable for a
   future explicitly designed continuation policy.
2. **Production predictions use the scheduler's pre-round cutoffs.** Pass the
   same actual cutoff to both perspectives and every fixture in that prediction
   batch. `evaluate_features` retains those cutoff identities for assembly.
   As in existing rating lookups, omitting cutoffs means row kickoff, which is
   appropriate only for an explicitly chosen match-by-match query. GAP does not
   infer round batches from nominal labels. Postponements use real availability
   and the actual chosen prediction boundary.
3. **State chronology is separate from batch prediction.** At each release,
   reconstruct the available prefix in kickoff order. Several matches involving
   one team get ordinary sequential equations (6)-(7), not pooled residuals.
   Predictions for their frozen batch do not consume any of them. A delayed
   result can change states at/after its release, never a saved earlier state.
   Equal-release earlier kickoffs are usable at the cutoff; kickoff itself must
   be strictly earlier. Two games for one team at the same kickoff fail because
   their chronological order is undefined.
4. **Season entry freezes inherited states.** Retained teams copy prior state;
   promoted teams average the previous league's relegated-out states and
   relegated teams average promoted-out states. Every component is averaged.
   Only membership known by entry and prior results available at entry are
   used. A late prior-season release after entry does not rewrite that transfer.
   This operational delayed-result policy is documented, not attributed to
   the paper. Unavailable donor metadata fails instead of resetting/shrinking.
5. **Burn-in is literal pre-match count greater than six.** P.85 says the home
   team has played six or fewer matches so far. Thus counts 0 through 6 are
   ineligible; count 7 is eligible. Earlier wording about the first six games
   could instead imply count 6 is eligible. The returned
   `paper_burn_in_eligible` is only this burn-in criterion, not a complete paper
   evaluation-population filter. No fixture or state update is discarded by it.
   `games_seen` counts known completed games in the current season; separately,
   `stat_updates` counts complete paired observations for that statistic.
6. **Incomplete statistics skip only their own paired update.** An observed
   zero updates; missing is not zero. Both perspective rows must mirror each
   other. Negative/infinite counts fail. Missing release times skip all updates
   for that match. Availability must be explicit (or `result_available_at`);
   reviewed completion bounds are honored even with an earlier supplied proxy.
7. **No unverified total-shot subtraction.** Existing source identities include
   `shotsOffGoal`, `shotsOnGoal`, `totalShotsOnGoal` and `blockedScoringAttempt`.
   Their names alone do not prove the additive definition. Prefer a verified
   direct off-target field. `derive_shots_off_target` requires caller-confirmed
   `total_is_on_plus_off=True` and nonempty definition evidence, with aligned
   columns. A total containing a separate blocked-shot category is unsuitable.
   Negative differences fail. No live provider mapping was configured here.
8. **The fitting mask and numerical settings are explicit.** `gap_mae` and
   `tune_gap` require a boolean objective mask and fixture cutoffs. Excluded
   objective rows still update state when their counts become available. The
   equation (9) denominator is eligible matches, not twice that count. Whether
   burn-in belongs in the fitting mask is not unambiguously specified.
   `tune_gap` requires an initial triple, `maxiter`, `maxfev`, `xatol`, `fatol`,
   fit cutoff and next-season start. It uses SciPy Nelder-Mead in
   `(log(lambda), logit(phi1), logit(phi2))`; unrepresentable candidates receive
   infinite objective, rather than projecting onto the open-domain boundaries.
   Starts, limits, tolerances, sample identities/mask, cutoffs, input hash,
   definitions and season plans are recorded. `adaptive=False`, the SciPy
   version and the actual transformed initial simplex are also recorded. An
   optional `initial_simplex` supplies a finite, nondegenerate 4-by-3 array;
   otherwise each starting coordinate is perturbed by 5 percent (zero by
   0.00025). These are implementation choices, not paper settings. Tolerances
   apply in transformed coordinates. These numerical choices are not
   published constants. Check `success` before considering a real fitted result.

The cross-venue prose on p.82 says a positive phi implies cross-venue influence;
the equations explicitly weight that influence by **1 minus phi**. They agree
on the allowed open domain but the prose is less precise about the weight.
The implementation follows the displayed equations. BA's 45-day half-life,
existing 180-day decay, season factors, priors, outcome loss and odds do not enter
GAP. Normalization remains an external training-only operation.

Before real fitting, decide replay versus continuation, burn-in convention and
objective mask, numerical settings, complete preceding season folds, provider
statistic definitions, membership/donor evidence and the delayed-season-entry
policy. No empirical accuracy, performance improvement or real-data coverage is
established by these tests.

## Native storage and implementation limits

`run.save(new_directory)` writes `gap.json`, `state.json` and
`snapshots.parquet` using the existing atomic artifact publisher and checksums.
`GAPRatingRun.load(directory)` preserves vintage checks, streams, season/league
identities, state policy, statistic definitions, source cutoff/hash and feature
schema. It refuses changed artifacts. There is deliberately no `ratings.json`:
`RatingRun.load` cannot silently turn this into an unrestricted generic run.
Existing artifacts are never overwritten. This task populated no real-data cache.

This is a correctness-first prefix replay. It reconstructs histories for saved
availability boundaries, so work/storage can grow quadratically with the number
of events. It has not been benchmarked on real data. A saved run can forecast
later fixtures within its vintage using the state through `metadata['data_cutoff']`;
it does not fetch subsequent results. Rebuild explicitly to assimilate them.
No incremental continuation or automatic data preparation is hidden in imports,
feature evaluation, configuration inspection or loading.

## Implementation handoff

Branch: `codex/wheatcroft-gap`. Isolated worktree:
`C:\Users\luisi\.codex\worktrees\wheatcroft-gap\xDiyo`.
Verified clean base: `0cb13cac57255f5e2f8e0eeb692e2b6f22344381`, matching local
and remote `main` at the start of this task (2026-10-23 JST).

The patch adds `ratings/gap_core.py`, `gap_data.py`, `gap.py`,
`gap_training.py`, the GAP fixture expression/exports/evaluator route, this
usage guide and synthetic tests. Existing rating equations and presets are
unchanged. Test output and PDF renderings remain in this worktree's own temporary
or ignored scratch paths. The existing Python environment was used read-only,
with bytecode writes disabled and BLAS/OpenMP threads limited to one.

No real-data optimization, model training, backtest, sweep, feature rebuild,
seasonal refit, GPU operation, merge, deployment or external upload was run.
The existing seasonal-refit supervisors were inspected read-only and were not
stopped, restarted, reconfigured or otherwise changed.


## Verification

Final result: **188 passed in 9.76 seconds** on the focused CPU-only suite.
The documented synthetic example also passed; `git diff --check` was clean.

The focused command covers GAP numerical/replay/features/training tests,
`test_ratings.py`, `test_rating_transitions.py`, `test_bayesian_features.py`,
`test_bayesian_replay.py`, `test_bayesian_kickoff_replay.py`,
`test_bayesian_replay_transitions.py`, and `test_fixture_assembly.py`.
The optimizer smoke test uses only two synthetic matches, `maxiter=3` and
`maxfev=8`. The usage example above is executed independently without fitting.
Full model-training suites, real-data replay throughput, predictive accuracy,
GAT training and XGBoost training are not tested by this implementation task.
