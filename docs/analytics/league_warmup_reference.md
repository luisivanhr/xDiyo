# League and transition API reference

Companion to the [practical guide](league_warmup.md) and
[coverage matrix](league_warmup_documentation_checklist.md). Signatures below describe
the implemented opt-in APIs. `history` is the observed DataFrame from
`build_team_history`; expressions return numeric columns only when evaluated.
The examples use the `history`, `corners` and `root` prepared in the practical guide.

## Imports and shared output contract

```python
import pandas as pd
from xdiyo_analytics.features import (
    Stat, ForAgainst, H2H, Lag, RollingMean, RollingStd, RollingZScore,
    League, LeaveOneOut, LeaguePopulation, WarmStart, SeededEMA,
    Hard, LinearFade, ObservationCount, blend_moments,
    build_team_seasons, TransitionContext, evaluate_features,
    MatchResultGlicko, StatGlicko, Rating,
)
from xdiyo_analytics.ratings import (
    Glicko2, GlickoTransition, RatingTransition, RatingRun, build_ratings,
)
```

`evaluate_features` preserves input row order and index, including duplicate labels.
One-column expressions use the supplied output name. Expanded periods/perspectives
append the source identity; generated rating columns include stream, role and state
field. Missing numeric outputs remain missing. Input observations and attrs are not
rewritten. `Stat(None, ...)` expands available periods independently, without merging
them or treating `ALL` as a derived total of periods.

History requires stable event/team/opponent/competition/season IDs, UTC-compatible
`kickoff_at`, `status` and `side`. Use exact integer/object or nullable integer types
for IDs; do not first convert large IDs through floats. `history.attrs['stat_columns']`
must identify role, period, group_name, key and field for selected numeric statistic
columns. Round-dependent policies also require nonmissing configured round keys.
Ratings require both home/away rows per fixture and consistent result availability.
`build_team_history` supplies these conventions from prepared exports.

## League and LeaveOneOut

```text
League(source, unit='team', schedule='completed_rounds', window_unit='rounds',
       round_keys=('competition_id', 'season_id', 'round'))
LeaveOneOut(source, exclude='team_contributions')
```

| Parameter | Meaning, permitted values and boundary |
| --- | --- |
| `League.source` | `Stat` or `ForAgainst(Stat)`. A population descriptor, not a standalone exported predictor. |
| `unit` | `'team'`: one contribution per team row; `'match'`: team+opponent statistic from home rows only. Match mode requires `Stat` and a unique matching opponent column. A missing component gives a missing total. |
| `schedule` | `'completed_rounds'` freezes the minimum target-round cutoff and requires all supplied fixture rows complete; `'kickoff'` uses each target row's cutoff. |
| `window_unit` | `'rounds'`, `'matches'` or `'days'`; the outer reducer's positive integer `window` measures this unit. See the guide for exact endpoints and ordering. |
| `round_keys` | Hashable tuple of columns identifying a round. Include competition, season and any stage needed to disambiguate repeats. Required for completed scheduling or round windows. |
| `LeaveOneOut.source` | A `League` descriptor. |
| `exclude` | `'team_contributions'` removes focal-team rows; `'fixtures'` also removes its opponents' rows. Match totals require `'fixtures'`. Exclusion follows window selection without refill. |

```python
population = LeaveOneOut(League(corners, unit='match'), exclude='fixtures')
match_total_loo = evaluate_features(history, {'total': RollingMean(population, 3)})
```

`RollingMean`, `RollingStd`, and `RollingZScore` reduce these populations. `Lag`,
ordinary `EMA`, H2H league grouping and arbitrary derived League sources are not
supported. Invalid units/schedules/windows/exclusions raise explicit errors when
evaluated. Reduction results have the source column count and original index.

## Rolling reducers and reference

```text
RollingMean(source, window=5, min_periods=1)
RollingStd(source, window=5, min_periods=1, ddof=1)
RollingZScore(source, window=5, min_periods=1, ddof=1, reference=None)
```

`window` is a positive integer; ordinary team windows count matches and league
windows use `window_unit`. `min_periods` is a positive finite-observation count;
ordinary team reducers reject a count above window, while a league round may contain
many observations. `ddof` is a nonnegative integer and std needs `n > ddof`.
Missing/nonfinite observations are ignored by moments but retain their window slots.
No eligible observations means missing output; std/Z with insufficient count are
missing. Constant populations have std zero and Z missing.

`reference` is the new optional Z-score expression. Team default is the latest
eligible observed source value, including that value in the window. League Z needs
an explicit reference, for example `Lag(corners)`. A reference produces one column
per population column, paired by position. Raw observed references, even wrapped
in H2H/WarmStart, are rejected. Historical or known-context references are permitted;
matching units and semantic identities are not inferred from equal column counts.

```python
reference_example = evaluate_features(history, {
    'team_z': RollingZScore(corners, 3, reference=Lag(corners)),
    'league_z': RollingZScore(League(corners), 3, reference=Lag(corners)),
})
```

## WarmStart and SeededEMA

```text
WarmStart(source, policy=None)
SeededEMA(alpha=.5, handoff=Hard(rounds=1), league_weight=.5,
          round_keys=('competition_id', 'season_id', 'round'))
```

`WarmStart` changes only its child definition; `policy=None` evaluates the child
unchanged. Use a `SeededEMA` policy on `RollingMean`/`RollingStd`/`RollingZScore`
whose observed source is Stat/ForAgainst or a League/LOO population. Use a rating
transition adapter on `MatchResultGlicko` or `StatGlicko`. A saved `Rating(name)`
accepts a no-op wrapper but cannot receive a transition policy after production.

| SeededEMA parameter | Meaning and validation |
| --- | --- |
| `alpha` | Finite `(0,1]`, weight of each new finite observation; missing values skip updates. |
| `handoff` | A `Hard`, `LinearFade` or `ObservationCount` instance. Other policies are rejected by this implementation. |
| `league_weight` | Finite `[0,1]`, mover mean blend toward eligible previous destination-league mean. It does not select rating cohorts or alter variance weighting. |
| `round_keys` | Tuple for completed-round counts on team features. League features use their League descriptor's keys. ObservationCount does not count rounds; Hard(0) skips warm-up lookup. |

Defaults, previous-team/league priors, no-prior fallback, missing variance, current
season observation selection, update order and equations are detailed in
[ordinary feature warm starts](league_warmup.md#ordinary-feature-warm-starts).
Each period and perspective has its own moments. H2H and extra team grouping keys
restrict applicable team priors. League priors use the declared observation unit
and exclusion. Warmed mean/std/Z preserve the ordinary expression's columns/index.

```python
policy = SeededEMA(alpha=.25, handoff=Hard(2), league_weight=.5)
warm_z = evaluate_features(history, {
    'warm_z': WarmStart(RollingZScore(corners, 3), policy),
    'warm_league': WarmStart(RollingMean(League(corners), 3), policy),
})
```

## Handoff classes and moment helper

```text
Hard(rounds=1)                         .weight(rounds, observations)
LinearFade(start=1, rounds=2)           .weight(rounds, observations)
ObservationCount(strength=5.)          .weight(rounds, observations)
blend_moments(mean_a, variance_a, mean_b, variance_b, weight)
```

Handoff constructors are frozen/hashable. Hard `rounds` and LinearFade `start` are
nonnegative integers; LinearFade `rounds` is a positive integer; booleans do not
count as integers. ObservationCount `strength` must be finite and positive.
`weight` accepts the evaluator's completed-round and finite-new-observation counts
and returns a scalar seeded-component weight. Hard/Fade ignore the observation
argument; ObservationCount ignores rounds. Their [exact equations and numeric
endpoints](league_warmup.md#all-three-handoffs) differ deliberately.

`blend_moments` returns a two-tuple `(mean, population_variance)` and validates
`weight` in `[0,1]`. It includes between-mean variance and returns the exact chosen
pair at weights 0/1. Missing active moments propagate; it does not fabricate a
variance or apply a sample correction. Supply compatible numeric population moments.

```python
weights = [LinearFade().weight(r, 0) for r in range(4)]
assert weights == [1., 1., .5, 0.]
assert blend_moments(2., 4., 10., 16., .25) == (8., 25.)
```

## Context inputs

`evaluate_features` adds `team_seasons=None` and `season_starts=None`:

```text
evaluate_features(history, features, *, group_by=('team_id','competition_id'),
                  cutoffs=None, available_at=None, team_counts=None, ratings=None,
                  team_seasons=None, season_starts=None)
```

Existing options keep their meanings: `features` is a nonempty name→expression map;
`group_by` sets ordinary team history groups; `cutoffs`/`available_at` accept an
aligned Series or column name (None uses kickoff); `team_counts` optionally supplies
standings denominators; `ratings` maps names to prebuilt runs. New context is created
lazily when an enabled warm-up/transition needs it. Policies are part of cache keys,
with full generated rating states shared across field/perspective extraction.

`team_seasons` accepts a DataFrame or list of records, unique on
`competition_id, season_id, team_id`. IDs remain exact. Columns:

| Column | Requirement |
| --- | --- |
| `competition_id`, `season_id`, `team_id` | Required nonambiguous identity for each supplied record. |
| `movement` | Optional; one of `retained`, `promoted`, `relegated`, `unknown`, `other_entry`. |
| `previous_competition_id`, `previous_season_id` | Optional predecessor pair for transfer/own-team prior lookup. Missing either triggers ordinary retained-team inference when possible, otherwise no predecessor. |
| Additional evidence fields | Preserved in context; explicit supplied movement is not independently verified by TransitionContext. Use the adapter below to derive evidence-backed flags. |

Absent records infer retained membership only from the preceding observed season
in the same competition; otherwise movement is unknown. A new appearance does not
mean promotion. Context inference is weaker than complete, consecutive membership
evidence; it does not certify a full competition inventory.

`season_starts` is either a mapping `(competition_id, season_id) → datetime` or
a DataFrame with `competition_id, season_id, entry_at`, unique per league-season.
Timestamps convert to UTC. Missing mappings use earliest query cutoff (evaluator)
or earliest kickoff (direct replay). Explicit starts may be earlier, never after
the first relevant prediction. Do not use missing/NaT entry times when a transition
must occur.

```python
# Illustration with synthetic IDs; substitute evidence-backed IDs for actual use.
team_seasons_example = [
    {'competition_id': 10, 'season_id': 2, 'team_id': 9001,
     'movement': 'promoted', 'previous_competition_id': 20, 'previous_season_id': 11},
]
starts_example = pd.DataFrame([
    {'competition_id': 10, 'season_id': 2, 'entry_at': '2024-08-01T00:00:00Z'},
])
early_cutoffs = history.kickoff_at - pd.Timedelta(days=2)
early_ratings = evaluate_features(history, {
    'rd': WarmStart(MatchResultGlicko(side='for', fields=('rd',)), GlickoTransition(phi_scale=1.2)),
}, cutoffs=early_cutoffs)
```

## Movement adapter

```text
build_team_seasons(history, competition_info, *, season_years=None,
                   complete_memberships=(), overrides=())
```

Adapts prepared-history membership to `utils.team_seasons.Membership` and
`derive_team_seasons`; it neither reads legacy CSVs nor collects data.
`competition_info` maps each competition ID to `{'tier': positive_int, 'system': str}`.
`season_years` optionally maps `(competition_id, season_id)` to full `(start,end)`
years. Otherwise one `source_season` or `season_year` label per partition is parsed,
such as `23_24`, `2023/2024`, `2023-24` or `2024`. Two-digit starts use the 2000s;
supply explicit full years for other centuries or ambiguous labels.

`complete_memberships` is a collection of league-season keys explicitly declared
complete by the caller. The default is observed-partial membership. A retained team
must appear in the consecutive-year predecessor; promotion/relegation additionally
needs absence from a complete destination predecessor and exactly one observed
origin in an adjacent tier of the same system. Unknown origin, ambiguous origin,
incomplete predecessor or missing division evidence stays unknown.

`overrides` is a list of unique matching team-season records with supported
`movement` and nonempty `evidence`. Optional predecessor IDs are preserved by this
adapter. Unsupported/duplicate overrides, invalid years/IDs/tiers and ambiguous
membership definitions are rejected by the existing evidence core.

Returns one object-dtype DataFrame row per observed team-season with identity,
previous IDs, movement, nullable `got_promoted`/`got_demoted`, evidence/source,
membership_status and derivation_version. Object IDs avoid float rounding when
predecessors contain nulls. Unknown flags remain unknown.

```python
competition_id = int(history.competition_id.iloc[0])
team_seasons = build_team_seasons(history, {
    competition_id: {'tier': 1, 'system': 'England'},
})
assert set(team_seasons.movement) <= {'retained', 'unknown'}
```

## LeaguePopulation helper

```text
LeaguePopulation(history, *, cutoffs=None, available_at=None)
population.round_info(keys)
population.rows(league, window, exclude=None)
```

This public helper caches shared positional row plans. It does not choose a
statistic, policy or reducer. Constructor time arguments use the evaluator's
alignment rules; cutoffs after target kickoff are rejected.

`round_info(keys)` takes a hashable column tuple and returns `(info, row_keys)`.
`info[key]` is `(integer_row_positions, minimum_prediction_anchor,
maximum_release_if_all_valid_else_NaT, maximum_kickoff)`. `row_keys` gives the round
key for every input row. Missing round keys raise. A missing measurement does not
itself make a fixture incomplete; status and time validity determine completion.

`rows(league, window, exclude=None)` returns one NumPy array of integer positions
per target row, after schedule/window selection and optional exclusion. These are
positions, not index labels; empty/missing-time queries yield empty arrays.
`exclude` accepts None/team_contributions/fixtures and obeys match-unit constraints.
Plans are cached by population configuration and window, with common anchors reused.
Treat cached arrays as read-only and the source history as immutable during use.

```python
population_plan = LeaguePopulation(history)
positions = population_plan.rows(League(corners), 3, exclude='team_contributions')
assert len(positions) == len(history)
```

## TransitionContext helper

```text
TransitionContext(history, *, cutoffs=None, available_at=None,
                  team_seasons=None, season_starts=None, population=None)
context.record(row)
context.prior_rows(season, boundary)
context.completed_rounds(row, round_keys, *, time=None)
```

`population` optionally supplies an already aligned LeaguePopulation; it must use
the same history/timing contract. Otherwise one is created. Context determines
league-season entry anchors, previous observed seasons by anchor order and per-team
movement records. It supplies context; it never discovers a policy from columns.

`record(row)` takes a zero-based integer position and returns its record containing
competition/season/team, entry_at, movement and predecessor IDs plus supplied fields.
`prior_rows(season, boundary)` takes a `(competition_id, season_id)` key (or None)
and returns integer positions of finished observations with kickoff strictly before
boundary and release no later than it. Missing/unobserved seasons yield an empty array.
`completed_rounds(row, round_keys, time=None)` counts fully eligible rounds of that
row's current season. Omitted time uses its cutoff; a frozen League caller supplies
the round anchor. Completion uses all supplied round rows and caches the count.
Treat records, cached arrays and source history as read-only while evaluating.

```python
context = TransitionContext(history)
first_record = context.record(0)
previous_key = context.predecessors[(first_record['competition_id'], first_record['season_id'])]
prior_positions = context.prior_rows(previous_key, first_record['entry_at'])
assert len(prior_positions) == 0  # The first loaded season has no loaded predecessor.
```

## GlickoTransition

```text
GlickoTransition(phi_scale=1., movement_phi_scale=1., shrinkage=.5,
                 top=3, bottom=3, rank_by='standings', aggregate='mean')
policy.apply(state, *, context, cohort)
```

Multipliers must be finite and at least one. `phi_scale` inflates a known prior
state at season entry; `movement_phi_scale` additionally inflates promoted/relegated
state. `shrinkage` is finite `[0,1]`; 0 retains location, 1 adopts an available cohort
location. `top`/`bottom` are positive integer cohort sizes (not booleans).
`rank_by` accepts standings/rating; `aggregate` accepts mean/median.

`apply` takes a full public/internal Glicko state and context including movement and
has_prior_state. Cohort records contain team_id, state and position. It returns a
new `rating, rd, sigma, mu, phi` state; input state/cohort remain unchanged. Sigma is
unchanged; public/internal conversions and multiplier/shrinkage equations are in
[the guide](league_warmup.md#glicko-policies-and-rating-only-cohorts).
Missing standings exclude those candidates; no eligible cohort keeps location.
Standings ties use stringified team ID as a deterministic tie-break. Cohort freezing
and removal of incoming teams are the replay helper's responsibility.

```python
movement_policy = GlickoTransition(
    phi_scale=1.1, movement_phi_scale=1.2, shrinkage=.4,
    top=3, bottom=3, rank_by='rating', aggregate='median',
)
```

## RatingTransition custom protocol

```text
RatingTransition.apply(state, *, context, cohort) -> dict[str, finite_number]
```

An adapter returns **exactly the initial engine's numeric field names**, each finite.
`state` and cohort states are copies. Context includes identity/predecessor fields,
entry_at, movement, `stream` and `has_prior_state`. Cohort entries are
`{'team_id': ..., 'state': {...}, 'position': number_or_None}` for eligible
destination prior-season members. Unknown/missing priors may produce an empty list.
Custom code owns its missing-prior and coordinate semantics; never apply Glicko's
mu/phi operations to arbitrary embeddings. An absent policy is a no-op.

Policies used inside feature ASTs must be immutable and hashable, as must the
associated engine/configuration. Direct replay can use an adapter object with a
callable `apply`. Example for an engine exposing flat `signal` and `uncertainty`
fields (not a Glicko engine):

```python
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class SignalTransition:
    uncertainty_scale: float = 1.2

    def apply(self, state, *, context, cohort):
        return {**state, 'uncertainty': state['uncertainty'] * self.uncertainty_scale}

example_state = SignalTransition().apply(
    {'signal': 2., 'uncertainty': 3.}, context={}, cohort=[],
)
assert example_state['signal'] == 2. and math.isclose(example_state['uncertainty'], 3.6)
```

This is a transition-only example, not a graph trainer or complete rating engine.
`GlickoTransition` is rejected for custom engines. Invalid field sets or nonfinite
returned values raise rather than silently changing the run's schema.

## build_ratings and persisted transitions

```text
build_ratings(history, *, stat=None, engine=None, higher_is_better=True,
              scope=('competition_id',), available_at=None, transition=None,
              team_seasons=None, season_starts=None, transition_context=None)
```

Existing arguments select W/D/L (`stat=None`) or a Stat comparison, the engine
(`None` means Glicko2), greater-is-better direction, shared match scope and release
times. `transition` explicitly enables an adapter. `team_seasons`/`season_starts`
use the [schemas above](#context-inputs); `transition_context` can reuse a prepared
TransitionContext with matching history and timing. When supplied, that context
already owns its membership/start settings.

With transitions, scope must be a subset of competition_id/season_id, including
empty scope. A scope that collapses simultaneous entries for the same team is
rejected; include competition_id to separate such entries. Explicit predecessors
allow retained/moving state transfer even with season in the key. The default
competition scope does not automatically calibrate different leagues.

Output remains a RatingRun: numeric full-state snapshots, initial_state, scope,
streams and metadata. When enabled, snapshots add `snapshot_kind` and
`snapshot_order` (transition 0, result 1). Both can share a timestamp. All entry
events at that timestamp read old states before commits; equal-time results then
update against their shared preceding states. Once-only entry is per stream/team/
season. Focal/other incoming teams are excluded from prior-season cohorts;
departing prior-season members remain eligible.

`RatingRun.features(history, *, cutoffs=None, side='both', fields=None)` selects
eligible snapshots in recorded-time/order sequence. Full fields persist even when
only one is exported. Missing query dates yield missing features; unrated teams
receive initial state. `save(directory)` writes snapshots.parquet and ratings.json,
including kind/order and transition repr/boundary metadata. It refuses existing
artifacts rather than overwriting them. `RatingRun.load(directory)` restores these
without replay. Source-selection provenance should accompany the run separately;
the metadata is not a reconstruction of historical data availability.

## Module support functions

These functions/classes are available from their defining modules, but are not
re-exported as the main user-facing feature/rating API. Prefer the public evaluator
or replay unless implementing an extension. They introduce no separate user policy.
The shortened module paths in the table are under `xdiyo_analytics`:

```python
from xdiyo_analytics.features.league import league_values, reduce_league
from xdiyo_analytics.features.warmup import seeded_moments, evaluate_warm_start
from xdiyo_analytics.ratings.transitions import TransitionReplay
assert seeded_moments(3., 5., [10.], .5) == (6.5, 14.75)
```

| Import and signature | Purpose, inputs and output |
| --- | --- |
| `features.league.league_values(history, source, frame, unit)` | Converts a selected Stat/ForAgainst numeric frame to team contributions or match totals. Uses stat_columns metadata to locate exactly one opposite-side column. Returns an aligned frame; missing total components remain missing. |
| `features.league.reduce_league(node, population, evaluate)` | Reduces a RollingMean/Std/Z node using a LeaguePopulation and callback returning `(frame, h2h_scope)`. Returns an aligned numeric frame; rejects unsupported reducers, H2H population changes and invalid count/ddof/reference shapes. |
| `features.warmup.seeded_moments(mean, variance, observations, alpha)` | Returns `(mean, variance)` after finite-observation updates; nonfinite observations skip, alpha=1 replaces missing spread with a complete observation's zero spread. Policy constructors validate alpha; this helper expects a valid alpha. |
| `features.warmup.evaluate_warm_start(node, context, evaluate, candidates, *, h2h=False, group_by=())` | Evaluator integration returning `(frame, scope)` from an explicit SeededEMA wrapper, shared context, expression callback and eligible-row callback. Handles frozen priors, source restrictions, handoff and missing fallbacks. |
| `ratings.transitions.TransitionReplay(context, scope, policy, initial)` | Prepares entry events for one replay definition, validates supported scopes and collisions. `key(record, previous=False)` forms scope/team keys (None if required predecessor keys are missing). `apply(time, states, games_seen, latest, stream)` reads frozen old states/cohorts, validates returned numeric state, commits the dictionaries and returns transition snapshot records. These state dictionaries are intentionally mutated by the replay integration. |

The new reference field and helpers do not implement arbitrary population algebra,
raw current-match predictors, automatic cross-stream mapping, custom graph training,
James-Stein priors, CV warm-up allocation, targets or the general experiment
orchestrator. Those remain distinct future work.
