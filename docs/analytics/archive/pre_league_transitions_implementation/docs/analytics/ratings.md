# Result, statistic and reusable state ratings

Rating expressions produce ordinary numeric columns aligned with the team-history
index. Use them in training `X`, prediction inputs and reporting.

```python
from xdiyo_analytics.features import (
    Stat, MatchResultGlicko, StatGlicko, evaluate_features,
)

corners = Stat("ALL", "Match overview", "cornerKicks")
X = evaluate_features(history, {
    "wdl": MatchResultGlicko(),
    "corners": StatGlicko(corners),
})
```

Both team and opponent perspectives are exported by default. For example,
`wdl::result::team::rating` and `wdl::result::opponent::rating` are the two
pre-prediction strengths. Choose `side="for"`, `"against"` or `"both"`, and
`fields=("rating", "rd")` when fewer outputs are wanted. The named feature rule
still applies: one output column takes the requested name; multiple columns add
stream/perspective/field suffixes.

## What each stream measures

- `MatchResultGlicko` uses finished-match `result` values W/D/L as 1/.5/0.
  The [history builder's score and penalty semantics](team_history.md) still apply.
- `StatGlicko(Stat(...))` compares the team's value with its opponent's value:
  greater/equal/less maps to 1/.5/0. Set `higher_is_better=False` for a quantity
  where a smaller value is preferred.
- Missing/nonfinite statistic pairs skip that stream's update for both teams.
  They do not become draws or zero counts. Unfinished matches and unusable times
  also do not update the state.
- `Stat(None, ...)` creates an independent stream for every supplied period.
  Result, corner and half-specific ratings never share their rating states.

A corner-comparison rating measures comparative corner performance. It is not
a corner-count forecast and does not use the victory margin. Selecting its
columns does not select an outcome distribution or a fitted prediction model.

## One numerical engine, explicit scales

The analytics `Glicko2` adapter uses the unchanged `utils.glicko_rating.Glicko2`
as its single numerical engine. Defaults are initial rating **1500**, RD **350**,
volatility **0.06**, tau **1.0**, and convergence tolerance **1e-6**.
Parameters remain configurable. See the [engine comparison](glicko_engine_comparison.md).

| Snapshot field | Scale and meaning |
| --- | --- |
| `rating` | Public rating scale, initially 1500. |
| `rd` | Public rating deviation, initially 350. |
| `sigma` | Volatility, initially .06; shared by public/internal representations. |
| `mu` | Internal coordinate `(rating - 1500) / 173.7178`. |
| `phi` | Internal deviation `rd / 173.7178`. |

The legacy object's fields named `mu` and `phi` are on the **public** scale;
they correspond to snapshot `rating` and `rd`. Do not compare them directly
with the snapshot's internal `mu` and `phi`.

The [official worked example](https://www.glicko.net/glicko/glicko2.pdf) uses
tau=.5 explicitly, with an initial rating of 1500 and RD 200. The adapter returns
rating 1464.0506705, RD 151.5165241 and volatility .0599959843, consistent with
the paper's rounded example. The paper describes rating periods; it does not
establish this football workflow's event-period choice as an optimal setting.

## Replay and prediction timing

`build_ratings` expects the complete paired home/away rows from
`build_team_history`. One match is processed once, with both teams updating from
the same preceding states. Events are replayed by result-availability time, then
kickoff/identity. Equal-release-time events form a simultaneous batch; a team
with several events in that batch receives one multi-opponent update.

The default is **event-driven periods**, with no automatic calendar inactivity
inflation. Lookup also adds no idle inflation. The optional engine method
`advance_periods(state, periods, cap_to_prior=True)` exposes legacy-compatible
idle advancement; a caller must choose its period mapping and avoid counting an
active update period twice. It is not inserted into replay implicitly.

`scope=("competition_id",)` keeps competitions separate and crosses seasons.
Use `scope=()` to follow a team across competitions, or add `"season_id"` to
reset independently by season. Rating nodes own this scope; the evaluator's
`group_by` controls lag/rolling history. `H2H` around a rating node is currently
unsupported and raises an error.

Availability timestamps are **fully optional**. The default is a retrospective
kickoff proxy, not evidence of historical result-completion or publication time.
Explicit availability must not precede kickoff, and both perspectives must
agree. Missing availability excludes that match's update.

`cutoffs=None` uses each target's kickoff. A saved state is eligible when its
recorded time is at or before cutoff and every contributing kickoff is strictly
earlier. Earlier results released exactly at cutoff are usable; target/same-kickoff
results are excluded. `latest_kickoff_at` retains cumulative contributions through
both the team's and its opponents' prior states, including delayed releases.
Unrated teams receive the configured initial state; missing query times produce
missing output. See [cutoff formats and frozen rounds](feature_cutoffs.md).

## Build once, save full state, select columns later

```python
from pathlib import Path
from xdiyo_analytics.ratings import Glicko2, build_ratings, RatingRun
from xdiyo_analytics.features import Rating

run = build_ratings(history, engine=Glicko2(), stat=corners)
directory = Path("experiment/my_rating_definition_v1")  # A new destination.
run.save(directory)
loaded = RatingRun.load(directory)

X = evaluate_features(history, {
    "corner_strength": Rating("saved_corners", fields=("rating", "rd")),
}, ratings={"saved_corners": loaded})
```

Saving writes `snapshots.parquet` and `ratings.json` and refuses existing
artifacts. All state fields survive, independently of the fields selected for
the model. `loaded.features(query_rows, cutoffs=..., side="both", fields=...)`
also works directly. Build from the complete history; query a subset afterward.

Generated expressions cache each source/engine/scope/direction stream within
one evaluation. To reuse across calls, supply a `RatingRun`. Stored runs use
their own recorded times; pass `available_at` when building the run or evaluating
generated nodes. It does not rewrite the timestamps of a supplied run.

Keep the source selection and rating definition with the experiment. New or
revised history requires replay and a new saved artifact; incremental appending
and source-version reconciliation are not implemented. The version-pinned
[quickstart notebook](../../notebooks/03_ratings_quickstart.ipynb) reuses the two
verified runs under `experiment/ratings_demo/Premier_League_24_25/`.

## Custom ratings and future embeddings

`PairwiseRatingEngine` requires `initial_state()` and
`update_period(state, games)`, where each game contains an outcome and a
pre-period opponent state. Return the same finite numeric fields on every update.

Other producers can supply `RatingRun(snapshots, initial_state, scope, streams,
metadata)` directly. Snapshots identify stream, scope, team, recorded time and
latest contributing kickoff; each embedding dimension can be a named numeric
field. `Rating(name, fields=None)` exports all these fields, or selects a subset.
The producer owns fitting boundaries and valid availability/provenance.

This supports storing and querying future learned states. Graph attention
network training, graph updates, embedding generation and temporal cross-fitting
are separate future work; the pairwise protocol does not implement them.

## Verification

The focused suite passes **37 tests** and the full analytics suite **178**.
Permanent tests include the official example, exact legacy parity for 36 rating
periods and 288 explicit idle comparisons, simultaneous prior-state updates,
delayed releases, scopes, periods, large IDs, custom engines/vectors and storage.

The pinned 380-match Premier League example produces **760 snapshots per stream**
and a **760 × 13** feature frame. Independent raw-Parquet replay and **38,000 lookup
cells** agree within documented solver precision; saved/reloaded features are
exactly equal. Default-tolerance public-field differences from a high-precision
bisection reference are below 0.0000065. With solver tolerance 1e-12, they fall
below 1.3e-11. These precision checks are distinct from exact legacy-engine parity.

The package wheel builds and imports in an isolated interpreter without installs.
See [verification evidence, source hashes and saved runs](ratings_check.json).
