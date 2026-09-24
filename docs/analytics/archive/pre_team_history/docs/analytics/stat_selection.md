# Select statistics for an experiment

Start with a `load_season` or `load_seasons` result containing `statistics` and,
when needed, `pregame`. Discover names with `list_stat_bundles()`:

```python
from xdiyo_analytics.data import list_stat_bundles, select_stats

list_stat_bundles()
selected = select_stats(
    experiment,
    bundles=["attack_totals", "defense_first_half", "standings"],
    stats=[("ALL", "Match overview", "ballPossession")],
)
selected.statistics.head()
selected.pregame.head()
```

Bundles and exact `(period, group_name, key)` triples form a **union**. Overlapping
requests do not repeat a selected row. The same key in different groups stays
distinct. Original observations, missing values, column types, row order and
indices remain intact. Missing event/side observations are not fabricated.

The result is another `SeasonData`: matches pass through when loaded, selected
statistics remain in `.statistics`, and standings remain in `.pregame`.
Individual shots are selected separately. There is no pivot, aggregation, join,
opponent-statistic calculation, history construction or feature fitting.

## Categories and periods

Attack and defense each use one period-independent list of `(group_name, key)`
pairs. Their `*_all` selection contains every supplied period; the other variants
filter that same selection.

| Category | Every supplied period | Provider `ALL` | Provider `1ST` | Provider `2ND` |
| --- | --- | --- | --- | --- |
| Every statistic | `all_all` | `all_totals` | `all_first_half` | `all_second_half` |
| Attack | `attack_all` | `attack_totals` | `attack_first_half` | `attack_second_half` |
| Defense | `defense_all` | `defense_totals` | `defense_first_half` | `defense_second_half` |

`all_stats` aliases `all_all`; `all_totals_only`, `all_first_period_only` and
`all_second_period_only` alias the corresponding `all_*` period filters.
All-stat selections include uncategorized and newly supplied measures.

**Totals use stored `ALL` observations; they never sum the halves.** A statistic
supplied only in `1ST` can still appear in the first-half selection. Additional
provider periods also remain in `*_all`.

### Initial category definitions

These are editable starting classifications, not a claim that every measure is
available in every publication. Defense means **observed defensive metrics**;
it does not compute the opponent's attack or swap team sides.

| Category | Group | Exact keys |
| --- | --- | --- |
| attack | Attack | `accurateThroughBall`, `bigChanceMissed`, `bigChanceScored`, `fouledFinalThird`, `offsides`, `touchesInOppBox` |
| attack | Shots | `blockedScoringAttempt`, `expectedGoals`, `expectedGoalsOnTarget`, `hitWoodwork`, `shotsOffGoal`, `shotsOnGoal`, `totalShotsInsideBox`, `totalShotsOnGoal`, `totalShotsOutsideBox` |
| attack | Match overview | `bigChanceCreated`, `cornerKicks`, `expectedGoals`, `totalShotsOnGoal` |
| attack | Passes | `accurateCross`, `finalThirdEntries`, `finalThirdPhaseStatistic` |
| defense | Defending | `ballRecovery`, `errorsLeadToGoal`, `errorsLeadToShot`, `interceptionWon`, `totalClearance`, `totalTackle`, `wonTacklePercent` |
| defense | Goalkeeping | `diveSaves`, `goalkeeperSaves`, `goalsPrevented`, `highClaims`, `penaltySaves`, `punches` |
| defense | Match overview | `goalkeeperSaves`, `totalTackle` |

Inspect or edit the defaults through public `STAT_CATEGORIES`. For one call,
replace named categories or add new ones with `categories=`; other defaults
remain available:

```python
categories = {
    "attack": [("Shots", "shotsOnGoal")],
    "possession": [("Match overview", "ballPossession")],
}
list_stat_bundles(categories=categories)
selected = select_stats(
    experiment, bundles=["attack_first_half", "possession_all"],
    categories=categories,
)
```

There are no separate per-period category lists. `all` is reserved for every
available statistic and does not depend on category membership.

## Standings and unavailable inputs

`standings` selects `event_id`, `side` and `position` from pregame, plus available
`team_id`, `observed_at`, `source_league` and `source_season`. It preserves the
provider's match-specific positions and missing rows/values; it does not build
a complete league table. It can be selected without loading statistics.

Unknown bundles, unavailable requested tables and an exact statistic absent
from the entire input raise `KeyError`. Category members absent from the input
are skipped; an empty category selection returns an empty statistics frame with
the original schema. Exact names and casing matter.

See the [executed notebook](../../notebooks/01_loader_walkthrough.ipynb),
[loading guide](season_loading.md) and [verification record](stat_selection_check.json).
