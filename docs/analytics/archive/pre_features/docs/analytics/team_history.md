# Build observed team histories

```python
from xdiyo_analytics.histories import build_team_history

history = build_team_history(selected)
history[["event_id", "side", "team_id", "opponent_id", "kickoff_at",
         "goals_for", "goals_against", "result"]].head()
```

Pass a `select_stats` result or loaded `SeasonData` with matches and optional
statistics/pregame. The returned DataFrame has **two rows per match**, one for
each team's perspective. Matches remain even when all their statistics,
positions, scores or dates are missing. Source frames and files are unchanged.

## Row identity, time and context

`source_league` and `source_season`, when present, combine with `event_id` to
identify a match. The same event ID in another partition stays separate.
`side` identifies the original home/away perspective. Each row contains:

| Columns | Meaning |
| --- | --- |
| `team_id`, `opponent_id` | The current perspective and its opponent. |
| `team_name`, `opponent_name` | Names when supplied by matches. |
| `kickoff_at` | Full timezone-aware UTC datetime converted from `kickoff_utc` Unix seconds. |
| `goals_for`, `goals_against` | The corresponding provider `home_score_current` / `away_score_current` values. |
| `result` | Score-based `W`, `D` or `L` only when status is `finished` and both current scores exist. |
| `team_position`, `opponent_position` | Match-specific pregame positions, when pregame was supplied. |

Available competition/tournament/season identifiers, season year, round and
status fields also pass through. The frame is ordered globally by UTC kickoff,
then partition/event identity and side, with a fresh index. Filter by team ID
when inspecting one team's rows.

**Time is not normalized to midnight.** Hours, minutes and seconds remain in
`kickoff_at`. Missing or unrepresentable timestamps become `NaT` and sort last.
The timestamp refers to scheduled/revised provider kickoff; it does not establish
when a result or statistic became available.

Current scores remain present for unfinished or unknown statuses, but `result`
stays missing. A tied current score yields `D` even if penalty scores differ;
the function does not resolve a shootout winner or substitute normal-time scores.

## Statistic columns

Each supplied statistic becomes separate team and opponent columns:

```text
team::period::group::key::field
opponent::period::group::key::field
```

For example, `team::ALL::Match overview::totalShotsOnGoal::value` and
`team::ALL::Shots::totalShotsOnGoal::value` remain distinct. Periods, groups and
keys are preserved; the builder accepts arbitrary selected measures. The
opponent column uses the other side's observed value for that same match and
identity. It does not calculate a new defensive statistic.

Components are percent-escaped so embedded `::` and `%` cannot collide with the
column format. Spaces and underscores remain readable. Use
`history.attrs['stat_columns']` to recover exact original labels: each column
maps to `role`, `period`, `group_name`, `key` and `field`.
`history.attrs['source']` retains the input selection provenance.

`stat_fields` defaults to `("value",)`. Include available `total` or `display`
explicitly when needed:

```python
history = build_team_history(selected, stat_fields=("value", "total", "display"))
```

Fields retain their supplied values and missingness; display strings are not
parsed and totals are not used to scale values. Missing observations on one or
both sides remain missing. An absent statistics table contributes no statistic
columns; an absent pregame table contributes no position columns.

## Errors and next steps

Missing required tables/columns raise `KeyError`. Invalid field choices,
duplicate match/observation identities, observations outside the supplied
match/side population and supplied team IDs that disagree with a match side
raise errors. Statistics are pivoted without aggregation; duplicates are never
averaged or silently discarded.

This is a table of **observed outcomes**, not prediction-ready features. Lags,
rolling windows, historical availability, cutoffs and model fitting remain
separate work.

See the [statistic-selection guide](stat_selection.md),
[executed notebook](../../notebooks/01_loader_walkthrough.ipynb) and
[verification evidence](team_history_check.json).
