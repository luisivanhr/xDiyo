# Discover tables and columns

```python
from pathlib import Path
from xdiyo_analytics.data import inspect_season

data_root = Path("C:/Users/luisi/Documents/Programming/Python/xDiyo/data/xDiyo_data")
catalogue = inspect_season(data_root, "Premier_League_24_25")

catalogue[["rows", "column_count", "description"]]
catalogue.loc["statistics", "columns"]         # List of names
catalogue.loc["pregame", "column_types"]       # Name-to-type dictionary
catalogue.loc["shots", "column_summary"]       # Readable names and types
```

There is one catalogue row per available table. The fields are `rows`,
`column_count`, `description`, `columns`, `column_types` and `column_summary`.
Names, physical order and types come from the selected files. Inspection reads
metadata rather than observations; it does not perform a whole-file audit.

Use `record_path` to inspect an existing saved selection. Inspection never
creates a record or directory. A missing file or record raises
`FileNotFoundError`; inconsistent or unreadable metadata raises `ValueError`.

## Table reference

These 18 descriptions explain known table roles. The catalogue tells you which
tables are present in the selected season; others may be absent or empty.
Unfamiliar tables are listed with their actual columns and a fallback description.

| Table | Description |
| --- | --- |
| `matches` | Match identity, teams, competition, season, scheduled kickoff and scores. |
| `statistics` | Team statistics by match, period, group, statistic key and side; measures are key values, not separate columns. |
| `shots` | Individual shots: players, classifications, timing, coordinates, expected-goal values and quality indicators. |
| `pregame` | Match-specific team positions in position, plus provider form/context in value_json; not a complete league standings table. |
| `coverage` | Collection status and latest attempt status for match components and sides. |
| `availability` | Reported player availability/absence context, reasons, status and classification. |
| `lineup_teams` | Team lineup context: formation, roster status, phase and confirmation. |
| `lineup_players` | Player lineup entries: identity, playing position, shirt number, phase and substitute flag. |
| `player_statistics` | Player statistics by match/team/phase and key, with numeric and JSON values. |
| `heatmap_points` | Team spatial points with coordinates, weights, kind and source order. |
| `incidents` | Match incidents: type/class, player, side and reported timing. |
| `goal_sequences` | Goal-linked sequences: scorer/team, classification, action count, timing and shot linkage. |
| `goal_actions` | Actions within goal-linked sequences: actors, recipients/evidence, coordinates and completion/assist flags. |
| `passing_edges` | Player-to-player links within goal sequences, with evidence and action ordering. |
| `timing` | Period timing evidence, source values, quality and added-time/elapsed-time candidates. |
| `comments` | Commentary text with reported timing, side, source order and source payload. |
| `team_seasons` | Team-season membership/movement evidence, including promotion/relegation flags when supplied. |
| `legacy_fields` | Preserved legacy keys and JSON values with source context. |

## Where team positions belong

`pregame.position` is the provider's match-specific team position, with `event_id`,
`side` and `team_id` providing context. The original team context is retained in
`value_json`. This is not a complete league standings table. Missing pregame rows
or positions remain missing.

`lineup_players.position` means a player's playing position. Collection timestamps
do not prove that pregame values were available at a historical prediction cutoff.

## Optional coverage

`coverage` is optional. Loading and inspection work for exports that never
declare it, and for a new version that omits both its descriptor and file while
older versions remain intact. The catalogue follows the selected manifest.
Explicitly requesting an absent coverage table from either loader raises
`KeyError`. Do not remove a collected file behind an unchanged manifest; omit the
descriptor and file together in a future export. The initial multi-season
experiment selects matches, statistics and pregame only.

Use [load_season or load_seasons](season_loading.md) to load the chosen tables. Measures such as
`totalShotsOnGoal` are values of `statistics.key`, not physical column names;
select their period and group deliberately. Keep individual `shots` separate.
