# Season inspection and standings

Use `inspect_season` to discover a published season's tables before loading data.
The function handles version selection internally. It returns a pandas catalogue
indexed by table name, with one row per declared table.

```python
from pathlib import Path
from xdiyo_analytics.data import inspect_season, load_season_table

data_root = Path("C:/Users/luisi/Documents/Programming/Python/xDiyo/data/xDiyo_data")
catalogue = inspect_season(data_root, "Premier_League_24_25")

catalogue.index.tolist()                    # Available tables
catalogue.loc["matches", "columns"]          # Ordered column-name list
catalogue.loc["shots", "column_types"]       # Name-to-Arrow-type dictionary
catalogue.loc["pregame", "column_summary"]   # Names and types in a string
catalogue.loc["pregame", "description"]      # Documented table role
catalogue[["declared_rows", "column_count", "description"]]
```

## Table reference

These descriptions match the table roles documented by `inspect_season` in
`src/xdiyo_analytics/data/inspection.py`. This is a reference of known tables;
each season's catalogue determines which are actually available. A listed table
may be absent or empty in a particular export.

| Table | Description |
| --- | --- |
| `matches` | Match identity, teams, competition, season, scheduled kickoff and scores. |
| `statistics` | Team statistics by match, period, group, statistic key and side; measures are key values, not separate columns. |
| `shots` | Individual shots: players, classifications, timing, coordinates, expected-goal values and quality indicators. |
| `pregame` | Match-specific team positions in `position`, plus provider form/context in `value_json`; not a complete league standings table. |
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

The shots table remains a separate input for models that need individual shot
events. Select statistics by their period, group and key rather than combining
repeated measure names across groups. See [Where standings belong](#where-standings-belong)
for the distinction between pregame team positions and player lineup positions.

When adding or changing a table description in the library, update this reference
in the same change so notebook discovery and written documentation stay aligned.

## Return fields

| Field | Meaning |
| --- | --- |
| `description` | Documented role of a known schema-2 table; explicit fallback for unknown tables. |
| `declared_rows` | Manifest row count, checked against the Parquet footer. |
| `file_bytes` | File size, checked against the manifest. |
| `column_count` | Number of physical columns. |
| `columns` | Ordered list of physical column names. |
| `column_types` | Dictionary of column names and stored Arrow type strings, including nested types. |
| `column_summary` | Comma-separated `name: type` string for display or reporting. |

The descriptions explain table roles; they do not guarantee that every field is
present, populated or historically available. Actual columns always come from the
selected files. Unknown tables and typed empty tables remain visible.

## Use the same selection for inspection and model inputs

```python
record = Path("experiment/pl_24_25.json")
matches = load_season_table(data_root, "Premier_League_24_25", record_path=record)
catalogue = inspect_season(data_root, "Premier_League_24_25", record_path=record)
shots = load_season_table(data_root, "Premier_League_24_25", table="shots", record_path=record)
```

The first successful loader call creates the record. Inspection only reads an
**existing** record; a missing record raises `FileNotFoundError` without creating
directories. Keep records outside `data_root`. The saved selection can be reused
after the public pointer advances, but its pinned source files must remain available.
Without a record, each call resolves the publication current at that call.

## What inspection checks

Inspection validates publication metadata through the existing resolver, bounds
table paths to the selected version directory, and compares file sizes and footer
row counts with the manifest. It reads schemas without decoding observation rows.
Missing files raise `FileNotFoundError`; invalid metadata, escaped paths, mismatched
sizes/counts, unreadable footers and repeated column names raise `ValueError`.

**It does not verify whole-table hashes or validate row values.** A readable footer
does not prove that data pages are intact. Use `load_season_table` for verified data
reads. Source metadata and these limits appear in `catalogue.attrs['xdiyo']`;
ordinary pandas transformations may discard attributes. Inspection writes no files.

## Where standings belong

Standings-related positions currently belong to **`pregame.position`**. The collector
copies each team's `position` from the provider's pregame-form response and keeps
the team's full response object in `value_json`.

```python
pregame = load_season_table(
    data_root, "Premier_League_24_25", table="pregame", record_path=record,
)
pregame[["event_id", "side", "team_id", "position"]].head()
```

| Column | Meaning |
| --- | --- |
| `event_id` | Match identifier. |
| `side` | Home or away team. |
| `team_id` | Team identifier. |
| `position` | Provider's match-specific team position, preserved without normalization. |
| `value_json` | Original team context, which can include form, average rating and other provider values. |
| `observed_at` | Collection timestamp; not proof of availability before the historical match. |

The intended pregame grain is `(event_id, side)`. This is not a complete standings
table with every team's points, played games and goal difference at each round.
The inspected Premier League 2024/25 selection declares no separate `standings`
table. It has 740 pregame rows for 370 of 380 matches, with no missing positions
among those present rows. The absent 10 matches remain absent evidence; they are
not assigned a fallback position. These counts describe that saved selection.

`lineup_players.position` has a different meaning: a player's playing position.
Any league-table reconstruction, position normalization or historical cutoff policy
needs separate design. Collector references: `scraping/sofascore/parsers.py` pregame
branch and `scraping/sofascore/schema.py` pregame schema.

## Statistics keys are not physical columns

Measures such as `totalShotsOnGoal` are values in `statistics.key`. Discovering
available measures requires reading statistics and inspecting combinations of
`period`, `group_name` and `key`. The inspector lists the physical schema only.
Keep shot-event tables and model-specific statistics selections separate.

An executed walkthrough of the actual function and examples is in
[notebook Checkpoint 5](../../notebooks/01_loader_walkthrough.ipynb).
