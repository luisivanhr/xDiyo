# Load the tables your experiment needs

```python
from pathlib import Path
from xdiyo_analytics.data import inspect_season, load_seasons

project = Path("C:/Users/luisi/Documents/Programming/Python/xDiyo")
data_root = project / "data/xDiyo_data"
seasons = ["22_23", "23_24", "24_25"]

catalogue = inspect_season(data_root, "Premier_League_24_25")
catalogue[["rows", "column_count", "description"]]

experiment = load_seasons(
    data_root, seasons, leagues=None,
    tables=["matches", "statistics", "pregame"],
    record_dir=project / "experiment/initial_population/selections",
)
experiment.matches.head()
experiment.statistics.head()
experiment["pregame"].head()
```

## Select the population and tables

`seasons` accepts a consecutive-year label such as `"24_25"` or a list/tuple of
distinct labels. `leagues=None` discovers all leagues from top-level publication
sidecars. Supply an exact league name, or a list/tuple of names, to filter them.
Publications are loaded in the supplied season order, then league-name order.
An entirely missing requested season or league raises an error. Individual
league-season combinations may be absent; no rows are fabricated for them.
An invalid or incomplete selected publication raises instead of disappearing.

`tables` accepts a single name or a list/tuple of distinct names and defaults to
`"matches"`. Statistics or shots may be loaded alone. Exactly the requested
tables are returned, separately and in request order, in `experiment.tables`.

The convenient `.matches`, `.statistics`, `.pregame` and `.shots` properties return
their table or `None` when unrequested. Bracket access works for every table:
`experiment["shots"]`. Requested unavailable tables raise an error; missing data
is never replaced with an invented table. `experiment.provenance['sources']`
lists the exact publications and versions loaded by this call.

Each row gains Arrow string `source_league` and `source_season` columns. Existing
columns with either name raise a collision error. Source values, source column
order, within-publication row order, nullable Arrow types and nested values are
preserved. Optional columns are combined across publications, staying missing
where absent; the combined table gets a fresh index. There is no global row
deduplication, joining, filling or normalization. Choose a statistic by
its period, group and key: the same key may appear in several groups. The `Shots`
statistics group is separate from the individual-event `shots` table.

## Reuse saved selections

The example saves one version-1 record named `<league>_<season>.json` in
`record_dir`, outside `data_root`. Each successful partition creates its record;
later calls reuse it without overwriting it. A later failed partition can leave
earlier successful records in place. Choose a new directory for fresh versions.

Records pin versions, **not discovery membership**. A newly available matching
publication may join a later call, and removed sidecars are no longer discovered.
Retain the returned source list with experiment evidence to document its exact
population. Records contain selection metadata, not copies of the data; keep
the selected source versions available.

### One publication

```python
from xdiyo_analytics.data import load_season

season_name = "Premier_League_24_25"
selection = project / "experiment/initial_population/selections/Premier_League_24_25.json"
season = load_season(
    data_root, season_name, tables=["matches", "statistics", "pregame"],
    record_path=selection,
)
```

For `load_season`, a new record is saved only after the entire single-publication
load succeeds. Omit `record_path` when no saved selection is needed.

`inspect_season(..., record_path=selection)` can reuse an existing record, but
inspection does not create one. Existing `load_season_table` calls remain a thin
single-table shortcut through the same loader.

## First import — 16 September 2026

All available leagues for `22_23`, `23_24` and `24_25` produced **39 publications
across 13 leagues**, with no missing league-season combinations among those
leagues. The separate frames contain:

| Table | Rows | Columns including origin labels |
| --- | ---: | ---: |
| matches | 13,976 | 37 |
| statistics | 3,303,574 | 20 |
| pregame | 27,072 | 10 |

The [executed notebook](../../notebooks/01_loader_walkthrough.ipynb) displays
match counts by league and season. The [import evidence](multiseason_import_check.json)
records all versions, row counts, 39 selection records and missing combinations.
Every loaded table partition matched its direct source Parquet data, including
values, row order, types and nulls. These are collected cohorts, not an independent
check of complete provider fixtures. No features, joins or model fitting occurred.

## Checks and errors

Readable files, size/count consistency and basic match/statistic structure are
checked automatically. Cross-table checks run when matches and statistics are
requested together. Add `verify_hashes=True` when you want a whole-file
fingerprint audit. Neither mode establishes complete provider coverage or
historical prediction-time availability.

Unavailable table names raise `KeyError`, missing files raise `FileNotFoundError`,
and invalid selections or unusable data raise `ValueError`. No new selection
record is saved for a partition whose read or basic check fails. Earlier successful
partition records can remain after a later failure. Source data is never written.

Start with the [short notebook](../../notebooks/01_loader_walkthrough.ipynb) or the
[table and column reference](season_inspection.md).
