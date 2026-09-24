# Load the tables your experiment needs

```python
from pathlib import Path
from xdiyo_analytics.data import inspect_season, load_season

project = Path("C:/Users/luisi/Documents/Programming/Python/xDiyo")
data_root = project / "data/xDiyo_data"
season_name = "Premier_League_24_25"

catalogue = inspect_season(data_root, season_name)
catalogue[["rows", "column_count", "description"]]

season = load_season(
    data_root, season_name, tables=["matches", "statistics", "pregame"],
)
season.matches.head()
season.statistics.head()
season["pregame"].head()
```

`tables` accepts a single name or a list/tuple of distinct names. It defaults to
`"matches"`. Statistics may be loaded alone. Exactly the requested tables are
returned, separately and in request order, in `season.tables`.

The convenient `.matches`, `.statistics`, `.pregame` and `.shots` properties return
their table or `None` when unrequested. Bracket access works for every table:
`season["shots"]`. Requested unavailable tables raise an error; missing data is
never replaced with an invented table. `season.provenance` provides a compact
source reference when needed.

Values, column/row order, nullable Arrow types and nested columns stay unchanged.
No joins, filling or normalization happen during loading. Choose a statistic by
its period, group and key: the same key may appear in several groups. The `Shots`
statistics group is separate from the individual-event `shots` table.

## Reuse a selection later

```python
selection = project / "experiments/my_experiment/pl_24_25.json"
season = load_season(
    data_root, season_name, tables=["matches", "statistics", "pregame"],
    record_path=selection,
)
```

The first successful whole load creates the record; later calls reuse it.
Keep the record outside `data_root`. Existing version-1 records remain compatible
and are never overwritten. Choose a new filename for a fresh selection. Retain
the source files: a selection record does not contain copies of the data.

`inspect_season(..., record_path=selection)` can reuse an existing record, but
inspection does not create one. Existing `load_season_table` calls remain a thin
single-table shortcut through the same loader.

## Checks and errors

Readable files, size/count consistency and basic match/statistic structure are
checked automatically. Cross-table checks run when matches and statistics are
requested together. Add `verify_hashes=True` when you want a whole-file
fingerprint audit. Neither mode establishes complete provider coverage or
historical prediction-time availability.

Unavailable table names raise `KeyError`, missing files raise `FileNotFoundError`,
and invalid selections or unusable data raise `ValueError`. No new selection
record is saved when a read or basic check fails. Source data is never written.

Start with the [short notebook](../../notebooks/01_loader_walkthrough.ipynb) or the
[table and column reference](season_inspection.md).
