# Analytics source layout

New reusable code belongs in `xdiyo_analytics/`. The current directory is
`xdiyo_analytics/data/`, for prepared-export reading and validation.

The first module, `data/exports.py`, contains `ResolvedSeason`,
`resolve_season_export` and `read_season_table`. These capture one named season's
manifest metadata and read one verified table with nullable types preserved.
`data/matches.py` adds structural match validation with a timing-issue report.
`data/loading.py` connects these steps through the public convenience API:

```python
from xdiyo_analytics.data import load_season_table

matches = load_season_table(data_root, "Premier_League_24_25")
# Optional: save once and automatically reuse this season selection on later calls.
matches = load_season_table(
    data_root, "Premier_League_24_25", record_path="experiment/pl_24_25.json",
)
```

Users supply no versions or hashes. Records must be outside `data_root`; they
pin metadata, so retain the referenced source files. Each table loads separately.
With `load_season_table`, matches receive structural validation and statistics
validation is an explicit call described below. Source and validation details are
in `matches.attrs['xdiyo']`, which pandas
transformations may discard. Saved records provide persistent selection replay.

## Load several separate tables from one selection

```python
from xdiyo_analytics.data import load_season

season = load_season(
    data_root, "Premier_League_24_25", tables=("matches", "statistics", "pregame"),
    record_path="experiment/pl_24_25.json",
)
season.matches                            # Same object as season.tables['matches']
season.validation["statistics"]            # Automatic cross-table validation
season.shots                              # None: unrequested
```

`data/season.py` resolves once and returns exactly the selected tables in a
`SeasonData` bundle. Statistics require explicitly selected matches. Requested
missing tables fail; no tables are added implicitly. Matches/statistics receive
domain validation; pregame/shots receive file integrity checks with domain status
`not_implemented`. New records are saved only after every read and validation
succeeds, using the writer shared with `load_season_table`. See the
[season bundle loading guide](../docs/analytics/season_loading.md) for properties,
errors, record compatibility, copied metadata and mutable report snapshots.

## Discover tables and columns

```python
from xdiyo_analytics.data import inspect_season

catalogue = inspect_season(data_root, "Premier_League_24_25")
catalogue.index.tolist()                   # Table names
catalogue.loc["matches", "columns"]         # Column-name list
catalogue.loc["shots", "column_summary"]    # Names/types as a string
catalogue.loc["pregame", "description"]     # Readable table description
```

Inspection reads schemas and footer counts, not observation rows or whole-table
hashes. It can reuse an existing `record_path` but never creates one. Standings-related
team positions belong to `pregame.position`; a complete league standings table is
not supplied by that table. See the [season inspection and standings guide](../docs/analytics/season_inspection.md)
for return fields, errors, saved-selection examples and the standings distinction.

## Validate statistics against matches

```python
from xdiyo_analytics.data import validate_statistics

statistics = load_season_table(
    data_root, "Premier_League_24_25", table="statistics", record_path="experiment/pl_24_25.json",
)
report = validate_statistics(statistics, matches)
```

Use the same saved selection for both inputs. `data/statistics.py` checks exact
identities, full `(event_id, period, group_name, key, side)` grains, match
references and supplied team IDs. Its immutable report lists missing team/value
positions and match sides without any statistics. Inputs remain unchanged;
repeated keys in different groups stay separate. `load_season_table` does not call
this cross-table validator automatically; `load_season` does when matches and
statistics are selected together. See the
[statistics validation guide](../docs/analytics/statistics_validation.md) for
inputs, report fields, errors, examples and coverage/timing limits.

The package is mapped to this source directory in `pyproject.toml` and imports
normally in `.misc314`.
See [IMPLEMENTATION_PROGRESS.md](../IMPLEMENTATION_PROGRESS.md) for the contract,
future package responsibilities, evidence and next steps.

The walkthrough lives in `notebooks/01_loader_walkthrough.ipynb`; new behavior
checks will live in `tests/analytics/`. Notebook cells import reusable functions
as they are implemented. Existing scraping and utility code remain in place.
