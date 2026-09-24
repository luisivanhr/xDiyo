# Load and select football data for an experiment

Use `load_seasons` for an experiment spanning leagues and seasons, and
`inspect_season` to browse one publication's tables and columns.
Use `select_stats` to combine statistic bundles and exact measures after loading.

```python
from pathlib import Path
from xdiyo_analytics.data import inspect_season, load_seasons

project = Path("C:/Users/luisi/Documents/Programming/Python/xDiyo")
data_root = project / "data/xDiyo_data"
catalogue = inspect_season(data_root, "Premier_League_24_25")
catalogue[["rows", "column_count", "description"]]

experiment = load_seasons(
    data_root, ["22_23", "23_24", "24_25"], leagues=None,
    tables=["matches", "statistics", "pregame"],
    record_dir=project / "experiment/initial_population/selections",
)
experiment.matches.head()
experiment["statistics"].columns.tolist()
```

`leagues=None` uses all available leagues; pass an exact name such as
`"Premier_League"` or a list to narrow the selection. Tables stay separate, with
original values, nullable types and within-publication row order. Added
`source_league` and `source_season` columns identify each row. A table name alone
also works: `tables="statistics"`. Shots are independently selectable.

The first import on 16 September 2026 loaded **39 publications across 13 leagues**:
**13,976 matches**, **3,303,574 statistics rows** and **27,072 pregame rows**.
All 13 leagues have a publication for each selected season. These are collected
cohort counts, not a provider-wide fixture completeness claim.

## Choose statistics and standings

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

Selections form a union and preserve group identity, observations, nulls and row
order. Each category has one list shared by all periods: `*_all` keeps every
supplied period, while totals and halves filter `ALL`, `1ST` and `2ND`.
Totals are not sums of halves. `all_stats` includes uncategorized measures.
Defense means observed defensive metrics, not calculated opponent statistics.
Standings stay separate; matches pass through when loaded. Categories are
editable through `STAT_CATEGORIES` or a per-call `categories=` mapping.

- [Short notebook](../notebooks/01_loader_walkthrough.ipynb)
- [Loading and saved selections](../docs/analytics/season_loading.md)
- [Table descriptions and column discovery](../docs/analytics/season_inspection.md)
- [Statistic bundles and category definitions](../docs/analytics/stat_selection.md)
- [Recorded first-import evidence](../docs/analytics/multiseason_import_check.json)

`record_dir` saves one selection per publication outside the source directory.
Records pin versions; later discovery may include newly available publications.
Use `load_season(..., record_path=...)` for one explicitly named publication.
Full-file fingerprint checking is optional with `verify_hashes=True`.
