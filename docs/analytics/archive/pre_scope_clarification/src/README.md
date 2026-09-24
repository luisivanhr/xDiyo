# Load football data for an experiment

Use `load_seasons` for an experiment spanning leagues and seasons, and
`inspect_season` to browse one publication's tables and columns.

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

- [Short notebook](../notebooks/01_loader_walkthrough.ipynb)
- [Loading and saved selections](../docs/analytics/season_loading.md)
- [Table descriptions and column discovery](../docs/analytics/season_inspection.md)
- [Recorded first-import evidence](../docs/analytics/multiseason_import_check.json)

`record_dir` saves one selection per publication outside the source directory.
Records pin versions; later discovery may include newly available publications.
Use `load_season(..., record_path=...)` for one explicitly named publication.
Full-file fingerprint checking is optional with `verify_hashes=True`.

`coverage` is optional and is not requested for this experiment. A future export
may omit both its descriptor and file; leave existing collected versions intact.
