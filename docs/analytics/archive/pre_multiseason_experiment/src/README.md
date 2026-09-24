# Load football data for an experiment

Use `inspect_season` to see available tables and `load_season` to load the ones you need.

```python
from pathlib import Path
from xdiyo_analytics.data import inspect_season, load_season

data_root = Path("C:/Users/luisi/Documents/Programming/Python/xDiyo/data/xDiyo_data")
catalogue = inspect_season(data_root, "Premier_League_24_25")
catalogue[["rows", "column_count", "description"]]

season = load_season(
    data_root, "Premier_League_24_25", tables=["matches", "statistics", "pregame"],
)
season.matches.head()
season["statistics"].columns.tolist()
```

Tables stay separate, with original values, nullable types and row order.
A table name alone also works: `tables="statistics"`.

- [Short notebook](../notebooks/01_loader_walkthrough.ipynb)
- [Loading and saved selections](../docs/analytics/season_loading.md)
- [Table descriptions and column discovery](../docs/analytics/season_inspection.md)

Add `record_path` outside the source directory when an experiment should reuse
the same season selection later. Full-file fingerprint checking is optional with
`verify_hashes=True`. Existing `load_season_table` callers use the same loading path.
