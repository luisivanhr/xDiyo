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

## Build team histories

```python
from xdiyo_analytics.histories import build_team_history

history = build_team_history(selected)
history[["event_id", "side", "team_id", "opponent_id", "kickoff_at",
         "goals_for", "goals_against", "result"]].head()
```

The result has two observed rows per match with team/opponent statistics and
positions. Statistic columns preserve period/group/key identity; exact labels
are available in `history.attrs['stat_columns']`. UTC kickoff retains the full
time of day. Goals use current scores, and W/D/L requires a finished match with
both scores; penalty winners are not inferred. Missing observations retain rows.
Use the feature evaluator below to calculate historical predictors from these rows.

## Evaluate named features

```python
from xdiyo_analytics.features import (
    Stat, ForAgainst, Lag, RollingMean, IsHome, NormalizedStanding,
    evaluate_features, league_season_team_counts,
)

corners = Stat("ALL", "Match overview", "cornerKicks")
team_counts = league_season_team_counts(history)  # Full population before splits.
features = evaluate_features(history, {
    "home": IsHome(),
    "standing": NormalizedStanding(),
    "corners_lag1": Lag(ForAgainst(corners, "both")),
    "corners_mean5": RollingMean(corners, window=5),
}, team_counts=team_counts)
features.head()
```

Outputs keep the history index. Default history combines venues and crosses
seasons within team/competition. Windows count eligible matches; reductions use
finite values inside the window, while lags keep missing positions. Standings
normalize first place to 1 and last to 0, with missing/invalid values defaulting
to zero. The library also includes std, Z-score, H2H, nesting and optional EMA.

`cutoffs=None` and optional availability times need no extra inputs. Earlier
finished kickoffs are the default **retrospective availability proxy**; they do
not establish exact result-completion or publication times. For an earlier
boundary, pass `cutoffs=history["kickoff_at"] - pd.Timedelta(days=2)` after
importing pandas as `pd`. Current context is not reconstructed at that earlier time.

- [Short notebook](../notebooks/01_loader_walkthrough.ipynb)
- [Loading and saved selections](../docs/analytics/season_loading.md)
- [Table descriptions and column discovery](../docs/analytics/season_inspection.md)
- [Statistic bundles and category definitions](../docs/analytics/stat_selection.md)
- [Team-history columns and semantics](../docs/analytics/team_history.md)
- [Feature quickstart notebook](../notebooks/02_feature_quickstart.ipynb)
- [Feature expressions and timing rules](../docs/analytics/features.md)
- [Exact cutoff formats and frozen rounds](../docs/analytics/feature_cutoffs.md)
- [Feature verification evidence](../docs/analytics/features_check.json)
- [Recorded first-import evidence](../docs/analytics/multiseason_import_check.json)

`record_dir` saves one selection per publication outside the source directory.
Records pin versions; later discovery may include newly available publications.
Use `load_season(..., record_path=...)` for one explicitly named publication.
Full-file fingerprint checking is optional with `verify_hashes=True`.
