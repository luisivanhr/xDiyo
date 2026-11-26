# Prepared scoreline history

This is a private, immutable xDiyo input. It materializes paired goal rows and
one movement record per competition, season and team. Reading the output does
not join the original providers or fit a model.

```python
from xdiyo_analytics.data.scoreline_history import load_scoreline_history

history, team_seasons = load_scoreline_history("scorelines_history")
inputs = {
    "history": history,
    "team_seasons": team_seasons,
    "available_at": "replay_available_at",
}
# For feature/prediction APIs with a cutoffs= argument, use "prediction_at".
# No model is run here.
```

- Native IDs stay numeric and unchanged. Historical seasons and fixtures use
  deterministic signed-64-bit local IDs. These are not SofaScore identifiers.
  Ambiguous annual archives (Liga MX and Colombia) have a distinct annual local
  competition rather than being assigned to an arbitrary native phase.
- Canonical season labels are inclusive year pairs such as `2019/2020` or
  `2022/2022`. Native exact kickoff remains available. Archive `kickoff_at`
  is the start of the reported analysis day, an ordering bucket needed by the
  library. It does not claim exact kickoff; archive `kickoff_utc` stays null.
- The built-in `result_available_at` lower bound mirrors the prior-day release
  guard, not an observed publication timestamp. `prediction_at` is the UTC
  analysis start-of-day cutoff. No same-day score is consumed.
- Existing native movement labels take precedence. Otherwise, use observed
  membership in the immediately preceding season. A club returning to a
  second division after a gap, with prior second-division history and no
  known preceding-season membership, is treated as promoted from below.
  Both predecessor IDs remain null in this case, and whenever the prior club
  has no results in the prepared history. No per-entry provenance
  or fictitious lower division is added.
- Cohort selection is unchanged. The prepared view uses the already-approved
  replay population only. The deferred index identifies broad snapshot
  fixtures outside that population; synthetic club IDs do not promote them
  into the prepared history. The original score-basis labels remain intact.
  Do not declare this input regulation-certified or use it for intraday forecasts.

To reproduce from the pinned corrected snapshot, the synthetic-ID overlay
Parquet, and its associated native publications:

```sh
python -m xdiyo_analytics.data.scoreline_history \
  --scorelines scorelines_with_archive_ids.parquet \
  --parent-manifest path/to/corrected-scorelines/manifest.json \
  --native-root path/to/pinned/native-matches \
  --league-inventory docs/analytics/data/library_league_inventory_20261116.csv \
  --output data/scorelines/prepared-v1
```

Keep generated parquet outside Git. The manifest pins the inputs and output
bytes and the loader checks both consumed tables by default.
