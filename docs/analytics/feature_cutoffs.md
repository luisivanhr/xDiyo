# Optional feature cutoffs

`evaluate_features` accepts `cutoffs=None` by default. You do not need to provide
result-availability timestamps. Examples below assume `history` comes from
`build_team_history` and contains the selected full-match corner statistic.

```python
import pandas as pd
from xdiyo_analytics.features import Stat, RollingMean, evaluate_features

expressions = {
    "corners_mean_5": RollingMean(
        Stat("ALL", "Match overview", "cornerKicks"), window=5,
    ),
}
```

## Default: no explicit prediction time

```python
features = evaluate_features(history, expressions)  # cutoffs=None
```

Each row uses its kickoff as the boundary for earlier finished matches. The
match being predicted and simultaneous kickoffs are excluded. With no explicit
availability times, earlier finished kickoffs are the retrospective assumption;
they do not establish exact result-completion or publication times.

## Predict two days before each match

Pass a pandas datetime Series with one cutoff per history row and the same index:

```python
cutoffs = history["kickoff_at"] - pd.Timedelta(days=2)

features = evaluate_features(history, expressions, cutoffs=cutoffs)
```

The timestamps remain timezone-aware UTC. Both team rows of a match receive the
same cutoff. Every cutoff must be at or before its row's kickoff. Missing target
times give missing historical features.

## Store the times in a column

A string names a datetime column in the supplied history:

```python
history_with_cutoffs = history.assign(
    prediction_at=history["kickoff_at"] - pd.Timedelta(days=2),
)
features = evaluate_features(
    history_with_cutoffs, expressions, cutoffs="prediction_at",
)
```

In particular, a bare date string passed as `cutoffs` is interpreted as a column
name, not a timestamp. For a scalar timestamp use
`pd.Timestamp("2024-08-14 12:00:00", tz="UTC")`; it applies to **every** row and
must be no later than any target kickoff. An aligned Series is usually the useful
choice when evaluating several seasons together. Datetime sequences in row order
are also accepted. Convert numeric Unix timestamps with an explicit unit first.

## Freeze the predictions for one round

```python
round_start = history.groupby(
    ["competition_id", "season_id", "round"],
)["kickoff_at"].transform("min")

features = evaluate_features(
    history, expressions,
    cutoffs=round_start - pd.Timedelta(days=2),
)
```

This example issues predictions two days before the earliest stored kickoff in
each round. It assumes usable round labels; stage distinctions and postponed-game
schedule policies should be reflected in the grouping/time definitions you choose.

Cutoffs affect historical operators. Current-match home flags and pregame
standings remain context; this API does not reconstruct changes to that context
as of a past issuance time. `available_at` is a separate optional datetime input
for historical observations whose availability times are known.
