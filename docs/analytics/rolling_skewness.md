# Rolling skewness

`RollingSkewness(source, window=5, min_periods=3)` measures asymmetry in
earlier eligible observations. Positive values indicate a longer upper tail;
negative values indicate a longer lower tail. It is available in the builder's
feature dropdown, statistic bundles, and Python/JSON/notebook recipe exports.

```python
from xdiyo_analytics.features import (
    Stat, RollingSkewness, League, LeaveOneOut, WarmStart, SeededEMA, LinearFade,
)

corners = Stat('ALL', 'Match overview', 'cornerKicks')
team_skew = RollingSkewness(corners, window=10)
loo_skew = RollingSkewness(LeaveOneOut(League(corners)), window=5)
warmed_skew = WarmStart(
    team_skew,
    SeededEMA(mode='uniform', alpha=.5, handoff=LinearFade(start=1, rounds=2)),
)
```

## Definition and missing values

For normalized observation weights, define the raw moments
`M1 = E[X]`, `M2 = E[X²]`, and `M3 = E[X³]`. Then:

```text
variance = M2 - M1²
third central moment = M3 - 3*M1*M2 + 2*M1³
skewness = third central moment / variance**1.5
```

Ordinary windows assign equal weight to each finite observation. This is
**population skewness**, equivalent to SciPy's `skew(..., bias=True)`, without
a small-sample correction. It has no `ddof` parameter. At least
`max(3, min_periods)` finite observations and positive variance are needed.
Undefined results remain missing, including constant windows.

The implementation uses centered moments and their stable mixture equations,
which are equivalent to combining the three raw moments but avoid cancellation
from subtracting large raw squares and cubes. It never averages skewness scores.

## Historical windows and LOO

The same team/competition grouping, cutoff, availability, H2H and missing-match
rules as `RollingStd` apply. Current and simultaneous matches cannot contribute.
A missing value still occupies a match position; an older match does not refill
the window. Without explicit availability, the existing earlier-finished-kickoff
proxy applies. Heatmaps are handled cell by cell; their venue selector is unchanged.

For League/LOO, choose the league window first, then exclude the focal team's
contributions or complete fixtures, then compute **all three moments from the
remaining observations**. Exclusion does not refill the window. Counts are
observations, not rounds, so a one-round window can have enough observations.
Match totals require `exclude='fixtures'` and count each complete total once.

## Warm starts

Every finite new observation updates the first three moments with the same EMA
weight. Missing observations are skipped. During a partial handoff, blend the
seeded and ordinary distributions' moments, including the between-means terms,
then normalize. Weight zero restores the exact ordinary rolling result. With
`alpha=1`, a single-point state has zero variance and missing skewness.

Supported policies follow the existing population boundaries:

- **uniform:** seed from the team's final eligible predecessor window.
- **w_league_prior:** for promoted/relegated teams, use the existing destination
  donor selection and fallback rules. Average donor means and within-team second
  and third central moments. This corresponds to recentering each donor's
  distribution to the cohort mean before pooling. Differences between donor
  means are excluded from the prior's shape, matching the within-team variance
  convention. Each donor needs at least `max(3, min_periods)` finite observations.
- **legacy:** keep the existing team/mover prior mean and destination-league
  variance; use that same league sample's third central moment. This translates
  the league distribution to the selected prior mean. League/LOO warm starts
  apply the same exclusion to both prior and current-season observations.

Explicit `uniform`/`w_league_prior` policies still reject pooled League/LOO
sources, as they do for standard deviation: a pooled population has no unique
team predecessor. Use ordinary LOO or `SeededEMA(mode='legacy')` for these sources.

Skewness always uses population moments. `variance_estimator`, `prior_strength`
and `variance_fade` continue to control SD/Z-score behavior; they do not change
skewness. In particular, skewness uses moment-mixture fade while the existing
explicit-mode SD/Z-score estimate-interpolation behavior is unchanged.
The explicit warm-start audit records the seed's third central moment and the
skewness estimator/fade convention.

## Feature bank

`FeatureBankPreset(skew_windows=(5, 10))` adds full-period team skewness columns.
Use `'skew'` in `loo_reducers` or `h2h_reducers` for the corresponding populations.
The default `skew_windows=()` preserves existing presets. New columns end in
`_skew5`, `_skew10`, etc.; home/away assembly follows the existing team-feature rules.
