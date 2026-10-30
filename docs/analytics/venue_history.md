# Historical venue views

Berrar, Lopes and Dubitzky (2024), Section 5.5, equations (4)-(5) and Table 7
(printed pages 8181-8183), distinguish each team's recent total, home and away
matches. These are independent of which side the team takes in the target fixture.

```python
from xdiyo_analytics.features import MatchScore, RollingMean, DerivedStandings

features = {
    'scored_at_home': RollingMean(MatchScore(side='for'), 3, venue='home'),
    'conceded_at_home': RollingMean(MatchScore(side='against'), 3, venue='home'),
    'scored_away': RollingMean(MatchScore(side='for'), 3, venue='away'),
    'home_form': DerivedStandings([3], venue='home'),
    'away_form': DerivedStandings([3], venue='away'),
}
```

`for`/`against` selects the observed quantity. `venue` selects the focal team's
historical appearances before applying the operator's window:

| Venue | Eligible appearances |
| --- | --- |
| `all` (default) | Home and away |
| `same` | The same venue as the target team's current fixture |
| `home` | Home only, even if the target team currently plays away |
| `away` | Away only, even if the target team currently plays home |

The same choices work with `Stat`, `ForAgainst`, spatial summaries, `Lag`,
`RollingMean`, `RollingStd`, `RollingSkewness`, `RollingZScore`, `EMA` and
`RollingWeightedMean`, including H2H and supported warm-start compositions.
Fixed-venue warm starts filter predecessor and donor histories too. Existing
cutoff, availability, grouping and missing-slot rules remain: a missing value in
an eligible home match occupies a home-window slot; an away match does not.
EMA recurs only over the selected appearances. Direct league-population sources
still require `venue='all'`; venue selection here is a team-history operation.
Raw score/statistic sources remain unsafe predictors until historically wrapped.

For the paper's Liverpool example before Manchester City on 1 April 2023,
the last three home goals scored are 2, 2, 7 (mean 11/3), whereas the last three
away values are 2, 0, 0 (mean 2/3). Away goals conceded average 1/3 and home
goals conceded average zero. The target result is excluded.

`DerivedStandings` builds a separate full-roster form table from each team's
selected-venue results and then normalizes rank as `(N-R)/(N-1)`. It does not
average past ranks. Its season reset, complete-membership/schedule validation and
immediate eligible-result publication remain in force (see the versioned publication repair). Its `side='against'` selects the
opponent whose rank is returned; see [derived standings](derived_standings.md).

The builder exposes these venue options for ordinary and spatial histories and
derived rankings. JSON/Python exports preserve them. `examples/venue_history.py`
provides nine team features: scored, conceded and normalized form rank for all
venues (six matches), home only (three) and away only (three). Native match layout
produces **18 predictors**, retaining both current teams' three historical views.
The example prepares data without fitting a model. Its short-history settings and
season-scoped ranking follow xDiyo conventions; it is not a full experimental
reproduction of the paper.

```powershell
$env:PYTHONPATH='src'
python -m examples.venue_history --data-root data/xDiyo_data
```

## Verification (28 October 2026)

The focused historical-feature, standings, warm-start, weighted, spatial and UI
regression selection passed **497 tests**; two optional Playwright browser tests
were skipped because that Python package was absent. Separately, the in-app
browser loaded this native recipe, changed both score and ranking venue selectors,
restored and saved it, and produced JSON exactly equal to the original recipe,
with no console errors. This was a desktop UI check, not mobile coverage.

The example also prepared the published Premier League 2022/23 population:
**380 matches and 18 predictors**. No model was fitted. Regressions include the
paper's Liverpool values, independent venue-form-table arithmetic, late releases,
season resets, missing slots, future-mutation safety, warm-start priors,
JSON/Python recipe preparation, legacy/extended saved standings and spatial
reporter metadata. Existing default standings tests remain unchanged in behavior.
