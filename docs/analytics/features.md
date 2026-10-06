# Evaluate named football features

See [historical goals](match_scores.md) for native-current goals scored/conceded
using `RollingMean(MatchScore(...), window=20)` without a statistics table.

See [heatmap features and fixture views](heatmaps.md) for spatial grid pooling,
Gaussian smoothing, shared pitch orientation and `RegionMass` summaries.
See [exact point summaries](spatial_point_summaries.md) for historical mean
longitudinal position and longitudinal/lateral spatial standard deviations.
See [regional shares and spatial distributions](spatial_distributions.md) for
thirds/channels/rectangles, entropy and concentration summaries.
See [spatial distances and fitted map representations](spatial_extensions.md)
for Phase C/D features and configurable grid resolution.
`Lag`, `RollingMean`, `RollingStd`, `RollingSkewness`, `RollingZScore` and `EMA` accept
`venue="all"` or `venue="same"` for both ordinary and spatial team histories.

See [arithmetic composition and fitted identity indicators](feature_composition.md)
for reusable absolute values, sums, products, differences, ratios and categorical
team/league columns.

Build observed rows with `build_team_history`, then pass named expressions to
`evaluate_features`. By default the result has the same row order and index as
the history. Set `keyed=True` to retain the values/order with match/team/side
identifiers in a named MultiIndex for [dataset assembly](datasets.md).
The [identity option reference](datasets_reference.md#feature-identity-option)
describes its required keys and validation.

```python
from xdiyo_analytics.features import (
    Stat, ForAgainst, H2H, IsHome, NormalizedStanding, Lag,
    RollingMean, RollingStd, RollingZScore, evaluate_features,
    league_season_team_counts,
)

corners = Stat("ALL", "Match overview", "cornerKicks")
team_counts = league_season_team_counts(history)  # Full population before splits.
features = evaluate_features(history, {
    "home": IsHome(),
    "standing": NormalizedStanding(),
    "corners_lag1": Lag(ForAgainst(corners, "both")),
    "corners_mean5": RollingMean(corners, window=5),
    "corners_std5": RollingStd(corners, window=5),
    "corners_z5": RollingZScore(corners, window=5),
    "h2h_corners_lag1": Lag(H2H(corners)),
}, team_counts=team_counts)
```

No cutoff or availability input is required. The defaults use each target's
kickoff as its cutoff and earlier finished kickoffs as a **retrospective
availability proxy**. Stored kickoff/status does not establish the exact time
when a result or statistic was completed or published.

## Source selection and output names

| Expression | Meaning |
| --- | --- |
| `Stat(period, group, key, field="value")` | One exact observed statistic identity, from the team's perspective. Use numeric `value` or an explicitly selected numeric field such as `total`. |
| `Stat(None, group, key)` | Expand every supplied period separately. `ALL` remains the provider's full-match value; halves are not summed. |
| `MatchScore(score_field="current", side="for")` | Observed native-current goals; `for`, `against`, or `both`. Requires a historical operator. |
| `ForAgainst(source, side="for")` | Select team (`for`), opponent (`against`) or both columns for Stat, MatchScore, Heatmap or SpatialPointSummary. |
| `H2H(expression)` | Restrict historical operations within the expression to the ordered team/opponent pair, respecting venue reversals. |
| `IsHome()` | Current row's known home context: home 1, away 0; unknown side is missing. |
| `NormalizedStanding(side="for", missing_value=0.0)` | Current row's supplied pregame standing, normalized using the full league-season team count. Also accepts `against` and `both`. |

`Stat`, `MatchScore` and `ForAgainst` cannot be output roots, including when wrapped only in
`H2H`; put a historical operator around observed values. This prevents returning
the target match's realized statistic directly as a feature.

A one-column expression uses its requested name. Expanded expressions use
`requested_name::source_column`, for example
`corners_lag1::opponent::ALL::Match overview::cornerKicks::value`.
Context expansions use labels such as `standing::opponent_standing`.
Select source identities through `history.attrs['stat_columns']`; output attrs
record expressions, grouping and the availability mode. Inputs remain unchanged.

## Match windows and missing values

| Operator | Calculation and defaults |
| --- | --- |
| `Lag(source, periods=1)` | Value from the nth latest eligible match. Missing values do not renumber matches. |
| `RollingMean(source, window=5, min_periods=1)` | Mean of finite values inside the last `window` eligible matches. |
| `RollingStd(source, window=5, min_periods=1, ddof=1)` | Standard deviation with divisor `n - ddof`, requiring `n > ddof`. `ddof=0` gives the population form. |
| `RollingSkewness(source, window=5, min_periods=3)` | Third central moment divided by population variance to power 3/2. At least three finite observations and positive variance; supports [LOO and moment-based warm starts](rolling_skewness.md). |
| `RollingZScore(source, window=5, min_periods=1, ddof=1, reference=None)` | Defaults to the latest eligible value minus the window mean, divided by its std. An explicit historical/known-context reference can replace the numerator; League populations require it. |
| `EMA(source, span=5, min_periods=1)` | Optional exponentially weighted mean over eligible history, with alpha `2 / (span + 1)`. |

Windows count **matches**, including matches with a missing selected value.
Reductions skip missing/nonfinite values inside that window; they do not reach
farther back to fill it. `min_periods` counts finite observations. No eligible
values, insufficient observations, or an undefined std produces missing output.
A Z-score also stays missing when the latest match's value is missing or the
window has zero spread. There is no global zero imputation.

EMA starts at the first finite observation and skips missing observations
without decaying the previous value. It uses all eligible history, not a fixed
`span`-match truncation. It is available but is not selected as a pilot requirement;
it adds no prior, seasonal blending or warm-start policy.

Expressions can be nested, for example `RollingMean(Lag(corners), window=5)`.
Each historical child's value is evaluated at that historical row's own
configured cutoff, before the outer window uses it. Nesting can therefore
require more history and retain additional missing values.

## Eligibility, grouping and optional cutoffs

Historical candidates must be finished, have a known kickoff **strictly before**
the target cutoff, and, if `available_at` is supplied, have a known availability
time **at or before** that cutoff. The target match and simultaneous kickoffs at
the boundary are excluded. Missing target kickoff/cutoff gives no eligible
historical rows. The target itself need not already be finished.

The default `group_by=("team_id", "competition_id")` combines home/away history
and crosses seasons. Add `"season_id"` or `"side"` to restrict those dimensions.
`team_id` is required; `H2H` adds opponent identity. Group identities cannot be
missing. `eligible_history_rows(...)` exposes the same selection as integer row
positions, suitable for `.iloc`, even when index labels are duplicated.

```python
import pandas as pd

cutoffs = history["kickoff_at"] - pd.Timedelta(days=2)
earlier_features = evaluate_features(
    history, {"corners_mean5": RollingMean(corners, 5)}, cutoffs=cutoffs,
)
```

Cutoffs must be no later than the target kickoff. An aligned datetime Series,
datetime sequence, scalar timestamp or datetime column name is accepted;
`cutoffs=None` uses kickoff. A bare date string is a column name, so use
`pd.Timestamp(..., tz="UTC")` for a scalar. Datetimes are converted to UTC;
numeric timestamps require explicit-unit conversion first. `available_at` is a
separate, fully optional input with the same formats. Missing explicit
availability excludes the affected observation.

See [exact cutoff formats and frozen-round examples](feature_cutoffs.md).
Cutoffs govern historical operators. Context nodes use the supplied current
match context; they do not reconstruct standings as of an earlier issuance time.

## Standings use the full league-season population

For position `p` among `n` teams, the value is `1 - (p - 1) / (n - 1)`:
first place is 1 and last place is 0. Compute `league_season_team_counts(history)`
from the full selected population **before filtering teams or splitting folds**,
and pass those counts to evaluation. It counts distinct team IDs from both team
and opponent columns within competition/season; it does not estimate a denominator
from observed ranks. Without `team_counts=`, evaluation infers counts from the
history supplied to that call.

Missing/nonfinite/out-of-range positions, missing/nonfinite denominators and
denominators at most 1 use `missing_value`, default **0**. Consequently zero can
mean last place or fallback; it does not prove a standing was observed. This
standing-specific choice does not change missing-value rules for lag/rolling
features. No prior or missing-season proxy is applied.

## Checked example

League populations, window-before-exclusion LOO and explicit `WarmStart` policies
are now available. `SeededEMA` supports all three handoffs (`Hard`, `LinearFade`,
`ObservationCount`), with consistent mean/variance updates and unchanged unwrapped
defaults. See the [practical guide](league_warmup.md), [complete API/helper reference](league_warmup_reference.md),
[coverage checklist](league_warmup_documentation_checklist.md), and separate
[league/warm-up notebook](../../notebooks/04_league_warmup_quickstart.ipynb).

Rating expressions are also available: `MatchResultGlicko`, `StatGlicko(stat)`
and `Rating(name)` for a supplied saved/custom run. They export numeric state for
both sides into the same feature frame. Rating streams have their own scope and
replay semantics; see the [ratings and persistence guide](ratings.md) and
[ratings notebook](../../notebooks/03_ratings_quickstart.ipynb).

The [eight-cell notebook](../../notebooks/02_feature_quickstart.ipynb) uses the
pinned Premier League 2024/25 publication: 380 matches, 760 team rows and 20 teams.
The original feature batch passed **141 tests**, including 32 new feature cases;
the ratings extension brought that suite to **178 tests**. The league/transition
extension now passes **275 tests**, including 97 new independent cases; its
[evidence](league_warmup_check.json) covers two pinned seasons and all handoffs.
Independent real-data arithmetic matches 22 feature columns at default and
two-day-earlier cutoffs; the named-column cutoff matches the aligned Series.
The two-day change leaves these real feature values unchanged in this snapshot;
synthetic tests exercise cutoffs that change eligible history.

See [recorded verification and source hashes](features_check.json).
# Season progress

`SeasonProgress()` is known fixture context, named `season_progress` in the
example below. It does not need a lag or rolling wrapper. Choose `mode="rounds"`
(default) or `mode="kickoff"`.

## Assigned-round mode

For a fixture assigned to round \(r\) in a season containing \(R\) rounds:

\[
\operatorname{season\_progress}=\frac{r}{R}.
\]

The first round is \(1/R\), not zero; the final round is one. Both teams in the
same fixture receive the same value. This is a normalized **round position**,
not the fraction of matches completed or elapsed calendar time.

**Postponed fixtures retain their original assigned round.** For example, in a
38-round season, a round-5 fixture played during round 12 still receives
\(5/38 \approx 0.1316\), not \(12/38\). Moving its kickoff does not change this
feature. The source `round` column must retain that assignment; the library
does not reconstruct an original round if the provider has overwritten it.

By default, the denominator is the maximum numeric assigned round, separately
for each `(competition_id, season_id)`, in the **full loaded fixture schedule**.
Scheduled, unfinished and postponed fixtures are included. Neither observed
scores nor completion times enter the calculation. Evaluate before filtering
labels, prediction fixtures, folds, or report rows. The builder already evaluates
features before dropping missing targets and creating folds.

```python
from xdiyo_analytics.features import SeasonProgress, evaluate_features

features = evaluate_features(history, {
    "season_progress": SeasonProgress(),
})
# With a partial schedule, explicitly supply the scheduled season length:
features = evaluate_features(history, {
    "season_progress": SeasonProgress(total_rounds=38),
})
```

The automatic mode assumes that the loaded schedule includes the final assigned
round. It cannot detect an incomplete schedule, and does not infer the season
length from team counts or the number of completed matches. Use the override
for such inputs; it applies to every row evaluated with that expression. For
mixed season lengths, use complete schedules or evaluate each league-season
with its own explicit override. An abandoned season's last played round is not
necessarily its originally scheduled final round.

Round numbers must run consecutively across the intended season scale. Stages
that restart round numbering require an upstream consistent season-round scale;
an override alone cannot resolve that ambiguity. Missing/nonnumeric rounds give
missing values. Numeric rounds outside positive integers, or above an explicit
total, raise an error rather than being clipped. Missing league/season identity
also yields missing automatic progress. The override must be a positive integer.

**UI:** Features & ratings → add feature → **Season progress**. Name the output
`season_progress`, select **rounds**; leave **Total rounds override** disabled for inference, or
enable it and enter the planned total. Recipes, Python/notebook exports and saved
feature definitions use the same expression. Match layout emits equal Home/Away
columns, following the usual feature assembly contract.

## Kickoff-day mode

Let \(D\) be the fixture's UTC kickoff calendar date, \(D_0\) the first kickoff
date and \(D_1\) the last kickoff date in its league-season schedule. Then

\[
\operatorname{season\_progress}_{\mathrm{kickoff}}
=\frac{(D-D_0).\mathrm{days}+1}{(D_1-D_0).\mathrm{days}+1}.
\]

Both endpoints are inclusive: the opening day is day 1 and the final day has
progress 1. Time of day is ignored after conversion to UTC. A season whose
fixtures all fall on one day has progress 1 for every fixture.

**Postponed fixtures use the kickoff date when they actually take place in this
mode.** For example, for a schedule from January 1 through January 20, a round-5
fixture postponed until January 16 receives \(16/20=0.8\). A round-12 fixture
played January 11 receives \(11/20=0.55\). The same fixtures in rounds mode keep
their assigned-round values, irrespective of these dates. For an upcoming match,
the available scheduled kickoff date is used; no unknown actual date is guessed.

```python
features = evaluate_features(history, {
    "season_progress": SeasonProgress(mode="kickoff"),
})
# Optional boundaries for a partial schedule (both apply to every input row):
feature = SeasonProgress(mode="kickoff", start_date="2026-08-01", total_days=300)
```

In the UI, select **kickoff** to see **Total days** and **Start date** overrides.
Leave them disabled to infer the earliest and latest kickoff per league-season.
Start date is a date string; total days is a positive inclusive calendar-day
count. Either override may be used independently; remaining boundaries come
from the full schedule. Use both if the input omits the season's opening and
closing fixtures. `total_rounds` belongs only to rounds mode; `total_days` and
`start_date` belong only to kickoff mode.

Missing or unparseable kickoff dates remain missing. Unknown grouping identity
gives missing inferred progress. Dates outside explicit boundaries raise an
error. All-missing schedules remain missing.

**Schedule revisions:** inference uses the schedule currently supplied, including
future fixtures. If a postponement moves the final kickoff, the inferred duration
and other fixtures' normalized values may change on re-evaluation. To reproduce
what was knowable at an earlier prediction date, supply that schedule version or
freeze boundaries explicitly. A final historical schedule is not evidence that
all its revised dates were known in advance. Features are computed before fold
filtering; a split mask does not freeze historical schedule revisions.
