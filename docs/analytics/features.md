# Evaluate named football features

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
| `ForAgainst(stat, side="for")` | Select existing team (`for`), opponent (`against`) or both columns. Its source must be a `Stat`. |
| `H2H(expression)` | Restrict historical operations within the expression to the ordered team/opponent pair, respecting venue reversals. |
| `IsHome()` | Current row's known home context: home 1, away 0; unknown side is missing. |
| `NormalizedStanding(side="for", missing_value=0.0)` | Current row's supplied pregame standing, normalized using the full league-season team count. Also accepts `against` and `both`. |

`Stat` and `ForAgainst` cannot be output roots, including when wrapped only in
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
