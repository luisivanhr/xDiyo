# Arithmetic features and fitted identity indicators

The public feature library provides `Column`, `Constant`, `Abs`, `Sum`, `Product`, `Difference`,
`Ratio`, `combine_features` and `IdentityIndicators` for reusable numeric
combinations and categorical team/league indicators.

The experiment builder's feature catalog exposes `Constant`, `Abs`, `Sum`, `Product`, `Difference`
and `Ratio`, including nested operands and a numeric zero-denominator fallback.
The builder also exposes `Column` and arithmetic under **Derived match features**
for prepared columns. It does not silently add identity columns.

## Historical expressions

For per-observation interactions and paired weighted histories, see
[Interactions and weighted historical means](weighted_feature_interactions.md).
`RollingWeightedMean(source, weights, ...)` is available in the native feature
builder; `Product` remains the pointwise interaction primitive.

```python
from xdiyo_analytics.features import (
    Stat, ForAgainst, RollingMean, Constant, Sum, Product, Difference, Ratio,
    evaluate_features,
)

corners = Stat("ALL", "Match overview", "cornerKicks")
own = RollingMean(corners, window=5)
conceded = RollingMean(ForAgainst(corners, side="against"), window=5)
features = evaluate_features(history, {
    "combined_level": Sum(own, conceded),
    "interaction": Product(own, conceded),
    "balance": Difference(own, conceded),
    "for_against_ratio": Ratio(own, conceded),
    "recent_minus_long": Difference(own, RollingMean(corners, window=20)),
    "offset": Sum(own, Constant(1)),
}, keyed=True)
```

Finite numeric scalars can replace `Constant`, for example `Sum(own, 1)`. For
row-aligned values \(a_i,b_i\),

\[
\operatorname{Sum}(a,b)_i=a_i+b_i,
\qquad
\operatorname{Difference}(a,b)_i=a_i-b_i,
\]

\[
\operatorname{Ratio}(a,b)_i=
\begin{cases}
a_i/b_i,&a_i,b_i\text{ finite and }b_i\ne0,\\
c,&a_i,b_i\text{ finite and }b_i=0\text{ with finite fallback }c,\\
\mathrm{NaN},&\text{otherwise}.
\end{cases}
\]

`Ratio(..., zero_value=None)` produces missing output for a zero denominator.
A finite fallback replaces only that case; it does not fill missing/infinite
observations. Nonfinite results, including overflow, also become missing.
Constants and explicit fallbacks must be finite.

`Product(left, right)` multiplies row-aligned values: `a[i] * b[i]`.
Use `Product(own, 2)` to scale a feature or `Product(Product(a, b), c)` for
three factors. Missing/nonfinite inputs remain missing even when the other
factor is zero; overflow becomes missing. Both the named-feature and derived
match-feature UI catalogs expose Product, with normal JSON/Python/notebook exports.

Each operand must produce exactly one column. Choose an explicit period,
perspective and rating field rather than implicitly combining expanded outputs.
Temporal arithmetic respects existing history/cutoff rules: `Sum(Stat(...), 0)`
cannot turn the current observed statistic into a prediction feature. Operands
must have compatible history scopes; apply H2H to the combined expression
consistently. Nested temporal operators retain historical-row semantics.

### Absolute value

`Abs(source)` takes the absolute value of one output column. Negative finite
values become positive, zero stays zero, and missing/nonfinite values remain
missing. It does not impute values or change their historical eligibility.

```python
from xdiyo_analytics.features import Abs, Difference, RollingMean, SpatialPointSummary

offset = Difference(SpatialPointSummary("mean_x", side="for"), 50)
features = {
    "mean_distance_from_midfield": RollingMean(Abs(offset), window=20, min_periods=1),
    "distance_of_mean_position": Abs(RollingMean(offset, window=20, min_periods=1)),
}
```

These orders differ: offsets of -10 and +10 give a mean absolute offset of 10,
but an absolute mean offset of 0. `Abs` also composes with `Lag`, `RollingStd`,
`RollingSkewness`, `EMA`, H2H and arithmetic wrappers. Fixed eligible-match
windows still include missing maps as slots. Raw observed summaries inside
`Abs` remain unsafe until historically wrapped.

Native evaluation retains spatial source identities, coverage and recursive
calculation metadata; the heatmap reporter labels the result as an absolute
value. Team/fixture scope is inherited from the source. Both feature catalogs
expose `Abs`, and JSON, Python and notebook recipe exports retain it. For an
already assembled predictor, use `Abs(Column("home::corner_mean5"))` in
`combine_features` or select **Abs** under **Derived match features**.

## Combining already assembled columns

```python
from xdiyo_analytics.features import Column, Sum, Difference, Ratio, combine_features

X = combine_features(X, {
    "corners_level_sum": Sum(Column("home::corner_mean5"), Column("away::corner_mean5")),
    "corners_level_difference": Difference(Column("home::corner_mean5"), Column("away::corner_mean5")),
    "home_away_ratio": Ratio(Column("home::corner_mean5"), Column("away::corner_mean5")),
})
```

Substitute actual column names. The function returns a new frame and keeps old
columns by default; `keep_existing=False` returns only the new ones. Input values
and attributes are not mutated. Index order, explicit match/team keys and duplicate
row labels are preserved. Definitions are added to `attrs["derived_features"]`.

`Column` reads the supplied original feature frame. Nest calculations instead of
referencing other new outputs in the same call. Conflicting names are rejected.
This function does **not** establish temporal eligibility: its inputs must already
be legitimate prediction features. It must not move labels into model inputs.

## Fitted team and league indicators

```python
from xdiyo_analytics.features import IdentityIndicators

encoder = IdentityIndicators(
    columns=("home_id", "away_id", "source_league"),
    prefixes={"home_id": "home_team", "away_id": "away_team", "source_league": "league"},
)
encoder.fit(training_metadata)
train_indicators = encoder.transform(training_metadata)
test_indicators = encoder.transform(test_metadata)
```

For each fitted category \(c\),

\[
I_c(x_i)=\begin{cases}1,&x_i=c,\\0,&\text{otherwise}.\end{cases}
\]

Missing and unseen values produce all-zero indicators without expanding the
fitted schema. Category order follows first occurrence in fitting rows.
`categories_` and `feature_names_` expose that schema. Large integer IDs are
compared exactly and stay categorical; they are not float-valued magnitudes.
Output-name collisions are rejected; prefixes distinguish home/away categories.
Team-match layouts can use `team_id`/`opponent_id` instead.

This DataFrame helper is **not an automatically fitted model preprocessor**.
Fit it on the applicable training population, then transform evaluation/future
rows. Fold-specific encoding requires a fresh helper fitted inside each training
fold. `TrainingRunner` does not automatically attach/refit it. Preserve the fitted
encoder alongside the model for future predictions.

## Notebook migration and verification

The shared notebook 19/20 helper now uses these APIs for its existing home/away
sums, differences, short-minus-long trends and league indicators. It preserves
names, ordering, formulas and float32 conversion. The league encoder fits only
outer-training metadata, sorted by league name to retain the previous alphabetic
column order. No team dummy columns or ratios are added automatically.

Derived definitions and fitted identity categories are recorded in
`dataset.definitions` and preparation configuration for run identity. Notebook
files and user outputs were not changed by this migration.

**103 affected tests passed in 9.24 seconds**, including arithmetic/ratio edge
cases, temporal cutoff isolation, raw-stat and multi-output rejection, shuffled
keyed assembly, exact large IDs, unseen categories and existing historical,
warm-up and notebook tests. A separate synthetic notebook copy completed Run All
in **6.47 seconds**, including direct parity checks against every previous pandas
sum/difference/trend formula before the float32 cast. It retained 293 prepared
inputs, with the existing 80% test selector requesting 235, and exercised reports
and model reload. No full real-data grid was executed. The proof is under
`.pytest_tmp/classifier_composition_notebook_check/`.
