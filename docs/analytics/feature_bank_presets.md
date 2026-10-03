# Reusable feature banks and fixture context

`FeatureBankPreset`, `RestDays`, `CalendarFeature` and `evaluate_context_features`
are public imports from `xdiyo_analytics.features`. Notebook19/20's helper now
delegates to them; notebook configuration cells and saved outputs are unchanged.

## Ordered feature-bank preset

```python
from xdiyo_analytics.features import (
    FeatureBankPreset, SeededEMA, Hard, RestDays, evaluate_features,
    combine_features, evaluate_context_features,
)
from xdiyo_analytics.ratings import GlickoTransition

preset = FeatureBankPreset(
    windows=(3, 5, 10, 20), lags=(1, 2, 3), spans=(5, 10),
    periods=("ALL", "1ST", "2ND"),
    include_loo=True, loo_windows=(3, 5, 10), loo_reducers=("mean", "std", "z"),
    include_h2h=True, h2h_windows=(3, 5, 10), h2h_reducers=("mean", "std", "z"),
    warm_policy=SeededEMA(alpha=0.5, handoff=Hard(rounds=1), league_weight=0.5),
    rating_warm_policy=GlickoTransition(phi_scale=1.1, shrinkage=0.5),
)
# identities must come from the chosen development population, not test coverage.
definitions = preset.build(identities)  # ordered (period, group, key) triples
if preset.include_rest:
    definitions["rest_days"] = RestDays()
features = evaluate_features(history, definitions, keyed=True)
# After the usual label creation and assemble_dataset(..., layout="match"):
derived = preset.postassembly_definitions(
    features.columns, dataset.X.columns,
    feature_scopes=features.attrs['feature_scopes'],
)
dataset.X = combine_features(dataset.X, derived)
calendar = evaluate_context_features(dataset.metadata, preset.calendar_definitions())
```

Concatenate `calendar` after existing predictors, preserving its index. Numeric
dtype, infinity handling and fitted identity indicators are separate preparation
choices. The preset itself does not inspect labels, fit selectors, create folds,
discover team memberships or infer identity vocabularies.

The default bank preserves notebook19's original choices:

- Configurable `(group,key)` `stats` shortlist and `periods`. The default list has
  22 statistics; see the [UI mapping](xgboost_classifier_ui_mapping.md#4-features).
- ALL periods: for/against Lag1/2/3, mean3/5/10/20, std5/10, Z5, EMA5/10.
- Half periods: `half_keys` limits the statistics; `half_lags=(1,)` and
  `half_windows=(5,10)` limit their operations. `std_windows` and `z_windows`
  control the ALL spread/Z families. Mean/lag/EMA windows remain independently configurable.
- League corners use completed rounds and `league_windows=(3,5)` for mean/std.
  LOO and H2H default to mean3/5; notebook20 explicitly enables mean/std/Z3/5/10.
- Normalized standings and optional result/corners rating+RD streams.
- Warm policies default to None. When enabled, additional `warm::` columns retain
  every baseline; `warm_stat_keys` can restrict the warmed rolling statistics.
  A rating transition policy affects independent warmed streams only.
- `include_rest`, `include_calendar`, `include_combinations` default True.
  `.build()` returns historical expressions only; rest is explicitly appended by
  the caller. The other switches make their corresponding definition maps empty.

Names and insertion order are stable. `.postassembly_definitions()` preserves
the original helper's home/away sums/differences for names containing mean5,
mean10, standings, Glicko or rest; trends subtract mean20 from mean3. It requires
match layout and preserves the source columns. This naming-based combination
policy is explicit; it is not an arbitrary automatic cross-product of features.

Combinations apply only to **team-level outputs**. Recipe preparation passes the
evaluator's `feature_scopes` metadata automatically, including in exported Python
and notebooks. For direct calls, pass that mapping as shown above; its keys are
the evaluated history column names, before match-layout prefixes. Omitted scope
entries retain the legacy team-level behavior.

For example, `BayesianFixture(fields=('expected_home_goals',))` named `goal_mean5`
remains one `fixture::goal_mean5` predictor and creates no home/away combinations.
A team rolling mean such as `ALL_Passes_accuratePasses_for_mean5` still produces
`home::...`, `away::...`, `sum::...` and `difference::...`. Scope metadata takes
precedence over alias patterns; values are never used to infer scope. The same
rule excludes fixture outputs from mean3-minus-mean20 trends.

## Rest days and calendar

For a fixture with kickoff \(t_i\), let \(E_i\) be its eligible historical rows
under the evaluator's grouping, prediction cutoff and result-availability rule.
Then

\[
\operatorname{RestDays}_i =
\frac{t_i-t_{\operatorname{last}(E_i)}}{24\text{ hours}}.
\]

Without eligible history the value is missing. An earlier prediction cutoff can
change which previous fixture is eligible; the interval still ends at the target
kickoff. Equal-time, unfinished and unavailable results do not enter the prior.
`H2H(RestDays())` restricts the eligible prior to the same ordered team/opponent
pair. The default RestDays crosses seasons within a team and competition.

`CalendarFeature(kind=...)` reads known fixture metadata:

\[
c_{\sin}=\sin(2\pi m/12),\qquad
c_{\cos}=\cos(2\pi m/12),
\]

where \(m\) is UTC calendar month1–12. `weekday` uses Monday0–Sunday6. `round`
converts the round field to numeric, leaving nonnumeric labels missing. Missing
dates stay missing. Evaluate calendar expressions directly in `evaluate_features`
when team-level copies are desired, or use `evaluate_context_features(metadata,
definitions)` after assembly for one shared calendar column per match.

## Hierarchical inventory

`preset.inventory(dataset.X.columns)` returns grouped counts by derived/direct
stage, baseline/warmed family, computation, window and period, with statistic,
for/against, home/away and rating-field sets. Unknown names are explicitly counted
as unclassified. The ordered-column SHA256 identifies schema ordering; it is not
a data-content fingerprint. It accepts actual selected columns as well as a full
prepared schema, so the two can be compared without printing thousands of rows.

The [verified classifier inventory](feature_inventory/README.md) distinguishes
the latest completed winner's **2,145 selected / 2,681 prepared** columns from
the current notebook's **2,681 configured inputs with selection disabled**.
Their expression bank and ordered prepared schema match; their grid and selection
settings differ. JSON inventories and the read-only reproducer are included.

Verification covers default formula/order preservation, custom family windows,
home/away combinations, cutoff/availability/H2H behavior, calendar formulas and
the unchanged notebook helpers' synthetic classification/quantile pipelines.
