# Full-statistics auxiliary quantiles

This opt-in extension stays on codex/gat-boundary-memory-stat-forecast.
It does not alter existing frozen experiments or enable memory/CNN/reconstruction.

## Native target contract

NumericStatisticTarget adds exact directed own-team columns, a semantic name,
fixture side, period/component, units, support, transform, fitting-only scaling,
quantile levels and weight. Only quantile heads are currently supported by this
generic type. Existing StatisticTarget recipes and numerical parameterization
remain unchanged.

Supports: count (nonnegative integer labels), nonnegative continuous, signed
continuous, and bounded continuous with finite increasing bounds. Count predictions
remain continuous quantiles, without rounding. Signed logits remain signed.
Nonnegative quantiles use sorted logits then softplus; bounded quantiles use sorted
logits then sigmoid mapped to the declared bounds. Quantiles can tie.

Transforms: identity, log1p, signed_log1p, affine. Inverses are identity, expm1,
sign(x)*expm1(abs(x)), and lower + range*x respectively. Affine is required for
bounded targets. Plain log1p is forbidden for signed support. Scaling is explicitly
standard deviation, robust interquartile range, or none; scales at most 1e-12 become
one. Centers/scales use only finite positive-weight **fitting** observations.
Validation and test never fit these statistics. Centering cancels in pinball residuals.

Missing observations are masked; missing columns, incorrect numeric types,
invalid support and source aliases repeated on the same fixture side raise.
A target with no positive-weight fitting observations raises. A completely missing
validation auxiliary population is unavailable (NaN), not a zero loss. No partial
imputation, hidden target selection, or outcome-only substitution is performed.

Loss averages quantiles, then weighted observed rows independently for each target,
then takes the convex target-weighted mean. Global auxiliary weight is applied once.
Empty target contributions are omitted arithmetically without changing configured
target weights; an entirely empty validation score remains unavailable.

## Full 74-channel configuration

full_totals_auxiliary.json is an executable native catalogue node plus the
supplied provenance, ordered home 37 then away 37: **74 targets, 222 outputs**.
All weights are one; quantiles are 0.25, 0.5, 0.75. No opponent duplicates,
Bayesian channels, availability flags or masks become targets.

The source mappings are the handoff's frozen expanded-totals mappings. Raw provider
source_item_json examples are retained in source_units_evidence.json:

- Possession and duelWonPercent::value are percentages in [0,100], affine.
- aerialDuelsPercentage, groundDuelsPercentage, dribblesPercentage and
  wonTacklePercent **value components are successful-event counts**, despite their
  names. Total components are attempt counts. Counts use log1p.
- Signed and absolute outfield covariance are separate continuous targets, identity
  transformed; units are products of normalized pitch coordinates on 0..100 axes,
  not metres squared. Native features/point_geometry.py supplies those units.
- These declarations do not create or repair source columns. The frozen directed
  history must contain them; no period substitution or alias guessing occurs.

Use recipe.full_totals_arms(base_config, frozen_bayesian_columns) with the accepted
expanded-totals configuration and its exact frozen 11 Bayesian fixture predictors.
It returns configurations for the two arms. Input masks and scalar features remain
owned by that configuration. Use selection_control(max_steps=..., ...) with an
inner validation population. validation_loss is probability-head multiclass CE:
auxiliary/combined losses are tracked separately and cannot govern patience or
checkpoint selection under this control. Derive each arm's refit budget from its
own new selection; this example does not supply a preselected budget.

    from xdiyo_analytics.graphs import NumericStatisticTarget, StatisticsForecast
    target = NumericStatisticTarget(
        semantic_name="outfield_cov_xy", source_column="team::outfield_cov_xy",
        units="squared normalized pitch-coordinate units (0..100 axes)",
        support="signed_continuous", transform="identity", scaling="standard",
        side="home", period="ALL", component="outfield_cov_xy",
        quantiles=(.25, .5, .75), weight=1.,
    )
    forecast = StatisticsForecast(targets=(target,), route="auxiliary", weight=1.)

The UI target list accepts both native target types and previews K and quantile
output counts. JSON/Python exports preserve the full schema. Fitted model state
includes ordered target/source identity, original units, transforms, scale/center,
positive-weight fitting coverage, label masks, levels and output order. These
participate in model identity. Loading reordered or changed target schemas or
inconsistent transform manifests fails. Forecast reports retain original units.

## Scope and acceptance

Only bounded synthetic CPU fits and read-only source/preparation checks are part of
this implementation. No GPU audit, predictive improvement, optimizer-resume
compatibility or research experiment result is claimed. See verification.md for
test results and preparation measurements. No frozen artifacts are rewritten.
