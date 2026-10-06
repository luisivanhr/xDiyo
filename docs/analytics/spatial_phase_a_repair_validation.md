# Phase A geometry and arithmetic repair validation

Date: 6 October 2026. Phase B was committed and pushed as `e37ca7a` first.
This work adds the optional section 5.3 fields and repairs arithmetic lineage.
It leaves between-match variability to `RollingStd` and does not change default
feature recipes, fit models, download archives or claim predictive improvement.

## Before and after

The evaluation dispatch loaded from commit `e37ca7a` was compared with the
repaired dispatch in the same process, using current point-source/report helpers.
The three fixtures and expressions are the exact repair-prompt reproduction.
All numerical assertions passed on both paths:

| Expression | Before: descriptor/audit/report | After |
|---|---|---|
| `RollingMean(SpatialPointSummary(),20)` | Present / present / passes | Passes |
| `RollingMean(Product(SpatialPointSummary(),Constant(2)),20)` | Missing / missing / raises | All present; passes |
| `Product(RollingMean(SpatialPointSummary(),20),Constant(2))` | Missing / missing / raises | All present; passes |

The old failure was `HeatmapReporter needs declared spatial feature metadata`.
The new regressions assert the expected `[NaN, NaN, 100, 100, 100, 100]` values
for the two transformed expressions, their neutral descriptors, audit counts,
assembly and reporting when each is the only feature.

## Contract and tests

See [spatial point summaries](spatial_point_summaries.md) for formulas, units,
degeneracy tolerances, rotation, angle caveats and multi-source coverage semantics.
The new coverage tests distinguish raw selection identity, each historical
expression/window, and final output finiteness. They cover all four arithmetic
operators, constants, distinct point selections, nested/shared expressions,
missing/empty/low-count maps, zero denominators/fallbacks, overflow, subset reports,
shuffled keyed rows, H2H restrictions, causal boundaries, cache order independence,
JSON/Python recipes, prepared Parquet and native recovery bundles.

The geometry tests cover repeated points, linear 10–90% quantiles, covariance,
ordered eigenvalues/spreads, anisotropy, principal angle, singular/zero/isotropic
maps, 180-degree rotation invariance, all optional UI field choices and recipe
round-trips. Existing tests retain the baseline six predictors, default kinds,
timing/availability, finite-value minimum periods and fixed eligible windows.

## Environment and commands

Windows project interpreter:
`C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe`.
Python 3.14.0, NumPy 2.4.6, pandas 2.3.3, scikit-learn 1.9.0,
PyArrow 22.0.0, pytest 9.1.1.

From the repository root:

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe' -m xdiyo_analytics.ui.build_inventory
& 'C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe' -m pytest tests/analytics/test_point_geometry.py tests/analytics/test_point_geometry_extensions.py tests/analytics/test_spatial_arithmetic.py tests/analytics/test_feature_composition.py tests/analytics/test_features.py tests/analytics/test_heatmaps.py tests/analytics/test_spatial_fixture_features.py tests/analytics/test_spatial_distribution.py tests/analytics/test_datasets.py tests/analytics/test_keyed_features.py tests/analytics/test_ui_inventory.py tests/analytics/test_ui_recipe_cutoffs.py tests/analytics/test_ui_feature_discovery.py tests/analytics/test_ui_workflow.py tests/analytics/test_table_attribute_recovery.py tests/analytics/test_ui_preparation_parity.py -q -p no:cacheprovider
```

Result: **414 passed**, with two expected float32 overflow warnings from the
existing UI preparation parity cases. No failures or skips. A subsequent focused
geometry run also checks repeated non-binary decimal coordinates for degenerate
axes (those extended fields use exact zero variance for constant coordinates).
That command was `python -m pytest tests/analytics/test_point_geometry.py
tests/analytics/test_point_geometry_extensions.py tests/analytics/test_spatial_arithmetic.py
tests/analytics/test_spatial_distribution.py -q -p no:cacheprovider`, using the same
interpreter/environment: **116 passed**, no warnings or failures.

The saved inventory diff is restricted to `SpatialPointSummary` choices/help.
The viewer's `JS` string also passes `node --check` via standard input. This is a
source/syntax and generated-artifact validation, not a live browser inspection.
No external adversarial review suite or full repository test run is claimed.
