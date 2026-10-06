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

## Separate catalogue compatibility repair

The spatial repair is commit `bf0307a`. A separate follow-up reconciles only
`ColumnTransformer.force_int_remainder_cols` with the installed constructor,
including its actual default. Constructor inspection is cached at registration;
ordinary saved-schema reads remain cached. Existing field help and unrelated
catalogue entries remain unchanged. This does not rewrite recipes or remove
recipe-construction validation, and does not claim that a removed constructor
argument can still be passed to scikit-learn 1.9.

The installed runtime is scikit-learn 1.9.0. Earlier presence/default contracts
are tested with synthetic constructors (boolean and `"deprecated"` defaults),
not by installing 1.8.0. The reported older-runtime failure cannot occur when
its constructor exposes the field: the schema now includes it. A stale inventory
cannot expose that removed option on a constructor without it.

```powershell
& 'C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe' -m pytest tests/analytics/test_ui_catalog_compatibility.py tests/analytics/test_ui_inventory.py tests/analytics/test_ui_workflow.py tests/analytics/test_point_geometry_extensions.py -q -p no:cacheprovider
```

Same environment variables as above. Result: **64 passed**, no warnings or
failures. No remaining failures in the executed validation suites.

## Changed files

Spatial implementation and presentation:

- `src/xdiyo_analytics/features/point_geometry.py`
- `src/xdiyo_analytics/features/spatial_lineage.py`
- `src/xdiyo_analytics/features/evaluation.py`
- `src/xdiyo_analytics/reporting/heatmaps.py`
- `src/xdiyo_analytics/reporting/spatial_view.py`
- `src/xdiyo_analytics/ui/build_inventory.py`
- `src/xdiyo_analytics/ui/inventory.json`

Spatial tests and documentation:

- `tests/analytics/test_point_geometry.py`
- `tests/analytics/test_point_geometry_extensions.py`
- `tests/analytics/test_spatial_arithmetic.py`
- `docs/analytics/spatial_point_summaries.md`
- `docs/analytics/spatial_phase_a_repair_validation.md`

Separate compatibility follow-up:

- `src/xdiyo_analytics/ui/catalog.py`
- `tests/analytics/test_ui_catalog_compatibility.py`
- This validation record.

Unrelated collector/data and feature-inventory working-tree changes were not
included. Only Phase B was pushed; these follow-up commits remain local.
