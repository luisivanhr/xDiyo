# Native historical principal-axis channels

`SpatialPointSummary("axis_cos2")` and `SpatialPointSummary("axis_sin2")`
encode each match's principal axis as `(cos(2θ), sin(2θ))`. The raw axis is
unoriented: θ and θ+π describe the same line. Encoding happens before historical
averaging, using exactly the same selected points and validity rule for both
channels. Existing `axis_angle` and generic `RollingMean` behavior are unchanged.

## Working recipe

The complete reusable feature mapping is in
[`examples/axial_point_features.py`](../../examples/axial_point_features.py).
It supplies both native Python `feature_specs()` and JSON-compatible
`recipe_features()` mappings. Its example is exercised through native preparation
and exported Python in the regression tests. To print the builder feature mapping:

```powershell
$env:PYTHONPATH='src;.'
python examples/axial_point_features.py
```

Python configuration:

```python
from xdiyo_analytics.features import SpatialPointSummary, RollingMean

features = {
    f"activity_{field}_r20": RollingMean(
        SpatialPointSummary(field, kinds=("player",), side="for", min_points=1),
        window=20, min_periods=1, venue="all",
    )
    for field in ("axis_cos2", "axis_sin2")
}
```

In the builder, add **RollingMean → SpatialPointSummary** twice. Select
**Principal-axis cosine (double angle)** and **Principal-axis sine (double angle)**.
Use Player, For, All venues, Window 20 and Min periods 1 on both. The builder
loads `heatmap_points` automatically and serializes both fields in JSON/Python
exports. Add this mapping deliberately to a new recipe; the example does not
edit existing recipes or frozen experiments.

Fixture assembly produces exactly four predictors, in this order:

1. `home::activity_axis_cos2_r20`
2. `home::activity_axis_sin2_r20`
3. `away::activity_axis_cos2_r20`
4. `away::activity_axis_sin2_r20`

Feed the means C and S directly to the model. Do not normalize them back to unit
length. Do not pool historical point clouds or impute raw angles before encoding.

## Validity, orientation and time

The population covariance eigenvalue gap must exceed
`64 * float64_epsilon * covariance_trace`, exactly as for `axis_angle`.
Missing/empty/insufficient-point, constant, isotropic and numerically isotropic
maps produce NaN in both channels. A rank-one line has an identifiable axis.
No missing angle becomes a zero vector or an arbitrary eigenvector direction.

Both channels are dimensionless. θ is in radians modulo **π**, and the encoded
double angle has period **2π**. The metric is the exported normalized 0–100 x/y
coordinate system, not physical metres. A common rescaling is harmless; changing
x and y scales differently changes the geometry.

The canonical team frame and against-side 180° rotation are preserved. Rotating
180° changes neither channel. Reflecting either coordinate preserves cosine and
negates sine; swapping x/y negates cosine and preserves sine. Final target venue
does not rotate these scalar features.

Existing grouping, cutoffs, availability and fixed eligible-match window rules
apply unchanged. Missing or undefined maps occupy window slots. `min_periods`
counts finite inputs within those slots, with identical masks for both channels.
Current-match channels remain unsafe predictors until historically wrapped,
including when embedded in arithmetic. Spatial source/timing lineage, raw map
coverage, historical coverage and final usability remain attached.

## Optional reporting summaries

```python
from xdiyo_analytics.reporting import axial_direction_summary

# C and S must be prepared means with identical history settings.
summary = axial_direction_summary(C, S)
angle_radians = summary["mean_axis"]
R = summary["consistency"]
```

The helper accepts scalar or array inputs. It returns
`mean_axis = (0.5*atan2(S,C)) modulo π` and `consistency = hypot(C,S)`.
It never normalizes the pair or adds predictors. If either input is missing,
both summaries are missing. With finite inputs and **R ≤ 64 × float64 epsilon**
(about `1.42e-14`), the mean axis is undefined. R remains the computed finite
resultant, including zero. This absolute tolerance is roundoff-sized for means
of unit double-angle pairs; it is not a statistical significance threshold.

| Historical axes | Mean axis | R |
|---|---|---|
| 1°, 179° | Near 0° modulo 180° | 0.999390827 |
| Identical axes | That common axis | Approximately 1 |
| 0°, 90° | Undefined | Approximately 0 |

R measures **between-match directional consistency**. Existing `anisotropy`
describes **within-match shape**. R is not silently included as a fifth predictor.
The helper is opt-in; the fixture viewer displays each prepared channel directly
and does not infer pairs from aliases or combine potentially different windows.

## HeatmapReporter coverage

Every implemented spatial family has a renderer, verified for both fixture teams:

| Family | Visualization |
|---|---|
| `Heatmap`: pooled or Gaussian; mass/density/count | Pitch heatmap |
| `RegionMass`: every named region and custom rectangle | Shaded region(s) and exact prepared value |
| All `SpatialPointSummary` fields, including both axial channels | Labelled scalar value with units |
| Normalized/raw `SpatialEntropy` | Labelled scalar value |
| All six `SpatialConcentration` metrics | Labelled scalar value |
| Supported scalar spatial arithmetic | Derived value and ordered calculation lineage |

Unavailable values show “Feature unavailable”. The audit covers **42 feature
configurations** and executes the shipped JavaScript grid, region, scalar and
missing-value paths with DOM/Plotly test doubles. This verifies renderer dispatch
and prepared values; it is not a screenshot or live-browser visual review.
Earlier proposed APIs such as `ConcentrationDiff` and `WeightedPresence` are not
implemented native components and are not counted as supported features.

## Coverage performance repair

Coverage now calculates `np.isfinite(frame.to_numpy(dtype=float, na_value=np.nan))`
once per output frame, then reads the array mask for each audit record. It no
longer performs metadata-bearing DataFrame scalar access per output cell.
Feature values, identities, record order, metadata and missingness are unchanged.

Reproducible synthetic benchmark (no dataset loading or model fitting):

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONDONTWRITEBYTECODE='1'
python examples/profile_spatial_coverage.py --output docs/analytics/axial_coverage_profile.json
```

The reference restores only the old per-cell coverage read; both paths otherwise
use identical current code and synthetic inputs. The 200 team rows and 13 fields
produce 2,600 output records. Complete values and metadata compare equal.

| Profile | Time with cProfile | Spatial scalar reads |
|---|---:|---:|
| Reviewed scalar coverage path | 11.5734 s | 2,600 |
| Array-mask path | 0.5037 s | 0 |

Measured speedup: **22.98×**. Environment: Python 3.14.0, pandas 2.3.3,
NumPy 2.4.6. These local timings are not a rerun under the report's pandas 2.2.3
environment and are not directly comparable with its 58.47-second measurement.
See [profile evidence](axial_coverage_profile.json) for cumulative-time hotspots.

## Validation and changed files

On 6 October 2026, the following regression run passed **492 tests** with two
expected float32 overflow warnings in existing UI parity tests. No failures or
skips. Interpreter: `C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe`.
This is the relevant-suite run, not a claim of a full repository run.

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest tests/analytics/test_axial_point_features.py tests/analytics/test_spatial_coverage_performance.py tests/analytics/test_spatial_reporter_catalogue.py tests/analytics/test_point_geometry.py tests/analytics/test_point_geometry_extensions.py tests/analytics/test_spatial_arithmetic.py tests/analytics/test_feature_composition.py tests/analytics/test_features.py tests/analytics/test_heatmaps.py tests/analytics/test_spatial_fixture_features.py tests/analytics/test_spatial_distribution.py tests/analytics/test_datasets.py tests/analytics/test_keyed_features.py tests/analytics/test_ui_inventory.py tests/analytics/test_ui_recipe_cutoffs.py tests/analytics/test_ui_feature_discovery.py tests/analytics/test_ui_workflow.py tests/analytics/test_ui_preparation_parity.py tests/analytics/test_table_attribute_recovery.py tests/analytics/test_ui_catalog_compatibility.py -q -p no:cacheprovider
```

| Files | Change |
|---|---|
| `src/xdiyo_analytics/features/point_geometry.py` | Two per-match axial channels using the existing identifiability mask |
| `src/xdiyo_analytics/features/evaluation.py` | One finite-value array mask for coverage |
| `src/xdiyo_analytics/reporting/heatmaps.py`, `reporting/__init__.py` | Optional public axial reporting summary |
| `src/xdiyo_analytics/ui/build_inventory.py`, `ui/inventory.json` | Selectable fields and angle-aware help |
| `examples/axial_point_features.py` | Native Python and JSON-compatible feature recipe |
| `examples/profile_spatial_coverage.py` | Reproducible synthetic scalar-reference/array-mask benchmark |
| `tests/analytics/test_axial_point_features.py` | Axial math, symmetry, validity, time safety and native export/persistence |
| `tests/analytics/test_spatial_coverage_performance.py` | Scalar-access guard and complete metadata/value differential regression |
| `tests/analytics/test_spatial_reporter_catalogue.py` | All spatial fields, both fixture panels and executed JavaScript renderer paths |
| `docs/analytics/spatial_point_summaries.md`, `axial_point_features.md`, `axial_coverage_profile.json` | Usage, semantics, validation and measured profile |

The legacy raw-angle mean is explicitly tested (1°/179° still gives 90° under
generic `RollingMean`). New channel means give the axial answer instead. Existing
recipes and unrelated working-tree changes were not edited; frozen experiments
were not rerun.
