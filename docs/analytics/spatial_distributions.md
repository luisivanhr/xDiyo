# Regional activity, entropy and concentration (Phase B)

These features extend the existing fixed-grid heatmap pipeline. They are available
in Python, the experiment builder, saved recipes, exported notebooks and the
fixture-based HeatmapReporter. They do not fit a model or choose parameters.

## Compact example

```python
from xdiyo_analytics.features import Heatmap, RegionMass, SpatialEntropy, RollingMean

outfield = Heatmap(
    grid_size=6, method="grid", normalization="mass",
    kinds=("player",), orientation="team",
)
features = {
    "activity_attacking_third_r20": RollingMean(
        RegionMass(outfield, "attacking_third"), window=20, min_periods=1,
    ),
    "activity_central_channel_r20": RollingMean(
        RegionMass(outfield, "central_channel"), window=20, min_periods=1,
    ),
    "activity_entropy_r20": RollingMean(
        SpatialEntropy(outfield, normalized=True), window=20, min_periods=1,
    ),
}
```

This produces six match predictors, three for Home and three for Away. Merge the
mapping into an existing feature recipe to retain its incumbent inputs. The
36 underlying grid cells are shared intermediate values, not additional model
columns unless explicitly selected as features.

The builder's new entropy/concentration source defaults to an **unsmoothed 6×6
outfield unit-mass grid**. Existing standalone `Heatmap()` defaults are unchanged.
Place the reducer inside **RollingMean** for the example above. Use one
concentration alternative at a time when comparing predictors.

## Regional integration

`RegionMass(source, region, rectangle=None)` supports these canonical team-frame
regions (own goal left, attacking toward increasing x):

| Name | Bounds on exported 0–100 coordinates |
|---|---|
| `own_half`, `opponent_half` | x below / above 50 |
| `defensive_third` | x from 0 to 100/3 |
| `middle_third` | x from 100/3 to 200/3 |
| `attacking_third` | x from 200/3 to 100 |
| `low_y_wide` | y from 0 to 100/3 |
| `central_channel` | y from 100/3 to 200/3 |
| `high_y_wide` | y from 200/3 to 100 |
| `wide_channels` | Union of the two wide channels |
| `central_attacking_third` | Attacking third intersected with central channel |
| `rectangle` | Explicit `(x_min, x_max, y_min, y_max)` |

Example custom rectangle:

```python
RegionMass(outfield, region="rectangle", rectangle=(60, 90, 25, 75))
```

In the UI, choose **Rectangle**, enable its bounds and enter the four coordinates.
Bounds must be finite, ordered and inside 0–100. Arbitrary lists of overlapping
rectangles are not a supported union API. The built-in wide union has disjoint
interiors and never double counts.

Integration treats each cell as a constant-density rectangle. Its contribution is
cell mass multiplied by the fraction of its area inside the region. Density
grids multiply by cell area first; count grids yield regional counts. Metadata
declares `backend="grid_area_overlap"`. This is **not exact raw-point membership**:
a custom boundary cutting through a cell apportions the entire cell by area.
The six-by-six grid aligns with the standard halves and thirds. Histogram
boundaries belong to the higher cell, with 100 included in the last cell.

Regions describe pitch location, not action type or a verified penalty box.
Low-y/high-y are coordinate labels, not asserted left/right touchline semantics.
Complete partitions are useful in reports, but their sum constraints create
redundant predictors. Avoid automatically adding all regions to a model.

## Entropy

`SpatialEntropy(source, normalized=True, base=e, mass_tolerance=1e-10)` uses:

```text
H = -sum(p * log_base(p)) over positive cell probabilities
normalized H = H / log_base(K)
```

K is the number of cells, 36 in the primary specification. Zero cells contribute
zero; no pseudocount is added. Normalized entropy is 0 for a one-cell map and 1
for a uniform map. Raw entropy uses natural logarithms by default; base 2 gives
bits. Log base cancels for normalized entropy.

### Operator order matters

```python
# Preferred: typical within-match dispersion.
RollingMean(SpatialEntropy(outfield), window=20)

# Different: dispersion of the historical mixture.
SpatialEntropy(RollingMean(outfield, window=20))
```

For two matches each concentrated in a different single cell, the first quantity
is zero; the second is `ln(2)/ln(36)`. Both use equal weight per usable match, not
weight proportional to the number of exported points. Neither pools raw points
across fixtures. The reporter caption and `calculation` metadata retain the full
order, for example “Rolling mean (20 matches) of Normalized entropy of 6×6 grid.”

Entropy measures spatial dispersion of recorded activity on a fixed partition.
It does not directly measure tactical quality, unpredictability or tempo. Keep
grid resolution, kinds, weighting and smoothing fixed in comparisons.

## Alternative concentration summaries

`SpatialConcentration(source, metric="hhi", mass_tolerance=1e-10)` returns one
selected scalar:

| Metric | Formula | Range |
|---|---|---|
| `effective_cells` | exp(natural entropy) | 1 to K |
| `effective_fraction` | effective cells / K | 1/K to 1 |
| `hhi` | sum(p²) | 1/K to 1 |
| `normalized_hhi` | (HHI − 1/K) / (1 − 1/K) | 0 to 1 |
| `largest_cell_share` | max(p) | 1/K to 1 |
| `occupied_fraction` | number of positive cells / K | 1/K to 1 |

Effective cells is a monotone transform of entropy. Occupied fraction is mainly
diagnostic because it is especially sensitive to sampling volume and smoothing.
HHI is an alternative to entropy, not an automatically appended companion.

## Validation, causality and orientation

- Distribution summaries accept mass or density grids. Configure mass explicitly
  rather than asking the reducer to normalize arbitrary counts.
- All-missing maps remain NaN. Partial NaNs, infinities, negative cells and
  nonpositive total mass raise with grid/row context.
- Total probability must be within `mass_tolerance` of 1. Only that harmless drift
  is normalized away. Tolerance is bounded between 0 and 1e-6 and is not a
  smoothing parameter. Outputs are bounded at mathematical endpoints to remove
  floating-point roundoff; invalid input probabilities are never clipped.
- New regions validate complete nonnegative maps, positive count totals and
  unit mass (fixed absolute tolerance 1e-10). Legacy half integration preserves
  its existing behavior for missing noncontributing cells.
- New reducers validate source league/season keys on both inputs, native point
  identities, kind selections and source-version consistency. Repeated
  coordinates remain repeated observations. Invalid selected coordinates raise
  with the offending map identity.
- Direct observed entropy, concentration and region features are rejected as
  predictors, including inside arithmetic or known-reference wrappers. Use
  native historical operators; no extra lag is needed before RollingMean.
- Existing cutoff, availability, simultaneous-match, missing-window, venue and
  cross-season grouping rules apply unchanged. No imputation or prior is added.
- Against grids use the native 180-degree **cell reversal**. Reversing an already
  binned map preserves entropy. Rotating raw coordinates and rebinning can differ
  for points exactly on internal edges; those are distinct operations.
- Entropy/concentration scalars are rotation-invariant and stay in the team
  frame. Regional values refer to the focal frame; the reporter rotates the
  shaded region on both axes when showing Away on a shared home-oriented pitch.

Source hashes/versions and configuration are retained in
`dataset.definitions["spatial_distribution_audit"]`, together with identity-keyed
cutoff/availability records and their hash. Formula/version, source-grid settings,
operator order, region bounds and units accompany spatial metadata. The existing
experiment record supplies the library code identity. Identical grid expressions
share evaluation work; no cache persists across unrelated evaluations.

## Reproduction and benchmark

```powershell
$env:PYTHONPATH='src'
python examples/spatial_distribution.py --output phase_b.json
```

This feature-only script uses the compact three-feature mapping above and native
loading/history/assembly. Its `recipe_features()` function provides serializable
UI nodes. No model is fitted. It uses the same explicit retrospective timing
assumptions as the Phase A example: frozen competition/season/round cutoff one
hour before the earliest kickoff, and availability at kickoff plus three hours.
These are not verified publication times.

Two fresh Windows processes on 6 October 2026 prepared Premier League 2023/24
(380 fixtures, 598,825 raw points, publication
`a13ab544b9ea4f0e82bfbc8969a3ed83`) with identical output hashes:

- Load and source verification: 0.49–0.55 seconds.
- Preparation and assembly: 1.43–1.67 seconds.
- Peak whole-process working set: 510–517 MiB, including imports and loading.
- Six output columns; 60 initial-history missing cells.
- Feature hash: `b01dfba871e1893fb9c14fc6f235577a01e2c1a759706b5b938d5c28a8d29642`.

This is a single-partition implementation benchmark, not an archive-scale memory
guarantee or evidence of predictive improvement. The loader materializes requested
tables. Distance/deviation features and fitted PCA remain later phases.
