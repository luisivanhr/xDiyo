# Spatial distances and fitted map representations (Phases C and D)

All additions are opt-in. Existing features, default recipes, heatmap defaults,
and frozen experiments are unchanged. No research models were fitted during
implementation. See `examples/spatial_extensions.py` for native configurations.

## Resolution

Every grid reduction uses the source's `Heatmap(grid_size=n)`, for any integer
`n >= 2`. There is no fixed list or maximum resolution. This applies to entropy,
all concentration metrics, regional integration, deviations, fixture distances,
and the grids supplied to PCA or clustering. Grids are square in the exported
0–100 coordinate system, not physical metres. Cell ordering is row-major [y,x].

The distribution-feature examples and builder defaults remain **6×6**; standalone
`Heatmap()` still defaults to **10×10**. Configure **Source → Grid size** in the UI.
No resampling or implicit resolution search occurs. Memory grows with `n²`.
For Gaussian sources, explicitly set a fine `resolution` divisible by `n`
(for example, output 7×7 with resolution 112). Sigma is in fine-grid cells.

Normalized entropy divides by `log(n²)`; effective fractions and normalized HHI
also use the actual cell count. Normalization does not make different grid
resolutions statistically interchangeable. Keep grid, smoothing, point kinds and
weighting fixed when comparing maps. Both operands of a native fixture distance
are constructed from the same source configuration.

## Phase C: deviation and distances

```python
from xdiyo_analytics.features import (
    Heatmap, RollingMean, SpatialHistoricalDeviation, SpatialFixtureDistance,
)

source = Heatmap(grid_size=6, kinds=("player",), orientation="team")
features = {
    "activity_deviation_r20": RollingMean(
        SpatialHistoricalDeviation(source, window=20, min_periods=1, venue="all"),
        window=20, min_periods=1, venue="all",
    ),
    "style_distance": SpatialFixtureDistance(source, comparison="style"),
    "home_context": SpatialFixtureDistance(source, comparison="home_context"),
    "away_context": SpatialFixtureDistance(source, comparison="away_context"),
}
```

Distances are the square root of Jensen–Shannon divergence, with base-2 logs,
and range from zero (identical distributions) to one (disjoint support). Zero
cells contribute zero; no pseudocount is added. Mass and density sources are
accepted; density is integrated using actual cell area. Entirely missing maps
remain missing. Partial missingness, negative or infinite cells, or invalid
total mass raise. `mass_tolerance=1e-10` only permits roundoff around total 1.

### Historical deviation

For historical match j, first average valid maps within the last `window`
matches eligible before **j's own configured cutoff**. Then compare j's observed
map to that baseline. Its score is an observed post-match value, so a raw
`SpatialHistoricalDeviation` is unsafe as a predictor, including inside `Abs`
or arithmetic. An outer `Lag`, `RollingMean` or `EMA` makes past scores available
under normal cutoff and release rules. A larger score means more deviation.

The inner baseline and outer reduction have separate windows/minimums. Missing
maps occupy eligible-match slots; neither step searches farther back to fill
them. A missing baseline or observation produces NaN. Round freezes, strict
earlier kickoffs, availability at cutoff, H2H and venue selection retain native
semantics. Appending future maps does not redefine old deviations.

### Fixture comparisons

Each team's mean uses the fixture's shared cutoff. For-map F is the team's own
historical activity; against-map A is historical opponent activity already
rotated into the focal team's frame. R reverses both axes of the binned grid:

| Comparison | Formula | Output |
| --- | --- | --- |
| `style` | JS(F_home, F_away) | One fixture scalar |
| `home_context` | JS(F_home, R(A_away)) | One fixture scalar |
| `away_context` | JS(F_away, R(A_home)) | One fixture scalar |

The source side must be `for`; the comparison selects the appropriate against
source internally. Final source orientation is canonicalized to `team` before
comparison. No rotated-point rebinning replaces native cell reversal.

Two team-row copies carry explicit fixture scope and identical values. Match
assembly emits each once as `fixture::<name>`. Different home/away cutoffs,
duplicate sides or inconsistent team identities raise; an absent partner row
or unavailable history yields NaN. H2H is supported for historical deviation,
but deliberately rejected around fixture comparisons. Missing-value and cutoff
consistency are still checked when fixture copies are assembled.

The HeatmapReporter discovers these features and shows exact prepared values.
Fixture distances have one panel with both team identities/badges. Team
deviations retain two panels. `spatial_distance_coverage` contains inner-window
eligible/selected/usable map counts, scoped to displayed fixtures. Recursive
calculation metadata distinguishes inner baseline from outer averaging. Source
hashes, timing, grid configuration and recipe identity travel with the dataset.

## Phase D: geometry and goalkeeper blocks

Extended geometry is already native: mean y, linear 10–90% widths/depth,
covariance/correlation, principal variances/spreads, anisotropy and identifiable
axes. Use the double-angle channels for historical axes. Between-match
variability remains a composition with `RollingStd`. See
[point geometry](spatial_point_summaries.md) and [axial features](axial_point_features.md).

`keeper_definitions()` in the example provides a separately named, optional
goalkeeper block with `kinds=("goalkeeper",)`: rolling mean x, robust depth and
entropy. Keeper maps normalize independently. Empty keeper maps stay missing;
neither role is silently substituted for the other. The default all-kind
`Heatmap()` behavior is unchanged. The existing geometry and grid reporter views
cover these outputs.

## Phase D: training-only PCA and clustering

`training.SpatialPCA` and `training.SpatialClusters` are native sklearn-compatible
preprocessors, available in **Model & training → Preprocessors**. They operate
on causal historical grids already assembled as predictors. Put them before
generic preprocessing when selecting columns by name.

Both accept `blocks`: ordered lists of complete cell columns, typically Home
then Away. Explicit blocks must be disjoint, square, equal-sized and identically
ordered. Use canonical **team** orientation with matching grid contracts. The
example's `embedding_recipe_step(prepared, family, method, dimensions)` discovers
and validates a prepared Home/Away grid family and builds the UI recipe node.
The UI also allows explicit ordered blocks using discovered columns. Omitting
blocks treats the entire input as a single vector, useful inside an explicit
ColumnTransformer; do this only if that input contains the intended map cells.

```python
from xdiyo_analytics.training import SpatialPCA, SpatialClusters

# home_cells and away_cells are complete, ordered columns from prepared metadata.
pca = SpatialPCA(n_components=3, blocks=[home_cells, away_cells])
clusters = SpatialClusters(n_clusters=3, blocks=[home_cells, away_cells], random_state=0)
# Select one and add it to the recipe's preprocessors. Do not fit before splitting.
```

Within each fit, stack the declared team maps into one shared training sample.
Fit median imputation, optional standard scaling (`scale=False` by default), and
one PCA basis or K-means codebook on that sample only. Entirely absent training
maps are excluded from estimating the representation; missing transform-time
maps receive training medians. A cell with no training evidence raises rather
than introducing a fabricated prior. `keep_other=True` passes unrelated columns
through for later preprocessing. No missingness indicators are added.

PCA returns component scores, using the full SVD solver. Clustering returns
Euclidean distances to the fixed shared centers, not arbitrary per-match cluster
labels. Both teams therefore share component/center ordering within one fitted
model. Bases can change between folds, and clusters are not tactical categories.
Use chronological validation to choose dimensions; no automatic search is added.

The native fold runner clones and fits the pipeline on fitting rows, then keeps
the imputer, scaler and encoder with the saved model. JSON/Python/notebook
exports serialize configuration, never fitted state. Model persistence retains
the fitted state. The pre-training HeatmapReporter displays the original prepared
maps; it does not pretend that fitted PCA scores or cluster distances are pitch
cells. Inspect `encoder_.components_` or `encoder_.cluster_centers_` for the
stored representation.

## Verification

Tests cover analytic JS values, directional rotation, cutoff freezes, late
releases, missing slots, large IDs, assembly, native recipe fitting on synthetic
data, model restoration, exports, reporter branches and sklearn estimator
checks. Grid tests include odd sizes and compatible Gaussian fine grids.
The installed sklearn version used for compatibility checks is 1.9.0; separate
dependency-version environments were not tested.

The broad regression run passed 625 tests, with two optional array-API checks
skipped. After the final temporal/fixture metadata correction, 136 focused
composition, assembly and reporter checks passed again. Known warnings were
the existing float32 overflow cases and sklearn's pytest parametrization warning.

`spatial_extensions_profile.json` records feature-only preparation of Premier
League 2023/24 (380 fixtures, 598,825 points) with Phase C and the keeper block.
Preparation/assembly took 5.89 seconds after a 0.47-second load; whole-process
peak working set was approximately 552 MiB. This is a preparation benchmark,
not evidence of predictive improvement or an archive-scale memory guarantee.
