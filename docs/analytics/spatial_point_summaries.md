# Exact historical heatmap summaries

Phase A adds `SpatialPointSummary` to the public feature library, recipe catalogue
and experiment builder. It describes exported activity points, not tracking
trajectories, passes, or a causal measure of tactical pressure.

## Three descriptors, six match predictors

```python
from xdiyo_analytics.features import SpatialPointSummary, RollingMean

features = {
    f"activity_{field}_r20": RollingMean(
        SpatialPointSummary(field=field, kinds=("player",), side="for"),
        window=20, min_periods=1, venue="all",
    )
    for field in ("mean_x", "sd_x", "sd_y")
}
```

Match-layout assembly produces these columns, in order:

1. `home::activity_mean_x_r20`
2. `home::activity_sd_x_r20`
3. `home::activity_sd_y_r20`
4. `away::activity_mean_x_r20`
5. `away::activity_sd_x_r20`
6. `away::activity_sd_y_r20`

No point counts, coverage flags, goalkeeper summaries, grid cells or imputation
indicators enter this matrix automatically. Team-layout output keeps three
columns on each team row. Existing `Heatmap()` defaults and `RegionMass` behavior
are unchanged.

In the builder, add **RollingMean**, choose **SpatialPointSummary** as its source,
then choose the descriptor and point kinds. Repeat for the three descriptors.
Keep window 20, minimum periods 1 and all venues for the example above. The
builder loads `heatmap_points` automatically, even without a `Heatmap` node.
Python/notebook exports use the same registered components. The existing
HeatmapReporter discovers these scalar features and displays prepared values
for the selected fixture's Home and Away teams.

## Numerical contract

Coordinates use the exported `[0,100]` pitch axes: x is longitudinal, y lateral.
Every selected observation has unit weight, including repeated coordinates.
The optional source `weight` column is intentionally ignored. Native kind labels
are `player` and `goalkeeper`; the new feature defaults to player only and rejects
unknown or duplicate selections. Selecting both pools all selected points with
unit weights, rather than giving the two roles equal aggregate weight.

For N selected points:

- `mean_x = sum(x) / N`.
- `sd_x = sqrt(sum((x - mean_x)**2) / N)`.
- `sd_y` uses the same formula for y.

These are population standard deviations (`ddof=0`), computed with centered sums.
A one-point map has zero spread. The units are percentage points of pitch length
or width, not metres. This differs from the existing temporal `RollingStd`, whose
default is `ddof=1` across historical observations.

`RollingMean(SpatialPointSummary("sd_x"), ...)` averages **per-match spatial
standard deviations**, with equal weight per usable match. It does not calculate
the spread of pooled points or the spread of an averaged grid. No grid binning,
smoothing or spatial normalization is involved in these exact descriptors.

## Optional geometry fields

The original three descriptors and six match columns above remain the baseline.
Select these additional `SpatialPointSummary(field=...)` values explicitly in
Python or the builder:

| Field | Per-match calculation |
|---|---|
| `mean_y` | Mean lateral position |
| `depth_80`, `width_80` | 90th minus 10th percentile of x or y; NumPy `method="linear"` |
| `cov_xy` | Mean of `(x-mean_x)*(y-mean_y)`; population denominator N |
| `corr_xy` | Covariance divided by `sd_x*sd_y`; missing if either spread is zero |
| `major_variance`, `minor_variance` | Ordered covariance eigenvalues, largest first |
| `major_spread`, `minor_spread` | Square roots of those eigenvalues |
| `anisotropy` | `(major_variance-minor_variance)/(major_variance+minor_variance)`; missing at zero spread |
| `axis_angle` | `0.5*atan2(2*cov_xy, var_x-var_y)` modulo pi, in `[0,pi)` radians |

Eigenvalues use a symmetric solver; eigenvectors and their arbitrary signs are
never exposed. Negative eigenvalues within `64*float64_epsilon*trace(covariance)`
are roundoff and become zero; more negative values raise. Angles are missing
when the eigenvalue gap is at most that tolerance, including isotropic and
zero-spread maps. Correlations are bounded to `[-1,1]` against rounding drift.
A valid map can therefore have an undefined correlation or angle. Its raw-map
status remains `ok`; usable-value counts reflect the undefined descriptor.

Covariance/eigenvalues have squared normalized-coordinate units. Principal
spreads have normalized-coordinate units; correlation and anisotropy are
dimensionless. These axes depend on x/y scaling and are not measured formation
angles. The lateral touchline convention remains unverified. Ordinary rolling
operators on angles calculate ordinary scalar statistics, **not circular angle
statistics**; a mean near the zero/pi wrap can be misleading.

For an alternative robust recipe, replace `sd_x` and `sd_y` with `depth_80` and
`width_80`; they are not appended to the baseline automatically. Between-match
variability continues to use `RollingStd(SpatialPointSummary(...), window=...)`.

## Orientation and timing

All point summaries stay in the focal team's canonical frame: own goal left,
attacking toward increasing x. For `side="against"`, historical opponent points
are rotated 180 degrees into that frame. Thus against mean x is `100 - opponent
mean_x`, and against mean y is `100 - opponent mean_y`. All spreads, robust
widths, covariance, correlation, eigenvalues, anisotropy and axis angle modulo pi
are unchanged by that 180-degree rotation. `side="both"`
emits both perspectives. Target Home/Away venue **does not rotate these scalar
values**. This differs intentionally from final shared-pitch grid visualization.

The raw descriptor is an observed source and cannot be used directly as a
predictor, including through arithmetic wrappers. Wrap it with `Lag`, a rolling
operator or `EMA`. The existing eligibility engine enforces earlier finished
fixtures, the prediction cutoff, result availability, target exclusion and
simultaneous-match exclusion. No extra `Lag` is required before `RollingMean`.

The rolling window counts eligible matches, including matches whose map is
missing. Reduction uses only finite values within that fixed window. A missing
latest map is not skipped by `Lag`. Historical `min_periods` counts usable
descriptor values. `venue="same"` restricts eligibility before applying the
window; `venue="all"` combines venues. Default grouping follows team and
competition across seasons. No season reset, warm-start prior or population
fallback is added by this source.

Cutoff and availability settings are supplied by the caller. The feature-only
example freezes each competition/season/round at its earliest kickoff minus one
hour, and assumes map availability at kickoff plus three hours. These are
**retrospective research assumptions**, not verified source publication times.
Loading the full schedule matters for that round anchor, including postponements.

## Missingness, identity and diagnostics

- Missing required table: raises a loading/evaluation error.
- Missing team map: NaN and status `missing_team_map`.
- Present map without selected kinds: NaN and `empty_selected_kind`.
- Fewer than the optional `min_points` observations: NaN and `low_point_count`.
- Invalid selected coordinates: raises with map identity; never clips or silently
  drops points. Unselected kinds are filtered before coordinate validation.
- Valid selected map: status `ok`, including a one-point zero-spread map.

Source league/season keys must occur on both history and point tables whenever
either input contains them. Map identity includes those keys plus event/team.
History identities must be unique. Duplicate native `point_order` identities and
conflicting source versions/hashes raise; identical coordinates alone are never
deduplicated. Custom tables without `point_order` cannot establish duplicate-point
identity and advertise that limitation in provenance.

`features.attrs["spatial_point_audit"]`, retained in
`dataset.definitions["spatial_point_audit"]`, stores:

- Source/configuration/formula fingerprints, available source versions and hashes,
  point kinds, exact coordinate ranges and per-map counts/status.
- Per-target historical eligible-match, window-match and usable-value counts.
- Identity-keyed cutoff/availability records, their hash and grouping policy.

The HeatmapReporter adds downloadable `point_map_coverage`,
`point_history_coverage`, `point_output_coverage` and `point_fixture_coverage`
tables, scoped to displayed fixtures. Both-team map coverage is grouped by
feature, source selection, league, season and perspective. Counts are diagnostics
only, not predictors or automatic cohort
filters. Preserve the incumbent cohort and its train-only imputation policy when
comparing models; any covered-only subset needs the same baseline comparison.

Within an evaluation, fields and perspectives sharing kinds/minimum count reuse
one exact map-summary pass. There is no global cache: changed points, history,
configuration or timing cannot reuse summaries from a previous evaluation.
Metadata travels through assembly and prepared-feature Parquet round-trips.

## Arithmetic lineage and coverage

Single-output `Sum`, `Difference`, `Product` and `Ratio` preserve spatial
provenance both inside and outside historical operators, including constants,
two spatial operands and nested/shared expressions. The reporter shows the exact
prepared scalar as **Derived spatial value**, with the ordered calculation tree.
It does not reuse an operand's physical label, units, side or pitch frame.
Contributing source descriptors retain those properties separately in `sources`.
For example `RollingMean(Product(mean_x, 2), 20)` and
`Product(RollingMean(mean_x, 20), 2)` retain different calculation trees even
when their values coincide. No numerical arithmetic or eligibility rules change.

Coverage is not added across branches:

- Each raw selection has a `source_id` over the source fingerprint, point kinds,
  minimum count, perspective, unit weighting and formula version. Selecting two
  fields on the same maps shares that raw selection; a different kind, threshold,
  side or source fingerprint does not. Repeated references are deduplicated by
  selection and fixture/team identity **within each output feature**.
- Each historical stage has a `history_id` over its expression, H2H scope and
  source selections. Nested stages and distinct windows remain separate records;
  records also identify the source column and target fixture/team. `source_ids`
  identifies its contributing point selections. `eligible_matches` and
  `window_matches` count candidate matches; `usable_values` counts finite inputs
  in that particular window. The legacy `usable_maps` field remains an alias
  only for direct point-summary children; it counts usable descriptor values,
  not every valid raw map. Branch counts must never be summed as disjoint maps.
- `point_output_coverage.usable_output` reports finiteness of each final named
  scalar on each target team row. A zero denominator or overflow can yield an
  unusable derived value while all raw maps remain valid. A finite ratio fallback
  can be usable; missing operands never trigger that fallback.

`point_map_coverage` describes raw maps attached to displayed fixture identities;
historical-window counts describe the prior inputs used for those targets. Their
populations differ intentionally. Full source-map records remain in the audit.
Both-team usability counts two distinct focal teams per source selection; shared
operands never multiply those counts. Identity-based filtering also works when
keyed rows are shuffled or a report displays only a subset of fixtures.

Source/timing fingerprints and the lineage survive native recipe JSON, Python
export, match assembly, Parquet and recovery bundles. Parquet JSON normalizes
existing tuple-valued grouping/identity keys into lists; identities and values
are retained. Cached children are copied when attaching derived metadata.

## Feature-only reproduction and measured cost

From the repository with its Python environment and `PYTHONPATH=src`, run:

```powershell
python examples/point_geometry.py --data-root data/xDiyo_data --league Premier_League --season 23_24 --output benchmark.json
```

The script verifies published source hashes, prepares the six columns through
native history/evaluation/assembly, prints source/configuration and output hashes,
and records time and peak process memory. It never fits a model. Run again in a
fresh process and compare `feature_sha256`, `source_sha256` and `timing_sha256`.
`recipe_features()` in the same example supplies the compact serializable UI
feature mapping; merge it into a recipe to retain existing predictors.

For the original Phase A implementation, on Windows, Premier League 2023/24, publication
`a13ab544b9ea4f0e82bfbc8969a3ed83`, two fresh-process runs on 6 October 2026 gave:

| Measure | Result |
|---|---|
| Raw points | 598,825 (558,048 player; 40,777 goalkeeper) |
| Team maps / fixtures | 760 / 380, all usable for both teams |
| Load including hash verification | 0.47–0.56 seconds |
| Feature preparation and assembly | 1.75–1.87 seconds |
| Peak process working set, including imports/loading | 487–498 MiB |
| Output | 380 rows × 6 features, 60 initial-history missing cells |
| Feature hash, identical in both processes | `a100311096fed5c7759b1089782bb394ec0f9c640b3471470bb4afe02088dea0` |

These measurements cover one partition, not a full-archive preparation or a
predictive-performance claim. The native loader still materializes requested
tables. Audit memory before loading the entire archive at once.

[Phase B](spatial_distributions.md) adds regional shares, entropy and concentration.
Style distances, goalkeeper-only research blocks and fitted PCA remain later
research phases; they are not part of this Phase A addition.
