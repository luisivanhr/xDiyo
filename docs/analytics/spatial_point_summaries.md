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

## Orientation and timing

All three scalars stay in the focal team's canonical frame: own goal left,
attacking toward increasing x. For `side="against"`, historical opponent points
are rotated 180 degrees into that frame. Thus against mean x is `100 - opponent
mean_x`, while both spatial standard deviations are unchanged. `side="both"`
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
- Per-target historical eligible-match, window-match and usable-map counts.
- Identity-keyed cutoff/availability records, their hash and grouping policy.

The HeatmapReporter adds downloadable `point_map_coverage`,
`point_history_coverage` and `point_fixture_coverage` tables, scoped to displayed
fixtures. Both-team map coverage is grouped by feature, league, season and source
perspective. Counts are diagnostics only, not predictors or automatic cohort
filters. Preserve the incumbent cohort and its train-only imputation policy when
comparing models; any covered-only subset needs the same baseline comparison.

Within an evaluation, fields and perspectives sharing kinds/minimum count reuse
one exact map-summary pass. There is no global cache: changed points, history,
configuration or timing cannot reuse summaries from a previous evaluation.
Metadata travels through assembly and prepared-feature Parquet round-trips.

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

On Windows, Premier League 2023/24, publication
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

Entropy, additional regions, style distances, goalkeeper-only research blocks
and fitted PCA are later phases; they are not part of this Phase A addition.
