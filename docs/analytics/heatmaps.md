# Heatmap features and fixture views

`Heatmap` turns a team's exported spatial points into numeric cells. Historical
operators turn those observations into predictors. `HeatmapReporter` displays
those actual predictors on a pitch, with a **team name, Home/Away role and local
badge**. It works in notebook reports and the experiment builder.

[Exact point summaries](spatial_point_summaries.md) provide compact historical
mean position and spatial spread directly from raw points, with no grid
approximation. These scalars stay team-relative even for Away fixtures.
For thirds, channels and rectangles, plus entropy/concentration of fixed grids,
see [Phase B spatial distributions](spatial_distributions.md).

## Orientation: a shared home-oriented pitch

Following the user's external provider check on 29 September 2026, the input
convention is each source team's own goal on the left. The library now applies
an explicit **180-degree rotation**, reversing both axes:

\[
R(x,y)=(100-x,100-y),\qquad [R(H)]_{ab}=H_{g-1-a,g-1-b}.
\]

`Heatmap(orientation="home")` is the default. The final feature grids share one
fixture frame: **Home's goal is left; Away's goal is right**.

| Final feature | Home-team grid | Away-team grid |
| --- | --- | --- |
| `for` | Native source orientation | Rotate native source 180 degrees |
| `against` | Rotate native opponent source 180 degrees | Native opponent orientation |

Against means opponents faced in the focal team's historical fixtures, not the
current opponent's historical form. A for/against selection applies equally to
both fixture panels: for shows both focal teams' own histories; against shows
both focal teams' histories of opponents faced.

**Ordering matters:** first rotate historical against maps into the focal team's
frame, then calculate lag/rolling/EMA values there, and finally rotate an Away
prediction row into the home-oriented frame. Historical Home/Away appearances
are never mixed in opposite coordinate frames before averaging. `orientation="team"`
retains each focal team's own-goal-left frame for final outputs instead. Against
still rotates into that focal frame. The low-level `heatmap_grid` helper simply
pools its supplied raw points; it does not know fixture roles or rotate them.

This is a source-coordinate convention, not a claim about exact real-world
positions in the upcoming match. Different providers must be brought into the
same input convention before using it. No half-period information is inferred.

## Two processing methods

| Method | Processing | Main controls |
| --- | --- | --- |
| `grid` | Count points in equal square cells of the exported coordinate plane. | `grid_size` |
| `gaussian` | Bin on a finer grid, smooth with a Gaussian and reflected boundaries, then sum fine cells into output cells. | `grid_size`, `resolution`, `sigma` |

The reference plotting approach used `resolution=100`, `sigma=2.6`, unit point
weights and normalization to unit mass. `Heatmap(grid_size=100, method="gaussian")`
reproduces that numerical grid. A smaller output grid, such as the default 10,
pools those smoothed cells and produces fewer predictors. The reference plot's
nonlinear color contrast is not part of the numerical feature. The reporter uses
a shared linear scale across teams and sides within each map.

`grid_size=10` produces 100 columns per perspective. `side="both"` produces 200;
assembling both teams into one match row doubles that again. Start with a coarse
grid to keep experiments manageable. `grid_size` is at least two. For Gaussian
processing, `resolution` must be a multiple of `grid_size`, and `sigma` must be
positive. Sigma is measured in fine-grid cells.

### Units and equations

Let point coordinates be \((x_i,y_i)\in[0,100]^2\), with weight \(w_i\).
For cell \(C_{ab}\), its count and the total weight are

\[
H_{ab}=\sum_i w_i\,\mathbf{1}\{(x_i,y_i)\in C_{ab}\},
\qquad W=\sum_i w_i.
\]

`use_weights=False` sets every weight to one. With `use_weights=True`, a supplied
weight is used; a missing weight is one. Negative or nonfinite weights and
coordinates outside the exported range raise an error rather than being clipped.
Boundary coordinates equal to 100 belong to the final cell.

The three normalizations are

\[
\text{count}_{ab}=H_{ab},\qquad
\text{mass}_{ab}=\frac{H_{ab}}{W},\qquad
\text{density}_{ab}=\frac{H_{ab}}{W A},\quad A=(100/g)^2,
\]

where \(g\) is `grid_size`. Thus counts sum to \(W\), masses sum to one,
and density times cell area sums to one. Density is per exported coordinate
area, not per physical square metre. Counts may reflect differences in export
sampling intensity; they are not automatically possession or event rates.

For Gaussian processing, let \(H^{(r)}\) be the fine histogram with resolution
\(r\). Smooth it before summing its cells into the coarser output grid:

\[
S=G_\sigma *_{\mathrm{reflect}} H^{(r)},\qquad
\widetilde H_{ab}=\sum_{(u,v)\in B_{ab}} S_{uv}.
\]

Apply the same normalization to \(\widetilde H\). Reflected boundaries preserve
total mass to floating-point precision. Returned arrays use `[y, x]` indexing.
This is a smoothed histogram, not a fitted continuous KDE.

`kinds=None` includes all exported point kinds. Use `kinds=("player",)` or
`("goalkeeper",)` to choose a subset. Empty, absent and zero-total-weight maps
are all missing, never an all-zero spatial distribution. Empty cells in a
genuinely observed map are valid zeros.

## Historical feature API

The operation is called **Rolling mean** for heatmaps and ordinary statistics
alike. Both use the existing `RollingMean` expression; heatmaps are its input,
not a separate kind of mean.

```python
from xdiyo_analytics.data import load_seasons
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.features import Heatmap, RegionMass, Lag, RollingMean, EMA, evaluate_features

data = load_seasons(
    "data/xDiyo_data", ["22_23", "23_24", "24_25"],
    tables=["matches", "statistics", "pregame", "heatmap_points"],
)
history = build_team_history(data)
features = evaluate_features(history, {
    "spatial_last": Lag(Heatmap(grid_size=10), periods=1),
    "Rolling mean": RollingMean(
        Heatmap(grid_size=10, method="gaussian", side="both"), window=5,
    ),
    "spatial_ema": EMA(Heatmap(grid_size=10), span=5),
}, heatmaps=data["heatmap_points"], keyed=True)
```

The cells also support `RollingStd`, `RollingZScore`, and H2H history grouping.
They use the existing eligibility, cutoff and result-availability rules. A raw
`Heatmap` at the feature root is rejected because it observes the current match.
`Lag(..., periods=1)` uses the latest eligible match, even if its map is missing.
Rolling reductions use available maps within the chosen match window; missing
maps do not pull older fixtures into that window. Standalone `heatmap_grid(points,
spec)` is available for descriptive processing of one observed team-match map.

`side="for"` describes the team's points. `side="against"` describes the points
of opponents faced in its historical fixtures. It does **not** select today's
opponent's own historical form. `ForAgainst(Heatmap(...), side="both")` is also
supported. Season/league and event/team IDs align maps with histories; IDs are
not converted through floating point.

Coordinate orientation follows the explicit convention above; no automatic
source-direction detection or first/second-half split is performed. League
population pooling and seeded warm-up priors for heatmap cells remain unsupported.

## Venue history

The existing `Lag`, `RollingMean`, `RollingStd`, `RollingZScore` and `EMA` accept
`venue="all"` (default) or `venue="same"`. This also works with ordinary statistics.
In the **UI**, the venue selector is shown only for expressions containing a
Heatmap source, including nested regional summaries. Switching to an ordinary
statistic removes the venue override and restores all-venue history. The Python
API retains its existing capabilities.
In the **UI**, the venue selector is shown only for expressions containing a
Heatmap source, including nested regional summaries. Switching to an ordinary
statistic removes the venue override and restores all-venue history. The Python
API retains its existing capabilities.

```python
same_venue = RollingMean(Heatmap(method="gaussian"), window=5, venue="same")
all_venues = RollingMean(Heatmap(method="gaussian"), window=5, venue="all")
```

Before a home fixture, same venue selects previous eligible home appearances;
before an away fixture it selects previous away appearances. Filtering happens
**before** taking the last five matches. Existing cutoff, availability, missing
observation and partial-window rules still apply. H2H adds its opponent restriction.
Each nested historical operator owns its own venue choice. League populations
reject same venue because that option concerns a focal team's history.

## RegionMass and spatial metadata

`RegionMass` integrates a grid over `own_half` or `opponent_half`. Those names
always refer to the focal team, regardless of final display orientation. Own
half is therefore left for Home and right for Away on a shared pitch.

```python
recent = RollingMean(Heatmap(side="both", method="gaussian"), window=5)
features = evaluate_features(history, {
    "recent_activity": recent,
    "own_half_mass": RegionMass(recent, region="own_half"),
    "opponent_half_mass": RegionMass(recent, region="opponent_half"),
    "previous_own_half": Lag(RegionMass(Heatmap(), region="own_half")),
}, heatmaps=data["heatmap_points"], keyed=True)
```

A raw `RegionMass(Heatmap())` cannot be a prediction feature without a historical
operator. With mass normalization the regional value is a proportion; with count
normalization it is a count; density is integrated using cell area. More generally,
applying it to transformed values integrates those values; a sum of cell Z-scores
or standard deviations is not a probability mass.

Let \(f_c=|C_c\cap A|/|C_c|\) be the fraction of cell \(c\) inside region \(A\).
Then

\[
M_A=\sum_c f_c p_c\quad\text{(mass)},\qquad
M_A=\sum_c |C_c\cap A|d_c\quad\text{(density)}.
\]

Counts use the same fractional weighting as mass. This treats each cell as
piecewise constant: a middle cell crossing halfway in an odd-sized grid contributes
half to each region. The halves sum to the full grid's mass/count when available.
Missing contributing cells make the regional result missing; they are not zero.

Spatial output descriptions travel from `features.attrs["spatial_features"]`
into `dataset.definitions["spatial_features"]`. The reporter reads these
**descriptors, not naming patterns**. Each descriptor declares `kind`, `columns`,
`orientation`, `side`, `feature`, `normalization` and historical `operators`.
Grids additionally declare `grid_size`; regions declare `region`. Dataset assembly
adds `family` and `fixture_side`, pairing the Home/Away outputs of one feature.

Custom operators can preserve this contract to expose a `grid`, `region`, or
`scalar` output. Grids with negative values use a shared diverging scale; region
outputs highlight the relevant area beside their number; scalar outputs show a
number. `ConcentrationDiff`, `WeightedPresence` and CNN embeddings are future
operators, not implemented feature constructors. Embeddings will need a dedicated
representation; unsupported kinds are not rendered as fabricated pitch maps.

## Fixture-level pre-training reporter

```python
from xdiyo_analytics.analysis import PreTrainingAnalysis
from xdiyo_analytics.reporting import HeatmapReporter, TeamCatalog

analysis = PreTrainingAnalysis({
    "Spatial inputs": HeatmapReporter(
        type="per_fold", partition="test", cache_size=8,
        catalog=TeamCatalog.from_json("docs/analytics/team_assets/catalog.json"),
        show_badges=True,
    ),
})
report = analysis.run(dataset, split_plan=split_plan)
report.show()  # Local notebook: spatial values are fetched on demand.
# When finished with the notebook's live report:
report.close_live()
```

Choose a fold, a fixture and a spatial feature. The two panels show the **exact
prepared feature values for that fixture**, with Home/Away, names and embedded
badges. There is no additional averaging or retraining. Both panels use the
selected feature's for/against perspective. Wide screens show them side by side;
narrow screens stack them. Missing panels remain explicitly unavailable.

- `maps=None` includes every supported spatial feature family. Select discovered
  family names to restrict it. Incomplete selected-away grid cells raise an error;
  inspect intact grids before individual-cell feature selection.
- `teams=None` includes all fixtures; a list of team IDs keeps fixtures involving
  those teams, still displaying both participants.
- `max_fixtures=None` keeps every selected fixture. Set a positive number to cap
  retained fixtures per report scope. Inclusion counts are displayed explicitly.
- `cache_size=8` bounds recently viewed fixture/feature pairs in each active view.
- Overall and per-fold scopes use the existing train/test/score/all row contract.
- The fixture selector initially shows the last available fixture. Search matches
  team names and available league/season/round identifiers. Match IDs remain
  internal and are omitted from the visible labels.
- Grid panels share one fixed color scale per feature across the report scope.
  Display coordinates remap 0–100 to 105 × 68 without changing feature values.
- Download the selected pair's numerical values as JSON. Fixture tables remain
  available through the ordinary report downloads.

**Select preview fixtures after feature evaluation.** Build histories and evaluate
features from the full eligible match population, then subset both the feature
rows and metadata for reporting. Filtering to a few chosen teams' fixtures first
can remove their opponents' earlier matches and produce missing historical maps.
The linked preview computes histories from the full Premier League 25/26 season,
then displays 24 fixtures involving Chelsea or Manchester City.

### Live versus offline memory

The builder retains computed numerical artifacts on the Python side and fetches
only the selected pair over its token-protected read-only endpoint. Its report
iframe remains sandboxed. The local notebook viewer uses a token-protected
loopback service for the same purpose. Neither performs feature computation or
model fitting when a selector changes. Two plot panels are reused; hiding a fold
or study releases its plots and browser cache. Server-side numerical data still
occupies memory: on-demand rendering avoids sending all grids to the browser,
not the need to retain the feature values somewhere.

`report.live()` returns a handle with `.url` and `.close()`. `report.close_live()`
closes the notebook service and releases its retained data. Local loopback URLs
require the notebook kernel and browser to run on the same machine. For remote
notebooks use `report.to_notebook(live=False)` or an offline export.

```python
report.to_html("spatial_subset.html", spatial_limit=100)  # Per spatial scope.
report.to_html("spatial_full.html")  # All retained fixtures; potentially large.
report.to_notebook(live=False)  # Self-contained data instead of a local service.
```

Standalone files embed their included numerical grids, while still reusing two
plots. Ordinary experiment saves cap the automatic HTML spatial preview at **100
fixtures per scope** and display the included/total counts. Full numerical spatial
artifacts remain in the recovery bundle for live inspection and explicit exports.
Opening saved results does not refit. `max_fixtures` on the reporter, unlike the
HTML export limit, deliberately reduces what the report retains.

## Experiment builder

1. In **Features & ratings**, add a historical computation such as Rolling Mean.
2. Select **Heatmap** as its source. Choose the grid method, size, normalization
   and perspective. Gaussian settings appear only for Gaussian processing.
3. The recipe automatically requests `heatmap_points` when a Heatmap source is
   present, retaining it even when statistic bundles filter the statistics table.
   Seasons without an exported heatmap table need to be excluded or prepared
   with that table; an empty table represents unavailable maps.
4. On the historical operator, choose **All venues** or **Same venue**. Heatmap's
   orientation selects the shared home-oriented pitch or a team-relative frame.
5. For regional summaries, add **RegionMass**, select the historical Heatmap
   expression as its source, and choose **Own half** or **Opponent half**.
6. In **Pre-training analysis**, discover feature columns, add **Heatmap Reporter**,
   and select its scope and row partition. Optionally select spatial feature
   families, including RegionMass; fixture limits and cache size are configurable.
7. Team names and the configured local badge catalog are connected automatically.
   Choose a fixture to display both teams. The same recipe exports to Python or
   a notebook through the existing workflow.

Existing recipes do not acquire heatmap features automatically. No model is
retrained when you switch fixtures or features in a computed report. Restart an
already running notebook kernel and relaunch its builder to load updated classes.
