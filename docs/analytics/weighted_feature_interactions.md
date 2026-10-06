# Interactions and weighted historical means

Implemented against checkout `c86aba5b149312899ffc660a41c69095aaf028dd`, after
heatmap Phases C/D. This is opt-in feature composition. Existing recipes, source
data and frozen research experiments are unchanged.

## Per-match interactions

Use the existing `RollingMean(Product(value, other), window=20, min_periods=1)`.
Each operand must select one numeric column and refer to the intended focal team
row. Explicitly select statistic period/field and for/against perspective. Both
operands are evaluated against the same history; they are not joined by length.
Source league/season identities and the native eligibility rules are retained.

Products are formed **before** the historical mean. For observations `[0, 10]`
and `[10, 0]`, the mean interaction is 0; the product of means is 25. Negative
inputs are valid; missing times zero and overflow are missing. Neither percentages
nor other units are silently converted. Multiplying percent-valued possession by
a position produces a value 100 times the fraction-valued interaction. Units are
the product of the two input units. Spatial lineage records the full expression,
and the reporter labels it as a derived value rather than an unchanged centroid.

The example `examples/weighted_feature_interactions.py` provides native feature
definitions and JSON-ready `recipe_features()`, including two non-possession
interactions and two non-possession weighting examples:

- Corners times centered longitudinal position `2 * mean_x / 100 - 1`.
- Goals scored times opponent-half minus own-half unit mass.
- Historical position weighted by corners or by goals scored.
- Historical double-angle channels weighted by corners.

The count weights intentionally give high-corner or high-goal matches more
influence. They are not measures of ball time. Zero-goal matches occupy window
slots but do not contribute to the goal-weighted mean. These are configurable
examples, not additional default features or selected research settings.

## Native paired reducer

```python
from xdiyo_analytics.features import RollingWeightedMean, SpatialPointSummary, Stat

feature = RollingWeightedMean(
    source=SpatialPointSummary("mean_x", side="for", kinds=("player",)),
    weights=Stat("ALL", "Match overview", "cornerKicks", field="value"),
    window=20,
    min_periods=1,
    venue="all",
)
```

For paired finite values and nonnegative weights in the selected window:

`weighted_mean = sum(weight * value) / sum(weight)`.

- Select the last `window` eligible matches first. Missing and zero weights do
  not extend that window to older matches.
- A missing/nonfinite value or weight removes the pair from **both** sums.
- `min_periods` counts finite pairs with **strictly positive weights**. Ordinary
  `RollingMean.min_periods` instead counts finite values.
- Zero total weight or too few positive contributors returns NaN. No epsilon,
  imputation, clipping of weights or invented fallback is used.
- A finite negative weight available in the selected window raises, even when
  its paired value is missing or late. Unavailable or out-of-window weights do
  not trigger that validation. `Abs(signed_feature)` is possible only as an
  explicit transform, with a different interpretation.
- Normal calculations rescale weights and values before accurate summation.
  Extreme magnitudes, subnormal intermediate contributions and severe signed
  cancellation use exact integer-ratio accumulation, rounded to float64 only
  after division. This retains small representable means through underflow,
  overflow-prone products and cancellation. There is no change to
  the existing `Product` overflow contract.
- A common positive scale on all weights leaves the result unchanged. A shift
  of all values shifts the mean equally; the result remains within the positive
  contributors' range, subject to ordinary floating-point rounding.

Do not replace this with a ratio of independently computed rolling means: that
can include an unavailable value's weight in the denominator.

## Cutoffs, scopes and input availability

The shared native window uses finished matches, strictly earlier kickoff,
availability at or before the prediction cutoff, and excludes the target fixture.
Default groups are team and competition across seasons. `venue="same"` restricts
the eligible population before selecting the window. `H2H(feature)` applies the
same ordered opponent scope to both inputs; mismatched operand H2H scopes raise.
Already historical features and rating states are evaluated at each past row's
own cutoff before the outer reduction.

Keep your explicit `cutoffs` and `available_at` when calling `evaluate_features`
or configuring the recipe. Group-first-kickoff minus one hour and historical
kickoff plus three hours are research proxies, not verified publication times.
Without `available_at`, the native retrospective kickoff proxy remains unchanged.

`source_available_at` and `weights_available_at` optionally name existing datetime
columns in **team history**, independently of the shared fixture availability:

```python
feature = RollingWeightedMean(value_source, weight_source, window=20,
    source_available_at="value_published_at",
    weights_available_at="weight_published_at")
```

These are masks **inside** the fixed eligible-match window. A missing publication
time means unavailable. Both inputs must be available by the target cutoff.
The supplied publication columns must correspond to the explicitly selected
for/against perspective; the reducer does not infer publication times by name.
The weight's availability alone governs negative-weight validation. Do not place
input-specific publication times into the shared fixture eligibility if you want
late inputs to keep their original window slots. These optional columns must be
provided by the caller; the builder does not invent publication data or add
arbitrary columns to the data loader.

For an interaction with distinct publication times, prepare a paired-publication
column: the later of the two times, or missing if either time is missing. Use
`RollingWeightedMean(Product(a, b), 1, source_available_at="pair_published_at", ...)`
to apply that pair mask inside the fixed window. Unit weights give the paired
ordinary mean. If the same shared availability applies to both operands, the
ordinary `RollingMean(Product(a, b), ...)` recipe is sufficient.

Raw observed components remain unsafe as current-match predictors. Do not add an
extra lag around these reducers merely to establish historical safety. Existing
whole-source validation still rejects malformed point maps before windowing.

## Spatial interpretation

Each match's exact point summary uses its existing point selection and unit-weight
contract first. The historical reducer weights **matches**, so a more densely
sampled map receives no extra influence. For an exact centroid this equals the
centroid of the same weighted mixture of unit-mass point distributions. A grid
centroid is a discretized approximation. Multiplying all point weights within
one map cannot change its normalized centroid.

Coordinates use the exported 0–100 pitch-coordinate metric, with the canonical
team frame and native against-side 180-degree rotation. They are not metres.
Missing/empty/insufficient maps and undefined geometry stay missing. Source
version conflicts and malformed coordinates retain native validation errors.

For principal axes use weighted means of `axis_cos2` and `axis_sin2` with identical
history and availability settings, never raw-angle averaging across the wrap.
The optional reporter helper `axial_direction_summary(C, S)` computes the angle
in radians modulo pi and consistency `sqrt(C*C + S*S)`. At resultant length
`<= 64 * float64 epsilon` the direction is undefined, while consistency is valid.
Undefined axes are missing in both input channels. No normalization back to a
unit vector and no automatic extra predictor are applied.

## UI, metadata and limits

Select **Rolling Weighted Mean** in the native feature builder, then choose Source
and Weights from the normal feature controls. JSON/Python/notebook exports retain
both expressions and settings. It produces one feature per team, hence a Home and
Away column in match layout. It also composes with arithmetic and other native
historical wrappers. The existing heatmap reporter discovers spatial descendants
and shows scalar results with captions identifying what is averaged and weighted.

`weighted_history_audit` travels with prepared frames, dataset definitions and
saved artifacts. Per-output records contain eligible/window counts, paired finite
and positive/zero-weight counts, missing/nonfinite and unavailable reasons,
total weight, log total weight, effective sample size and output usability.
Missing/nonfinite counts and unavailable counts are disjoint for each input;
positive and zero counts require the whole pair. If total weight exceeds float64,
the stored total is null with `total_weight_overflow=true`; its log and the mean
remain usable. Effective sample size uses positive paired weights only.
These diagnostics are not automatically added predictors. Heatmap reports expose
the diagnostic table for their selected fixture population.

Metadata retains the expression order, formula version, grouping, timing policy,
both inputs, known source units, and an input/timing/source-metadata fingerprint.
Undeclared nonspatial units remain explicitly undeclared; none are inferred from
names. Spatial raw-map coverage is distinct from paired output usability and
shared child lineage is deduplicated. Existing preparation signatures include
these expression definitions and data fingerprints; local evaluation caches are
rebuilt for each supplied history.

Formula version 2 fixes extreme signed underflow. Audit collection follows the
actual scoped evaluation dependencies, including cache hits. Reusing one
expression for ordinary and H2H histories therefore retains separate records,
independent of feature order or enclosing arithmetic/historical wrappers. Saved
version-1 artifacts are not rewritten. Rescaling invariance is subject to
rounding in the supplied floating-point weights themselves: a rescale that
already rounds an input to zero or infinity cannot be reversed by the reducer.

The MVP accepts scalar values and scalar weights. Multi-column/grid broadcasting,
league/LOO population reducers, seeded warm-up, weighted EMA and arbitrary kernels
are not supplied by this reducer. Use the existing separate APIs for those tasks.
No fitting, model-selection sweeps, data refresh or experiment rewrite is needed.

## Verification

The geometry/composition/history/assembly/persistence/UI regression suite passed
**663 tests**, with two optional sklearn array-API checks skipped because
`SCIPY_ARRAY_API` was unset. Three pre-existing warnings concern pytest generator
parametrization and intentional float32 overflow checks. The new deterministic
tests cover paired support, signs/zeros/missingness, fixed windows, late inputs,
scopes and identities, rating as-of semantics, extreme magnitudes, spatial
reflection/axial cancellation, source validation, and JSON/Python/notebook and
prepared-frame persistence. The final venue-control change is additionally tested
against the actual JavaScript visibility function; that final feature/UI/metric
regression run passed **116 tests**. No research model was fitted
and no source dataset was refreshed.

The follow-up edge-case repair passed **463 regressions**, covering ordinary/H2H
audit isolation in either feature order, nested wrappers, spatial report records,
signed underflow and subnormal contributions under exact power-of-two rescaling,
and row-order-independent cancellation. Shared evaluator checks also cover rating
states, seeded warm-up, league reducers, keyed assembly and UI recipes. The normal
weighted arithmetic path remains in use for ordinary non-cancelling inputs.
