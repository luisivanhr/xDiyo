# Explicit statistical warm starts and movement features

## Compatibility and configuration

`WarmStart(operator, SeededEMA(...))` opts in for that expression only. No
wrapper, or `policy=None`, leaves ordinary features unchanged. The first four
positional arguments of `SeededEMA` remain `alpha`, `handoff`, `league_weight`,
`round_keys`. Old recipes without `mode` use `mode="legacy"`: their full-season
priors, mover blending, population-moment updates and original handoff semantics
are unchanged. See [legacy warm-up](league_warmup.md).

| Mode | Initial state |
|---|---|
| `legacy` (default) | Existing full previous-season behavior |
| `uniform` | Team's final eligible rolling window at the season-entry boundary |
| `w_league_prior` | Uniform for nonmovers; pure destination-cohort replacement for known promoted/relegated teams |

`league_weight` is explicitly inactive in both new modes, even if a legacy value
is retained in an imported recipe. There is no hidden 50/50 blend. New modes do
not automatically enable or change Glicko transitions.

### Parameters

| Parameter | Meaning |
|---|---|
| `alpha` | Finite per-observation update weight in `(0,1]` |
| `handoff` | `Hard(rounds)`, `LinearFade(start, rounds)` or `ObservationCount(strength)` |
| `bottom` | Promoted-team destination cohort size |
| `top` | Relegated-team destination cohort size |
| `round_keys` | Full league-round identity used to measure completed rounds |
| `variance_prior` | `"within_team"`, the approved average of donor variances |
| `variance_estimator` | `"population"` for `ddof=0`; `"weighted_sample"` for `ddof=1` |
| `prior_strength` | Explicit effective strength greater than 1 for a corrected abstract cohort variance prior; required for league-prior SD/Z-score with `ddof=1` |
| `variance_fade` | `"estimate_interpolation"`: interpolate compatible variance estimates |

`top` and `bottom` accept positive integers or `-1` for every eligible donor.
Booleans, zero, smaller negative integers and floats are rejected. The API keeps
constructor defaults for convenience, but no production alpha, cohort size,
fade duration or corrected prior strength has been selected or tuned. Example
factories require these choices as arguments.

## Frozen boundary and source scope

The boundary **B** defaults to the earliest supplied prediction cutoff in each
competition-season. `season_starts` can supply an earlier boundary. Supply full
context before selecting prediction rows, or explicit boundaries, so a truncated
input cannot silently move the intended season start.

The own seed uses the identified **adjacent predecessor** competition-season,
including the final finished match if its kickoff is strictly before B and its
availability is at or before B. It is not the last pre-match feature row, which
would omit that final match. A mover needs origin identities to use origin
history; a stale visit to the destination does not establish its own seed.

The seed window retains the wrapped source, for/against perspective, window,
`min_periods`, H2H opponent, venue and custom grouping. Eligible matches with
missing measures still occupy window positions. Each column reduces its own
finite observations without extending the window to refill missing measures.
Grouping must contain `team_id` and `competition_id`. Adding `season_id` limits
the seed to the predecessor season; otherwise older observations in the same
predecessor competition may complete the window. Older seasons must have a
known start year strictly before the predecessor's year; if that ordering is
unavailable, only the identified predecessor is used. Newer origin seasons are
excluded even when they contain matches before B. Other group values are matched
to the prediction row. `venue="same"` matches its home/away side.

Historical children of nested expressions keep each historical row's own
cutoff. The new evaluator never unwraps a lag/rolling child into a raw statistic.
Two dated observations for the same team, competition and kickoff are rejected
as ambiguous on the new path. Undated target rows remain missing even with an
explicit finite cutoff; multiple undated fixtures do not block dated rows.
Legacy tie ordering is unchanged. IDs stay exact,
including unsigned IDs larger than the floating-point exact-integer range.

## Cohort construction

Use the destination league's immediately previous-season membership. Exclude
the focal team and known incoming promoted, relegated or administrative entrants;
departing prior members may remain eligible. Every donor's paired mean/variance
uses its own final eligible boundary window with the same source and scope.
Supplied entrant evidence applies even if that team has no current-season
fixture row. The complete evidence is retained in the audit and input hash.

For positive counts, rank eligible donors by their latest valid provider
**pregame standings position** available before B. This is a standings proxy,
not a reconstructed final table. Promoted teams use the largest positions;
relegated teams use the smallest. Exact team identity breaks ties deterministically
(lexicographic representation of the integer ID). Eligibility, finite values,
`min_periods` and sufficient dispersion support are checked before selecting the
top/bottom members. If fewer exist, use all available. `-1` requires no ranking
evidence. Every selected team has equal weight, regardless of its match count.

For H2H/custom scopes the same target opponent/group filters apply to donors;
they are never relaxed to invent a cohort. An empty compatible cohort follows
the fallback below. Snapshot construction never updates donors using another
incoming team's state or current-season result.

### Fallback

| Condition | Behavior |
|---|---|
| Uniform without a valid own seed | Ordinary result, including missing |
| League-prior mover without usable cohort | Own seed if valid, otherwise ordinary/missing |
| Known mover without origin history | Destination cohort can still initialize it |
| Unknown movement | No cohort assignment; evidenced own predecessor or ordinary fallback |
| Administrative entry | Preserve `other_entry` and its false flags; no mover cohort |

A missing-seed fallback stays on the ordinary path for that season and source
scope. Later results do not create a new entry seed. Every boundary seed is
frozen; later availability or standings cannot revise it retrospectively.

## Moment and estimator equations

For a finite own boundary sample of size \(n\), initialize

\[
\mu_0=\frac1n\sum_i x_i,\qquad
C_0=\frac1n\sum_i(x_i-\mu_0)^2,\qquad Q_0=\frac1n.
\]

Require \(n\ge\text{min\_periods}\), and for SD/Z-score also \(n>d\), where
\(d=\text{ddof}\). The population estimator returns \(v=C\) and requires
\(d=0\). The corrected weighted estimator requires \(d=1\) and returns

\[
v=\frac{C}{1-Q},\qquad Q=\sum_i p_i^2<1.
\]

For real boundary observations this equals the ordinary sample variance.
Other ddof values are explicitly unsupported in new modes. Ordinary and legacy
operators keep their existing ddof handling.

For a cohort of \(J\) eligible donors with compatible estimates \(v_j\), use
the **user-approved within-team target**:

\[
\mu_0=\frac1J\sum_j\mu_j,\qquad
v_0=\frac1J\sum_j v_j,\qquad s_0=\sqrt{v_0}.
\]

There is no between-donor mean term, and SDs are not averaged. Donor means 2 and
8 with population variances 1 and 9 give mean 5 and variance 5 (SD \(\sqrt5\)).
Their population-mixture variance would be 14, a different, unselected target.
Mean and variance share the same eligible donor set for each dispersion column.

A population cohort sets \(C_0=v_0\). For a corrected abstract cohort prior,
the caller supplies effective strength \(s>1\); initialize

\[
Q_0=1/s,\qquad C_0=v_0(1-Q_0).
\]

This is an explicitly assumed effective weight state, not a claim that the
cohort is a real pooled sample. Strength is neither donor count nor the sum of
donor match counts. It is required for league-prior corrected dispersion even
when a particular row ultimately takes the own-seed fallback. Mean-only
features need no variance strength.

Each eligible finite current-season source observation updates the float64
central state once, in chronological order:

\[
\delta=x-\mu,\qquad \mu'=\mu+\alpha\delta,
\]
\[
C'=(1-\alpha)(C+\alpha\delta^2),\qquad
Q'=(1-\alpha)^2Q+\alpha^2.
\]

Missing/nonfinite observations cause no update, decay or observation-count
increment. With \(\alpha=1\), mean becomes the latest value, population
variance is zero and \(Q=1\); corrected variance is missing. Zero spread yields
a missing Z-score. Materially invalid/nonfinite moment states raise; signed
observations are allowed. Missing/inadequate dispersion is never filled with zero.
The \(\alpha=1\) update assigns \((\mu',C',Q')=(x,0,1)\) directly, avoiding
cancellation or overflow from subtracting a very large previous mean.

## Handoff

Let \(r\) be completed current-season league rounds and \(j\) the number of
finite new observations for this source column. The EMA weight is

\[
w_{\rm hard}=\mathbf1\{r<R\},\qquad
w_{\rm fade}=\min(1,\max(0,1-(r-a)/R)),\qquad
w_{\rm count}=\frac{s}{s+j}.
\]

Here \(a\) is fade start, \(R\) its duration, and \(s\) the observation-count
handoff strength (separate from variance `prior_strength`). Postponed rounds
remain incomplete. ObservationCount approaches zero asymptotically; it has no
finite complete handoff.

With ordinary rolling estimates \(\mu_R,v_R\), the new modes explicitly use
**estimate interpolation**:

\[
\mu_* = w\mu_E+(1-w)\mu_R,\qquad
v_* = wv_E+(1-w)v_R,
\]
\[
\operatorname{SD}=\sqrt{v_*},\qquad
Z=\frac{x_{\rm reference}-\mu_*}{\sqrt{v_*}}.
\]

The default reference is the latest eligible historical source value, included
in its window; a supplied reference must pass the existing historical/known-context
safety check. Missing latest/reference stays missing. This fade combines variance
estimates, not distributions; no between-mean or overlap-weight correction is
applied after fading. Legacy moment blending remains unchanged.

While \(w>0\), an unavailable ordinary estimate leaves the usable EMA state in
place. At \(w=0\), return the **exact ordinary output and missingness**. Ordinary
history keeps its configured grouping across seasons. A returning team's
ordinary destination history can therefore include old destination visits;
that is distinct from its adjacent-origin seed.

## Movement features

```python
features = {
    "was_promoted": TeamMovement("promoted"),
    "was_relegated": TeamMovement("relegated"),
}
```

These are known context leaves, usable directly: numeric 1/0 when known and
missing when unknown. Known retained teams are 0/0; administrative `other_entry`
is also 0/0, without implying retention. No imputation or extra venue feature is
added. Existing keyed assembly supplies home/away prefixes in match layout and
the focal team's values in team-match layout.

Explicit `team_seasons` records override native history **by exact
competition/season/team ID**, including explicit unknown/null evidence. Absent
records use native `team_season_entry`, `team_got_promoted`, `team_got_demoted`.
Without either source flags remain unknown. Conflicting statuses/flags, duplicate
record keys, inconsistent native values and partial predecessor IDs fail clearly.
Full transition evaluation additionally validates adjacent predecessor identity.
Native flags alone do not contain origin IDs: request the `team_seasons` table
for movers' own-history warm starts. Absence from a loaded subset never establishes
promotion, relegation or a false flag.

Each flag has independent missingness: `False/None` produces `0/missing`, and
`True/None` produces `1/missing`. Known negative flags without a movement status
do not establish retention or administrative entry. A true flag can establish
the corresponding movement for warm-start routing. A known status supplies
flags omitted from a status-only record, but an explicitly null flag stays
missing. Evidence records expose `movement_basis` (`status` or `flags`).

## Coverage matrix

| Family | New-mode support / intentional boundary |
|---|---|
| Stat, all periods, numeric fields, ForAgainst | Mean/SD/Z-score per emitted column, independent missing masks |
| MatchScore current / for / against / both | Same generic rolling adapters, existing score basis |
| Heatmap grid/Gaussian | Cell-wise moments; original units/dimensions/metadata; final away 180-degree rotation retained |
| RegionMass | Reduce each match to a regional observation **before** SD/Z-score; summing cell SDs is not regional SD |
| RollingMean | Boundary mean, per-observation updates, exact ordinary endpoint |
| RollingStd / RollingZScore | Population ddof=0 or corrected weighted ddof=1; other ddof rejected |
| RollingSkewness | Population third/second central moments; at least three finite seed observations; moment-mixture fade. See [rolling skewness](rolling_skewness.md) for cohort and LOO conventions |
| Arithmetic Sum/Product/Difference/Ratio/Constant | Per-match derived observations can be reduced; warmed leaves can feed arithmetic. Original scalar-column restrictions apply |
| Nested temporal children | Historical child values retain their own cutoff and operator semantics |
| H2H / venue / custom groups | Filters retained in own, donor, update and ordinary windows; require team + competition grouping |
| League / LeaveOneOut population source | New team-seed mapping explicitly rejected: population units/schedules have no unique team predecessor. Ordinary and existing legacy reducers remain unchanged |
| Direct EMA wrapper | Explicitly rejected: native span/initialization has a separate state contract. Native EMA and EMA as a historical child remain available |
| Direct Lag wrapper | Explicitly rejected: exact nth-match semantics remain unchanged, including missing observations. Lag can be a historical child |
| Ratings / Glicko | Existing independent transition policies unchanged; Glicko count validation still excludes -1 |
| Known context / preparation | Existing IsHome, standing, rest/calendar/identity behavior unchanged; TeamMovement added |

For pooled League/LOO expressions, the rejection refers to a rolling operator
whose selected population is League/LOO. An already reduced historical scalar
can still be a nested observation and retains its own source computation.

## Recipes, UI, provenance and examples

The feature tree exposes **Team Movement**, and Warm Start → Seeded EMA exposes
mode, handoff and relevant cohort/variance controls. Cohort counts appear for
league-prior mode; legacy blend weight appears only for legacy. Corrected cohort
dispersion exposes optional prior strength, which must be explicitly supplied
before evaluating that configuration. Choose matching `ddof` on SD/Z-score.
Python and notebook export use the same native recipe components.

[Example factories](../../examples/explicit_warmup.py) provide both modes,
shots/shots-on-target/goals, multi-period corners, derived goal differences,
spatial/region examples and flags. Call `feature_definitions(...)` or
`native_recipe(...)` with explicit production choices. To prepare only:

```python
recipe = native_recipe(data_root, mode="w_league_prior", alpha=alpha,
    fade_start=fade_start, fade_rounds=fade_rounds, bottom=bottom, top=top,
    variance_estimator="population", prior_strength=None)
prepared = prepare_recipe(recipe)  # no model fitting
```

For uniform, change only `mode="uniform"`. Corrected SD/Z-score recipes use
`variance_estimator="weighted_sample"`, matching `ddof=1` and explicit cohort
`prior_strength` when applicable. Factories do not modify any saved model recipe.

`feature_frame.attrs["warm_start_audit"]` retains seeds, boundary, exact team/
season/predecessor IDs, movement evidence, parameters, scope, selected donors,
pregame ranks, donor moment/count states, event IDs, last eligibility times and
fallback reasons. Arithmetic outputs retain descendant seed audits.
`movement_evidence` retains classification, including administrative/unknown.
Keyed assembly carries these into `dataset.definitions` and native recovery
preserves the policy. `warm_start_input_hash` binds history, movement/transition
records, eligibility times and resolved statistical/spatial source values; normal preparation identity also includes the
source/configuration. Changes cannot silently reuse an old prepared warm state.

Without explicit `available_at`, eligibility continues to use earlier finished
kickoffs as a **retrospective assumption**, not proof of publication time.
No fitted preprocessing, model selection or training is performed by warm-up.

## Deterministic seed examples

| Synthetic case | Result |
|---|---|
| Own history 0,0,12,18; window 2 | Boundary mean 15 |
| Last value 18 unavailable at B | Window 0,12; mean 6, frozen even after 18 becomes available |
| Seed 10; alpha .5; new 14,8 | EMA 12, then 10 |
| Own 20; donor means 4,8 | Pure cohort seed 6; new 10 at alpha .5 → 8 |
| Donor means 2,8; population variances 1,9 | Mean 5; variance 5; SD sqrt(5) |
| Mean 4, variance 9; alpha .25; new 10 | Mean 5.5; population variance 13.5 |
| Same update with initial Q=.5 | Q=.34375; corrected variance 13.5/.65625 |
| LinearFade(1,2), completed rounds 0,1,2,3 | Weights 1,1,.5,0 |

See [acceptance tests](../../tests/analytics/test_explicit_warmup.py) and
[UI/preparation checks](../../tests/analytics/test_ui_explicit_warmup.py).
Validation outcomes are recorded in [warm-start validation](explicit_warmup_validation.md).
