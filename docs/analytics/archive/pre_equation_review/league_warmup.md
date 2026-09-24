# League features, warm starts and rating transitions

Use these APIs to produce aligned numeric features from observed team histories.
Every warm-up/transition policy is opt-in. Existing expressions without a wrapper
and `build_ratings(..., transition=None)` retain their previous behavior.
All three handoffs described below are implemented.

Start with the separate [minimal notebook](../../notebooks/04_league_warmup_quickstart.ipynb).
The [API and helper reference](league_warmup_reference.md) lists exact signatures,
schemas, extension points and restrictions. The [coverage checklist](league_warmup_documentation_checklist.md)
maps each addition to its explanation. Verification evidence is recorded in
[league_warmup_check.json](league_warmup_check.json).

## Load a bounded, pinned history

```python
from pathlib import Path
from xdiyo_analytics.data import load_seasons, select_stats
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.features import (
    Stat, ForAgainst, Lag, League, LeaveOneOut, RollingMean, RollingStd,
    RollingZScore, WarmStart, SeededEMA, Hard, LinearFade, ObservationCount,
    MatchResultGlicko, StatGlicko, Rating, evaluate_features,
)
from xdiyo_analytics.ratings import GlickoTransition, build_ratings, RatingRun

root = Path('C:/Users/luisi/Documents/Programming/Python/xDiyo')
data = load_seasons(
    root / 'data/xDiyo_data', ['23_24', '24_25'], leagues='Premier_League',
    tables=['matches', 'statistics', 'pregame'],
    record_dir=root / 'experiment/initial_population/selections',
)
history = build_team_history(select_stats(
    data, stats=[('ALL', 'Match overview', 'cornerKicks')],
))
corners = Stat('ALL', 'Match overview', 'cornerKicks')
```

This selection has 760 matches and 1,520 team rows. It preserves missing pregame
positions; the 1,480 pregame rows do not provide a complete final standings table.
Each saved selection pins its publication. No collector or source export changes
are needed. Membership and round completion refer to the supplied fixtures;
loading a truncated round cannot establish that the provider's whole round is complete.

## League populations and windows

`League(corners)` describes a population; wrap it in a rolling reducer to produce
a feature. The default is team contributions, completed-round scheduling, and a
round-count window. One fixture contributes two team observations. Use
`unit='match'` for one home-plus-away total per fixture; both values must be present
for a finite total. `ForAgainst(corners, 'both')` provides separate team/opponent
columns in a team population. It is not supported for match totals.

```python
league = League(corners)
loo = LeaveOneOut(league)
X = evaluate_features(history, {
    'team_mean': RollingMean(corners, 3),
    'league_mean': RollingMean(league, 3),
    'league_std': RollingStd(league, 3),
    'loo_mean': RollingMean(loo, 3),
    'total_mean': RollingMean(League(corners, unit='match'), 3),
    'relative_form': RollingZScore(loo, 3, reference=Lag(corners)),
})
```

The `window` argument counts the selected unit:

| `window_unit` | Window selection before LOO |
| --- | --- |
| `'rounds'` | Last N eligible round groups. Completed mode orders by completion time; kickoff mode by each group's latest eligible kickoff. |
| `'matches'` | Last N distinct fixtures, retaining all their selected team rows. Missing measurements still occupy a fixture slot. |
| `'days'` | Kickoffs in `[prediction_anchor - N days, prediction_anchor)`. The lower endpoint is included. |

Windows may cross seasons within one competition. Round keys default to
`('competition_id', 'season_id', 'round')`. Supply a tuple including a stage
column when round numbers repeat across stages. Keys must be present and nonmissing.
Partial windows are usable; nothing fills missing historical observations.

### Completed-round versus kickoff scheduling

`schedule='completed_rounds'` uses the **minimum supplied prediction cutoff among
all rows of the target round**. Its shared reference snapshot stays frozen for
every fixture in that round. A prior round qualifies only when every supplied
fixture row is finished, has usable kickoff/availability times, all its kickoffs
are strictly earlier than the anchor, and all releases are no later than it.
While another round is incomplete, the last eligible completed-round baseline stays
available. Postponed rounds enter only once their supplied fixtures qualify;
round-number order alone does not establish completion.

For example, round 1 is complete Monday. Round 2 has a Wednesday game and a postponed
Saturday game. Predictions issued Thursday still use round 1 in completed mode.
Kickoff mode can already use Wednesday's result when eligible. If a target round
starts Thursday and ends Sunday, completed mode keeps Thursday's snapshot for both
fixtures; kickoff mode can change before Sunday.

```python
rolling_live = RollingMean(
    League(corners, schedule='kickoff', window_unit='matches'), 5,
)
by_stage = League(corners, round_keys=(
    'competition_id', 'season_id', 'stage', 'round',
))  # history must contain stage before this expression is evaluated
```

### LOO and pooled moments

LOO means leave-one-out of the observation population, not LOCO model-importance
refitting. The order is **select window → exclude → aggregate**, with no refill.
`exclude='team_contributions'` removes the focal team's rows and retains opponent
contributions from its fixtures. `exclude='fixtures'` removes both sides of every
fixture involving that team. Match-total populations require fixture exclusion.

Suppose the selected fixtures contain `A:10, C:20` and `B:30, D:40`. For A:

| Population | Count | Mean | Sample variance (`ddof=1`) |
| --- | ---: | ---: | ---: |
| Full population | 4 | 25 | \(500/3\) |
| Remove A's contribution | 3 | 30 | 100 |
| Remove A's whole fixture | 2 | 35 | 50 |

Counts and moments are recomputed after exclusion. Standard deviations cannot be
subtracted. Three-round std pools all individual observations around their pooled
mean; it is not the std of three round means. Unequal round sizes therefore receive
observation weighting. Reducers ignore missing/nonfinite values after selecting
the window. `min_periods` counts finite observations, not rounds; it may exceed a
league round-window length. Std/Z require \(n>d\), where \(d\) is `ddof`
(default 1).

Explicitly, let \(W_t\) be the selected window and \(E_u\) the rows excluded
for focal team \(u\). For each statistic column, retain its finite observations
in \(I_{t,u}=W_t\setminus E_u\), and let \(n\) be their count. Then

\[
m_R=\frac{1}{n}\sum_{i\in I_{t,u}}x_i,\qquad
s_d^2=\frac{1}{n-d}\sum_{i\in I_{t,u}}(x_i-m_R)^2,
\qquad \operatorname{std}_d=\sqrt{s_d^2},
\]

where \(d=\texttt{ddof}\). The mean requires the configured minimum count;
variance additionally requires \(n>d\). For a round window, the sum runs over
individual observations in all selected rounds. Whole-fixture LOO removes both
team rows, or the single total when the observation unit is a match.

### Cutoffs and Z-score references

`cutoffs=None` uses each row's kickoff. A column name or aligned Series can specify
an earlier prediction time, including two days before kickoff:

```python
import pandas as pd
early = history.kickoff_at - pd.Timedelta(days=2)
early_X = evaluate_features(history, {'league': RollingMean(league, 3)}, cutoffs=early)
```

Pass `available_at='result_available_at'` only if that column carries actual usable
release times. With `available_at=None`, earlier finished kickoff is a retrospective
proxy, not proof that the result was known then. Eligible historical kickoffs are
strictly before the cutoff; releases may equal it. Missing query times yield missing
outputs; cutoffs after kickoff are rejected. Completed mode additionally freezes
all target-round rows at their earliest cutoff.

A team rolling Z-score defaults to its latest eligible observed value, included in
its rolling window. `reference=Lag(corners)` supplies that value explicitly.
A league Z-score **requires** an explicit historical or known-context expression.
Raw `Stat`/`ForAgainst` references are rejected, including when hidden inside
`H2H` or a no-op `WarmStart`. References pair positionally with population columns
and must have the same column count; matching units and period semantics are the
caller's responsibility. Missing reference, insufficient observations or zero
spread produce missing Z-scores.

For the chosen eligible reference \(x_{\mathrm{ref}}\),

\[
z=\frac{x_{\mathrm{ref}}-m}{s}.
\]

Ordinary rolling features use \(m=m_R\) and \(s=\operatorname{std}_d\).
Warm features use their blended mean and \(s=\sqrt{v}\). This expression is
evaluated only for finite inputs and strictly positive spread.

## Ordinary feature warm starts

```python
warm_features = evaluate_features(history, {
    'ordinary': RollingMean(corners, 3),
    'hard': WarmStart(RollingMean(corners, 3), SeededEMA(handoff=Hard(1))),
    'fade': WarmStart(RollingMean(corners, 3), SeededEMA(handoff=LinearFade(1, 2))),
    'count': WarmStart(RollingStd(corners, 3), SeededEMA(handoff=ObservationCount(5))),
})
```

`WarmStart(expr)` with `policy=None` is a no-op. Enabling `SeededEMA()` selects
`alpha=.5`, `league_weight=.5`, and `Hard(rounds=1)`. These are enabled-policy
defaults; unwrapped expressions still have no warm-up. `Hard(0)` returns the ordinary
child without additional warm-up round lookup; a League child still needs its own
population keys. `SeededEMA` supports rolling mean/std/Z
over a `Stat`, `ForAgainst(Stat)`, or its League/LOO population, not arbitrary
derived signals, lags or rating producers.

### Frozen priors and missing data

For a team feature, the mean prior comes from its eligible previous team-season
observations; without them, an available previous destination-league mean is used.
Known promoted/relegated teams blend their previous-team mean toward the previous
destination-league mean using `league_weight` (0 keeps the team mean, 1 uses the
league mean). Variance uses the matching previous-season **league population**,
or destination league for movers. Ordinary features never use rating top/bottom
cohorts. Period, statistic field, perspective and observation unit stay separate.

Writing \(\lambda=\texttt{league_weight}\), the initial team mean is

\[
m_0=\begin{cases}
(1-\lambda)\bar{x}_{\mathrm{team,prev}}
 +\lambda\bar{x}_{\mathrm{destination,prev}},
 &\text{known mover with both means available},\\
\bar{x}_{\mathrm{team,prev}},&\text{otherwise, when the team mean exists},\\
\bar{x}_{\mathrm{destination,prev}},&\text{otherwise, when the league mean exists}.
\end{cases}
\]

When neither mean exists, ordinary rolling is used. A missing destination mean
leaves an available team mean unchanged. The initial variance is the empirical
population variance of the matching previous league observations (destination
league for a mover):

\[
v_0=\frac{1}{N_L}\sum_{j=1}^{N_L}
 (x_{L,j}-\bar{x}_L)^2.
\]

It is centered on the league's own mean, even if \(m_0\) is team-specific.
With no finite league observations, \(v_0\) is missing.

League features seed from the previous league population; League LOO applies its
exclusion to that prior too. Match priors count totals once. Team history grouping
such as venue and H2H restricts its team mean prior and subsequent history.
All prior observations must qualify at the frozen season-entry boundary.

If no finite mean prior exists, return the ordinary rolling result. Missing prior
variance stays unknown during fractional-alpha warm-up; it is never silently zero.
If the rolling component is absent or below `min_periods`, keep the available seeded
moments. No padding or full-window requirement is introduced. At weight zero, the
original rolling operator—including its `min_periods` and `ddof`—is returned exactly.

### EMA and mixture equations

Let \(m,v\) be the seeded mean and **population variance**, \(x\) a new finite
observation, and \(\alpha\) the `alpha` parameter. Each eligible current-season
observation at/after entry updates:

\[
\begin{aligned}
\delta &= x-m,\\
m_{\mathrm{new}} &= m+\alpha\delta
                   =(1-\alpha)m+\alpha x,\\
v_{\mathrm{new}} &= (1-\alpha)\left(v+\alpha\delta^2\right).
\end{aligned}
\]

Missing/nonfinite observations are skipped without decay. At \(\alpha=1\), a finite
observation completely replaces the prior with \((x,0)\), including an unknown prior
spread. Updates follow eligible history order; league observations follow the
supplied chronological team-row order, so a sequential EMA is order-sensitive
within tied kickoffs. No averaging of a simultaneous league batch is implied.

For seeded moments \((m_E,v_E)\), ordinary rolling population moments
\((m_R,v_R)\), and seeded-component weight \(w\in[0,1]\), combine first and
raw second moments:

\[
m=w m_E+(1-w)m_R,\qquad
q=w(v_E+m_E^2)+(1-w)(v_R+m_R^2),\qquad v=q-m^2.
\]

Expanding the variance gives

\[
\boxed{v=w v_E+(1-w)v_R+w(1-w)(m_E-m_R)^2}.
\]

The between-means term is required. For \((m_E,v_E)=(2,4)\),
\((m_R,v_R)=(10,16)\), and \(w=0.25\), the result is \(m=8\), \(v=25\).
Endpoints return the selected pair without propagating missing values from the
unused pair. Weighted standard deviation is \(\sqrt{v}\). Warm Z uses these same warmed
moments and its eligible reference. After handoff, ordinary rolling ddof resumes.

The rolling component used in a blend is the population moment
\(v_R=n^{-1}\sum_i(x_i-m_R)^2\), computed directly from finite observations.
If starting from an ordinary sample variance \(s_d^2\), the equivalent conversion
is \(v_R=(n-d)s_d^2/n\), provided \(n>d\). A single observation still has a
defined empirical population variance of zero even when its sample variance with
\(d=1\) is undefined. The blend is a mixture of distributions; it is not the
sampling variance of an estimated mean and does not require independence of its
two components.

### All three handoffs

\(r\) counts completed current-season league rounds at the prediction time; \(n\)
counts finite new observations of this feature, after any population exclusion.
The weights below refer to the seeded EMA, not to the original prior alone.

| Policy | Seeded-component weight \(w\) | Example |
| --- | --- | --- |
| `Hard(k)` | \(w(r)=\mathbf{1}\{r<k\}\) | For \(k=1\): weights \(1,0,0\) at \(r=0,1,2\). |
| `LinearFade(start, rounds)` | \(w(r)=\min\{1,\max\{0,1-(r-r_0)/L\}\}\) | Defaults give \(1,1,0.5,0\) at \(r=0,1,2,3\). |
| `ObservationCount(strength)` | \(w(n)=\kappa/(\kappa+n)\) | For \(\kappa=5\): weights \(1,0.5,0.25\) at \(n=0,5,15\). |

Here \(k\) is `Hard.rounds`, \(r_0\) is `LinearFade.start`,
\(L\) is `LinearFade.rounds`, and \(\kappa\) is `ObservationCount.strength`.

The hard handoff is the simplest baseline and can jump at its boundary. One-round
warm-up mainly supplies opening-round priors; at least two rounds let later
predictions use the updated EMA. LinearFade provides a finite fade. ObservationCount
approaches zero asymptotically and does not require round counting. All are usable,
not just proposed alternatives. No model-quality superiority is claimed.

## Rating transitions and reuse

```python
transition = GlickoTransition(phi_scale=1.2)
rating_X = evaluate_features(history, {
    'ordinary': MatchResultGlicko(fields=('rating', 'rd')),
    'seasonal': WarmStart(MatchResultGlicko(fields=('rating', 'rd')), transition),
    'corner_seasonal': WarmStart(StatGlicko(corners, fields=('rating',)), transition),
})
run = build_ratings(history, transition=transition)
saved = root / 'experiment/league_warmup_demo/Premier_League_23_25/result_phi_1_2'
if not (saved / 'ratings.json').exists():
    run.save(saved)
loaded = RatingRun.load(saved)
reused = evaluate_features(history, {'rating': Rating('seasonal')}, ratings={'seasonal': loaded})
```

A transition acts once on full state before selecting fields or perspectives.
Generated streams distinguish result/statistic/period, engine configuration,
scope, comparison direction and transition policy. Warmed and unwarmed variants
have separate caches/states. `WarmStart(Rating(name), policy)` is unsupported:
transition a saved run through `build_ratings(transition=...)` before reuse.

### Glicko policies and rating-only cohorts

Defaults are `phi_scale=1`, `movement_phi_scale=1`, `shrinkage=.5`, `top=3`,
`bottom=3`, `rank_by='standings'`, `aggregate='mean'`. On known prior state, multiply
RD by `phi_scale` while preserving location. For a promoted/relegated team,
additionally multiply RD by `movement_phi_scale`, then blend the previous rating
with the destination prior by `shrinkage`. Sigma is unchanged.

The public and internal coordinates are related by

\[
\mathrm{rating}=1500+173.7178\,\mu,\qquad
\mathrm{RD}=173.7178\,\phi.
\]

Retained inflation preserves \(\mu\). A mover without an existing
state starts from the engine's initial state; only movement inflation applies.

With \(c=\texttt{phi_scale}\), \(c_M=\texttt{movement_phi_scale}\),
\(s=\texttt{shrinkage}\), and a nonempty eligible rating cohort \(C\),

\[
\mu_C=\begin{cases}
|C|^{-1}\sum_{j\in C}\mu_j,&\texttt{aggregate='mean'},\\
\operatorname{median}_{j\in C}(\mu_j),&\texttt{aggregate='median'},
\end{cases}
\]
\[
\mu_{\mathrm{new}}=(1-s)\mu_{\mathrm{old}}+s\mu_C
\quad\text{for a mover with an available cohort},\qquad
\sigma_{\mathrm{new}}=\sigma_{\mathrm{old}},
\]
\[
\phi_{\mathrm{new}}=\phi_{\mathrm{old}}
 c^{\mathbf{1}\{\text{prior state exists}\}}
 c_M^{\mathbf{1}\{\text{promoted or relegated}\}}.
\]

For retained teams or an empty mover cohort, \(\mu\) stays unchanged.
These are season-entry adjustments; subsequent match updates still use the
existing Glicko-2 engine.

Promoted teams use the bottom-n destination **prior-season** members; relegated
teams use top-m. Standings ranking uses the latest eligible provider pregame
position, not a reconstructed final table. Rating ranking uses this exact stream's
rating. Mean/median summarize selected rating values. Candidates with missing or
invalid standings are omitted in standings mode. A small cohort uses all available
eligible members; an empty cohort leaves location unchanged, while applicable
uncertainty inflation still occurs.

Priors are frozen before the entry batch, excluding the focal and other incoming
teams. Prior-season members now departing the league can remain in the cohort.
An explicit previous competition/season permits state transfer; it does not
automatically calibrate otherwise incomparable league rating scales. The policy
operates on the supplied common numeric scale. Custom cross-stream/calibration
rules require an explicit custom adapter; no automatic mapping is implemented.

### Entry timing and saved snapshots

The evaluator defaults entry to the earliest supplied prediction cutoff per
league-season. Direct replay defaults to first kickoff; pass `season_starts` when
preparing a run for earlier predictions. A start after the first query is rejected.
See [exact context schemas](league_warmup_reference.md#context-inputs).

At a timestamp, transitions run before equal-time result updates; all transitions
read frozen pre-batch states. A result released exactly at entry is processed after
the transition. Transition snapshots have `snapshot_kind='transition'` and
`snapshot_order=0`; result snapshots have `'result'` and order 1. Lookup can use
the transition at opening kickoff but excludes a snapshot containing a result
whose contributing kickoff is at/after the prediction boundary. Counts, cumulative
contributing-kickoff provenance, order and transition metadata survive save/load.

## Scope and limits

The independent check covers two pinned Premier League seasons and synthetic
movement/availability boundaries. It verifies arithmetic and temporal contracts,
not historical publication times, season-final standings, calibration quality or
predictive benefit. Newly seen teams remain unknown without movement evidence.
The existing ID-based movement core is reused through the new adapter.

Custom engines need their own transition adapters; Glicko coordinate formulas do
not apply to arbitrary embeddings. James-Stein borrowing, arbitrary model/graph
state dynamics, incremental replay, general orchestration, targets and pilot fitting
remain outside this batch. See the reference for supported compositions and scopes.
