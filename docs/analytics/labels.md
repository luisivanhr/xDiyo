# Observed labels and explicit settlement

`create_labels` constructs realized outcomes for later training and reporting.
Each named result carries its numeric values, row identities, observation unit
and perspective. Features use `evaluate_features`; labels use `create_labels`.
Labels intentionally contain the actual outcome and apply no prediction cutoff,
lag, rolling window or warm-up policy.

Start with the separate [minimal notebook](../../notebooks/05_labels_quickstart.ipynb).
The [API reference](labels_reference.md) gives complete signatures, defaults,
schemas and errors. The [coverage checklist](labels_documentation_checklist.md)
records verification. [Dataset assembly](datasets.md) now joins a selected label
to keyed features; model fitting remains later work.

## Load a pinned observed history

This example uses the saved Premier League 2024/25 selection. It reads prepared
exports and uses their recorded versions; it does not collect new data.

```python
from pathlib import Path
import pandas as pd
from xdiyo_analytics.data import load_seasons, select_stats
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.features import Stat
from xdiyo_analytics.labels import (
    TeamValue, MatchTotal, Outcome, Above, BetOption, create_labels,
)

root = Path('C:/Users/luisi/Documents/Programming/Python/xDiyo')
data = load_seasons(
    root / 'data/xDiyo_data', ['24_25'], leagues='Premier_League',
    tables=['matches', 'statistics'],
    record_dir=root / 'experiment/initial_population/selections',
    verify_hashes=True,
)
history = build_team_history(select_stats(data, stats=[
    (period, 'Match overview', 'cornerKicks') for period in ['ALL', '1ST', '2ND']
]))
corners = Stat('ALL', 'Match overview', 'cornerKicks')
corner_periods = Stat(None, 'Match overview', 'cornerKicks')
```

Supply the complete paired history: exactly one home and one away row for every
match, with reversed team/opponent IDs and agreeing statuses. Both rows with a
missing status are accepted but yield missing ordinary labels. Optional partition
identifiers disambiguate repeated event IDs. These requirements also apply when
requesting only one team's values. See the [history contract](team_history.md)
for score and statistic provenance.

## Create several target definitions

```python
definitions = {
    'team_corners': TeamValue(corners),
    'corner_total': MatchTotal(corners),
    'team_outcome': Outcome(),
    'home_outcome': Outcome(perspective='home'),
    'away_outcome': Outcome(perspective='away'),
    'corner_win': Outcome(corners),
    'over_9_5': Above(MatchTotal(corners), threshold=9.5),
    'over_10_option': BetOption(MatchTotal(corners), selection='over', line=10),
    'home_draw_option': BetOption(Outcome(perspective='home'), selection='draw'),
}
labels = create_labels(history, definitions)
summary = pd.DataFrame([
    {'name': name, 'rows': len(item.y), 'columns': item.y.shape[1],
     'unit': item.unit, 'perspective': item.perspective}
    for name, item in labels.items()
])
summary
```

`labels` is a dictionary of `LabelData`, not a final combined target matrix.
Every `LabelData.y` is a numeric DataFrame, including a single-column target.
One output column takes the requested name. Multiple columns append the selected
source column identity. Names must be nonempty strings.

### Unit, perspective and identity

| Definition | Unit | Perspective | Row order and IDs |
| --- | --- | --- | --- |
| `TeamValue`, `Outcome(perspective='team')` | `team_match` | `team` | All input rows/index labels; identity includes `team_id`. |
| `MatchTotal` | `match` | `total` | Input home-row order/index; metadata has `home_id` and `away_id`. |
| `Outcome(perspective='home')` | `match` | `home` | Same home-row order and home/away metadata. |
| `Outcome(perspective='away')` | `match` | `away` | Uses the paired away observation, aligned to the same home-row order. |
| `Above`, `BetOption` | Inherited | Inherited | Preserve their child label's layout. |

Unit specifies what one row represents. Perspective specifies whose outcome that
row describes. An away-perspective outcome still has one row per match; it does
not switch to the input away-row order. Duplicate pandas index labels are retained.
Use `identity_columns` to join later; index labels alone need not identify rows.

Match identity uses the available columns from `source_league`, `source_season`,
`competition_id`, `season_id`, `event_id`, in that order. `event_id` is required.
Team-match identity additionally uses `team_id`. Integer IDs retain their exact
input types, including unsigned values beyond floating-point integer precision.
Already-rounded floating-point IDs cannot be repaired by this function.

```python
target = labels['corner_total']
target.y.head(4)
target.metadata.loc[:, list(target.identity_columns) + ['home_id', 'away_id']].head(4)
```

`metadata` and `y` have the same index and row order. Metadata also keeps available
`kickoff_at`, `round`, `stage` and `status`. Team-match metadata includes focal
`team_id`, `opponent_id` and `side`; match metadata replaces these with home/away
IDs and removes `side`. `definition` retains the expression that produced the
result. Each returned table is an independent copy; source history is unchanged.

## Quantities and outcome equations

Write \(x_{i,p}\) for team \(i\)'s observed statistic in period \(p\), and
\(x_{o(i),p}\) for its opponent's matching statistic. An ordinary label is defined
only when the match status is exactly `finished` and its required observations
are finite. Otherwise it is missing, written \(\mathrm{NaN}\) below. Missing
observations do not become zero, a draw or a void.

### Team values and match totals

For an eligible team row,

\[
T_{i,p}^{\mathrm{for}}=x_{i,p},\qquad
T_{i,p}^{\mathrm{against}}=x_{o(i),p},\qquad
T_{i,p}^{\mathrm{both}}=(x_{i,p},x_{o(i),p}).
\]

The two `both` columns remain separate; a missing value on one side does not hide
a finite value on the other. For a match \(m\), with home team \(h\) and away team
\(a\), the total is

\[
Q_{m,p}=\begin{cases}
x_{h,p}+x_{a,p},&\text{finished, both values and their sum finite},\\
\mathrm{NaN},&\text{otherwise}.
\end{cases}
\]

Both sides are required even if the other side is present. A nonfinite sum,
including floating-point overflow, becomes missing. Match totals count each
fixture once.

```python
period_labels = create_labels(history, {
    'both_periods': TeamValue(corner_periods, side='both'),
    'totals_by_period': MatchTotal(corner_periods),
    'against': TeamValue(corners, side='against'),
})
period_labels['totals_by_period'].y.head(4)
```

`Stat(None, ...)` expands each supplied period independently. It neither sums
halves into `ALL` nor treats `ALL` as a new derived total. With `side='both'`, own
period columns precede opponent period columns. Period/group/key/field identity
stays in expanded column names. Select only comparable measures for later use.

### Outcomes

Let \(G_{i,p}\) mean that the match is finished and both comparison values are
finite. For a statistic comparison with higher values preferred,

\[
O_{i,p}=\begin{cases}
1,&G_{i,p}\text{ and }x_{i,p}>x_{o(i),p},\\
0,&G_{i,p}\text{ and }x_{i,p}=x_{o(i),p},\\
-1,&G_{i,p}\text{ and }x_{i,p}<x_{o(i),p},\\
\mathrm{NaN},&\text{otherwise}.
\end{cases}
\]

Thus, for eligible finite values, \(O_{i,p}=\operatorname{sgn}(x_{i,p}-x_{o(i),p})\).
The implementation compares the values directly, avoiding subtraction overflow.
`higher_is_better=False` reverses the finite outcome sign. It requires a `Stat`.

With `source=None`, `Outcome` uses the history's existing result:

\[
O_i=\begin{cases}
1,&\text{finished with W},\\0,&\text{finished with D},\\
-1,&\text{finished with L},\\\mathrm{NaN},&\text{otherwise}.
\end{cases}
\]

These are score-based W/D/L values from provider `score_current` fields. The
function does not reinterpret extra time, reconstruct regulation-only scores or
resolve shootouts. Statistic outcomes compare the selected statistic rather than
match victory or its margin. Home/away perspective changes row selection only;
it introduces no extra target-time filter.

```python
outcomes = create_labels(history, {
    'away_corner_outcome': Outcome(corner_periods, perspective='away'),
    'fewer_corners': Outcome(corners, higher_is_better=False),
})
outcomes['away_corner_outcome'].y.head(4)
```

### Strict thresholds

For a finite child value \(q\) and finite threshold \(\tau\),

\[
A_\tau(q)=\begin{cases}
1,&q>\tau,\\0,&q\leq\tau,\\\mathrm{NaN},&q\text{ is missing}.
\end{cases}
\]

Equality is false. `Above` can wrap quantity, outcome or another threshold label,
but cannot wrap a settled `BetOption`. It preserves expanded columns and layout.

## Explicit settlement and numeric encoding

`BetOption` describes generic outcome/threshold settlement. Its `y` encodes wins
and losses, while its same-shaped `settlement` table preserves `win`, `loss`,
`push`, `void` and `missing` separately. There are no odds, profit calculations or
implicit bookmaker rules.

| Selection | Required immediate child | Winning condition |
| --- | --- | --- |
| `yes`, `no` | `Above` | Child is respectively \(1\) or \(0\). |
| `win`, `draw`, `loss` | `Outcome` | Child is respectively \(1\), \(0\) or \(-1\). |
| `over`, `under` | `TeamValue` or `MatchTotal` | Child is respectively greater than or less than finite `line`. |

For a finite binary child \(a\), let \(b=1\) select yes and \(b=0\) select no:

\[
S_b(a)=\begin{cases}\mathrm{win},&a=b,\\\mathrm{loss},&a\ne b.\end{cases}
\]

For a finite outcome \(o\), use \(c=1,0,-1\) for win, draw and loss selection,
respectively. Let \(D\) be the configured draw treatment, loss or push:

\[
S_c(o)=\begin{cases}
\mathrm{win},&o=c,\\
D,&o=0\text{ and }c\ne0,\\
\mathrm{loss},&\text{otherwise}.
\end{cases}
\]

### Draws and threshold equality

For a finite quantity \(q\) and line \(\ell\), let \(E\) be the configured
equality result, either push (default) or loss. Then

\[
S_{\mathrm{over}}(q)=\begin{cases}
\mathrm{win},&q>\ell,\\\mathrm{loss},&q<\ell,\\E,&q=\ell,
\end{cases}
\qquad
S_{\mathrm{under}}(q)=\begin{cases}
\mathrm{win},&q<\ell,\\\mathrm{loss},&q>\ell,\\E,&q=\ell.
\end{cases}
\]

For a win/loss selection, an outcome draw is a loss by default; `draw='push'`
changes it to a push. An outright `selection='draw'` wins on the draw even when
`draw='push'` is configured. `on_equal` affects over/under only; `draw` affects
win/loss only. Neither option changes yes/no settlement.

Every numeric line is a literal single threshold. For example, a line of
\(9.25\) compares directly with \(9.25\); it does not split stakes between two
Asian lines. Split lines and parlays are outside this increment.

```python
options = create_labels(history, {
    'over_10': BetOption(MatchTotal(corners), 'over', line=10),
    'under_10_equal_loses': BetOption(MatchTotal(corners), 'under', line=10, on_equal='loss'),
    'home_win_draw_push': BetOption(Outcome(perspective='home'), 'win', draw='push'),
    'not_over_9_5': BetOption(Above(MatchTotal(corners), 9.5), 'no'),
})
pd.concat([options['over_10'].y, options['over_10'].settlement.add_suffix('_settlement')], axis=1).head(6)
```

### Voids, missingness and precedence

Let \(V\) be the explicitly supplied status list. Final settlement is

\[
S=\begin{cases}
\mathrm{void},&\text{match status belongs to }V,\\
\mathrm{missing},&\text{otherwise, child value is missing},\\
S_{\mathrm{selection}},&\text{otherwise}.
\end{cases}
\]

`void_statuses` defaults to an empty tuple. A tuple or list of nonempty status
strings is accepted; a supplied list is copied to a tuple. Matching is exact.
Explicit void status takes precedence even when the observation is absent, or
when a configured status would otherwise produce a win, loss or push. No status,
including cancellation, is implicitly void. Other unfinished/missing results
remain `missing`.

With configured finite encodings \(p\) and \(v\) for pushes and voids,

\[
y(S)=\begin{cases}
1,&S=\mathrm{win},\\0,&S=\mathrm{loss},\\
p,&S=\mathrm{push},\\v,&S=\mathrm{void},\\
\mathrm{NaN},&S=\mathrm{missing}.
\end{cases}
\]

Both \(p\) and \(v\) default to missing when their parameters are `None`.
Any explicit mapping must be finite; it need not lie between zero and one.
Numeric encodings can coincide, but the settlement strings remain distinct.

```python
mapped = create_labels(history, {
    'mapped_over': BetOption(
        MatchTotal(corners), 'over', line=10,
        push_value=0.5, void_value=0.0, void_statuses=['cancelled'],
    ),
})
mapped['mapped_over'].settlement.value_counts(dropna=False)
```

The status choice above is an explicit illustrative rule, not a provider or
bookmaker settlement guarantee. Keep both tables when auditing outcomes.

## Checks and next steps

Input rows are never dropped to improve target coverage. Missing labels remain
available to [assemble_dataset](datasets.md), which keeps them by default or
optionally removes complete matches with missing selected targets. Match and
team-match outputs can coexist in this dictionary. Select one `LabelData` and
an explicit matching layout when joining to keyed features. Label construction
and assembly do not split data, fit models, calculate profit or create reports.

Verification status and exact evidence are in the
[coverage checklist](labels_documentation_checklist.md).
