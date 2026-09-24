# Label API and data contract

Companion to the [practical guide and LaTeX equations](labels.md) and
[coverage checklist](labels_documentation_checklist.md). Every definition below
is exported from `xdiyo_analytics.labels`. `Stat` is shared with
`xdiyo_analytics.features`.

## Entry point and result

```text
create_labels(history, labels) -> dict[str, LabelData]

LabelData(y, metadata, unit, perspective, identity_columns, definition,
          settlement=None)
```

`history` is a pandas DataFrame with the complete paired observation population.
`labels` is a nonempty mapping from nonempty string names to `LabelExpr` objects.
Each mapping entry returns one `LabelData`; dictionary order follows the request.
No file I/O, automatic filtering, feature evaluation or dataset assembly occurs.

| `LabelData` member | Exact contract |
| --- | --- |
| `y` | Numeric DataFrame; one or more columns; missing/nonfinite outcomes are NaN. |
| `metadata` | DataFrame with the same index/order as `y`, exact identifiers and available context fields. |
| `unit` | `team_match` or `match`, according to the child definition. |
| `perspective` | `team`, `home`, `away` or `total`; independent of row unit. |
| `identity_columns` | Tuple of metadata names identifying the row, including `team_id` only for team-match output. |
| `definition` | The requested `LabelExpr`, with all its configured settings. |
| `settlement` | `None` except for `BetOption`; then a DataFrame with the same shape/index/columns as `y` and five distinct settlement strings. |

`LabelData` is a mutable container. Expression dataclasses are frozen and hashable
for reuse within one call. Returned value, metadata and settlement tables are
independent copies, including when several names share the same expression.
The evaluator cache lasts for the current `create_labels` call only.

### Input schema and pairing

| Input | Requirement or use |
| --- | --- |
| `event_id`, `team_id`, `opponent_id` | Required, nonmissing identifiers; no conversion through float. |
| `side` | Required; exactly `home` or `away`; one of each per composite match identity. |
| `status` | Required; paired rows must agree, including both missing. Only exact `finished` yields ordinary labels. |
| `source_league`, `source_season`, `competition_id`, `season_id` | Optional partition identifiers; if supplied, included in match identity and required nonmissing. |
| `result` | Required only for `Outcome(source=None)`; W/D/L recognized, other/missing values remain missing. |
| `kickoff_at`, `round`, `stage` | Optional copied context; no chronological sorting or cutoff is applied. |
| Statistic columns | Numeric observations referenced by `history.attrs['stat_columns']`; each selected team column requires one matching opponent column. |

Composite match keys use available partition columns in this order:
`source_league`, `source_season`, `competition_id`, `season_id`, `event_id`.
Paired rows must identify distinct teams and reverse their team/opponent IDs.
Missing sides, duplicate sides, inconsistent IDs/statuses and missing identifiers
raise errors. The contract requires paired population even for a team-only output.
Input tables already containing rounded floating IDs cannot recover lost precision.

The statistic metadata dictionary maps each column name to `role`, `period`,
`group_name`, `key`, `field`. Roles are `team` and `opponent`. Group/key/field
match exactly; `Stat(period=None, ...)` selects all supplied matching periods.
The selected column order follows metadata insertion order. A missing source
raises `KeyError`; a missing/ambiguous opponent metadata match raises `ValueError`.
Use `build_team_history(select_stats(...))` to obtain this schema.

### Output metadata and naming

Team outputs preserve every row/index label and include the match keys plus
available `kickoff_at`, `round`, `stage`, `status`, `team_id`, `opponent_id`, `side`.
Match outputs follow input home-row order/index, rename team/opponent IDs to
`home_id`/`away_id` and omit `side`. This remains true for away-perspective outcomes.
No focal-team identifier is assigned to match output.

A one-column result uses its mapping name. Expanded results use the requested
name followed by `::` and the source column identity. `Stat(None)` may therefore
produce multi-column `y`, and `side='both'` can double that count. No synthetic
periods or averaging are introduced. Empty, correctly typed histories yield empty
tables with the declared layout. Extra input columns are not copied automatically.

## Expression signatures and settings

`LabelExpr` is the marker base class for this namespace; it has no standalone
evaluation behavior. Raw `Stat`, feature expressions and unknown subclasses are
not valid top-level labels. `evaluate_features` does not consume these labels.

```text
TeamValue(source: Stat, side='for')
MatchTotal(source: Stat)
Outcome(source: Stat | None=None, perspective='team', higher_is_better=True)
Above(source: LabelExpr, threshold)
BetOption(source: LabelExpr, selection='yes', line=None, on_equal='push',
          draw='loss', push_value=None, void_value=None, void_statuses=())
```

| API/parameter | Allowed values, effect and boundaries |
| --- | --- |
| `TeamValue.source` | `Stat`; own/opponent observations selected by matching period/group/key/field. |
| `TeamValue.side` | `for` default, `against` or `both`; always team-match rows with team perspective. Both sides are separate columns, not a difference or sum. |
| `MatchTotal.source` | `Stat`; sum home and away once per match, only if both operands and sum are finite. |
| `Outcome.source` | `None` default uses existing W/D/L; `Stat` compares own/opponent observations independently per period. |
| `Outcome.perspective` | `team` default preserves all team rows; `home`/`away` produce match rows in home-row order. |
| `Outcome.higher_is_better` | Boolean; default `True`. `False` reverses a statistic comparison and requires a `Stat`; it cannot reverse the meaning of W/D/L. |
| `Above.source` | Quantity, outcome or another threshold label; cannot be a settled `BetOption`. Unit/perspective/columns are inherited. |
| `Above.threshold` | Required finite real number; strict comparison, equality is false; booleans rejected. |
| `BetOption.source` | Immediate child must match the selection family below; nested arbitrary settlement is unsupported. |
| `BetOption.selection` | `yes` default or `no` needs `Above`; `win`/`draw`/`loss` needs `Outcome`; `over`/`under` needs `TeamValue` or `MatchTotal`. |
| `BetOption.line` | Required finite real number for over/under; must be `None` for other selections. Booleans rejected; fractional/negative thresholds remain literal. |
| `BetOption.on_equal` | `push` default or `loss`; controls over/under equality only. |
| `BetOption.draw` | `loss` default or `push`; controls a draw for win/loss selections. Outright draw selection still wins on a draw. |
| `BetOption.push_value` | `None` default yields NaN; an explicit finite real number encodes pushes. Booleans rejected. |
| `BetOption.void_value` | Same numeric rules as `push_value`, applied only to explicit void settlements. |
| `BetOption.void_statuses` | Empty tuple default; tuple or list of nonempty strings. Lists normalize to tuples at construction without retaining the caller's mutable list. A lone string, set, null or invalid element raises `TypeError`. |

Invalid sources/compositions raise `TypeError`; absent columns/statistics raise
`KeyError`; invalid numeric/domain choices and malformed paired populations raise
`ValueError`. Most validation occurs during `create_labels`; void-status container
normalization and validation occur when constructing `BetOption`.

## Missing observations and settlement

Ordinary labels require exact `finished` status and finite required observations.
Individual team-value columns preserve an available side independently. Totals
and statistic outcomes need both sides. W/D/L outcomes require a recognized
result, with the history builder's current-score and shootout limitations.
Nonfinite arithmetic results remain missing. No rows are dropped or imputed.

Settlement precedence is explicit void status, otherwise missing child, otherwise
selection rules. Wins/losses encode one/zero; pushes/voids encode missing by default
or the configured finite mappings. `settlement` preserves the category even when
two categories have equal numeric encodings. Missing observations alone never
imply a void. A configured `finished` void status deliberately overrides an
otherwise valid outcome. Status comparisons are exact and case-sensitive.

All equations and worked examples are in the [practical guide](labels.md):
[quantities](labels.md#team-values-and-match-totals),
[outcomes](labels.md#outcomes), [thresholds](labels.md#strict-thresholds),
[ties](labels.md#draws-and-threshold-equality),
and [void/missing precedence and encoding](labels.md#voids-missingness-and-precedence).

## Examples and boundaries

The examples below continue the practical guide's `history` and `corners` setup.

```python
from xdiyo_analytics.labels import LabelExpr, LabelData

reused = TeamValue(corners, side='against')
separate = create_labels(history, {
    'against_count': reused,
    'against_above_5': Above(reused, 5),
    'against_under_5': BetOption(reused, 'under', line=5, on_equal='loss'),
})
assert all(isinstance(item, LabelData) for item in separate.values())
assert isinstance(reused, LabelExpr)
assert separate['against_count'].y.index.equals(history.index)
```

The dictionary can mix match and team-match targets. A later assembler must choose
target columns, match/team layout, identity joins and missing-label policy before
aligning with features. These labels do not choose prediction times, train/test
splits, transformations, models or reporting groups. There is no odds/profit
calculation, automatic bookmaker settlement, Asian split-line handling or parlay
composition. Raw observations and all existing feature/rating behavior are preserved.
