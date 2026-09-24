# Dataset assembly API reference

Companion to the [practical guide](datasets.md) and
[coverage checklist](datasets_documentation_checklist.md).

## Feature identity option

`evaluate_features` is imported from `xdiyo_analytics.features`.

```text
evaluate_features(history, features, *, group_by=('team_id', 'competition_id'),
                  cutoffs=None, available_at=None, team_counts=None, ratings=None,
                  team_seasons=None, season_starts=None, keyed=False)
```

Only `keyed` is new. It must be a boolean. Default `False` keeps existing values,
row order, index and behavior. `True` attaches available partition/match keys,
`team_id` and `side` as a named MultiIndex and records the names in
`result.attrs['identity_columns']`. Match/team identity must be unique and
nonmissing, `event_id` must exist, and sides must be home/away. This option does
not require a full paired population on its own; assembly validates the pairs
needed by the selected label. Other arguments retain their existing semantics.

## Assembly entry point

Both public dataset APIs are imported from `xdiyo_analytics.datasets`.

```text
assemble_dataset(features, label, *, layout, feature_columns=None,
                 target_columns=None, drop_missing_targets=False) -> ModelDataset
```

| Parameter | Required behavior and defaults |
| --- | --- |
| `features` | DataFrame with a named MultiIndex or explicit identifier columns containing every label match key, `team_id`, `side`. Anonymous index/order never substitutes. |
| `label` | Exactly one `LabelData` selected from `create_labels(...)`, not the complete label dictionary. |
| `layout` | Required `match` or `team_match`; must equal `label.unit`. No automatic unit conversion. |
| `feature_columns` | `None` selects all candidate feature columns. A string or sequence selects an ordered subset of distinct nonempty names. Explicit required key columns are excluded from candidates. |
| `target_columns` | Same selector contract, applied to `label.y` columns before filtering and settlement extraction. |
| `drop_missing_targets` | Boolean, default `False`. When true, remove a whole match if any selected target cell is missing on any of its selected-label rows. |

MultiIndex identity levels are resolved by name, so level order may differ.
Required keys must be nonmissing, and feature records must be unique per
match/team and match/side. If a complete required MultiIndex is present, it is
the identity source; otherwise explicit columns are used. Extra feature rows
are permitted but must still have valid, unique identities. Every label observation
requires its feature record before any target filtering is applied.

Feature values are copied without recoding, sorting, aggregation, imputation or
numeric dtype conversion. A downstream adapter must validate suitability for its
chosen model. Column labels selected as features or targets must be nonempty
strings. All participating DataFrames must have unique column names.

## Label metadata validation

`label.y` and `label.metadata` must share their row order/index. The identity tuple
must contain distinct columns including `event_id`. It includes `team_id` exactly
for team-match layout and never includes `side`. Every identity value must exist,
be nonmissing and uniquely identify its label row. Match keys are those identity
columns except `team_id`.

| Layout | Additional metadata and consistency requirements |
| --- | --- |
| `team_match` | `team_id`, `opponent_id`, `side`; both home/away rows per match; distinct opponents and reversed pair IDs; feature side agrees with label side. |
| `match` | Distinct, nonmissing `home_id` and `away_id`; independently matched feature home/away team IDs must agree. |

Settlement, when present, must be a DataFrame with unique columns and the same
index/order as `label.y`. Every selected target needs a settlement column.
An existing metadata column named `settlement::<target>` raises a collision error.
Unselected settlement columns are ignored. Metadata copies all supplied label
context columns; it does not use context timestamps as join filters.

Invalid `features`/`label` types and nonboolean flags raise `TypeError`. Malformed
label tables raise `ValueError`. Missing selected columns or required label
metadata columns raise `KeyError`. Unsupported/mismatched layout,
missing/duplicate identities, missing feature records, conflicting team/side IDs,
empty/duplicate selectors and misaligned tables raise `ValueError`.
Nonempty column selections are required even for an empty row population.

## ModelDataset fields and groups

```text
ModelDataset(X, y, metadata, layout, identity_columns, match_columns,
             target_perspective, definitions=<new empty dict>)
ModelDataset.groups -> pandas.Series
```

| Member | Contract |
| --- | --- |
| `X` | Selected team features or the home-then-away feature blocks; original feature dtypes/missing values preserved. |
| `y` | Selected label columns with unchanged names, values and dtypes. |
| `metadata` | Label metadata with exact ID/context dtypes and selected settlement columns. |
| `layout` | Explicit match or team-match observation unit. |
| `identity_columns` | Selected label's identity tuple, retained exactly. |
| `match_columns` | Identity tuple without `team_id`; used for grouping and whole-match filtering. |
| `target_perspective` | Selected label's team/home/away/total perspective, without rewriting feature orientation. |
| `definitions` | Deep copies of `features.attrs.get('features', {})` and `label.definition`, under `features` and `label`. Explicit caller-built frames may have no feature definitions. |
| `groups` | Computed Series named `match_group`, dtype object, indexed like metadata, containing exact match-identity tuples. Not ranker group sizes. |

All three output tables have the same fresh RangeIndex and follow selected label
order after optional filtering. Team-match feature names are retained. Match
feature names are prefixed with `home::` and `away::`, preserving feature selection
order within each block. Complete team pairs remain together. Empty inputs or
filtering away all matches yield correctly shaped empty tables and groups.

The container is mutable, but assembled tables and definitions are independent
of the inputs. Assembly clears DataFrame attrs on the output tables; definition
provenance is available through the explicit `definitions` field instead.
It does not persist files, split data, train, predict or impute.

## Filtering and reporting boundaries

Target filtering uses missingness of selected columns only, at whole-match level.
It leaves missing feature values untouched and does not waive record validation.
Finite push/void encodings survive; settlement is metadata rather than a filter
instruction. Manually supplied infinite target values are not treated as missing
by assembly. Normal `create_labels` outputs already normalize nonfinite outcomes.

Use identity tuples for joins, metadata times/rounds for later temporal decisions,
and layout/perspective to interpret reports. A ranker may require contiguous
groups or size counts; a separate adapter must produce those deliberately.
Assembly makes no train/test split or model-specific ordering choice.

The guide includes executable [keyed/explicit forms](datasets.md#keep-feature-identities-attached),
[both layouts](datasets.md#choose-the-layout-explicitly),
[selection order](datasets.md#select-columns-before-assembling-and-filtering),
[pair-preserving filtering](datasets.md#preserve-pairs-when-filtering-targets),
and [settlement/group examples](datasets.md#metadata-settlement-and-downstream-groups).
