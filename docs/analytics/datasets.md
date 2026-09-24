# Assemble features and one selected target by identity

`assemble_dataset` joins already evaluated team features to one `LabelData`.
It produces aligned `X`, `y` and metadata for later splitting, reporting and fitting.
Choose `layout='match'` or `layout='team_match'` to agree with the selected label.
No observation-unit conversion, time split, imputation or model fitting is performed.

Use the [minimal notebook](../../notebooks/06_dataset_assembly_quickstart.ipynb),
[complete reference](datasets_reference.md) and [coverage checklist](datasets_documentation_checklist.md).

## Prepare a pinned history

```python
from pathlib import Path
import pandas as pd
from xdiyo_analytics.data import load_seasons, select_stats
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.features import Stat, ForAgainst, IsHome, Lag, RollingMean, evaluate_features
from xdiyo_analytics.labels import TeamValue, MatchTotal, Outcome, BetOption, create_labels
from xdiyo_analytics.datasets import ModelDataset, assemble_dataset

root = Path('C:/Users/luisi/Documents/Programming/Python/xDiyo')
data = load_seasons(
    root / 'data/xDiyo_data', ['24_25'], leagues='Premier_League',
    tables=['matches', 'statistics'],
    record_dir=root / 'experiment/initial_population/selections', verify_hashes=True,
)
history = build_team_history(select_stats(data, stats=[
    (period, 'Match overview', 'cornerKicks') for period in ['ALL', '1ST', '2ND']
]))
corners = Stat('ALL', 'Match overview', 'cornerKicks')
periods = Stat(None, 'Match overview', 'cornerKicks')
```

## Keep feature identities attached

```python
feature_definitions = {
    'home_context': IsHome(),
    'recent_corners': Lag(ForAgainst(corners, side='both')),
    'mean_corners': RollingMean(corners, 3),
}
features = evaluate_features(history, feature_definitions, keyed=True)
labels = create_labels(history, {
    'team_corners': TeamValue(periods),
    'match_corners': MatchTotal(periods),
    'away_outcome': Outcome(perspective='away'),
    'over_10': BetOption(MatchTotal(corners), 'over', line=10, push_value=0.5),
})
features.index.names
```

`keyed=True` retains feature values and row order but replaces the output index
with a named MultiIndex. Its levels are the available `source_league`,
`source_season`, `competition_id`, `season_id`, `event_id`, then `team_id`, `side`.
Match/team identities must be unique and nonmissing, with home/away sides.
`keyed=False` remains the default and preserves the original history index.

Assembly accepts this MultiIndex or explicit identifier columns. An anonymous
index, matching row count or apparent row order is insufficient. Keep identifiers
attached when shuffling, storing or transforming feature rows.

```python
shuffled_features = features.sample(frac=1, random_state=193)
match_data = assemble_dataset(
    shuffled_features, labels['match_corners'], layout='match',
)
explicit_feature_columns = shuffled_features.reset_index()
same_match_data = assemble_dataset(
    explicit_feature_columns, labels['match_corners'], layout='match',
)
pd.testing.assert_frame_equal(match_data.X, same_match_data.X)
```

All match keys declared by the label, plus `team_id` and `side`, are required in
the feature identity. Extra feature records are allowed. Every selected label
observation must have a matching feature record; match layout needs both sides.
Missing feature **values** pass through, while missing or duplicate identity
records raise errors. No rows are paired by a possibly duplicated pandas index.

## Choose the layout explicitly

Let \(k_m\) be the full match-identity tuple and \(f(k_m,t,s)\) the feature vector
for team \(t\) on side \(s\). The alignment is

\[
X_{m,t}=f(k_m,t,s_t)\quad\text{for team-match layout},
\]
\[
X_m=\left[f(k_m,h_m,\mathrm{home}),\ f(k_m,a_m,\mathrm{away})\right]
\quad\text{for match layout}.
\]

Here \(h_m\) and \(a_m\) are the label metadata's home/away team IDs. The joined
rows follow label order, and each output table receives a fresh RangeIndex.

| Layout | Feature columns | Label requirement |
| --- | --- | --- |
| `team_match` | Original selected feature names, one row per team-match. | `label.unit` is `team_match`, with complete, consistent home/away label pairs. |
| `match` | All `home::<feature>` columns, followed by all `away::<feature>` columns. | `label.unit` is `match`, with distinct, nonmissing home/away IDs. |

An away-outcome label retains away target perspective, but the feature blocks
still stay in home-then-away order. Assembly neither aggregates team labels into
a match nor duplicates a match label into team rows.

### Home/away blocks versus for/against statistics

Home/away names identify the focal team whose feature row is being placed in the
match matrix. For/against names describe the observations used to build that
team's feature. For example, the home block's previous **against** corners are
corners conceded by the home team in its previous eligible match. They are not
the current away team's feature. `ForAgainst` defaults to `side='for'`; choose
`side='both'` explicitly to produce both historical perspectives.

```python
team_data = assemble_dataset(features, labels['team_corners'], layout='team_match')
away_data = assemble_dataset(features, labels['away_outcome'], layout='match')
assert team_data.X.shape == (760, 4)
assert match_data.X.shape == (380, 8)
assert away_data.target_perspective == 'away'
match_data.X.head(4)
```

## Select columns before assembling and filtering

`feature_columns` and `target_columns` accept one string or an ordered sequence
of distinct nonempty names. Their default is all feature/target columns; explicit
identifier columns are excluded from default feature selection. Feature selection
uses the original names, before adding home/away prefixes. Selected target names
are unchanged. Empty, duplicate or unknown selections raise errors.

```python
all_period_target = next(
    name for name in labels['match_corners'].y if '::ALL::' in name
)
selected = assemble_dataset(
    features, labels['match_corners'], layout='match',
    feature_columns=['mean_corners', 'home_context'],
    target_columns=all_period_target,
)
assert list(selected.X) == [
    'home::mean_corners', 'home::home_context',
    'away::mean_corners', 'away::home_context',
]
selected.y.head(4)
```

## Preserve pairs when filtering targets

The default `drop_missing_targets=False` keeps prediction fixtures with missing
outcomes. When enabled, filtering considers only selected target columns and
removes an entire match if any selected target is missing on either team.

For selected target columns \(J\), and selected-label rows \(R_m\) belonging to
match \(m\), define

\[
B_m=\bigvee_{i\in R_m}\bigvee_{j\in J}
\mathbf{1}\{y_{ij}\text{ is missing}\},\qquad
\text{keep row }i\iff B_{m(i)}=0.
\]

Thus both rows survive or both are removed for team-match data. Missing values
in unselected target columns do not matter. Missing features never trigger this
filter. Feature records are validated before filtering, so a missing target does
not excuse a missing feature record.

```python
eligible_targets = assemble_dataset(
    features, labels['team_corners'], layout='team_match',
    drop_missing_targets=True,
)
assert eligible_targets.groups.value_counts().eq(2).all()
```

`create_labels` normalizes nonfinite observed outcomes to missing. Assembly itself
checks missingness, not finiteness: an infinity manually placed in `LabelData.y`
is not recoded or automatically removed. Finite push/void encodings are retained;
assembly applies no implicit settlement filter.

## Metadata, settlement and downstream groups

```python
option_data = assemble_dataset(features, labels['over_10'], layout='match')
option_data.metadata[['event_id', 'home_id', 'away_id', 'settlement::over_10']].head(6)
option_data.groups.head(4)
```

`metadata` copies the label's identities and context, preserving exact ID dtypes,
including unsigned IDs above signed 64-bit range. Context can include kickoff,
league/season, round and stage. No cutoff or chronological reinterpretation occurs.
For bets, selected target settlements are added as `settlement::<target>` metadata
columns. They never become features.

`ModelDataset.groups` returns a Series of complete match-identity tuples:

\[
g_i=k_{m(i)}.
\]

These tuples distinguish identical event IDs in different supplied partitions.
Each tuple appears twice for complete team-match pairs. They identify groups;
they are **not ranker group sizes**. The [splitters](splits.md) choose whole-match
membership and chronological or retrospective grouping. A later ranker/report
adapter must choose its required format. Reporters can consume `layout`,
`target_perspective`, identity/context metadata, and original definitions without
guessing a row's meaning. `definitions` records feature descriptions from
`features.attrs['features']` and the selected label expression.

## Scope and evidence

The assembler creates one selected model dataset. It introduces no prediction
cutoffs, temporal folds, preprocessing, imputation, model adapter or fitting.
The [feature timing contract](features.md) and [label observation contract](labels.md)
remain separate. Verification covers identifier joins, layouts, missingness and
preservation; it does not establish predictive performance.

See [datasets_check.json](datasets_check.json) for the stable-snapshot results.
Use the separate [splits and CV guide](splits.md) and
[split API reference](splits_reference.md) to prepare fold membership from this dataset.
