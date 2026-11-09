# International statistic names

The authority is the native SofaScore schema-2 `statistics.parquet`, including
its key, group, period and display label. The audit reads the published Premier
League 2024/25 table and the original workbook headers. `schema-verification.json`
records the exact file hashes, observed native labels and publication checks.

| International field | Native group | Native key | Native label |
|---|---|---|---|
| corners | Match overview | cornerKicks | Corner kicks |
| yellow_cards | Match overview | yellowCards | Yellow cards |
| red_cards | Match overview | redCards | Red cards |
| xg | Match overview | expectedGoals | Expected goals |
| ball_possession | Match overview | ballPossession | Ball possession |
| total_shots | Match overview | totalShotsOnGoal | Total shots |
| shots_on_target | Shots | shotsOnGoal | Shots on target |
| shots_off_target | Shots | shotsOffGoal | Shots off target |
| fouls | Match overview | fouls | Fouls |
| goalkeeper_saves | Match overview | goalkeeperSaves | Goalkeeper saves |

Each field expands to `home_<field>_<period>` and `away_<field>_<period>`, with
`ft -> ALL`, `1h -> 1ST`, `2h -> 2ND`. [mapping.csv](mapping.csv) lists all 60
source columns and both orientations explicitly. The output convention is
`team::<period>::<group>::<key>::value` / `opponent::<period>::<group>::<key>::value`.
For a home-team row, home fields become team fields; for an away-team row they
become opponent fields. **Total shots are never renamed to shots on target.**
Off-target shots are not calculated by subtraction. No units, numeric values,
zeros, missing cells, identities or timestamps are changed. Possession stays
on its original 0–100 scale. No numeric coercion occurs in the new rename path.

## Native use and reversible lineage

```python
from xdiyo_analytics.data import homogenize_international_statistics

# A fixture-wide table as seen by its home team; every unmatched column survives.
renamed = homogenize_international_statistics(table, perspective="home")
lineage = renamed.attrs["international_statistic_names"]
original_names = renamed.rename(columns={v: k for k, v in lineage["mapping"].items()})
```

Persist the JSON-serializable lineage alongside exports that do not retain
DataFrame attrs. `international_statistic_mapping("home"/"away")` exposes the
full explicit dictionary. Repeated conversion preserves its lineage; changing
perspective requires restoring source names first. Native-name collisions and
duplicate input columns raise rather than merge or choose one observation.

`build_team_history(SeasonData({"matches": canonical_matches}, provenance))`
now recognizes these wide statistic columns directly and produces the same
team/opponent fields and `stat_columns` metadata as native long statistics.
It requires the ordinary canonical fixture/team IDs and timing fields: it does
not guess workbook IDs or manufacture kickoff times. The existing history
builder's context selection remains intact; arbitrary odds columns are not
automatically admitted as model predictors. On the wide renamer, odds and all
other unmatched columns retain their original names and values.

Unknown statistic-like home/away period names, e.g. `home_shots_ft`, remain
unchanged and are listed in lineage `unresolved`. They are not equated with
total or on-target shots. Scores remain governed by their existing score-basis
contract. Native display-group duplicates (e.g. shots in both overview and
Shots) are not combined; this mapping uses the existing importer groups.

## Existing publications and compatibility

The existing catalogue importer already uses these ten native identities.
The three local international publication versions each contain 263,400
statistic rows and already use the correct keys. This change does not rewrite
their immutable data or hashes. The audit reconstructs source-named wide
statistics, then compares both preparation paths across all 9,676 fixtures
(19,352 team rows) in each publication, including missing statistics and IDs.

The historical results TXT contains no shot/corner/card statistics. Those
observations remain missing. The workbook's statistics sheets have no other
unresolved statistic fields. Their unmatched identity/context columns and all
40 odds field names are preserved by the renamer and listed in the receipt.
The existing catalogue's long odds representation is unchanged.

Tests exercise every mapping, nullable integer/float values, true zero,
home/away reversal, exact large IDs, unknown names, collisions, row ordering,
lineage and the native GAT event-store/numeric-edge preparation. No model fit
is performed. GAT's existing xG exclusion and named-dataset membership/timing
gates are retained; compatible column names alone do not establish eligibility
for an experiment. There are no model, countdown or ranking changes.

## Reproduce the read-only schema audit

With `src` on `PYTHONPATH`:

```powershell
python examples/audit_international_statistic_names.py `
  --native-root data/xDiyo_data `
  --workbook '<original workbook.xlsx>' `
  --catalogue '<private named catalogue directory>' `
  --output docs/analytics/dataset-catalogue/statistic-names
```

The audit verifies source hashes and asserts original files remain unchanged.
Only aggregate receipts and the name mapping are written; it does not rebuild
publications, experiments or models.

## Test receipts

The 65 focused cases pass on both Python 3.14/pandas 2.3.3 and Python
3.12/pandas 3.0.1. Run `python -m pytest
tests/analytics/test_international_statistic_names.py -q` with `src` on PYTHONPATH.

The broader command runs that file plus `test_team_history.py`,
`test_statistics_workbook_backfill.py`, `test_named_datasets.py`,
`test_named_dataset_repairs.py`, `test_named_movements.py`, `test_features.py`,
`test_datasets.py` and `test_graph_integration_gates.py` under `tests/analytics`.
Including the focused additions and successful wheel reruns, final unique
outcomes are **309 passed** on Python 3.14; **307 passed, 2 failed** on Python 3.12.

The two pandas-3 failures occur in the archived `5ee87b57` GAT reference's
`graphs/statistics.py` (`ValueError: output array is read-only`) before candidate
comparison, for `gat_statistics` and `gat_combined`. Both reference comparisons
pass on pandas 2.3.3. This scope does not change frozen models. Synthetic CPU
regressions exercise bounded updates and checkpoint persistence; no research
experiment is launched. Initial packaging failures came from sandbox pip temp
directory permissions; both installed-wheel tests pass when rerun outside
that sandbox, with `PIP_NO_INDEX=1` and isolated installation directories.

`verification.json` reconciles the original suite receipts with targeted
reruns, records source hashes and names both limitations. Initial failure logs
are retained alongside successful reruns.
