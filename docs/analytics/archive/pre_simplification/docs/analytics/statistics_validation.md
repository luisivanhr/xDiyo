# Validate team statistics against match identities

`validate_statistics(statistics, matches)` checks the relationship between two
already loaded tables and returns a frozen `StatisticsValidationReport`. It
raises an error for malformed structure and reports gaps without changing rows.

## Load one saved selection, then validate explicitly

```python
from pathlib import Path
from dataclasses import asdict
from xdiyo_analytics.data import load_season_table, validate_statistics

project = Path("C:/Users/luisi/Documents/Programming/Python/xDiyo")
data_root = project / "data/xDiyo_data"
selection = project / "docs/analytics/loader_demo_selection.json"

matches = load_season_table(data_root, "Premier_League_24_25", record_path=selection)
statistics = load_season_table(
    data_root, "Premier_League_24_25", table="statistics", record_path=selection,
)
report = validate_statistics(statistics, matches)
asdict(report)
```

The existing record pins the same publication for both table loads.
`load_season_table` checks file integrity and validates matches. With this API,
statistics validation is an **explicit cross-table call**; it is not automatic, and
this call does not replace `statistics.attrs['xdiyo']['validation']`, whose
loader status remains `not_implemented`. Retain the returned report separately.
Individual-event `shots` remain a separate input for models that need them.

## Automatic validation through the season bundle

`load_season` now calls the same validator automatically when matches and
statistics are explicitly selected together:

```python
from xdiyo_analytics.data import load_season

season = load_season(
    data_root, "Premier_League_24_25", tables=("matches", "statistics"),
    record_path=selection,
)
season.validation["statistics"]  # {'status': 'structure_valid', ...report fields...}
```

The tables remain separate. The bundle retains a structured report and copies it
into the statistics frame's attributes. Standalone `validate_statistics` remains
available and does not attach its report or mutate either input. The single-table
loader's behavior is unchanged. See [season bundle loading](season_loading.md)
for selection dependencies, record handling and snapshot-report limits.

## Input contract

| Input | Required columns | Checks |
| --- | --- | --- |
| `statistics` | `event_id`, `period`, `group_name`, `key`, `side`, `value` | Valid event IDs, nonempty labels, exact sides, unique full grains, known match references. |
| `matches` | `event_id`, `home_id`, `away_id` | Positive integer identities, unique event IDs, different home/away teams for each match. |
| Optional `statistics.team_id` | No column required | Nonmissing IDs must be positive integers and match the referenced match's indicated side. |

Both arguments must be pandas DataFrames with unique column labels, including
optional labels. Nullable Arrow columns are supported. IDs must be actual positive
integer values: Booleans, strings, floats (including `7.0`), zero and negative
values fail. Required IDs cannot be missing. No conversion through float is used
to build match references, so large integer IDs remain exact.

`period`, `group_name` and `key` must be strings containing a non-whitespace
character. Validation preserves their original casing and surrounding whitespace;
it does not normalize them. `side` accepts exactly `home` or `away`.

Reference matches need only the three identity columns. The validator rechecks
that subset even when the statistics table is empty or references only some
matches. Competition/season identity, kickoff time, status and optional columns
are outside this reference check.

## Preserve the complete row identity

One statistics row is identified by:

```text
(event_id, period, group_name, key, side)
```

Two rows at that full grain raise `ValueError`, even if their numeric or display
values differ. The same key can appear under another period, group, side or match.
Case-distinct keys remain distinct.

In the saved Premier League 2024/25 export, event `12436410` has these `ALL`
period rows for `totalShotsOnGoal`:

| Group | Side | Value |
| --- | --- | ---: |
| Match overview | home | 5 |
| Match overview | away | 14 |
| Shots | home | 5 |
| Shots | away | 14 |

Keep those groups separate. Dropping `group_name` would make these valid entries
appear duplicated; adding values across groups could count the measure twice.
The `Shots` statistics group and the `shots` table are different inputs. A later
selector defines each model's period/group/key choices.

## Report fields

| Field | Meaning |
| --- | --- |
| `rows` | Number of input statistic rows. |
| `matches_referenced` | Number of distinct event IDs appearing in statistics, not the number of rows in the reference table. |
| `team_id_column_present` | Whether statistics actually contains the optional `team_id` column. |
| `missing_team_id_row_positions` | Tuple of zero-based statistic row positions with scalar null/NaN team IDs. If the column is absent, includes every input position. |
| `missing_value_row_positions` | Tuple of zero-based statistic row positions with scalar null/NaN values, including valid Arrow NaNs. |
| `match_sides_without_statistics` | Tuple of `(event_id, side)` pairs having no statistic rows at all, in reference-match row order and home before away within each match. |

Positions refer to the order passed into the function, regardless of repeated,
unsorted or MultiIndex labels. Inspect them with `.iloc`, for example:

```python
missing_values = statistics.iloc[list(report.missing_value_row_positions)]
missing_values[["event_id", "period", "group_name", "key", "side", "value", "display"]]
```

`display` is optional; select it only if the input includes it. A missing numeric
value can coexist with display text such as `7/10 (70%)`. The validator preserves
both and does not parse or fill values. Missing team IDs remain unknown.

A single row of any measure/period covers a match side for this report, including
a row with a missing value. A report with no uncovered sides therefore does not
establish completeness of each required model measure. The report is immutable;
the two DataFrames, their indices, source order, dtypes and attributes remain
unchanged on validation success or failure.

## Errors and empty inputs

| Condition | Result |
| --- | --- |
| Either argument is not a DataFrame | `TypeError`, naming the argument. |
| Required columns missing, or any column labels repeated | `ValueError`, naming the input and problem. |
| Malformed identity, label or side; duplicate full grain | `ValueError`, with positional evidence where relevant. |
| Malformed reference table, orphan event or supplied team mismatch | `ValueError`; reference identities must be consistent. |
| Missing optional team IDs or scalar null/NaN `value` | Successful report with gap positions. |
| Empty statistics retaining required columns | Zero rows/referenced matches; every reference side is uncovered. |
| Both tables empty and correctly shaped | Zero counts and empty gap tuples. |
| Nonempty statistics with an empty match reference table | Orphan-reference `ValueError`. |

Messages normally show at most the first five affected row positions; the report
retains all gap positions. Structural errors stop validation without returning a
partial report. They require investigation of the source or preceding
transformation; the validator does not repair inputs.

## Limits

This function validates structure and reports scalar missingness. It does not
enforce a numeric type/range policy, so nonmissing text, negative values or
infinity are not automatically errors. It does not verify measure-specific
completeness, provider fixture coverage, historical availability, match scope or
kickoff timing. It reads no files and checks no source hashes or version
provenance. Use the same saved selection and the existing loaders for those
separate source checks.

There is no joining, filtering, filling, aggregation, feature construction or
automatic attachment of the report to a DataFrame. A later selector must define
the statistics, context and completeness policies needed by each model.

## Checked state — standalone validator batch, 16 September 2026, Japan time

The results and hashes below record Checkpoint 6's original verified state.
The subsequent [bundle verification](season_loading_check.json) reran these tests
alongside the new API and records the updated package/helper fingerprints.
`statistics.py` itself remained unchanged.

- The full analytics suite passed **257 cases**, including **137 new statistics
  cases**, with the designated `.misc314` interpreter and the pytest cache
  provider disabled. Source fingerprints matched immediately before/after the
  final suite and through the real-data and notebook checks.
- The existing Premier League 2024/25 selection yielded **98,224 × 18 statistics**
  and **380 × 35 matches**. All 380 matches were referenced; missing team-ID rows,
  missing-value rows and match sides without any statistic rows were all zero.
  There were three periods, seven groups, 50 keys and four keys appearing across
  multiple groups. These are properties of this saved export, not a population
  or provider-wide completeness claim.
- Checkpoint 6 adds **23 executed cells, including 11 code cells**, to
  [the walkthrough](../../notebooks/01_loader_walkthrough.ipynb). It displays the
  imported code, real groups, synthetic gaps, caught errors and preservation
  checks. All 79 earlier cells and outputs were preserved; the total is 102.
  Only the new checkpoint ran in a fresh `misc314_py314` kernel; existing D-Tale
  cells were not executed. The nonfatal Windows/ZMQ selector-thread warning
  recurred.
- An initial fingerprint guard detected a source edit after the first baseline.
  The complete suite was rerun with immediate source captures; no pytest case
  failed and this verification task made no implementation edits. The guard
  failure was reported to the primary task. Final checks refer to the hashes
  below, not the earlier captured state.

| Source | Verified SHA256 |
| --- | --- |
| `src/xdiyo_analytics/data/statistics.py` | `4a13fda9ff4ee840af7c32afcade033ec48d6e15dcfb7af759893fc8b51180bf` |
| `src/xdiyo_analytics/data/__init__.py` | `054ea336427856a9471b18af3f42aa0c867f7c407fa1821c2f58aae60bde95e1` |

The [verification record](statistics_validation_check.json) contains all seven
analytics source hashes, the selection and selected-input hashes, suite output,
real-data summary and notebook execution/preservation evidence. Focused cases
are in [test_statistics.py](../../tests/analytics/test_statistics.py).
