# Load separate tables from one season selection

`load_season` returns a `SeasonData` bundle containing exactly the requested
tables, their shared source provenance and per-table validation reports. It
resolves one season version once, then reads every selected table against that
same reference. It does not join tables or create features.

## Example

```python
from pathlib import Path
from xdiyo_analytics.data import load_season

project = Path("C:/Users/luisi/Documents/Programming/Python/xDiyo")
season = load_season(
    project / "data/xDiyo_data",
    "Premier_League_24_25",
    tables=("matches", "statistics", "pregame"),
    record_path=project / "docs/analytics/loader_demo_selection.json",
)

season.matches.shape                    # (380, 35) in this saved selection
season.statistics.shape                 # (98224, 18)
season.pregame.shape                     # (740, 8)
season.shots is None                     # True: shots were not requested
season.validation["statistics"]          # Structural result and gap positions
season.provenance["loaded_tables"]       # ('matches', 'statistics', 'pregame')
```

These shapes describe the saved Premier League 2024/25 export. They are not
requirements imposed on other seasons. Use [season inspection](season_inspection.md)
to discover available tables and columns before choosing inputs.

## Inputs and table selection

```python
load_season(data_root, season_stem, *, tables=("matches",), record_path=None)
```

| Argument | Contract |
| --- | --- |
| `data_root` | Prepared-export root, here `data/xDiyo_data`. |
| `season_stem` | Explicit publication stem, such as `Premier_League_24_25`. |
| `tables` | Nonempty ordered sequence of distinct nonempty string names. The default loads matches only. |
| `record_path` | Optional record outside the source root. Replay it if present; create it only after successful loading and validation if absent. |

Lists and tuples work. Bare strings, sets, dictionaries and generators are not
accepted table sequences. Names are not trimmed or case-normalized. Selecting
`statistics` explicitly requires `matches` somewhere in the same selection;
either order works. No dependency or context table is added implicitly.

All requested names are checked against the selected manifest before reading
any table file. Request order is retained in `tables`, `validation` and
`provenance['loaded_tables']`. A declared additional table without a convenience
property remains accessible through `season.tables[name]`.

## Return value and missing/unselected tables

| `SeasonData` field | Meaning |
| --- | --- |
| `tables` | Dictionary from exactly the requested names to separate DataFrames. |
| `provenance` | Season stem, selected names, version, manifest path, publication/manifest SHA256 fingerprints and copied manifest metadata. |
| `validation` | Dictionary of structured per-table validation results. |

`.matches`, `.statistics`, `.pregame` and `.shots` return the corresponding
DataFrame object from `tables`. Their initial loaded-state behavior distinguishes:

| Situation | Result |
| --- | --- |
| Table not requested | Its convenience property is `None`. |
| Requested table undeclared | `KeyError` before table reads; no bundle returned. |
| Requested file missing | `FileNotFoundError`; no bundle returned. |
| Requested table valid with zero rows | An empty DataFrame with its physical columns, not `None`. |

The table dictionary is mutable: these properties reflect subsequent caller edits
to it as well. They do not retain a separate immutable request registry.

## One selected version across all reads

With no existing record, the public season sidecar is resolved once. With an
existing record, its saved publication is replayed. Each requested table receives
the same `ResolvedSeason`; the loader does not repeatedly call
`load_season_table` or consult the current publication between reads.

The existing reader rechecks the pinned canonical manifest, file size, SHA256
and decoded row count for each table. A publication advancing or disappearing
after initial resolution does not change the selection. A pinned manifest or
selected file that changes or disappears causes failure with no fallback.
This fixes version selection; it does not lock files against external edits.

Rows, physical column order, nullable Arrow dtypes, exact large IDs and missing
values are preserved. Each table retains its own grain and row count. Loading
uses memory for all selected DataFrames plus the reader's temporary file bytes.

## Validation coverage

| Selected table | Work performed | Domain status |
| --- | --- | --- |
| `matches` | File integrity plus `validate_matches`. | `structure_valid` on successful return. |
| `statistics` | File integrity plus `validate_statistics` against selected matches. | `structure_valid` on successful return. |
| `pregame`, `shots`, other tables | File integrity and decoding; no implemented domain validation. | `not_implemented`. |

Match reports contain row count and missing/invalid kickoff event IDs. Timing
gaps emit a warning and retain all rows. Statistics reports contain row/reference
counts, optional team-ID presence, missing team/value positions and match sides
without any statistic rows. Statistics gaps do not emit a warning or imply that
associated display text is unusable.

`structure_valid` does not mean complete, model-ready or historically available.
Statistics retain the full `(event_id, period, group_name, key, side)` identity;
repeated keys in different groups remain separate. See the
[statistics validator guide](statistics_validation.md) for exact checks and
gap semantics. Pregame and shots receive whole-file integrity verification,
which is stronger than schema/footer inspection, but no automatic reference,
football-semantic or completeness checks.

## Independent metadata, mutable contents and snapshot reports

`SeasonData` is frozen at the field level: assigning a new `season.tables` value
raises `FrozenInstanceError`. Its existing dictionaries and DataFrames remain
mutable. Editing a frame or table dictionary does not refresh provenance or
validation reports. Revalidate changed inputs when needed.

Each selected frame receives its own deep copies in
`frame.attrs['xdiyo']['source']` and `frame.attrs['xdiyo']['validation']`.
Editing a nested bundle dictionary does not change frame attributes, and editing
one frame's attributes does not change the bundle or sibling frames. These copies
prevent accidental coupling; they do not make an immutable data archive. Pandas
transformations can discard attributes.

## Saved records and single-table compatibility

`_write_season_record` in `data/loading.py` is shared by both loading APIs.
It uses the existing version-1 record format: saved publication bytes and source
fingerprints. Records contain metadata, not copies of table data; retain the
referenced files. A saved record can be replayed from a relocated source root.

The bundle loader calls the writer only after **all** reads and available
validation succeed. A late missing/corrupt table or structural validation failure
creates neither a new record nor its parent directory. Existing records are never
overwritten. Exclusive file creation can raise `FileExistsError` if another
caller creates the record first; retry reuses it. Other write/I/O failures can
still fail the call after validation, so this is not a transactional file store.

Records written by either API can be read by the other, and both serialize
identical bytes for the same season selection:

```python
from xdiyo_analytics.data import load_season_table, validate_statistics

statistics = load_season_table(
    project / "data/xDiyo_data", "Premier_League_24_25", table="statistics",
    record_path=project / "docs/analytics/loader_demo_selection.json",
)
# The single-table loader's statistics validation behavior remains unchanged.
statistics.attrs["xdiyo"]["validation"]["status"]  # 'not_implemented'
report = validate_statistics(statistics, season.matches)  # Explicit standalone use

# The bundle loader has already performed the cross-table validation.
season.validation["statistics"]["status"]               # 'structure_valid'
```

## Errors

| Condition | Exception |
| --- | --- |
| Wrong `tables` argument type | `TypeError`. |
| Empty/invalid/repeated names, missing statistics dependency, record inside source | `ValueError` before resolution/table reads as applicable. |
| Undeclared table | `KeyError` listing missing and available names, before table reads. |
| Missing selected file or pinned manifest | `FileNotFoundError`. |
| Changed/corrupt bytes, invalid Parquet, inconsistent metadata or domain structure | `ValueError`. |
| Concurrent first record creation | `FileExistsError`; winning record is not overwritten. |

No partial bundle is returned. Other filesystem errors propagate. The function
does not join, pivot, aggregate, filter, normalize or fill data. It defines no
experiment population, model features, prediction-time availability policy or
provider-wide completeness requirement.

## Verification — bundle batch, 16 September 2026

- **311 analytics tests passed**, including **54 new bundle cases** and all
  existing single-table and statistics tests after the writer extraction.
  Command: `C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe -m pytest tests/analytics -q -p no:cacheprovider`.
- The existing saved selection returned matches **380 × 35**, statistics
  **98,224 × 18**, and pregame **740 × 8**. Matches/statistics had no reported
  timing/team/value/side gaps. Pregame retained `not_implemented` domain status.
  A separate descriptive check found pregame rows for 370 matches, with 20 match
  sides absent and no null positions among present rows. No rows were fabricated.
- Checkpoint 7 in [the walkthrough](../../notebooks/01_loader_walkthrough.ipynb)
  contains **21 new executed cells, including 10 code cells**. All 102 prior cells,
  outputs and notebook metadata were preserved; the notebook now has 123 cells.
  Only the new checkpoint ran in a fresh `misc314_py314` kernel. D-Tale was not
  rerun. The nonfatal Windows/ZMQ selector-thread warning recurred.
- Earlier source displays and fingerprint outputs are historical evidence.
  Checkpoint 6's old batch-level source guard includes the intentionally changed
  `loading.py` and package exports and will flag those changes if rerun unchanged.
  Checkpoint 7 explains the extraction and uses this batch's new baseline.

| Source | Verified SHA256 |
| --- | --- |
| `src/xdiyo_analytics/data/season.py` | `39beaa1364b3a52b529ebaccfb4e0064db3a448fc7a66768f2716ec1814c9ca4` |
| `src/xdiyo_analytics/data/loading.py` | `6db4b88170229571cc4b76cfa2d1b130de665674069cfe29ee5c9f7517e11da1` |
| `src/xdiyo_analytics/data/__init__.py` | `577a8b37cadebfa41c53d953ecc7cb8d65a6da6d14e614651c5694ebbb67700e` |

All eight analytics source hashes matched this batch's baseline, the immediate
before/after test captures, and the real-data/notebook checks. Selected source
files and the saved record remained unchanged. The statistics validator source
was unchanged from its prior verified implementation.

See [season_loading_check.json](season_loading_check.json) for complete
fingerprints, diagnostics and notebook preservation evidence, and
[test_season.py](../../tests/analytics/test_season.py) for the focused cases.
