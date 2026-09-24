# Football analytics implementation progress

## Current agreement — 16 September 2026

**Latest workflow supersedes the detailed walkthrough below:** the user explicitly requested a simpler experiment-loading library and a minimal notebook. Normal usage now consists of `load_seasons`, `load_season` and `inspect_season`; no public validation reports/status dictionaries or lessons on validator internals. Keep essential checks internal, preserve separate tables and nullable data, and make full-table hashing opt-in. The old implementation history below is historical. The background task is authorized to archive the long notebook, replace it with a short usage notebook, migrate obsolete tests and consolidate documentation around the new API. Detailed function-by-function validator teaching is no longer requested.

Restart directly in the saved xDiyo project, using Astra with High effort. New library code belongs under `src/`. The previous implementation worktree was explicitly discarded; its draft code is not the starting implementation.

**Current experiment population:** use `C:/Users/luisi/Documents/Programming/Python/xDiyo/data/xDiyo_data` and **all available leagues for 2022/23, 2023/24 and 2024/25**. The user selected this population for the experiment and first feature implementations. This supersedes the earlier four-season inspection range and unselected league subset. The legacy directory remains out of scope. The current publication inventory contains 39 matching league-season exports across 13 leagues; these describe the collected cohorts, not a claim of complete provider fixtures.

The primary owns implementation and keeps the conversation focused on the practical experiment-loading workflow. The reusable background task owns tests, notebooks and documentation, records the checked source state, and sends callbacks only for actual failed tests. The user no longer wants validator-internal teaching or lengthy function-by-function checkpoints. Archive originals before replacing tests/guides/notebooks so edits stay recoverable.

**Background verification workflow (user-requested):** the delegated task owns tests and notebook edits for the assigned batch, preserves existing user-edited cells, and records results in the relevant documentation/tracker. Record which code state was checked and serialize notebook edits across batches. The primary agent owns implementation changes. The background task sends a message back to this task **only if a test fails**, including the failing case and useful diagnostic evidence; successful runs produce no callback. Notebook issues are recorded in the task/tracker and handled there within scope. Do not wait or repeatedly poll in the foreground. Mark code as implemented with verification pending until results are recorded; do not claim unrun tests passed. This workflow supersedes the earlier requirement to finish tests and the executed notebook before returning to discussion.

Keep this tracker current with completed work, the immediate next steps, decisions, checks, and deferred work. A function is not marked complete merely because its file exists.

Reusable background task: **Verify xDiyo functions and maintain walkthrough**, ID `01a0a5f9-b838-7080-b248-9cb58fb38a73`, host `local`, saved project checkout. Send subsequent approved verification/notebook batches there; callbacks to primary task `01a0a37d-7dc6-7422-9402-227a13350f37` are for failed tests only.

## Multi-season loading — implemented, verified and first population imported

## Statistic selection — implemented; background verification pending

The user requested general statistic selection rather than corner-specific logic, with composable attack/defense/all bundles and extra statistics/context. `data/selection.py` implements `select_stats`, `list_stat_bundles` and editable `STAT_CATEGORIES`. Attack and defense each have one period-independent list of `(group_name, key)` pairs. Their `all`, `totals`, `first_half` and `second_half` variants filter that shared selection; the halves are not derived from totals. `all_stats`, `all_totals_only`, `all_first_period_only` and `all_second_period_only` are supported aliases. All-stat selections include uncategorized measures; classifications are initial choices and can be overridden per call.

Selections form a union with exact `(period, group_name, key)` triples. Groups/periods, rows, values and missingness remain distinct; no pivot, join or aggregation occurs. `standings` selects pregame position plus available identity/timing context in its own table. Matches pass through when loaded, while the separate shots table is outside this selector. There is no dependency on coverage or source_key. Team-history construction, features and fitting remain the next implementation stages. Verification and the minimal notebook/docs update are delegated; do not claim tests passed until results are recorded.

### Previously verified multi-season import

`load_seasons(data_root, seasons, *, leagues=None, tables=("matches",), record_dir=None, verify_hashes=False)` discovers current publications for the requested season labels and optional exact league names, calls the existing loader once per selected partition and concatenates each requested table separately. The result is the same `SeasonData` interface. Added `source_league` and `source_season` columns identify every row's origin; existing columns with these names cause an explicit collision error. Source values/order/nulls remain unchanged; optional schema columns are combined with missing values, without artificial observations or joins.

Discovery is deterministic (input season order, then league name). Missing requested seasons/leagues or failed publications are errors; missing individual league-season combinations are not fabricated. `record_dir` optionally saves/reuses one existing-format record per successful partition, while provenance lists the exact loaded population. These records pin per-season versions, not discovery membership; future matching publications may expand a fresh discovery. Partial successful partition records may remain if a later load fails.

**Actual import, 16 September 2026:** all **39 publications across 13 leagues** for `22_23`, `23_24` and `24_25` loaded successfully. Separate frames contain matches **13,976 × 37**, statistics **3,303,574 × 20**, and pregame **27,072 × 10**, including the two origin columns. All 13 leagues have a publication in each of the three seasons; missing league-season combinations are empty for this snapshot. These are collected cohorts, without a provider-wide fixture completeness claim. Shots remain independently selectable. No features, joins or fitting were run.

- [x] Preserved all 36 existing slim API tests unchanged and retained 19 multi-season cases in `tests/analytics/test_multiseason.py` after the user's scope clarification. **55 tests passed** with `.misc314 -m pytest tests/analytics -q -p no:cacheprovider`. Tests cover top-level discovery, exact filters/order, absent combinations, source values/nulls/optional columns, independent shots, generic unavailable tables, reserved origin columns, records/version pinning/new discovery, later failures, and opt-in hash forwarding.
- [x] Compared all **117 loaded table partitions** with direct source Parquet. Values, physical source columns, types, row order and nulls matched exactly. All **195 selected sidecar/manifest/table fingerprints** remained unchanged; selected table hashes also matched their manifest descriptors.
- [x] Saved **39 version-1 selection records** in `experiment/initial_population/selections`, outside the source directory. The exact population and versions are recorded in `docs/analytics/multiseason_import_check.json`; future discovery can expand even when these records are reused.
- [x] Kept the notebook at **9 cells, including 6 code cells**, and executed it top-to-bottom in a fresh `misc314_py314` kernel. It shows the chosen seasons, one-season catalogue/column access, the actual multi-season load with saved selections, league-by-season match counts and small samples. The known nonfatal Windows/ZMQ selector-thread warning recurred; no cell errors or D-Tale controls occurred.
- [x] Archived the prior notebook and six affected guides/notes under `docs/analytics/archive/pre_multiseason_experiment/` before editing, with verified hashes and relative paths. Updated the README, loading/inspection guides, working notes and coaching history. Earlier evidence JSONs and the existing `pre_simplification` archive remain intact. No library implementation was edited by this worker, and no actual test/import assertion failed or callback was sent.

**Checked source:** `loading.py` SHA256 `b77d29fbe2b09c4036475beb2138b779d8fcddb712e1c9222082c8497b784bbf`. All five package-source hashes were identical before/after the suite, real import and notebook execution. The revised 55-case suite checked the same implementation. Source/selection fingerprints remain in the unchanged [first-import evidence](docs/analytics/multiseason_import_check.json); the [current verification record](docs/analytics/multiseason_scope_check.json) records the revised tests and documentation. The notebook change was limited to prose; all executed code cells and outputs remain unchanged. Materials immediately preceding this clarification are preserved in `docs/analytics/archive/pre_scope_clarification/`.

## Simplification rewrite — verified before the multi-season extension

The primary rewrote the data layer into `_source.py` (internal version/record handling), `loading.py` (one loading path and small internal checks), `inspection.py` (table catalogue/descriptions), and package exports. Removed the old resolver/reader modules and public validator/report classes rather than maintaining parallel implementations. Supported everyday calls are `load_season` and `inspect_season`; `load_season_table` remains a thin compatibility shortcut using the exact same path.

`SeasonData` provides `.tables`, bracket lookup, `.matches/.statistics/.pregame/.shots`, and compact `.provenance`. No `.validation` or `not_implemented` statuses remain. Table properties return None when not selected. Statistics can be loaded independently; cross-table identity checks run when matches are also requested. Column names, values, order, nullable Arrow types and table separation are preserved. Source exports stay untouched. No joins, feature construction or normalization were added.

`load_season(..., verify_hashes=False)` resolves once, checks readable files/size/count/basic structure, and optionally performs full SHA256 verification when requested. `record_path` continues to save/replay the existing version-1 record format; save occurs only after the entire load succeeds. `inspect_season` reads metadata and returns rows, column_count, description, columns, column_types and column_summary; a supplied record must already exist. It writes nothing. Metadata hashes needed for saved-selection replay remain internal.

**Verified simplification (16 September 2026):**

- [x] Archived all four original analytics test files, `src/README.md`, the loading/inspection/statistics guides, tracker and coaching history under `docs/analytics/archive/pre_simplification/`, preserving relative paths and verifying SHA256. No differing archive was overwritten and no files were removed. The original 123-cell notebook remains byte-for-byte in `01_loader_walkthrough_detailed.ipynb` there (SHA256 `ea90a29bf1d13d1a098789920d053723fa3d880790c3be68fba6d881a49a429d`); `notebook_source_displays.txt` also remains intact. See the archive preservation manifest.
- [x] Replaced obsolete tests with **36 compact public-API cases**, all passing with `.misc314 -m pytest tests/analytics -q -p no:cacheprovider`. Coverage includes separate/selected-only tables, standalone statistics and conditional cross-table checks, exact nullable/unsigned/nested data, group-qualified duplicates, broken-data errors, record creation after success, version-1 replay and pinning, no source writes, default reads versus opt-in hashing, and metadata-only discovery. No actual test failed and no callback was sent.
- [x] Compared saved Premier League 2024/25 matches **380 × 35**, statistics **98,224 × 18** and pregame **740 × 8** against direct source Parquet in both normal and hash-audit modes. Values, rows, columns, order and nullable Arrow dtypes matched exactly. Inspection returned 15 available tables. Selected files and all prior evidence JSONs remained unchanged.
- [x] Replaced the active notebook with **9 practical cells (6 code cells)** and executed it top-to-bottom in a fresh `misc314_py314` kernel. It covers setup, catalogue, column access, separate table loading, small samples and an optional saved selection. It contains no implementation dumps, diagnostic-report lessons, manifest/hash displays, checkpoint history or test-count lectures. Running D-Tale was untouched. The known nonfatal Windows/ZMQ selector-thread warning recurred.
- [x] Simplified `src/README.md`, `season_loading.md` and `season_inspection.md` around `inspect_season`/`load_season`, retained all 18 descriptions and the `pregame.position` explanation, and replaced the retired statistics guide with a notice linking to its archive and the quick start.

**Checked source:** `loading.py` SHA256 `e1682d89464d8a1c74159e14d75c457e8ec2f35b85b38d2ac5b6305e7987d693`; `_source.py` SHA256 `66bb91adc2c4d1280ef2e47830eca98e652dbe85fde4691c8a27fac7f47a6dad`; `inspection.py` SHA256 `ce80e30b42663dc763e944c3a6c5699c8ecec8b4af9ab5608fbdeb6167bc98e6`; `data/__init__.py` SHA256 `b3f2fccb2f2ee324d2fa549d0b6b1abc9d27677d3deadcb13a17969d9a7d5836`. Loading changed after the initial capture but before testing; these results apply to the identical immediate before/after test state and subsequent real-data/notebook checks. The worker made no implementation changes. Full evidence: `docs/analytics/simplified_loading_check.json`; earlier evidence JSONs remain historical and unchanged.

## Done (implementation history before simplification)

- [x] Removed the rejected implementation worktree and archived its task.
- [x] Preserved the main project, collected data, and existing scraper environment.
- [x] Created the `src/` directory in the main project. The first analytics function, `resolve_season_export`, is now implemented as recorded below.
- [x] Located and inspected the corrected source, `data/xDiyo_data`, following the user's clarification.
- [x] Rechecked that `C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe` runs and reports Python 3.14.0. D-Tale was initially absent and is now installed as recorded below.
- [x] Preserved `football_analytics_working_notes.md` and `first_pipeline_idea.md` as broader design references.
- [x] Inspected 55 publication manifests; 52 are in the requested four-season range. Verified hashes, row counts, scope IDs, unique event IDs and nonmissing kickoff timestamps for those 52 small matches tables (18,726 rows).
- [x] Inspected matches, statistics, pregame and coverage tables for Premier League 2021/22 and 2024/25, and Bundesliga 2024/25; all 12 selected table hashes and row counts passed. Saved schemas, examples, versions and bounded checks in `docs/analytics/restart_inspection.json`.
- [x] Installed D-Tale 3.22.0 in `.misc314`; comparison against the pre-install package inventory found no replaced existing packages. An eight-row loopback table served HTTP 200 and the test server was shut down. Full browser interaction is not yet tested.
- [x] Confirmed existing notebook kernel `misc314_py314` uses the selected interpreter.
- [x] Created `notebooks/01_loader_walkthrough.ipynb`, validated its notebook structure and executed it top to bottom with a fresh `misc314_py314` kernel. The optional D-Tale cell is disabled during automatic execution. A nonfatal Windows/ZMQ selector-thread warning appeared; execution completed successfully.
- [x] Documented the package layout and first function contract below. After inspecting the season, the user explicitly chose to proceed with the resolver.
- [x] Implemented `ResolvedSeason` and `resolve_season_export` in `src/xdiyo_analytics/data/exports.py`. The resolver reads only two JSON files, validates the inspected schema-2 layout and captures one explicit version.
- [x] Added 39 focused resolver cases in `tests/analytics/test_exports.py`; all passed with `.misc314`. An existing pytest cache-directory warning was nonfatal; no cache repair was attempted.
- [x] Verified the real Premier League 2024/25 metadata resolution (competition 17, season 61627, 380 declared matches, 15 declared tables). No Parquet was read by this function.
- [x] Added and executed 15 walkthrough cells under "Checkpoint 1: resolve one season" in a fresh `misc314_py314` kernel. They show the actual imported code in sections, compact real metadata, table declarations and one intentional error. Existing user-edited inspection/D-Tale cells were preserved and not executed. The known nonfatal ZMQ selector-thread warning recurred.
- [x] Refreshed the project's editable installation in `.misc314` with `--no-deps --no-build-isolation`; the analytics import resolves into `src/`. Existing scraper/utilities package declarations and `xdiyo-collect` entry point remain available. The scraper `.venv` was not altered.
- [x] Implemented `read_season_table(resolved, table_name)` after the user's go-ahead. It verifies the pinned canonical manifest and captured metadata, checks the selected file's bytes/hash/row count, then returns Arrow-backed pandas columns. It does not reread the public season pointer or implement match-domain validation.
- [x] All 59 resolver/reader tests passed in `.misc314` (39 resolver cases plus 20 reader cases), using `-p no:cacheprovider` to leave the existing cache issue alone. The new cases cover nullable/exact data, changed files, pointer advancement, optional flags, empty tables, physical index columns and no source writes.
- [x] Read the real Premier League 2024/25 canonical `matches` table through the new function: 380 rows, 35 columns, file order retained, nullable Arrow-backed types and a RangeIndex. All eight entirely missing extra-time/overtime/penalty columns remain present.
- [x] Added and executed 13 cells under "Checkpoint 2: read one verified table" in a fresh `misc314_py314` kernel. Actual code sections, a real match sample, all column types/missing counts and an intentional absent-table error are visible. Existing user-edited cells were preserved and not executed; the earlier D-Tale session was not changed. The known nonfatal ZMQ selector-thread warning recurred.
- [x] At the user's request, opened the actual `Premier_League_24_25.parquet` season in D-Tale: 380 matches and all 49 source columns. Verified the season-file SHA256 and visible browser grid. The 14 nested list columns are displayed as complete JSON strings; source files remain unchanged. The local inspection server is left running at `http://127.0.0.1:40000/dtale/main/premier_league_2024_25_all_columns`. Launcher: `notebooks/serve_season_dtale.py`; session details: `docs/analytics/dtale_season_session.json`. This is an inspection helper, not the reusable loader.

## Next discussion and implementation steps

1. Use the executed short notebook and `load_seasons` to work with the selected 13-league, three-season population. `inspect_season` and `load_season` remain useful for one publication. The detailed checkpoint history below is historical context.
2. Discuss the first model's selected statistics and optional context, including periods/groups/keys, table-specific validation and cross-table references before joins.
3. Define the first feature inputs and timing/history policies against these imported tables, retaining the exact source list and existing per-publication selection records with future experiment configuration. The league/season population is already selected; model fitting and joins remain future work.

## Current structure — first increment

```text
src/
  xdiyo_analytics/
    data/                  # current increment: prepared-export I/O and validation
      _source.py           # internal source selection and record handling
      loading.py           # SeasonData, load_season, compatibility shortcut
      inspection.py        # inspect_season and table descriptions
      __init__.py          # public imports
notebooks/
  01_loader_walkthrough.ipynb
tests/
  analytics/               # focused new checks alongside existing tests
docs/
  analytics/               # inspection and environment evidence
IMPLEMENTATION_PROGRESS.md # canonical agreement, decisions and next steps
```

Only directories needed for this increment are established now. `xdiyo_analytics` separates reusable analysis from the existing `scraping` and `utils` packages. Put source-format knowledge in `data/`; the notebook selects inputs, calls library functions and displays results. It does not implement loading or feature calculations. The active notebook contains the short experiment-loading workflow; earlier inspection, D-Tale and implementation walkthrough cells are preserved in the recovery archive.

Future package responsibilities, created only when needed: `histories/` for eligible team observations and rating snapshots; `features/` for selectable/composed definitions; `targets/` for outcomes and availability; `validation/` for temporal folds; `models/` for fit/predict adapters; `reporting/` for diagnostics and optional explanation results; `experiments/` for configuration and orchestration. The orchestrator connects these interfaces; it does not make every experiment use Lasso or corners. The separate Architecture/pipelines scaffold is inspiration, not a dependency or copied implementation.

`pyproject.toml` now explicitly includes the existing `scraping`, `scraping.sofascore` and `utils` packages plus `xdiyo_analytics` and `xdiyo_analytics.data`. A package-directory mapping locates only the new package under `src/`. This preserves the original package list and collector CLI while allowing ordinary notebook imports. Add further analytics subpackages to the explicit list when they are implemented.

Existing files outside the new source layout stay in place while collection continues. Moving the chosen scraper under the future source layout's scraping subfolder and cleaning up the project exterior are deferred.

## Data and environment decisions

- Use the existing published seasons in `data/xDiyo_data` now. The corrected source supersedes the initial restart's legacy adapter proposal and older instructions to wait for new exports. The loader's first adapter targets the inspected schema-2 collector publications.
- Promotion/relegation fields have not yet been added. They are expected to be permanently added to the data files later. Support the current absence explicitly and the later persisted columns; missing evidence must not silently mean a team was not promoted/relegated. Do not rewrite source files to fabricate the flags.
- Use `C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe` for analytics development and notebook execution. The initial sandboxed check was denied; the same read-only check succeeded with normal tool escalation.
- D-Tale 3.22.0 is installed in the selected environment. Its missing dependencies were added without replacing installed packages. Verification is recorded in `docs/analytics/environment_check.json`; keep later dependency changes equally scoped.
- Keep the existing project `.venv` intact for the running scraper. A replacement project environment with newer libraries is future work.

## Inspection findings and limits

Evidence: `docs/analytics/restart_inspection.json`. This is a dated snapshot, not a live collection monitor. It replaces the preliminary report for the incorrect legacy directory.

| League | 21/22 | 22/23 | 23/24 | 24/25 |
| --- | ---: | ---: | ---: | ---: |
| Bundesliga | 305 | 306 | 306 | 306 |
| Bundesliga 2 | 306 | 306 | 306 | 306 |
| Championship | 552 | 552 | 552 | 552 |
| Eredivisie | 306 | 306 | 306 | 306 |
| La Liga | 380 | 380 | 380 | 380 |
| La Liga 2 | 462 | 462 | 462 | 462 |
| Ligue 1 | 380 | 380 | 306 | 306 |
| Ligue 2 | 379 | 379 | 380 | 306 |
| Premier League | 380 | 380 | 380 | 380 |
| Premiership | 228 | 228 | 228 | 228 |
| Pro League | 312 | 328 | 240 | 311 |
| Serie A | 380 | 380 | 380 | 380 |
| Serie B | 380 | 380 | 380 | 380 |

Counts are actual match rows, not assertions of complete provider-wide seasons. All 52 publications report complete; all inspected matches have status `finished`. For 51 publications, actual match IDs agree with the saved expected IDs. Bundesliga 2024/25 uses a different reconciliation layout (306 discovered matches and no unresolved round reconciliation); it has no explicit expected-ID list, so that particular comparison is unavailable. No provider fixture discovery was repeated. In particular, 305/379-row cohorts and playoff/stage coverage need deliberate population policy before an experiment.

- **Grains:** matches use `event_id`; pregame uses `(event_id, side)` in these samples; statistics need `(event_id, period, group_name, key, side)` to distinguish stored entries. The sampled statistics have no duplicates at that full grain and no orphan event references. These are observed sample properties; later schemas must validate their own contracts.
- **Repeated statistics and model separation (clarified 2026-09-16):** keep the shots table separate from the other tables, with each predictive model explicitly selecting its sources. Repeated measure names across these separate inputs do not require deduplication. In the stored Premier League 2024/25 sample, the 1,520 `ALL/totalShotsOnGoal` rows occur within `statistics.parquet`: its `Match overview` and `Shots` groups each contain 760 event/side entries. The separate event-level `shots.parquet` contains 9,883 rows. Preserve table identity, group and original key casing so model-specific selection keeps these inputs distinct; do not sum or silently deduplicate across groups. `shotsOnGoal` is a separate key. This is a source-selection decision, not a loader defect or a blocker to the next discussion.
- **Missingness:** Premier League 2021/22 and 2024/25 each have 760 corner values but only 740 pregame rows (20 absent event/side records). Bundesliga 2024/25 has 306 matches and only 610 corner rows, rather than 612. Keep the matches; later joins expose missing statistics. No zero filling, fabricated rows of observations or normalization during loading.
- **Timing:** kickoff timestamps and collector observation timestamps exist in the corrected source. `kickoff_utc` means scheduled/revised provider start, not verified first whistle. Observation timestamps are retrospective collection evidence, not original historical publication times. Neither collection time nor scheduled start establishes result-availability time. A valid historical cutoff policy remains future work.
- **Promotion/relegation:** none of the 52 inspected manifests includes `team_seasons`, and their match schemas have no movement columns. Source fields added later should be preserved; until then optional context remains unknown. Do not automatically import the separate legacy flags or join an unselected external artifact.

## First function contract — implemented and verified

```python
resolve_season_export(data_root: Path, season_stem: str) -> ResolvedSeason
```

**Purpose:** select one explicitly named published season and capture the exact canonical manifest that later table reads will use. Separating selection from table I/O makes version choice visible before a large dataset is read.

**Inputs:** `data_root` points to the corrected export directory. `season_stem` is a single filename stem, for example `Premier_League_24_25`, with no extension or directory segments. The function reads `<stem>.manifest.json` and follows its `manifest` link once to the canonical version. It does not scan every version or choose leagues/seasons automatically.

**Return:** a small named `ResolvedSeason` dataclass containing resolved publication/manifest paths, SHA256 digests of the two JSON byte sequences, and their parsed metadata (`publication`, `manifest`). Metadata retains scope, schema/parser version, publication version, table descriptors, profile, provenance and coverage declarations. Reported counts remain explicitly unverified until table reading. Fields are accessed by name rather than tuple position. This record is not a DataFrame and does not claim that table bytes were checked. `frozen=True` prevents field reassignment, but does not recursively freeze the dictionaries; callers must treat captured metadata as read-only.

**Validation:** reject an empty/invalid stem, a missing sidecar or canonical file, malformed/non-object JSON, `complete` other than literal true, unsupported schema versions, missing essential scope/version/table descriptors, or a link outside the chosen root. Check that the named league/year and canonical scope agree, and that scope IDs/version match the inspected canonical directory layout. This first adapter accepts `league_YY_YY` filenames with consecutive full years stored in the scope. Table descriptors must declare local Parquet filenames, nonnegative integer row/byte counts and SHA256 hex digests; both manifests must agree on the declared match count. Optional feature metadata and promotion/relegation tables are not prerequisites. Use `FileNotFoundError` for absent files and descriptive `ValueError` for an invalid publication; other filesystem errors propagate. Do not catch an error and quietly select another season.

**Pinning:** read each JSON file once; compute its digest from those same bytes. Retain the exact canonical path and descriptor snapshot. Later reads use those captured descriptors and hashes, never reselect `CURRENT.json`. Capturing a digest gives a reproducible fingerprint; it does not independently authenticate a publisher. Subsequent conflicting file contents must produce a clear failure, not an implicit new version.

**Example from the actual source:** resolving `Premier_League_24_25` identified competition **17**, season **61627**, publication version **6d950fa327de4fc29c796c7cca101606**, and a manifest declaring **380** matches and **15** table descriptors. The matches table is relational (35 columns); the top-level nested season Parquet is an alternate representation, not another 380 matches to append.

**Verification:** 39 focused tests passed: metadata resolution with no Parquet files; missing/incomplete/malformed publication cases; scope/descriptor mismatch; ignored unreferenced versions and CURRENT; an existing result remains pinned when a fixture sidecar advances; optional flag table absent/present. The real-manifest notebook example also executed successfully. Table corruption, missing-value conversion and row-grain validation belong to subsequent functions' checks.

The resolver discussion was followed by authorization to proceed with the one-table reader below. The later convenience loader shares the same resolver checks through `_resolve_season_publication`; `ResolvedSeason` now also retains the exact publication bytes for saving a replayable snapshot without rereading a changing public pointer. The public resolver's arguments and behavior are unchanged. No understanding has been assessed or inferred from reading the explanation.

## Second function contract — implemented and verified

```python
read_season_table(resolved: ResolvedSeason, table_name: str) -> pd.DataFrame
```

**Inputs and selection:** use a resolver-produced reference and one exact canonical table name, such as `matches`. Read only that table. The canonical manifest must still match its captured SHA256; its parsed metadata must also agree with the captured dictionary, detecting accidental caller edits. The public sidecar can advance without invalidating this fixed version because the reader never rereads it.

**File verification:** resolve the selected filename within the pinned version directory, read its bytes once, check byte count and SHA256, decode those same bytes with `ParquetFile(BufferReader(...))`, then check the decoded row count. This avoids both a second table-file read after hashing and inferred Hive partition columns. Other table files need not be present. No source files are written.

**Return:** all physical columns and rows in stored order, using `to_pandas(types_mapper=pd.ArrowDtype, ignore_metadata=True)`. Nullable integer/Boolean/string/numeric and nested types survive; large integer IDs are not converted to floats. Missing, zero, False and valid NaN are not intentionally conflated. Stored pandas index metadata is not used to hide a physical column; the output gets a RangeIndex. Keep the resolved reference alongside the frame for provenance. Timestamps retain their stored units and meanings.

**Errors:** invalid reference type -> TypeError; invalid/empty table-name argument -> ValueError; undeclared table -> KeyError listing available names; missing pinned manifest/table -> FileNotFoundError; changed manifest/dictionary, file bytes/hash/row-count mismatch or invalid Parquet -> descriptive ValueError. Other filesystem failures propagate. Optional absence is explicit; no invented empty table or false promotion flag is returned.

**Limits:** this is a whole-table in-memory reader (compressed bytes plus decoded data), without projection or streaming. It does not validate event uniqueness, match identities, status meaning, chronology, fixture completeness or historical information availability. It does not sort, join, normalize, fill gaps, flatten nested values or create model inputs. D-Tale's earlier JSON display conversion is separate from library data loading.

**Verification:** 20 new reader cases and all 39 existing resolver cases passed (59 total). Real `matches` output is 380 by 35; Checkpoint 2 executes the imported function and displays its actual code, data sample and missingness. Match validation and features remain unimplemented.

**Checkpoint outcome:** the user approved proceeding with match validation and requested a simple loading API that handles versions and fingerprints automatically. The next batch is recorded below; function-by-function feedback remains the agreement.

## Third function contract — match validation implemented and verified

```python
validate_matches(matches: pd.DataFrame, resolved: ResolvedSeason) -> MatchValidationReport
```

Requires unique column labels and `event_id`, `home_id`, `away_id`, `competition_id`, `season_id`, `kickoff_utc`. Identity values must be positive integers, excluding Booleans, strings and floats. Reject repeated event IDs, identical home/away IDs and competition/season disagreement with the resolved manifest. Structural failures raise descriptive `ValueError`; incorrect argument types raise `TypeError`. The function does not mutate the DataFrame or perform file I/O.

The immutable report contains `rows`, `missing_kickoff_event_ids` and `invalid_kickoff_event_ids`. Null/NaN kickoff values are missing. Nonnumeric, Boolean, nonfinite, nonpositive or UTC-datetime-unrepresentable values are invalid. Report exact event IDs and preserve rows. No seasonal date cutoff, automatic drop/fill, fixture completeness, finished-status policy or historical availability is inferred. An empty table with required columns returns a zero-row report. Optional statistics, flags and other fields are not required.

## Fourth function contract — simple loading and saved selection implemented and verified

```python
load_season_table(data_root, season_stem, *, table="matches", record_path=None) -> pd.DataFrame
```

The user provides a data directory, season stem and table name, never a version ID or hash. The function resolves the publication, verifies the selected file and runs match validation for `matches`. Timing issues emit a warning with counts; detailed event IDs remain in `frame.attrs['xdiyo']['validation']`. Other tables receive integrity checks only and explicitly report domain validation as `not_implemented`. Keep table and statistics-group identities separate for model-specific selection.

All rows, columns, source order and nullable Arrow types are preserved. `frame.attrs['xdiyo']['source']` captures the selected manifest, fingerprints and source identity; the manifest includes table descriptors, scope, parser version, provenance and coverage metadata. Pandas transformations may discard attributes, so they are not a persistent experiment archive.

An optional readable `record_path` outside the source data directory saves the season's exact publication snapshot and fingerprints **after successful loading and validation**. The first call creates the parent directory/record; existing records are never overwritten. Later calls restore through the same resolver checks and verify the captured canonical fingerprint. One record covers one season and can be shared by its separate table loads; relocation of the data root is supported. A changed or absent pinned file fails with no fallback to newer data. The record stores metadata, not copies of tables. Concurrent first writes can raise `FileExistsError`; retry uses the created record. Without a record, each call resolves the publication current at that call. Use a new record filename for a deliberate refresh.

**Completed evidence (16 September 2026):**

- [x] 109 focused analytics tests passed in `.misc314` (59 existing cases plus 50 validation/loading/replay cases), with the pytest cache provider disabled. Covered exact large IDs, malformed identities, missing/invalid timing, duplicate match records, optional absence, shots independence, source preservation, pointer advancement/removal, relocated sources and changed/missing pinned-file failures.
- [x] All 52 current publications in 2021/22–2024/25 passed match validation: 18,726 rows, zero missing/invalid kickoffs. This is a loader compatibility check, not experiment population selection or a fixture-completeness claim. Evidence: `docs/analytics/loader_validation_check.json`.
- [x] Real Premier League 2024/25 matches (380 × 35) and shots (9,883 × 34) loaded separately from the same saved selection. Example record: `docs/analytics/loader_demo_selection.json`.
- [x] Added 23 notebook cells for Checkpoints 3 and 4, refreshed the resolver code displays after sharing its internal checks, and executed all 51 cells across Checkpoints 1–4 in a fresh `misc314_py314` kernel. The notebook has 65 cells. Earlier user-edited inspection/D-Tale cells were preserved and not executed; this was not a full-notebook rerun. The known nonfatal Windows/ZMQ selector-thread warning recurred.

**Checkpoint outcome:** the user requested table/column discovery before statistics selection. The next function implements that inspection step below; joins and features remain deferred.

## Fifth function contract — season inspection implemented and verified

```python
inspect_season(data_root, season_stem, *, record_path=None) -> pd.DataFrame
```

Return an alphabetically indexed table catalogue containing `description`, `declared_rows`, `file_bytes`, `column_count`, `columns`, `column_types` and `column_summary`. Physical names, column order and nested Arrow type strings come from the selected files. Known schema-2 roles have documented descriptions; unfamiliar tables remain visible with an explicit fallback. No version IDs or hashes are user inputs.

Resolve the current publication, or reuse an **existing** saved season record outside the data root. Unlike the loader, inspection creates no record/directories. Check each table path stays in the selected version directory, then compare file size and Parquet footer row count with the manifest. Read no observation rows and verify no whole-table hashes. Invalid data pages may still have readable metadata. Source provenance and these limits remain explicit in `catalogue.attrs['xdiyo']`. Missing records/files raise `FileNotFoundError`; invalid metadata, escaped paths, mismatched sizes/counts, unreadable footers and repeated column labels raise `ValueError`. Other I/O errors propagate.

**Standings clarification:** the collector places provider pregame team positions in `pregame.position`, keyed by match and side, with `team_id` and the original team context in `value_json`. `lineup_players.position` instead means playing position. The saved Premier League 2024/25 export has 740 pregame rows for 370 of 380 matches, with no null positions among those present; there is no separate declared `standings` table or complete round-by-round league table. Collection time is not evidence of original pre-match availability. No fallback positions, normalized standings or reconstructed table were added.

**Completed evidence:**

- [x] 120 focused analytics tests passed (109 existing plus 11 inspection cases): metadata-only reading, source preservation, existing-record replay, missing-record behavior, empty/unknown/nested schemas, corrupt metadata, size/count discrepancies and changed pinned manifests. A data-page corruption case demonstrates that inspection makes no full-integrity claim.
- [x] Real saved season inspection returned all 15 declared tables, including matches (35 columns), shots (34), statistics (18) and pregame (8).
- [x] Added and executed 14 notebook cells (6 code cells) for Checkpoint 5 in a fresh `misc314_py314` kernel. The notebook has 79 cells; all 65 earlier cells and outputs were preserved. This was not a full-notebook/D-Tale rerun. The known nonfatal Windows/ZMQ selector-thread warning recurred.
- [x] Added `docs/analytics/season_inspection.md` with API examples, a reference of all 18 documented table roles matching `inspect_season`, descriptions/types, saved-selection behavior, inspection limits and standings guidance; linked it from `src/README.md` and the notebook. Keep the written table reference aligned with future library description changes.

**Pause:** discuss this discovery function before selecting/validating the first model's statistics and context.

## Sixth function — statistics validation implemented and verified

`validate_statistics(statistics, matches) -> StatisticsValidationReport` is implemented in `src/xdiyo_analytics/data/statistics.py` and exported from the data package. It requires statistic identity columns plus `value`, rechecks the reference matches' identity subset, rejects orphan references, invalid sides, mismatched supplied team IDs and repeated `(event_id, period, group_name, key, side)` entries. Different groups may retain the same measure. IDs are exact positive integers; labels are nonempty strings with original casing preserved.

The immutable report provides row/reference counts, presence of the optional `team_id` column, missing team-ID/value row positions, and match sides having no statistic rows at all. Scalar null/NaN numeric values are reported without rejecting textual measures. No rows are modified, joined, filtered, filled or aggregated. Missing team IDs remain unknown. The function does not impose numeric ranges, per-measure completeness, timing or version checks. Use matches and statistics from the same saved selection. Standalone use remains explicit after `load_season_table`; the Seventh function, `load_season`, now invokes it automatically when matches/statistics are selected together.

**Completed evidence (16 September 2026, Japan time):**

- [x] Added 137 focused cases in `tests/analytics/test_statistics.py`; all **257 analytics tests passed** using `C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe -m pytest tests/analytics -q -p no:cacheprovider`. Covered exact large/signed/unsigned nullable Arrow IDs, required/repeated columns, malformed identifiers/labels/sides, full-grain duplicates versus permitted repeated keys, orphan references, malformed reference tables, supplied team agreement and missing/absent team IDs, scalar null/NaN with text, empty tables, side coverage, arbitrary indices and input preservation. No pytest case failed; no library implementation edits were made by the background task.
- [x] Replayed the existing `docs/analytics/loader_demo_selection.json` using `data/xDiyo_data`: **98,224 × 18 statistics** validated against **380 × 35 matches**. All 380 matches were referenced, with zero missing team-ID/value rows and zero sides without statistics. Three periods, seven groups and 50 keys were present; four keys occurred across groups. Selected input files, the selection record, DataFrames and attributes were preserved. Shots remained separate. Evidence: `docs/analytics/statistics_validation_check.json`.
- [x] Added and executed **23 Checkpoint 6 cells (11 code cells)** in a fresh `misc314_py314` kernel, showing the actual imported code, report fields, real group separation, synthetic exact IDs/gaps, caught errors, empty inputs, limits and preservation checks. The notebook now has **102 cells**. All 79 previous cells/outputs and notebook metadata were preserved; this was not a full-notebook or D-Tale rerun. The nonfatal Windows/ZMQ selector-thread warning recurred.
- [x] Added `docs/analytics/statistics_validation.md`, linked from `src/README.md` and Checkpoint 6, explaining the contract and the explicit validator call. Single-table `load_season_table` statistics metadata remains `not_implemented`; subsequent automatic bundle integration is recorded under the Seventh function.

**Verified source state:** `statistics.py` SHA256 `4a13fda9ff4ee840af7c32afcade033ec48d6e15dcfb7af759893fc8b51180bf`; `data/__init__.py` SHA256 `054ea336427856a9471b18af3f42aa0c867f7c407fa1821c2f58aae60bde95e1`. All seven analytics source hashes matched immediately before/after the final suite and through the real-data/notebook checks; complete fingerprints are in the evidence JSON. The first baseline captured `statistics.py` as `4fe815d138e19da42d3fe062eef322810f496261199ab4d769935e701746576f`, then a source edit was detected. The provenance guard failed before the first real-data check and was reported to the primary task. The suite was rerun against the current stable state; no pass is transferred to another source state. The background task sent no success callback.

**Next discussion:** select the first model's periods/groups/keys and optional context, then define completeness and timing policies before joins. No feature construction or experiment population was added.

## Seventh function — season bundle loading implemented and verified

`load_season(data_root, season_stem, *, tables=("matches",), record_path=None) -> SeasonData` is implemented in `src/xdiyo_analytics/data/season.py` and exported from the data package. It resolves one version exactly once, prechecks requested table availability, and reads all selected tables against that reference. It does not call the convenience single-table loader repeatedly or reselect the public pointer between tables.

The selection must be a nonempty ordered sequence of distinct table names. Statistics require matches explicitly in the selection; no tables are added implicitly. Matches and statistics run their existing validators. Other tables receive integrity checks and explicitly report domain validation as `not_implemented`. No joins, pivots, filtering, normalization or features are added.

`SeasonData.tables` maps exactly the requested names to separate DataFrames. Convenience properties `.matches`, `.statistics`, `.pregame` and `.shots` return the selected frame or None when not requested. A requested-but-unavailable table raises KeyError, never an invented empty frame. `.provenance` retains the shared manifest/version/fingerprints and loaded names; `.validation` contains per-table reports. The record is frozen at the field level but its frames/dictionaries remain mutable; reports describe the original loaded state. Each frame's metadata is independently copied from the bundle metadata.

Existing selection records replay normally. New records are saved only after all requested reads and validation succeed. Record serialization was extracted to `_write_season_record` in `loading.py` and reused by both loaders, preserving the existing record format and exclusive-create behavior. Missing timing warns and retains rows; statistics gaps remain in diagnostics. Source files are not written.

**Completed evidence (16 September 2026):**

- [x] Added **54 focused bundle cases** in `tests/analytics/test_season.py`; all **311 analytics tests passed**, including the previous 257 cases and original single-table tests after extracting the shared writer. Used `.misc314 -m pytest tests/analytics -q -p no:cacheprovider`. Cases cover one resolution under changing/removed publication pointers, exact requested order and property behavior, dependency/availability prechecks before table reads, automatic full-grain/reference/team validation, other tables' integrity-only status, timing warnings with exact IDs/nullable data retained, late failures with no new record/directory, source preservation, both record compatibility directions, relocation, existing-record failure/no fallback, exclusive creation, copied metadata and mutable snapshot reports. No tests failed and no callback was sent.
- [x] Bounded `load_season` replay of the existing `docs/analytics/loader_demo_selection.json` returned separate **matches 380 × 35**, **statistics 98,224 × 18**, and **pregame 740 × 8** from one pinned manifest. Matches/statistics passed structural validation with zero reported timing/team/value/side gaps. Pregame domain validation remained `not_implemented`; a separate descriptive check found rows for 370 matches, with 20 match sides absent and zero missing positions among present rows. Shots were unrequested. Selected file/record hashes were unchanged. Evidence: `docs/analytics/season_loading_check.json`.
- [x] Added and executed **21 Checkpoint 7 cells (10 code cells)** in a fresh `misc314_py314` kernel. All **102 earlier cells/outputs and notebook metadata were preserved**; the notebook now has **123 cells**. The new checkpoint displays actual `SeasonData`, `load_season` and `_write_season_record` source, properties/selection errors, reports, pregame gaps, copied metadata and report-snapshot behavior, single-table compatibility and preservation checks. Only Checkpoint 7 ran; D-Tale was not rerun. The nonfatal Windows/ZMQ selector-thread warning recurred.
- [x] Added `docs/analytics/season_loading.md` and linked it from `src/README.md`; updated `docs/analytics/statistics_validation.md` to distinguish standalone calls and automatic bundle validation. `load_season_table` behavior is unchanged. Earlier notebook source displays and fingerprint outputs remain historical; Checkpoint 7 explains that Checkpoint 6's old batch-level guard includes the intentionally changed helper/package files and would flag them if rerun unchanged.

**Verified new batch source state:** `season.py` SHA256 `39beaa1364b3a52b529ebaccfb4e0064db3a448fc7a66768f2716ec1814c9ca4`; `loading.py` SHA256 `6db4b88170229571cc4b76cfa2d1b130de665674069cfe29ee5c9f7517e11da1`; `data/__init__.py` SHA256 `577a8b37cadebfa41c53d953ecc7cb8d65a6da6d14e614651c5694ebbb67700e`. All eight analytics source hashes matched this batch's new baseline, immediate before/after suite captures and real-data/notebook checks. The statistics validator remained at `4a13fda9ff4ee840af7c32afcade033ec48d6e15dcfb7af759893fc8b51180bf`. Complete hashes, selected input hashes and execution evidence are in the new evidence JSON. The verification task made no library implementation edits and sent no success callback.

**Next discussion:** select each model's tables/periods/groups/keys and required completeness/timing policies before joins. Pregame/shots domain validators, features and experiment population selection remain separate work.

## End-to-end objective

A configuration should drive the orchestrator through loading, team histories and features, temporal folds, training/selection, and pre-training/post-training statistical reports. Models and reporters should remain extensible. The Lasso total-corners experiment is the first integration example, not a fixed architecture for every experiment.

The current pilot still uses selectable corner/shot lag and rolling feature definitions, match-result Glicko, partial eligible history and no seasonal warm-start blend. Preserve information cutoffs and keep realized targets/option outcomes separate from predictors. Detailed feature, rating, CV and reporting choices remain in the broader notes and are resolved when their implementation step is reached.

## Deferred

- Scraper relocation, cleanup of existing top-level files, and replacement/removal of the scraper environment.
- Permanent promotion/relegation augmentation of the source exports.
- Later feature/model families, richer explanation providers, CPCV and betting-strategy extensions.
- New scraping runs or a full real-data experiment as part of restart setup.
