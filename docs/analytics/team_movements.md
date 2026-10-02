# Promotion and relegation flags

These flags describe **how a team entered the current season**. `got_demoted`
means relegated from a higher division before this season began. It does not
mean the team will be relegated at the end of the current season.

## Loading

```python
from xdiyo_analytics.data import load_seasons
from xdiyo_analytics.histories import build_team_history

data = load_seasons(
    "data/xDiyo_data", ["22_23", "23_24", "24_25"],
    tables=["matches", "statistics", "team_seasons"],
)
movements = data["team_seasons"]
history = build_team_history(data)
```

`team_seasons` has one row per competition, season and team. It contains
`got_promoted`, `got_demoted`, `movement`, team name, predecessor competition and
season where observed, and evidence. `movement` is `retained`, `promoted`,
`relegated`, `other_entry` or `unknown`. Unknown flags stay missing, not false.

The current dataset includes `home_got_promoted`, `home_got_demoted`,
`home_season_entry` and corresponding `away_...` columns in both the native
matches table and each top-level season Parquet file. Ordinary matches loading
therefore includes the flags without requesting an extra table.
History receives `team_got_promoted`, `team_got_demoted`, `team_season_entry`
and the corresponding `opponent_...` columns. Adding context columns does not
automatically select them as model inputs. The separate native `team_seasons`
table carries the evidence and predecessors for the existing warm-up API:

```python
features = evaluate_features(history, definitions, team_seasons=movements)
```

In the experiment builder, select **Data → Tables → team seasons**. The builder
passes the table to feature evaluation and rating construction, including
opted-in warm-start features. It does not enable warm-up automatically. Rating
engines retain their explicit transition configuration; a supplied movement-table
override takes precedence over the loaded table.

## Derivation and assumptions

1. A team present in the same league in consecutive seasons is retained.
2. A team observed in the previous season's adjacent lower division is promoted;
   one observed in the adjacent upper division is relegated.
3. With reviewed previous same-division and upper-division rosters, a newcomer
   absent from both is **inferred promoted from below**. For a top division only
   the previous same-division roster is required. This is a membership-based
   inference, not a separately researched claim about every entrant.
4. Missing predecessor seasons and the first observed seasons use researched
   boundary evidence. Administrative entries are recorded separately. An
   unreviewed or ambiguous case remains unknown.

Team IDs, country/league systems and tier numbers control comparisons; a team
missing from the loaded experiment subset does not count as newly promoted.
Derivation uses the full reviewed source population. Roster completeness is
distinct from match completion: a live season can already contain all its teams.
No end-of-current-season result is used to classify current-season entry.

The current review covers 135 league-seasons and 2,610 team-seasons: 2,098
retained, 374 promoted, 137 relegated and one administrative entrant. 213 of the
promotions use rule 3. The 14 boundary reviews include the gap before Eredivisie
2017/18. Paris FC's 2017 Ligue 2 admission is `other_entry`; Eibar 2015, Venezia
2019 and Waasland-Beveren 2020 are retained with administrative notes.

Sources, exact IDs and exceptions are in
[boundary evidence](data/team_movement_boundary_evidence.json). The exact roster
population and original match checksums are in
[roster review](data/team_movement_roster_review.json).

## Updating recipes after dataset changes

At the user's request the current dataset has been updated **in place**, rather
than retaining two full dataset versions. Manifests contain the real new hashes.
Old pinned selection records will correctly reject the changed data until
explicitly refreshed. Unpinned recipes use the updated data normally.

In the builder, use **Data → Update recipe to current data**. This saves a new
recipe and selection-record directory, displays which references changed, and
switches the builder to that recipe. Restart a running notebook UI server to
load the new code. No training happens during refresh.

```python
from xdiyo_analytics.ui import refresh_recipe_data

update = refresh_recipe_data(
    "experiments/_recipes/old_recipe.json",
    "experiments/_recipes/updated_recipe.json",
    workspace=".",
)
recipe = update["recipe"]
print(update["changes"])
```

The helper accepts a recipe dictionary as well as a JSON path. It preserves
features, labels, splits, models, reporters and other experiment settings. Only
`data_root` (resolved to an absolute path) and `record_dir` change. An optional
`data_root=` selects a relocated dataset. Existing recipe files and records are
not overwritten, and an existing output path is rejected. Paths resolve relative
to `workspace`, as they do in the builder. Requested seasons/leagues must remain
available. With `leagues=None`, current discovery includes all matching leagues,
just as ordinary loading does; each selected source appears in the summary.

Refresh does not migrate auxiliary files, pinned odds or saved results. Existing
saved models/reports remain historical artifacts. Changed data or library code
can invalidate automatic training reuse. Refresh adopts current data; it cannot
reproduce an old run after its original data has been replaced.

Refresh uses the loader's season, league and table validation and discovery.
Unrelated manifest names are ignored; duplicate/empty selections and invalid
`include_awarded` values fail before output is written. With `verify_hashes=True`,
requested table bytes are checked before adopting their pins. Without that
option, refresh checks publication/table availability, not all table contents.
Recipe and record files are staged first. A caught publication/write failure
removes only the newly installed records so the same output path can be retried.
Publication never overwrites an existing recipe.

Inputs are recipe dictionaries or native JSON, not Python/notebook exports.
Refresh the JSON and regenerate those exports. All other configuration,
including custom code-revision metadata, run names and output directories,
remains unchanged: choose new-run provenance and output identity separately
when starting a new experiment.

## Rebuilding

From the repository root, with the normal analytics environment/PYTHONPATH:

```console
python examples/build_team_movements.py
```

The derivation builder checks reviewed IDs and match hashes, initially writing
additive evidence files under `_team_seasons`. Already materialized seasons are
recognized using the original evidence reference. Changed/new source populations
need another roster review. `derive_movements` and `save_movement_enrichment` are
public Python helpers.

Already materialized seasons are compared with the newly derived table,
including source-reference text and predecessor IDs. Any mismatch stops the
builder before it replaces the audit or writes other enrichments. Identical
evidence is idempotent. Audits therefore describe the actual persisted tables.
An explicitly reviewed correction can be published with
`build(..., rematerialize=True)` in `examples/build_team_movements.py`, or
`materialize_movement_flags(root, stem, reviewed_flags=frame)` for one season.
The builder CLI exposes the same explicit action as `--rematerialize`.
The latter requires exact roster coverage and updates flags, evidence and real
hashes together. The original matches hash remains recorded for roster review.

Boundary IDs may be positive integers or integer strings; both apply identically.
Duplicate boundary stems, absent teams, conflicting movement booleans and
incompatible predecessor divisions are rejected. Explicit warm-up overrides
must agree with available predecessor membership, adjacent year and league tier.

`xdiyo_analytics.data.publish_movements.materialize_movement_flags` embeds that
reviewed evidence into the native matches table and season Parquet, registers
the native team-season table, and updates checksums. It stages one season at a
time and keeps a small resumable journal and fingerprint audit under
`_team_seasons/materialization`. It keeps no duplicate season dataset. Existing
fields, nested observations and unrelated native tables are preserved.

The writer preserves each existing file's compression codec and nested
list-child names. Mixed codecs require an explicit supported choice rather
than a silent default. `compression="zstd"` explicitly recompresses the top-level
season export; native matches keep their codec. Each rewritten table is read
back and compared for exact values and schema metadata before replacement.
Compression levels are not recoverable from Parquet metadata: preservation
refers to the codec, not byte-for-byte reproduction of the original encoding.

## Warm-up across season boundaries

Automatic team and destination-league predecessors require an adjacent season
year, determined from `source_season`, `season_year` or `season_stem`. Missing
or unparseable year information does not authorize guessing from season IDs or
the latest available kickoff. Thus 2015/16 cannot seed 2017/18 when 2016/17 is
missing. Explicit null predecessor IDs remain null even when adjacent history
exists; omitted predecessor fields permit normal adjacent-season inference.
Both predecessor IDs must be supplied together or both null. Warm-up remains
opt-in, and no stale-season policy is implicitly enabled.

## Review repair, 2 October 2026

All 46 outdated source-reference rows in Championship 2016/17 and La Liga 2
2016/17 were corrected. All 2,610 classifications stayed unchanged. All 135
top-level exports now use ZSTD: 2,644,938,141 bytes became 1,440,234,065 bytes,
saving 1,204,704,076 bytes. Current nested field names were preserved; previously
normalized `element` names were not renamed again. Existing observation values
were preserved, and publication/native checksums were verified for 45,309 matches.

The real Eredivisie 2017/18 history was checked: none of its 18 teams acquires a
fallback predecessor across the missing 2016/17 season. Fresh derivation agrees
with every persisted flag and evidence row. No study models were refitted and
no existing experiment recipes or completed results were rewritten. Old pinned
recipes need the explicit refresh described above because these files have new,
truthful hashes. See [repair verification](data/movement_review_repair.json).

Focused synthetic verification (150 tests):

```powershell
$env:PYTHONPATH='.;src;tests/analytics'
& 'C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe' -m pytest tests/analytics/test_movement_review.py tests/analytics/test_recipe_refresh.py tests/analytics/test_movement_enrichment.py tests/analytics/test_transition_context.py tests/analytics/test_warmup.py tests/analytics/test_rating_transitions.py tests/analytics/test_loading.py tests/analytics/test_multiseason.py tests/analytics/test_season.py tests/analytics/test_awarded_loading.py -q -p no:cacheprovider --tb=short
```
