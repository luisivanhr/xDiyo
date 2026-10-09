# Promotion and relegation flags

These flags describe **how a team entered the current season**. `got_demoted`
means relegated from a higher division before this season began. It does not
mean the team will be relegated at the end of the current season.

To use these as model inputs, explicitly select `TeamMovement("promoted")` and
`TeamMovement("relegated")`, conventionally named `was_promoted` and
`was_relegated`. They preserve numeric 0/1/missing values. See
[explicit warm starts and movement features](explicit_warmup.md) for API/UI
examples, override precedence, and statistical transition policies.

## Loading

Japan's 2010–2026/27 publications also carry these fields. See the
[Japan enrichment](japan_enrichment.md) for boundary evidence and the special
handling of the 2026 transition tournament.

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

As of 8 October 2026, the review covers 221 league-seasons and 4,225 team-seasons:
3,387 retained, 602 promoted, 234 relegated and two administrative entrants.
The 2010/11–2014/15 extension adds 65 league-seasons, 23,658 match records and
1,250 team-season entries: 999 retained, 180 promoted, 70 relegated and one
administrative entrant. All entry classifications are resolved. The 36 boundary
reviews retain historical sources even where an adjacent season is now present.
The extension also supplies predecessor IDs for the thirteen 2015/16 seasons;
their existing classifications and observation files are unchanged.

Vicenza's 2014 Serie B admission and Paris FC's 2017 Ligue 2 admission are
`other_entry`. Dundee's 2012 entry is `promoted`, following UEFA's explicit
classification, with its invitation after Rangers' exclusion documented in
the evidence. Triestina 2010, Vicenza 2012, Eibar 2015, Venezia 2019 and
Waasland-Beveren 2020 remain retained under prior-season membership semantics.

The new extension's native tables, season exports and movement tables all use
**ZSTD**, never Snappy. Older observation files retain their existing compression
when only predecessor evidence changes. Source statistics coverage gaps remain
missing; this processing does not invent observations or fill optional tables.
The collection excluded one abandoned fixture in each of Championship 2014/15
and Pro League 2014/15, leaving 551 and 239 records respectively. Manifest
completion refers to the collected selection, not every scheduled fixture.

The reproducible preflight/publication script is
[`prepare_season_extension_20261008.py`](../../examples/prepare_season_extension_20261008.py)
(`--apply` to write); its table fingerprints, counts and verification record are
in [the extension audit](data/season_extension_20261008.json). Existing fields,
nested observations and unrelated tables are checked for preservation, and
manifests contain the actual updated hashes. Current movement publication also
preserves existing Arrow string widths when run under pandas 2 or 3.
The [final verification](data/season_extension_20261008_verification.json)
checks all 801 current extension Parquet files and records successful native UI
recipe preparation of 23,658 matches, 47,316 team-history rows, four movement
predictors and three temporal folds. It performs no model fitting. The movement,
recipe-refresh and loader regression suite passed all 54 tests.

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
New additive and native `team_seasons` tables explicitly use ZSTD. An update
that changes only predecessor/evidence metadata leaves matching observation
files untouched, including their bytes, timestamps and existing codecs.

## Warm-up across season boundaries

Automatic team and destination-league predecessors require an adjacent season
year, determined from `source_season`, `season_year` or `season_stem`. Missing
or unparseable year information does not authorize guessing from season IDs or
the latest available kickoff. Thus 2015/16 cannot seed 2017/18 when 2016/17 is
missing. Explicit null predecessor IDs remain null even when adjacent history
exists; omitted predecessor fields permit normal adjacent-season inference.
Both predecessor IDs must be supplied together or both null. Warm-up remains
opt-in, and no stale-season policy is implicitly enabled.

For a list of record dictionaries, omission is interpreted separately for each
team: another team's explicit null predecessor does not suppress inference for
a record that omits those fields. In a DataFrame, every column has a cell on
every row; null predecessor cells explicitly disable that predecessor. Use
record dictionaries when mixing omitted fields with explicit nulls.

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

## Historical extension, 5 October 2026

The following requested downloads have been enriched in `data/xDiyo_data`:

| Season | Leagues | Matches | Team-seasons |
| --- | --- | ---: | ---: |
| 2015/16 | Pro League, Premiership, Championship, La Liga 2, Bundesliga 2, Ligue 2, Serie B | 2,599 | 134 |
| 2016/17 | Pro League, Premiership, Eredivisie, Bundesliga 2, Ligue 2, Serie B | 1,891 | 106 |
| 2017/18 | Pro League, Premiership, Serie B | 900 | 50 |
| 2018/19 | Pro League, Premiership, Serie B | 779 | 47 |
| 2019/20 | Pro League, Premiership | 411 | 28 |
| **Total** | **21 league-seasons** | **6,580** | **365** |

The new team-season records contain 290 retained entries, 48 promotions and 27
relegations, with no unknowns. Seven 2015/16 boundary populations were researched
online because their adjacent 2014/15 rosters are absent. The linked
[boundary evidence](data/team_movement_boundary_evidence.json) records exact
team IDs, sources, review date and administrative exceptions. Ascoli is promoted
as explicitly classified by [Lega Pro's official notice](https://www.lega-pro.com/com/1516-143L.pdf);
Brescia and Virtus Entella remain retained after readmission. Elche is relegated
from the upper division despite the administrative reason.

All 21 top-level exports, available native tables, and new movement tables use
**ZSTD, not Snappy**. Match scores, timestamps, awarded status, nested observations
and existing optional tables were preserved. Missing/empty tables, partial
statistic coverage and the source's regular-season fixture scope remain as
downloaded. No absent statistics or fixtures were fabricated. Standings continue
to be normalized by the existing history/feature layer from available pregame
positions; missing positions remain missing. The one awarded fixture is retained
in storage and excluded by the loader's existing default, giving 6,579 eligible
matches unless `include_awarded=True`.

The new adjacent seasons also fill predecessor IDs for 18 already-reviewed
league-seasons. Their classifications, original observation files and season
exports are unchanged; only movement evidence and its manifest references were
updated. Eredivisie 2017/18 can now use its actual 2016/17 predecessor. No
cross-gap fallback was introduced. Other newly downloaded seasons outside the
explicit list were not enrolled in this review.

Validation covered hashes and compression for every available new table and
export, exact preservation of existing observation columns, all native tables
through the public loader, team-history construction, `TeamMovement` features,
and default awarded exclusion. Fresh derivation agrees with every persisted
reviewed team-season table. The focused suite above now passes **152 tests**.
See the [machine-readable extension audit](data/season_extension_20261005.json).

The bounded preparation script can verify this exact extension again without
writing data:

```console
python examples/prepare_season_extension_20261005.py
```

`--apply` publishes pending reviewed enrichments; already matching publications
are left unchanged. Existing recipes and completed model artifacts were not
edited or refitted. Dataset hashes are truthful; pinned recipes selecting
changed manifests should use the explicit refresh helper described above.
