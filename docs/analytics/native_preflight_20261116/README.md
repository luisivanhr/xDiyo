# Native SofaScore queries and cohort preflight

This extends the existing loaders, recipe refresh helper, countdown and derived
standings engines after `e82ffdff`. It creates no second dataset and changes no
source observations, source manifests, storage keys or frozen experiment inputs.
No experiment is fitted. Legacy `seasons`/`leagues` recipes keep their original
behavior; the explicit `query` route selects the versioned scoped policy below.

The current report covers **617 publications, 662 phase/cohort scopes and
172,274 fixtures**. Countdown is supported in 561 scopes; independent sporting
standings in 501. Requiring countdown, standings and movement flags together
leaves 501 `conditional_retrospective` scopes and 161 blocked scopes. These are
phase/cohort counts, not counts of distinct competitions. The report lists every
blocked reason and still exposes those publications to queries.

The identity inventory contains 50 competitions, 616 distinct provider-season
identities, 69 phases, 1,318 teams and 592 library annual groups. Captured names
are present for all competitions/seasons, 64 phases and 1,290 teams. The remaining
**five phase names and 28 team names lack captured evidence**; their stable IDs
and historical aliases remain discoverable with an explicit missing-evidence
status. No conflicts were silently resolved. One provider-season identity has
multiple native publications, so the publication and season counts differ.

## Discovery and identity

```python
from xdiyo_analytics.data import NativeLibrary, load_seasons

library = NativeLibrary("data/xDiyo_data")
competitions = library.discover("competitions")
seasons = library.discover("seasons", competitions="HNL")
teams = library.discover("teams", competitions=170, seasons=61243)
phases = library.discover("phases", competitions=11539)
annual_groups = library.discover("annual_groups", competitions=11539)

# Provider names, IDs and old storage aliases share one exact resolver.
selection = library.select(competitions="HNL", seasons=61243)
data = library.load(selection=selection, tables=["matches", "team_seasons"])
same = load_seasons("data/xDiyo_data", query={"competitions": 170,
                   "seasons": 61243}, tables=["matches", "team_seasons"])
```

Discover the captured name before querying a particular phase: names can change,
and Apertura/Clausura labels can occur in several competitions. The filters are
`competitions`, `seasons`, `phases`, `teams`, `annual_groups`, and `publications`
(exact old season stems). Lists select several entities. Filters resolve in that
order, except explicit publication aliases narrow the starting population.
Season and phase identities include competition ID. Numeric IDs are integers;
string names/aliases are Unicode-, case- and whitespace-normalized, never fuzzy.

`AmbiguousNativeName.candidates` contains the alternatives. Duplicate names and
conflicting same-time evidence never silently pick an entity. Canonical display
names come from captured SofaScore evidence, using the latest capture when names
changed. Source names and old storage keys remain aliases/provenance. Missing
captured names stay explicitly `missing`; historical aliases are not promoted
to verified names. Annual groups are library identities, **not** invented
SofaScore season IDs. A calendar-year season, a provider phase and its annual
group remain separate fields.

All publications are indexed even if their features are unsupported. Selection
pins bind exact publication membership, source-manifest hashes, phase/team
filters and the captured vocabulary fingerprint. Loading validates the pins;
refresh is explicit. Index instances are invocation-local snapshots, not global
caches. Reconstruct one after publication changes. Match identity/name indexing
also verifies the matches file's hash and row count.

Team queries select target fixtures involving those teams, while retaining the
whole selected phase's history for league-wide ranks and opponents. Direct
callers must respect `matches.native_target`; `prepare_recipe` applies it. Phase
queries retain that phase's source history and full verified table roster.
Per-table child filtering joins source league, source season and event ID.

## Readiness and conventions

```python
report = library.preflight(
    query={"competitions": 170, "seasons": 61243},
    required_features=["countdown", "standings", "movements"],
)
```

The [full report](preflight.json) and [CSV](preflight.csv) inventory each
publication/phase: IDs/names, annual grouping, coverage, rules, table treatment,
membership, movement completeness, result bounds, exclusions, assumptions and
blockers. A feature can be unsupported without hiding its dataset from discovery.
Required unsupported features block preparation; callers can select a ready
phase or omit the unsupported feature. `conditional_retrospective` is a visible
assumption classification, not proof of historical availability.

The new query route defaults to:

* `availability="retrospective_finished"`: earlier finished kickoffs are the
  established retrospective result/statistic availability proxy. Every reviewed
  completion/publication lower bound still applies. This convention is also
  passed explicitly to the standings engine; missing release fields do not
  silently create permanently empty tables. No source timestamps are written.
* `entry_information="retrospective_season_entry"`: season membership and
  promotion/relegation are reviewed retrospectively. Flags describe entry to
  this season from the previous season, never departure at its end. No historical
  announcement time is asserted.

`availability="explicit"` requires timezone-aware result release evidence and
uses it in histories, ratings and temporal training-label selection. Unknown
release times block preparation. Result release is not evidence of per-statistic
or heatmap publication: strict preparations requesting those tables are blocked
until that additional contract is supported. `entry_information="explicit"`
requires complete `entry_available_at` evidence, checked against every resolved
prediction cutoff, including warm-start use. Missing members or dates block it.

Result eligibility preserves `kickoff < cutoff`, `available_at <= cutoff` and
target exclusion, including delayed/resumed results. Explicit t−1 cutoffs remain
unchanged. Supervised targets must be finished, finite/labeled sporting fixtures;
awarded, abandoned, annulled and unplayed records are excluded from their history
and target population. Loaded observations and source status/scores stay intact.
An administrative result is never silently substituted for a played score.
Verified `resumed_completed` reviews supply the existing standings engine's
explicit played-score fields only when fixture IDs, played scores and the
timezone-aware completion bound agree. This fixes the adapter boundary for
Granada–Mallorca and Padova–Torino; it does not rewrite their kickoff or release
time. Other reviews remain unavailable until their table treatment is explicit.

## Verified scopes and unresolved rules

The adapter uses pinned reviewed cohort/phase evidence plus roster and schedule
validation. It does not infer formats from display names or count downloaded
fixtures to establish season length. Planned fixture counts, complete membership,
provider table IDs, round ranges, pair multiplicities and venue counts must agree.

Supported independent scopes use one, two, three or four meetings per pair.
This includes regular calendar-year and cross-year calendars, verified ordinary
Apertura/Clausura phases, reviewed incomplete snapshots and Scottish Championship
2020/21's three-meeting season. Countdown and standings have different scopes:

* Scotland's Premiership countdown stays continuous across 33+5 rounds. The
  regular pre-split independent form table is supported. Split standings with
  carry-over/rank constraints remain unavailable.
* Japan 2016 retains its verified immutable phase mapping and continuous
  34-round countdown. Its stage tables remain unavailable through this adapter.
* Japan's 2026 transition keeps independent 10-team, 18-round group clocks.
  Its shootout-dependent 3/2/1/0 standings are unavailable; an ordinary 3/1/0
  table would be wrong.
* Chile 2019 stays curtailed. Its planned calendar is not shortened to downloaded
  results. Administrative official-table outcomes remain distinct from played
  results and require their timed treatment before standings are enabled.
* Carry-over, halving, unequal schedules, unresolved playoffs, conflicting
  topology, missing phase membership and untimed administrative rulings fail
  clearly. Three supplemented Greek 2020/21 fixtures lack provider phase IDs;
  they remain discoverable, and all table/calendar features for that publication
  are blocked rather than silently omitting their dependencies.

The derived table is explicitly **independent 3/1/0 sporting form**, using the
existing points/GD/GF/GA competition-rank definition. It does not claim to
reproduce official deductions, tie-breaks or administrative tables. Table scopes,
rosters and adapter policy enter state/cache identity and saved metadata.

The existing closest-96-hour-cluster countdown is preserved. It uses the revised
published schedule, which may contain later schedule changes. The report exposes
this retrospective assumption; strict availability blocks countdown without
archived schedule-at-issuance evidence. It is not certified future-invariant.

Primary format references supplement the pinned cohort evidence:
[SPFL reduced 2020/21 schedule](https://spfl.co.uk/news/spfl-confirms-12-10-10-10-league-structure-for-n),
[SPFL rules](https://spfl.co.uk/admin/filemanager/images/shares/pdfs/MASTER%20-%20Rules%20and%20Regulations%20(CLEAN%20-%2011%20JUNE%202024).pdf),
[HNL regulations](https://hns.family/files/documents/33080/Propozicije%20natjecanja%20SuperSport%20HNL%2026-27.pdf),
[Slovenian first-league instructions](https://www.prvaliga.si/Doc/Info/2025/Navodila%20za%20organizacijo%20tekem%20v%201.%20SNL_V5.1_julij2025.pdf),
[J.League transition](https://www.jleague.jp/sp/special/2026specialseason/j1/).
These references do not authorize extrapolating a modern format to every older
season; each publication must also pass its pinned historical scope checks.

## Recipes, splits and GAT

For a new recipe replace legacy `data.seasons`/`data.leagues` with:

```json
{
  "data_root": "data/xDiyo_data",
  "query": {"competitions": 170, "seasons": 61243},
  "tables": ["matches", "team_seasons"]
}
```

The optional top-level `native_preprocessing` object specifies the two conventions
above. Preparation stores the report in `outputs.native_preflight` and dataset
definitions; JSON/Python exports retain the same query/settings/pins. Builder
server routes `native-discover`, `native-select` and `native-preflight` expose
the same resolver. Existing builder selectors and storage-key recipes are
compatible; this change does not add a second independent selection framework.

New query recipes' default round-based `TemporalSplit` includes the independent
calendar ID. Explicit round grouping that combines distinct calendars or repeats
a team within a round fails. Season-based gaps measured in rounds are likewise
checked. No provider round values are rewritten.

For GAT By Round on several independent phases, use the existing
`round_stage_column="countdown_calendar_id"` and include `round_stage` in
`TemporalGraphConfig.round_keys`, or select one independent phase. Ambiguous
round groups fail before graph preparation. A combined Japan 2016 phase calendar
still needs an explicit original-stage key for By Round; its continuous countdown
does not by itself prove a graph issuance grouping. The model and its mathematics
are unchanged. GAT's existing previous-day cutoff is retained.

Refresh hashes and membership with the existing helper, not hand-edited JSON:

```powershell
python examples/refresh_recipe_data.py recipe.json --output recipe_refreshed.json --previous-revision e82ffdff --dry-run
python examples/refresh_recipe_data.py recipe.json --output recipe_refreshed.json --previous-revision e82ffdff
```

Old recipe/record files are preserved. The refreshed recipe, compact native
selection and change report are new small metadata artifacts, not dataset copies.
Old pins still fail against changed manifests, including metadata-only changes.
Reproduce an old experiment from its frozen inputs or corresponding recorded Git
revision; refreshing does not migrate fitted models or prior experiment results.
Frozen experiment recipes are not refreshed by this work.

The committed [HNL example](demo_recipe.json) and
[refreshed pinned recipe](demo_refreshed.json) demonstrate this path: 180 targets,
three features and five temporal folds, with t−1 cutoffs and no fit.
[demo_preparation.json](demo_preparation.json) records the result. The generated
refresh output contains local absolute paths; use the portable original example
and refresh it into a new path on another machine.

## Reproduce verification

```powershell
python examples/preflight_native_library.py --require countdown standings movements --output docs/analytics/native_preflight_20261116/preflight.json
python examples/verify_native_preflight_20261116.py
python examples/verify_native_query_loading_20261116.py
```

[verification.json](verification.json) records entity/alias coverage, all-cohort
engine checks, independent sample-table comparisons, exact legacy parity with
`e82ffdff`, source hashes and frozen-control hash checks. [pytest.xml](pytest.xml)
and [ui-tests.log](ui-tests.log) record regression results. All fixtures are counted
in the coverage denominator, including unsupported/missing-phase records. The
verification script validates real data but never trains a model or writes it.
[query-loading.json](query-loading.json) additionally verifies 172,274 unique
fixture identities, identical name/ID loading, and no-fit GAT preparation on HNL
and Peru's two phases. The latter retains both native phase IDs and their
independent round groups.

Final checks: **446 Python regressions**, **21 JavaScript tests**, and
**630,432 independently recomputed standings scalar comparisons** passed.
Every supported countdown scope produced finite values in range; unsupported
scopes remain in the denominator. The real legacy Premier League 2024/25 check
matched the parent implementation's countdown values and standings values,
publications and state/cache identity exactly. All three frozen experiment
control hashes matched. Python 3.14.0/pandas 2.3.3 were used; the only Python
warning was the existing empty/all-NA history-concatenation FutureWarning.
[receipts.json](receipts.json) binds the test and audit artifacts to source bytes.

The availability assumptions above remain evidence limitations even when every
numerical check passes. These checks do not validate model quality or invent
historical schedule, score-publication or promotion-announcement archives.
