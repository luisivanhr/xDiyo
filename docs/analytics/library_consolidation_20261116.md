# Native library consolidation, 16 November 2026

The supplied inventory contains 48 leagues, 617 native publications and 172,274
matches. Preparation enriches the 328 new publications and preserves the 289
previously enriched publications. It does not rebuild the separate named
international/comprehensive catalogues or run an experiment.

## Season-entry flags

`home_got_promoted`, `away_got_promoted`, `home_got_demoted`,
`away_got_demoted`, and both `*_season_entry` columns use the established native
schema. `team_seasons` records the exact team, competition and season IDs,
movement category, observed predecessor IDs and evidence. These flags describe
entry **into** the season, never its end-of-season outcome.

Consecutive rosters identify retained teams and transfers between collected
adjacent divisions. Absence from reviewed same/upper division rosters supports
the existing explicitly labelled inference of entry from below. First observed
seasons and Argentina's missing 2020 boundary use external reviews. The review
JSON retains sources and all exact reviewed rosters and observation hashes.
Administrative reinstatements, licence changes, relocations with new provider
IDs, and expansion clubs in closed leagues are `other_entry`, not sporting
promotions. Existing source IDs and modern provider names are not rewritten.

The 5,402 new publication-level team-season rows comprise 4,612 retained,
666 promoted, 101 relegated and 23 other entries, with no unresolved flags.
This includes repeated evidence in paired short-tournament publications.
These are retrospective enrichment records: the external review date is not
invented historical publication evidence. Inferred membership comparisons are
distinguished from explicit boundary reviews in each row's evidence.

## Annual grouping without changing source identities

Both manifests carry versioned `season_calendar` metadata. The native loader
exposes `annual_league`, `annual_season`, `annual_phase` and nullable
`season_half` alongside the original match columns. Native `competition_id`,
`season_id`, `tournament_id`, rounds, dates and fixture IDs remain unchanged.
The annual columns are grouping metadata; they do not rewrite model season
identities or alter existing rank/countdown definitions.
These columns also survive team-history construction. The existing numeric
`season_phase` input (including Japan 2016) is separate and remains untouched.

* Mexico uses annual split-year cycles; Apertura is half 1, Clausura half 2.
  The source `Liga_MX_Apertura_26_26` belongs to annual `2026/2027`.
* Argentina and Colombia's calendar-year Apertura/Clausura publications share
  an annual grouping and annual entry flags. Combined 2026 publications retain
  their explicit source tournament-to-half mappings.
* Colombia's 2021 mid-year entrants receive their entry flags only in the
  publications where they occur. No earlier fixtures or memberships are added.
* Peru's Apertura/Clausura (and 2021 first/second stages) retain separate phase
  identities within the year. The additional 2017/18 Verano stages remain
  `Verano` with a missing half, rather than pretending there were only two stages.
* Calendar-year seasons stay calendar-year, including Brazil's delayed 2020
  season despite its source filename ending `20_21`.
* Japan's transition competition remains `2026_transition`, separate from
  `2026/2027`; its already reviewed movement flags are unchanged.

Unknown tournament identities in a reviewed multi-phase mapping fail on load.
There is no inference of phase identity from kickoff dates.

## Preservation and compatibility

New movement tables and rewritten new observations use ZSTD, never Snappy.
Previously enriched observation files remain byte-for-byte unchanged, including
older native tables that already use Snappy. Their codecs are recorded in the
receipt; this task does not recompress unrelated old files.

Original Arrow columns, nested observations, values, units, missingness,
schemas, source/availability flags and reconciliation evidence are compared
before and after preparation. The three Greek Flashscore supplements retain
their exact supplied fixture IDs in the ordinary native ID columns. Their
source evidence and existing provider-ID verification limitation are retained.
Nothing labels those observations as independently captured SofaScore responses.

Manifest coverage remains authoritative. The 32 explicitly marked snapshots
remain snapshots, not assertions that an ongoing season is complete. Other
curtailed, excluded-stage and missing-component dispositions are preserved.

Hashes change honestly when flags or calendar metadata change. Saved source
selections pinned to older manifests must be explicitly refreshed using the
existing recipe/source refresh workflow. A legacy publication without calendar
metadata keeps its original loader behavior. Old artifacts and experiments are
not modified or automatically rebuilt.

Both movement and calendar publication use fingerprinted recovery journals.
Rerun the preparation command after an interrupted write; it completes only
the already prepared files. Changed live/staged files fail validation.

## Reproduction

From the repository root, using the configured Python environment:

```powershell
python examples/prepare_library_20261116.py
python examples/prepare_library_20261116.py --apply
python examples/verify_library_20261116.py
python -m pytest tests/analytics/test_library_consolidation.py tests/analytics/test_movement_enrichment.py tests/analytics/test_mexico_enrichment.py tests/analytics/test_japan_enrichment.py tests/analytics/test_loading.py tests/analytics/test_awarded_loading.py -q
```

Inputs: [original inventory](data/library_league_inventory_20261116.csv) and
[reviewed rosters, boundaries and calendars](data/library_movement_review_20261116.json).
The generated [publication receipt](data/library_preparation_20261116.json)
records all native hashes, row counts, compression codecs, calendar identities,
the review hash and preparation script hash. Test receipts are retained beside it.

Final verification: **236 tests passed**, with one existing pandas concatenation
FutureWarning. The [independent final verifier](data/library_pipeline_smoke_20261116.json)
checked every publication and table fingerprint against the preparation receipt,
reloaded all 172,274 matches, and exercised five real-data movement-feature and
fixture-assembly paths plus combined Colombia 2017 loading. Japan 2016 countdown
values matched exactly with/without the additive metadata across 612 team rows.
The [environment receipt](data/library_preparation_environment_20261116.json)
records runtime versions, commands and test/verification hashes.

Boundary sources include the [UEFA 2017/18 club directory](https://editorial.uefa.com/resources/023c-0f842cbf29bd-62e198f93b00-1000/first_division_clubs_in_europe_2017_18.pdf),
national league/club records and the country references in the review JSON.
Provider identity changes are preserved rather than silently joined by names.
