# Played fixtures and collection completion

`complete: true` means that the saved collection cohort was published. It does
not prove that the collector discovered every played match. New publications
make this explicit in `complete_scope`. Missing `fixture_coverage` metadata means
that no separate coverage result is recorded, not that full coverage is verified.

`data.fixture_coverage.audit_fixture_coverage` checks a caller-supplied regular
season format against saved played rows, team pairings, home/away balance, and
independently reviewed required fixtures (teams, UTC date and final score).
`status: balanced` means those specified checks passed. It is not a claim about
every source value. Playoffs, awarded matches and curtailed seasons require their
own reviewed scope; do not apply ordinary round-robin totals to them.

## 5 October 2026 repair

Two independently corroborated played fixtures were absent from the published
match tables. Both appear in the supplied workbook's `Matches_Results`, as well
as its full-time and both half-time statistics sheets.

| Fixture | Actual kickoff, UTC | Original round | Final / halftime | Supplemental event ID |
|---|---|---|---|---|
| Aberdeen–Hamilton | 15 September 2015, 18:45 | 3 | 1–0 / 1–0 | 1000000072241 |
| Troyes–Gazélec Ajaccio | 8 August 2015, 19:00 | 1 | 0–0 / 0–0 | 1000000018849 |

Aberdeen's kickoff was 19:45 BST. Troyes' kickoff was 21:00 CEST. The workbook
stores 20:45 and 21:00 respectively on its Central European clock. Conversion
agrees with those UTC timestamps. The old Aberdeen round-3 provider stub
6773232 remained cancelled with an August kickoff; it was not used as the played
record. Original collection evidence is retained unchanged.

External corroboration:

- [Aberdeen match archive](https://afcheritage.org/matches/match-report?id=11258)
  and [match report](https://www.skysports.com/football/aberdeen-vs-hamilton-academical/report/342082).
- [Troyes result](https://www.statmuse.com/fc/match/8-8-2015-etr-vs-gaz-96719)
  and [contemporary report](https://www.maxifoot.fr/info-222205_150808/football.php).

Native played-event IDs could not be verified. Supplemental workbook fixtures
use the reserved namespace `1_000_000_000_000 + source_match_id`, retain native
team/competition/season IDs and have no fabricated provider `custom_id`.
These event IDs are local identifiers, **not provider match-page IDs**. A future
native recovery must reconcile the supplement rather than append a duplicate.

Each fixture adds 60 statistic rows using the existing native keys and periods.
Lineups, incidents, heatmaps and other uncaptured components remain empty.
Promotion/relegation fields come from each team's existing season membership.
All original match, statistics and nested rows were preserved. Updated Parquet
files use ZSTD, with genuine hashes and sizes recorded in both manifests.

The Scottish regular-season publication now has **198** matches (before the
championship/relegation split); Ligue 1 has **380**. Both pass the format and
required-fixture checks. Loading with hash verification, native/nested statistics
parity and rolling corners/goals feature construction passed for both seasons.

`fixture_enrichment` records the workbook hash, source row, local event ID and
limitations. The reviewed fixture inventory is in
`data/fixture_coverage_review_20261005.json` beside this document. Detailed local
receipts are retained under `data/audits/missing_fixtures_20261005/` at repository
root. The bounded repair script is `examples/repair_missing_fixtures_20261005.py`.
It needs the verified local workbook extracts, stages before installation and
uses a journal to resume interrupted writes without overwriting concurrent edits.

New runs use the repaired data. Previously pinned recipes need the existing
recipe-data refresh helper because the dataset's hashes have changed.
