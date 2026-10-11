# Mexican dataset enrichment

Preparation begun on 10 October 2026 covers **33 publications and 4,939 matches**:
16 Apertura/Clausura pairs from 2010/11 through 2025/26, plus the Apertura 2026
completed-match snapshot (`Liga_MX_Apertura_26_26`). The collector's selection,
IDs, kickoff times and optional coverage are preserved. One sourced final-score
correction is documented below. Clausura 2019/20
is the interrupted tournament; no unplayed fixtures are invented.

## Movement flags

All **595 team/tournament entries** are resolved: **574 retained, 18 promoted,
three administrative entries**, and no unknowns. There are no relegated entrants
because this collection contains only the top division. Flags describe entry
into the annual season, not relegation at its end. Both short tournaments retain
that annual classification; Clausura does not reset a promoted team to retained
merely because it played in the preceding Apertura.

Consecutive native ID rosters supply ordinary retention and promotion inference.
The [review](data/mexico_movement_review_20261010.json) binds the original matches
hashes and rosters and records researched exceptions:

- [Necaxa's 2010 promotion](https://www.clubnecaxa.mx/contenidos/historia)
  supplies the first-season boundary; the other 17 members were retained.
- [La Piedad's promoted franchise moved to Veracruz in 2013](https://www.jornada.com.mx/2013/05/29/deportes/a14n2dep),
  so Veracruz is promoted. Other existing native IDs remain unchanged through
  franchise moves and retrospective provider name changes.
- [AtlÃ©tico San Luis earned promotion in 2019](https://www.atleticodemadrid.com/noticias/el-atletico-de-san-luis-logra-el-ascenso).
  [FC JuÃ¡rez acquired Lobos BUAP's place](https://www.mediotiempo.com/futbol/liga-mx/oficial-juarez-fc-division-sustituye-lobos-buap),
  so JuÃ¡rez is `other_entry` in both 2019/20 publications.
- [Atlante purchased MazatlÃ¡n's franchise for 2026](https://www.milenio.com/futbol/liga-mx/atlante-confirma-regreso-a-la-liga-mx-compra-franquicia-del-mazatlan)
  and is `other_entry`. Missing source-division history is not fabricated.

Both sides' flags are embedded in native matches and nested Parquet. The
`team_seasons` table carries evidence and observed predecessor IDs. Annual
same-tournament predecessors are used only when their native competition IDs
agree. Clausura changes competition ID after 2010/11, so its 2011/12 predecessor
is explicitly null. No cross-ID warm-start is guessed.

Historical names can reflect later provider names, including MazatlÃ¡n on
Morelia-era records. This processing preserves those names and IDs; it does not
reconstruct a separate historical franchise identity model.

## Shared provider identity

The two 2010/11 tournaments share competition ID 11621 and season ID 2770.
Their fixture lists and native files are distinct. The loader accepts the
collector's explicitly declared `_tables/cohorts/<exact season stem>/...` path,
checking the cohort stem, league, competition, season, version and hashes.
Ordinary publication path validation remains in force.

When both are loaded, identical `team_seasons` evidence for those shared keys
is coalesced: **577 unique provider team-season identities** across all 33
publications. Only declared shared cohorts qualify. Every substantive field,
including missing predecessor IDs and evidence, must agree; conflicts raise.
All publications remain in `SeasonData.provenance`; the first copy supplies
the coalesced row's source labels. Match rows are never collapsed. Both
tournaments remain selectable in the UI.

## Missing statistics

The workbook covers Mexico from **2020/21 onward**. Earlier seasons retain their
original gaps. The importer separates annual Liga MX entries into native
Apertura/Clausura publications by fixture date and season label.

**23,670 missing values were filled in 1,834 matches**, across 13 publications:

| Statistic | Values filled |
| --- | ---: |
| `expectedGoals` | 8,440 |
| `redCards` | 8,322 |
| `fouls` | 6,520 |
| `yellowCards` | 298 |
| `goalkeeperSaves` | 54 |
| `shotsOnGoal` | 12 |
| `cornerKicks` | 6 |
| `ballPossession` | 6 |
| `totalShotsOnGoal` | 6 |
| `shotsOffGoal` | 6 |

Existing observations, including zeros and display-only values, take priority.
No missing period is inferred. Added cells retain workbook fingerprint,
worksheet, row and column evidence. Native long and nested statistics are
updated together with truthful manifest hashes.

There are 1,922 matched workbook fixtures and 206 unused unmatched rows,
including playoffs outside the native regular-season selection. The
[reviewed aliases](data/mexico_statistics_aliases.json) distinguish current
QuerÃ©taro ID 1951 from historical ID 1952. A further
[301 exact date exceptions](data/mexico_statistics_fixture_overrides.json)
resolve adjacent calendar dates: clocks differ by one or two hours, with two
cases differing by an additional two minutes. Ordered identities, tournament
membership and both final scores agree. The workbook timezone remains
unspecified; native kickoff times are unchanged.

### Reviewed final-score correction

Event **11341818**, Tigres–Querétaro on 2 September 2023, originally had native
final/display score 4–0 despite periods of 2–0 and 3–0. The workbook and
[Tigres' official report](https://www.tigres.com.mx/es/noticias/tigres/la-previa-ante-queretaro-3/)
confirm **5–0**. On 11 October 2026, both native and nested final/display home
scores were corrected to 5, with original values and source evidence retained
in the [review record](data/mexico_score_review_20261010.json) and manifests.
All other match fields are unchanged. The cause of the provider discrepancy
is unknown; this is a correction to the official played result, not an awarded
result. The fixture is now included in missing-only statistics backfilling
(16 added cells). All 1,922 mapped fixtures agree on final scores.

## Verification and use

- **464 Parquet files** passed hashes, bytes, row counts and **ZSTD** checks.
- All added cells match the workbook and persisted values; nested/native
  statistics agree for 3,868 matches. **7,775 protected files** are unchanged.
- Repeating the statistics import proposes **zero** additional values.
- Native UI preparation loaded all 4,939 fixtures, 9,878 history rows,
  six movement/rolling-goal predictors and 15 temporal folds.
- A corners/cards preparation produced 1,833 eligible matches, eight predictors
  and four folds, retaining normal missing-label exclusion.
- **218 regression checks passed** across pandas 2.2 and pandas 3, covering
  loading, movement evidence, backfilling, recipes and UI workflows.
- No experiments were fitted or changed.

See [verification](data/mexico_enrichment_verification_20261010.json),
[publication fingerprints](data/mexico_movement_publication_20261010.json), and
the [native corners recipe](data/mexico_preparation_recipe.json). Adjust its
data root for another checkout. Existing pinned recipes require explicit
[data refresh](team_movements.md#updating-recipes-after-dataset-changes).

## Reproduce

```powershell
python examples/backfill_statistics.py --workbook "C:/path/football_database.xlsx" --leagues Liga_MX_Apertura Liga_MX_Clausura --team-aliases docs/analytics/data/mexico_statistics_aliases.json --fixture-overrides docs/analytics/data/mexico_statistics_fixture_overrides.json --audit-root data/audits/mexico_enrichment_20261010/statistics
python examples/prepare_mexico_20261010.py
```

Both default to validation/proposals; `--apply` publishes reviewed changes.
The reviewed score correction is reproducible with
`python examples/repair_mexico_score_20261011.py --apply` before statistics
backfilling. Movement publication follows statistics. The workbook is never edited.
