# Missing statistics backfill

The published dataset can be enriched from a separately supplied statistics
workbook. Existing native observations take priority, including genuine zeros.
The original workbook is read-only.

## Completed backfill: 5 October 2026

All **174** published league-seasons across **13 leagues** were inspected.
**810,624** missing team/period/statistic values were filled across **50,587
matches** in **154 publications**. Twenty publications had no usable additions.
The workbook supplied 51,230 uniquely matched fixtures; 792 unmatched fixtures
were excluded. Seven statistic pairs containing invalid negative source counts
were rejected.

| Statistic | Values filled |
| --- | ---: |
| Corners | 17,634 |
| Yellow cards | 31,274 |
| Red cards | 248,572 |
| Ball possession | 18,550 |
| Expected goals | 229,618 |
| Total shots | 18,505 |
| Shots on target | 18,614 |
| Shots off target | 18,506 |
| Fouls | 189,563 |
| Goalkeeper saves | 19,788 |

Every imported value was independently checked against its workbook cell. All
154 updated publications passed native-loader, hash, nested-table parity and
ZSTD compression checks. Eleven focused tests passed. Loader → team history →
rolling-feature smoke checks passed for Bundesliga, Bundesliga 2, Championship
and Premiership 2015/16.

[Per-season coverage and verification summary](data/statistics_backfill_20261005.json).

## Matching and field mapping

The [Japan extension](japan_enrichment.md) adds calendar-year J1 imports and an
explicit `--leagues` scope. Its 2026 transition tournament is kept separate from
2026/27. The earlier European publications are unchanged by that scoped import.

Fixtures are matched using league, season, ordered home/away team identities and
calendar date. Provider match IDs are never treated as native event IDs. Existing
reviewed aliases and schedule exceptions are reused. The supplementary aliases in
`data/statistics_backfill_aliases.json` were reviewed against repeated same-date,
known-opponent fixtures; the support counts are recorded there. Ambiguous fixtures,
unmatched dates, source-sheet disagreements and many-to-one matches are excluded.
The workbook timezone is unspecified; ordinary matching uses the native UTC
calendar date, with no broad date-tolerance fallback.

Only completed, non-awarded native fixtures are eligible. Full-time statistics
refer to regulation time. The audited native publications contain no explicit
extra-time or penalty-shootout fixtures. The workbook's periods map as follows:
`FT → ALL`, `1H → 1ST`, `2H → 2ND`.

| Workbook statistic | Native key | Group for new rows |
| --- | --- | --- |
| corners | `cornerKicks` | Match overview |
| yellow_cards | `yellowCards` | Match overview |
| red_cards | `redCards` | Match overview |
| ball_possession | `ballPossession` | Match overview |
| xg | `expectedGoals` | Match overview |
| total_shots | `totalShotsOnGoal` | Match overview |
| shots_on_target | `shotsOnGoal` | Shots |
| shots_off_target | `shotsOffGoal` | Shots |
| fouls | `fouls` | Match overview |
| goalkeeper_saves | `goalkeeperSaves` | Match overview |

Possession stays on the native **0–100 percentage scale**; counts remain counts.
Blank source values stay missing. Negative/non-finite counts, fractional counts,
out-of-range percentages and inconsistent possession pairs are rejected. Counts
are not inferred from another period, and half-time possession is not averaged.

## Preservation rules

- Backfill is decided separately for each match, period, statistic and side.
  A match with complete corners can receive missing possession data.
- If the statistic is already populated in any native display group, its value
  is preserved. No contradictory duplicate is added under another group.
- A native display-only observation is also protected. When an existing row has
  no numeric value or meaningful display, only its missing value/display can be
  filled; its other native metadata remains unchanged.
- Missing rows use existing native names, fields and types. Their provenance
  identifies them as imported statistics, with the workbook SHA-256, worksheet,
  row, column and provider fixture ID. Filled existing rows retain this evidence
  in the separate cell audit.
- The native long statistics table and the nested season statistics are checked
  for agreement before publication and updated together. All populated native
  fields and every unrelated nested column are checked for preservation.
- Original collection coverage/status evidence remains unchanged. The manifests'
  `statistics_enrichment` records describe the additional imported observations.

## Run and audit

```powershell
python examples/backfill_statistics.py --workbook "C:/path/football_database.xlsx"
python examples/backfill_statistics.py --workbook "C:/path/football_database.xlsx" --apply
```

The first command produces a proposal without changing publications. Both commands
inspect all published league-seasons. Seasons outside the workbook's coverage and
unavailable source fields remain unchanged. Source extracts and detailed fixture,
cell and rejection audits are saved under the ignored
`data/audits/statistics_backfill/` directory. A per-season journal allows an
interrupted publication to finish on the next `--apply` run.

Parquet compression and schemas are preserved; the current publications use ZSTD.
The writer refuses Snappy. Actual manifest fingerprints and sizes are updated.
Old pinned recipe selections therefore need the existing explicit data-selection
refresh helper to use the enriched dataset; saved trained artifacts are not edited.
New recipes/loaders use the current publications normally, with no new UI fields.

For a recipe with pinned source selections, use
`xdiyo_analytics.ui.refresh.refresh_recipe_data(recipe, output_path, workspace=...)`
to write an explicitly refreshed recipe. This does not rewrite saved experiments.

These imported observations remain secondary-source measurements, particularly
for model-dependent quantities such as xG. Provenance is retained so their coverage
and contribution can be inspected independently.
