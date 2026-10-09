# Japan dataset enrichment

The 9 October 2026 enrichment covers all **35 publications and 12,494 matches**:
J1 and J2 calendar seasons from 2010 through 2025, the J1 2026 transition
tournament, and the J1/J2 2026/27 completed-match snapshots. Snapshot status and
the collector's original coverage evidence are preserved.

## Missing statistics

The supplied workbook has J1 statistics for 2019–2025, 2026 transition and
2026/27. It contains no J2 statistics or earlier J1 statistics. These gaps remain
missing. The import matched 2,623 fixtures by league, season, ordered team
identities and calendar date. Twenty transition playoff fixtures are outside the
published regular-stage selection and were excluded, without creating matches.
The 51 matched transition fixtures decided on penalties were also excluded by
the existing ordinary-finished-only backfill policy.

**37,040 missing statistic values were filled in 2,572 matches**, across nine
publications:

| Native statistic | Values filled |
| --- | ---: |
| `redCards` | 13,920 |
| `expectedGoals` | 12,006 |
| `fouls` | 9,240 |
| `yellowCards` | 1,766 |
| `goalkeeperSaves` | 96 |
| `shotsOnGoal` | 12 |

The other workbook fields, including corners and possession, supplied no missing
eligible values: existing native observations take priority. Genuine zeros,
display-only observations, team identities, kickoff times, scores, awarded flags
and all unrelated native fields are preserved. No halves or xG values are inferred.
Every added value retains worksheet/row/column and workbook fingerprint evidence.
The native long statistics table and nested season table are updated together.

All 2,572 eligible fixtures also agree with the workbook's final scores. The
reviewed abbreviated team names are recorded in
[the alias evidence](data/japan_statistics_aliases.json). Repeating the import
proposes **zero** additional values.

## Movement flags

All **697 team-season entries** are resolved: **575 retained, 78 promoted and
44 relegated**, with no unknown flags. Both sides' flags are materialized in
native matches and nested publications; the native `team_seasons` table carries
the classifications, predecessor IDs and evidence for warm starts.

Consecutive J1/J2 membership supplies the usual same-division and cross-division
comparisons. The first 2010 seasons and entrants from uncollected JFL/J3 history
were checked against [J.League's official tournament history](https://www.jleague.jp/corporate/about_competitions/tournament_history/).
Unobserved predecessor IDs remain null; they are not fabricated.

The [official 2026 membership announcement](https://www.jleague.jp/news/article/32852/)
applies the 2025 movement decisions to both the special tournament and 2026/27.
The [special-tournament rules](https://www.jleague.jp/news/article/31576/) provide
no additional promotion/relegation. Therefore both J1 publications use the same
entry flags and 2025 predecessor IDs, with distinct native season identities.
They are not used as each other's predecessor. Current-season results never
determine that season's entry flags.

The Japan review is separate from the earlier European review because the
transition publication shares a competition ID and calendar year with regular
J1. Its reproducible publisher is
[`prepare_japan_20261009.py`](../../examples/prepare_japan_20261009.py), and its
[review](data/japan_movement_review_20261009.json) binds all memberships to exact
native matches fingerprints.

## Loading and verification

The analytics loader now accepts both calendar labels (`19_19`) and adjacent-year
labels (`26_27`), while retaining manifest identity checks and rejection of
reversed or multi-year spans. Builder discovery already exposes the native
league names `J1`, `J2` and `J1_Transition`.

```python
from xdiyo_analytics.data import load_seasons

data = load_seasons(
    "data/xDiyo_data", ["19_19", "20_20", "21_21"], leagues=["J1", "J2"],
    tables=["matches", "team_seasons"], verify_hashes=True,
)
```

Request `statistics` only for selections where that table is available. Early
provider coverage remains sparse. The normal loader excludes one existing native
awarded match, yielding 12,493 rows across the full Japan selection; explicitly
setting `include_awarded=True` retains all 12,494.

Verification checked all **445 native/nested Parquet files**, their actual
hashes, byte sizes, row counts and **ZSTD** codecs. It checked every filled value
against the extracted workbook cell, native/nested statistics agreement for
7,158 matches with statistics tables, and both team flags on all 24,988 history
rows. All 3,752 protected files from unrelated publications were unchanged.
See [verification details](data/japan_enrichment_verification_20261009.json) and
[publication fingerprints](data/japan_movement_publication_20261009.json).

The native builder preparation checks exercise all Japan matches and movement
predictors, plus a separate rolling-statistics recipe. They do not fit models or
modify saved experiments. Existing pinned recipes require the usual explicit
[data refresh](team_movements.md#updating-recipes-after-dataset-changes).
Restart an already-running notebook builder to load the calendar-season fix.

## Reproduce

```powershell
python examples/backfill_statistics.py --workbook "C:/path/football_database.xlsx" --leagues J1 J1_Transition J2 --team-aliases docs/analytics/data/japan_statistics_aliases.json --audit-root data/audits/japan_enrichment_20261009/statistics
python examples/prepare_japan_20261009.py
```

Both default to validation/proposals. Add `--apply` to publish the reviewed
changes. Run the statistics import before movement publication so the final
fingerprint report describes the enriched files. The extractor versions its
cache separately from the workbook hash to avoid reusing older extracts that
omitted calendar-year leagues. The source workbook is never edited.
