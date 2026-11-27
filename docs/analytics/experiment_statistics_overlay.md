# Use collected match statistics in an experiment

Select the published release in the experiment recipe. The adapter loads the native matches and statistics, fills only missing team/statistic/period cells, then builds statistical histories, rolling features, graph inputs and labels. A missing value in one team’s row can be filled independently. Existing numeric zeros, display values and previously imported observations take precedence. Recipes without a selection keep their original inputs.

From the xDiyo checkout, print the JSON section for the local published release:

```powershell
.venv/Scripts/python.exe examples/prepare_collected_statistics.py
```

Paste the resulting `statistics_overlay` section into the recipe alongside `data`. For a Python experiment run from the repository root:

```python
from examples.prepare_collected_statistics import recipe_section
from xdiyo_analytics.ui import prepare_recipe

recipe.update(recipe_section())
prepared = prepare_recipe(recipe)
audit = prepared.dataset.definitions["statistics_overlay"]
```

The helper reads the [published manifest](../../data/statistics_backfill/2026-11-27/manifest.json) and checks all six compressed partitions. A fresh Git checkout contains the data it needs. Keep the manifest hash and requested keys with the experiment. On another machine, run the helper there to resolve the local path. Each published cell includes the hash of its original reviewed record. Raw captures remain in the collection archive. Both native league/season lists and SofaScore queries or selections are supported. The example selects corners, total shots, shots on and off target, and possession. It includes the earlier supplement’s verified periods for these keys. The campaign itself adds only full-time (`ALL`) observations.

### What the November 27 release contains

The release layers the earlier workbook, API-FOOTBALL and reviewed cached supplement with the campaign snapshot `20261127T082851Z` and the separately verified Omiya replay for event `9632204`. The latter supplies one additional home shots-off-target value of 9. Their combined 41,853 campaign cells are unique and do not overlap the earlier supplement. The gap inventory excludes ten cells on the unplayed, awarded PAOK–Olympiacos fixture.

The collector sometimes chose the provider reporting the highest team shot total. It preserved the selected provider’s values and comparison evidence. Of 15,626 cells with this policy, 15,279 have no comparison marked as verifying the blocked-shot definition. This can favor larger totals and change the interpretation of off-target counts. The release also includes FootyStats provider-reported possession, corners and shots on target. Consult the per-row `source_item_json` for source, fixture identity evidence and selection details. The per-statistic experiment receipt records fills by source, shot selection modes and how many occupied values were protected.

The adapter matches competition, season, fixture, team, period and statistic. A matched value enters the same history row as the native match statistics. The experiment's usual fixture eligibility and prediction cutoffs apply. Capture timestamps are optional source metadata. The overlay has no timing setting. Previously saved retrospective selections load directly. To switch a previously time-filtered selection to fixture matching, remove its `mode` and `as_of` fields and rebuild the prepared data.

### Remaining full-time team-stat cells

| Statistic | Since 2018-01-01 | Since 2020-01-01 |
|---|---:|---:|
| Corners | 33 | 0 |
| Total shots | 5,129 | 14 |
| Shots on target | 88 | 8 |
| Shots off target | 5,171 | 50 |
| Possession | 2,750 | 18 |
| **Total** | **13,171** | **90** |

Counts use actual UTC kickoff dates and the eligible played-fixture inventory.
