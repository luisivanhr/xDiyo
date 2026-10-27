# Season countdown: scheduled round progress

`SeasonCountdown()` is selectable as **Season countdown** in Features & ratings. It assigns an effective season round from the schedule available when making the prediction, before matches start. `SeasonProgress`, standings and the underlying dataset are unchanged.

## One fixed rule

1. Within each competition-season/calendar, group unique fixtures by immutable assigned season-round ordinal.
2. For each round, select its largest scheduled cluster spanning **at most 96 hours**. On equal fixture counts choose the earliest cluster start. Its interval runs from the earliest to the latest kickoff in that cluster.
3. A fixture inside its own round's main interval keeps its assigned ordinal, including before the first kickoff. An isolated advanced fixture cannot displace a larger main cluster.
4. Otherwise choose the closest round-cluster interval. Distance is zero inside an interval, and otherwise is distance to its nearest endpoint. Equal-distance ties choose the later season-round ordinal. Clamp the answer below by the fixture's original ordinal. With no usable interval, retain the original ordinal.
5. `season_countdown = (total_rounds - effective_round) / (total_rounds - 1)`.

Following the user's final clarification, there is **no prediction-batch override**. A mixed selection or its row order never assigns a common round. There is no activation threshold, `schedule_mode`, ambiguity flag, manual selector or additional feature parameter. Every eligible fixture with valid kickoff, round and prediction cutoff receives a value. Intentional playoff exclusions and missing required metadata retain missing outputs; malformed identities and conflicting schedule copies still fail validation.

For example, a round-26 catch-up scheduled inside round 35's main interval receives round 35 even if every match is marked `notstarted`. When it falls between round 35 and round 36, the closer interval wins, regardless of whether either round has begun. Ordinary round-35 fixtures stay at 35 even if a later round's interval overlaps.

Intervals are built once per competition-season/calendar. Bounded vectorized distance comparisons reuse them for all queries. Unique fixture identities prevent paired team rows from changing cluster counts. Subsets evaluated with the same full schedule receive identical values.

## Pre-match information contract

Upcoming scheduled dates are valid pre-match inputs. The feature never consults starts, played status, scores, results, standings or result-release timestamps. Prediction cutoffs retain native alignment and validity checks (no later than target kickoff); they do not filter upcoming schedule context.

Use the schedule available to the prediction. A historical export may contain subsequently revised dates. This feature does **not** reconstruct unavailable historical schedule knowledge, and recomputing from such an export can change earlier outputs. No archived schedule versions are required. Sparse schedules can make a catch-up look like its own round's main cluster; provide the full schedule before model-population filtering whenever possible.

## Verified competition formats

| Competition | Planned rounds / calendar |
| --- | --- |
| Ordinary double round robin | even n: `2(n-1)`; odd n: `2n` |
| Mexico Apertura / Clausura | even n: `n-1`; odd n: `n`; separate published tournament clocks |
| Scotland Premiership | 33 + 5 = 38, continuous through both split groups |
| Japan J1 2015 / 2016 | 34 across two stages |
| Japan 2026 transition | 18 per independent ten-team regional group |

Regular-season eligibility is explicit. Scotland permits tournament IDs 54, 92066 and 92067; Japan transition permits 171852 and 171853 and never borrows clusters across groups. Mexico's historical Clausura 2010/11 uses ID 28; later Clausura uses 16753. Belgian playoffs and unrecognized stages are excluded from both cluster construction and feature values.

J1 2015 numbering is already continuous. J1 2016 second-stage rounds receive offset 17 from **original phase identity**, never kickoff date. The packaged 306-fixture crosswalk was matched to the [official J.League table](https://data.j-league.or.jp/SFMS01/search?competition_frame_ids=1&competition_years=2016&lang=en) by home team, away team and assigned round. Custom fixtures need explicit `season_phase` 1 or 2 unless their identity is verified. Conflicting phase evidence is rejected. Native labels retain phase metadata so context and fixture features agree. See [phase provenance](j1-phase-provenance.json).

When inferring ordinary season length, provide complete `team_seasons` or the existing `team_count` override. An explicit `total_rounds` is sufficient without membership. Scotland and Japanese-transition fixed formats retain their verified denominators and reject contradictory team-count overrides. Cancelled/incomplete seasons keep planned lengths rather than the highest downloaded round.

## Native use and recipes

```python
from xdiyo_analytics.data import load_seasons
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.features import SeasonCountdown, evaluate_features

data = load_seasons("data/xDiyo_data", "23_24", leagues=["Premier_League"],
                    tables=["matches", "team_seasons"], include_awarded=True)
history = build_team_history(data)
features = evaluate_features(
    history, {"season_countdown": SeasonCountdown()},
    schedule_history=data.matches, team_seasons=data["team_seasons"], keyed=True,
)
```

The builder loads full schedule context and needed membership before modelling filters. This does not add fixtures to the training population. UI, JSON and Python exports retain only `format`, `total_rounds` and `team_count`. Match-layout output is `fixture::alias` once per match; team output repeats the same value; context features resolve the same calendar.

```json
{"features": {"season_countdown": {"component": "features.SeasonCountdown", "params": {}}}}
```

## Explicit legacy migration

Recipes containing removed `schedule_mode` fail with a migration instruction. `migrate_season_countdown_recipe(old_recipe)` returns a copied recipe and change records identifying `nearest_scheduled_cluster` and `values_may_change=True`. It visits nested/context features, rejects unknown modes, preserves explicit auxiliary input paths and is idempotent. Replace an old auxiliary file containing repeated archived snapshots with one consistent schedule. Review the result and save it to a new file; the helper never overwrites a recipe, prepared artifact or run.

Recipes without the removed setting retain their constructor shape, but recalculated values can change from the rejected majority policy. Saved recipes, feature artifacts and frozen experiments are not silently rewritten or retrained.

## Reproducible verification

From the repository root, with `PYTHONPATH=src`:

- `python docs/analytics/season-countdown/verify_scheduled.py`: after committing implementation, verifies exact source bytes against Git blobs, runs the regression suite and both scripts below, then writes `scheduled-verification.json`. Generated text receipts use canonical LF bytes to keep committed checksums reproducible.
- `python docs/analytics/season-countdown/audit_scheduled.py`: checks frozen expected eligible/excluded counts for all 289 publications / 92,983 fixtures, exact agreement with an independent exhaustive-cluster/per-target-distance oracle, and hand-inspected effective rounds. Unexplained missingness, all-NaN output, finite playoff values and incorrect representative rounds fail.
- `python docs/analytics/season-countdown/compare_scheduled.py`: compares every fixture with frozen majority-rule commit `643c2029`; emits changed fixture identities, old/new rounds and countdowns without requiring old-value parity.

`scheduled-effective-round-checks.json` records 14 independently specified examples, including Chelsea-Tottenham (nearest round 36), FC Tokyo-Hiroshima (nearest round 16), Bristol City-Watford (29), Cerezo-Nagoya (32), Millwall-Watford (26), both Scottish split groups, Mexico and both Japanese transition groups. Their rationales include scheduled interval endpoints.

All earlier receipts, including `majority-*`, `review-*`, `publication-audit.csv`, `tests.xml` and `verification.json`, remain historical evidence of their original releases. New results use `scheduled-*`. No models or frozen experiments are rerun.

Format references: [Scotland 33+5](https://spfl.co.uk/news/post-split-qa-season-202324), [Japan regional rounds](https://www.jleague.jp/news/article/32817/), [Mexico 2019 calendar](https://lmxwebsite.s3.amazonaws.com/docs/publicaciones/ligamx/CALENDARIO-LIGA-MX-TORNEO-APERTURA-2019.pdf).
