# Season countdown: observed league progress

`SeasonCountdown()` is one fixed causal clock, selectable as **Season countdown** in Features & ratings. It uses immutable assigned season-round ordinals and evidence that other fixtures actually started. It does not reconstruct an official calendar or use round-date clusters. `SeasonProgress` is unchanged.

## Fixed rule

1. Determine the expected fixtures in a round from complete league-season membership and the verified format. For ordinary single/double round robin it is `floor(n/2)`, including odd-team byes. Downloads never set this denominator.
2. A round activates after **strictly more than half** those expected fixtures have actually started. Its activation timestamp is the kickoff of the required start in chronological order. At a prediction cutoff, only kickoffs **strictly before** that cutoff count. A simultaneous group is therefore wholly absent at its kickoff and present immediately after it.
3. The observed round is the highest activated season round, defaulting to 1. The target effective round is `max(assigned season round, observed round)`.
4. `season_countdown = (total_rounds - effective_round) / (total_rounds - 1)`.

For 20 teams, ten fixtures are expected and six must start. One advanced round-38 fixture cannot finish the league clock for other fixtures. An advanced target itself still uses its assigned round, as required by the maximum rule. A postponed target can advance once newer rounds meet the majority threshold. A sparse download might establish no newer round at all; it never reduces the threshold.

Activation tables are built once per competition/season/calendar. Batch binary searches use the supplied prediction cutoffs, including recipe `cutoff_hours`, without rebuilding a schedule per target. Fixture output is `fixture::alias`; team-row output repeats the same value when the cutoffs agree. Context features after assembly use the same aligned cutoffs. Conflicting paired metadata or fixture cutoffs are rejected.

## Evidence and causality

A bare scheduled timestamp does not prove play. The native fixture table must retain `status`, optional `play_status` and `is_awarded` evidence:

- Ordinary `finished`, `inprogress`, `in_progress`, `interrupted` and `abandoned` statuses establish a start.
- Explicit `played`, `abandoned`, `resumed`, `completed` or `played_then_awarded` reviews establish play, including a later awarded outcome.
- `not_played`, `unplayed`, `cancelled`, `canceled` and `not_started` reviews veto a start. Otherwise an awarded result alone is insufficient evidence.
- Scheduled, postponed, cancelled, unknown or missing status alone does not count. Missing kickoff/ordinal cannot activate a round. The feature does not read scores, standings or result-release times; a started and later resumed match counts from its actual initial start, not from completion.

The cutoff defaults to target kickoff and cannot be later. A later fixture date, while still later than an earlier cutoff, cannot change that earlier output. No future kickoff contributes to its observed round. This assumes the source preserves truthful actual kickoff and played-status evidence; it cannot undo a subsequent correction to an incorrectly recorded historical start. Complete membership, format and immutable assigned rounds are season metadata. This is an **observed progress convention**, not an exact official calendar.

## Formats and thresholds

| Competition | Planned rounds | Expected fixtures per round |
| --- | --- | --- |
| Ordinary double round robin | even n: `2(n-1)`; odd n: `2n` | `floor(n/2)` |
| Mexico Apertura / Clausura | even n: `n-1`; odd n: `n`; separate published tournament clocks | `floor(n/2)` |
| Scotland Premiership | 33 + 5 = 38, continuous through split | 6 across both split groups; majority 4 |
| Japan J1 2015 / 2016 | 34 across two stages | 9; majority 5 |
| Japan 2026 transition | 18 per independent ten-team group | 5 per group; majority 3 |

Regular-season eligibility is explicit. Scotland permits tournament IDs 54, 92066 and 92067; Japan transition permits 171852 and 171853. Other stages/playoffs cannot activate rounds and have missing countdowns. Mexico's historical Clausura 2010/11 uses ID 28; later Clausura uses 16753. Belgian playoffs remain excluded. If an imputer is used downstream, exclude those fixtures from the modelling population as well.

J1 2015 numbering is already continuous. J1 2016's second stage receives offset 17 based on **original phase identity**, never actual kickoff. The packaged 306-fixture crosswalk was matched to the [official J.League table](https://data.j-league.or.jp/SFMS01/search?competition_frame_ids=1&competition_years=2016&lang=en) by home team, away team and assigned round. Custom fixtures need explicit `season_phase` 1 or 2 if their identity is not verified. Conflicts with verified identity are rejected. See [phase provenance](j1-phase-provenance.json).

For ordinary formats, provide the complete `team_seasons` table or the existing `team_count` override. **Overriding `total_rounds` alone no longer removes that requirement:** it cannot establish a majority threshold. Fixed Scotland and Japanese-transition formats establish twelve and twenty season teams respectively; contradictory team-count overrides are rejected. Cancelled/incomplete seasons retain planned lengths rather than the highest downloaded round.

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

The builder loads the full fixture history and needed membership before modelling filters; source rows excluded from the prediction population can still establish starts. This does not include them as training targets. `schedule_history` remains an optional ordinary fixture table, **not** an archived schedule/snapshot store. No snapshot timestamps or `schedule_mode` are needed. UI, JSON and Python exports expose only the existing format, total-round and team-count settings.

```json
{"features": {"season_countdown": {"component": "features.SeasonCountdown", "params": {}}}}
```

## Explicit migration

Old recipes containing `schedule_mode` fail with a migration instruction. They are not silently interpreted as the new policy. Use the copy-only helper, inspect its change records, then save a new recipe:

```python
import json
from pathlib import Path
from xdiyo_analytics.ui import migrate_season_countdown_recipe

old = json.loads(Path("old-recipe.json").read_text(encoding="utf-8"))
updated, changes = migrate_season_countdown_recipe(old)
print(changes)  # each removed mode explicitly records that values may change
with Path("majority-recipe.json").open("x", encoding="utf-8") as stream:
    json.dump(updated, stream, indent=2)
```

The helper visits nested features and context features, rejects unknown legacy modes, does not mutate the original and is idempotent. It preserves explicit auxiliary input paths. If an old path contains repeated archived snapshots, replace it with one consistent fixture history or remove that override to use the native dataset. Legacy custom recipes with only a total-round override also need full membership or `team_count`. This is a semantic migration, not old-value parity or permission to reuse frozen experiment outputs.

## Validation

Run from the repository root with `PYTHONPATH=src`:

- `python docs/analytics/season-countdown/audit.py`: independent eligibility/count checks and a slow per-target count oracle for all 289 publications / 92,983 fixtures. It must match every output exactly and reject unexplained missingness, all-NaN output, finite playoff values and wrong hand-verified rounds. Writes `majority-publication-audit.csv`.
- `python docs/analytics/season-countdown/compare_majority.py`: compare against fixed pre-change commit `dc11b3a`, report changed fixtures/direction rather than assert old-value parity. Writes `majority-value-changes.csv`, `majority-change-counts.csv`, `majority-comparison.json`.
- `python docs/analytics/season-countdown/verify_j1_phases.py --html PATH_TO_OFFICIAL_HTML`: reproduce the immutable official phase join without writing dataset files.

`expected-publication-counts.csv` and `majority-effective-round-checks.json` contain independently reviewed count and 14 representative-round expectations, including three changed fixtures with counted starts. They are not regenerated from the new outputs. The tests additionally cover strict majority/instant boundaries, advanced singletons, later-date mutations, partial downloads, status evidence, season resets, special formats, one activation build per calendar, duplicate indices, recipe migration and team/fixture/context/export consistency.

Earlier `review-*`, `published-value-parity.json`, `publication-audit.csv`, `tests.xml` and `verification.json` are **historical receipts**, not certification of this replacement. The old comparison script belongs to its original release. New receipts use the `majority-` prefix. No model fit or frozen experiment is needed.

Format references: [Scotland 33+5](https://spfl.co.uk/news/post-split-qa-season-202324), [Japan regional rounds](https://www.jleague.jp/news/article/32817/), [Mexico 2019 calendar](https://lmxwebsite.s3.amazonaws.com/docs/publicaciones/ligamx/CALENDARIO-LIGA-MX-TORNEO-APERTURA-2019.pdf).


## Changes measured against dc11b3a

The new policy changes **41 of 92,983 fixtures**, all toward less observed progress (a higher countdown). Eligibility and missingness remain unchanged: 92,855 finite league fixtures and 128 excluded Belgian playoff fixtures.

For example, Bristol City-Watford on 29 January 2013 was assigned round 24. Only five of the twelve round-29 fixtures had started before its kickoff, short of the required seven, so its effective round is 28 instead of the old cluster rule's 29. Cerezo Osaka-Nagoya on 6 November 2018 similarly uses round 31 instead of 32: round 32 had no earlier starts, one simultaneous start and eight later starts. These are expected conservative differences, not parity failures. See the complete `majority-value-changes.csv` and the independent round-check rationales.
