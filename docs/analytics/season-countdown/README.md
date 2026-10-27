# Season countdown

`SeasonCountdown()` is a native fixture feature, selectable as **Season countdown** in Features & ratings. JSON and Python recipe exports retain its parameters. Match layout emits one `fixture::season_countdown` column; team layout retains the same value on both team rows. `SeasonProgress` remains unchanged.

```python
from xdiyo_analytics.data import load_seasons
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.features import SeasonCountdown, evaluate_features

data = load_seasons("data/xDiyo_data", ["23_24", "24_25"],
                    leagues=["Premier_League"],
                    tables=["matches", "team_seasons"], include_awarded=True)
history = build_team_history(data)
features = evaluate_features(history, {"season_countdown": SeasonCountdown(schedule_mode="retrospective")},
                            schedule_history=data.matches,
                            team_seasons=data["team_seasons"], keyed=True)
```

Recipe fragment:

```json
{"features": {"season_countdown": {"component": "features.SeasonCountdown", "params": {"schedule_mode": "retrospective"}}}}
```

In retrospective mode, recipes automatically load the complete published matches and membership tables before modelling filters, including awarded fixtures needed to identify the calendar. This does not add awarded fixtures to a model that excludes them. Standalone callers must supply the complete roster, or an explicit `team_count`/`total_rounds`. Unknown leagues need an explicit `format` or `total_rounds`; incomplete observed schedules cannot establish a league's format or roster.

## Definition and postponed fixtures

`(total_rounds - effective_round) / (total_rounds - 1)` gives 1 at round 1 and 0 at the final round. Team counts come from complete league-season membership, not observed teams after filtering. Cancelled or partially downloaded seasons retain their planned length; no denominator comes from the maximum downloaded round. `total_rounds` overrides the denominator; `team_count` overrides membership counting for single/double round robin. Overrides apply to every selected season.

Each round's calendar anchor is the earliest kickoff in its largest cluster spanning at most 96 hours. Ties choose the earliest cluster. This prevents an isolated advanced match or late catch-up from setting the calendar for an otherwise normally scheduled round. Anchors are inferred from unique fixtures, so paired team rows have no extra weight. At a fixture's actual kickoff, its effective round is the greater of its assigned season round and the highest round whose anchor has started. The start is inclusive. Between rounds the most recently started round remains active; after the final round the value is zero. Advanced fixtures retain their assigned round.

Example from the published data: Chelsea-Tottenham, event 12226774, was assigned round 26 but kicked off during round 35. The countdown is `(38-35)/37 = 3/37`, not `12/37`.

This anchor rule is a documented **schedule inference**, not an official round reassignment. Unusual mass rescheduling or ambiguous sparse calendars can require an authoritative schedule supplied through `schedule_history` (`input.Table` in advanced settings). Missing kickoff/round values remain missing; invalid numeric rounds and conflicting fixture copies raise. Query identities, rounds, timestamps and shared stage metadata must agree with the supplied schedule.

The feature never reads goals, statistics, results, availability of results, or standings. It now requires an explicit schedule contract:

- `schedule_mode="as_of"` (default): provide `schedule_history` with `schedule_snapshot_at` UTC publication timestamps. Each timestamp identifies a **complete, coherent competition/season schedule snapshot**, including planned future fixtures. For each query, the latest snapshot published at or before its prediction cutoff supplies the anchors. Later revisions are ignored. Cutoffs default to target kickoff and must not be later. Team-row and context-feature recipe paths honor the same explicit cutoffs, including `cutoff_hours`.
- `schedule_mode="retrospective"`: explicitly accept the final revised schedule as context. This preserves the original clock calculation, but **does not establish historical leakage safety**. The builder automatically loads full exported matches before modelling filters only when this assumption is selected. The example above uses this mode deliberately.

The bundled exports do not contain historical schedule publication timestamps. They cannot be silently upgraded to as-of data. Existing countdown recipes without `schedule_mode` now fail closed unless snapshots are supplied; add `"schedule_mode": "retrospective"` to retain their former assumption. JSON/Python exports retain this choice.

The caller must provide truthful publication times and complete snapshots, not assign a capture time separately to each fixture. The feature validates snapshot rows and query consistency; it cannot prove a source's completeness or when a schedule was first known. Query kickoff/round/stage must agree with the selected snapshot; otherwise it rejects the input instead of inventing a historically known target schedule. Team membership is assumed known before the season, or must be supplied from the same issuance-known source. This mechanism does not reconstruct missing schedule archives.

For Japan 2016, phase identity comes from [the official source join](j1-phase-provenance.json), packaged in `features/data/j1_2016_phases.csv`. Phase 1 round 10 remains ordinal 10 even if postponed past July 2; its effective round can still advance according to the actual kickoff. Explicit phase values conflicting with the verified fixture identity are rejected.

## Audited formats

The 289 publications contain 92,983 fixtures across 19 league labels. They are **not all double round robin**. See [publication audit](publication-audit.csv); rerun `PYTHONPATH=src python docs/analytics/season-countdown/audit.py` from the repository root. The audit reads scheduling/membership data only and does not alter dataset files or run experiments.

| Competition | Planned rounds / clock |
| --- | --- |
| Premier League, Championship, Bundesliga/2, La Liga/2, Ligue 1/2, Serie A/B, Eredivisie, ordinary J1/J2, Belgian regular league | Double round robin: `2*(n-1)` for even n, `2*n` for odd n. |
| Scotland Premiership | 38 rounds: 33 before the split, followed by 5 league rounds. The split does not restart the clock. Some exports contain only the first 33 rounds. |
| Mexico Apertura / Clausura | Single round robin: `n-1` for even n, `n` for odd n. The separately published tournaments each have their own league-season clock. Usually 17 rounds; 19-team Apertura 2019 has 19 rounds including byes. Liguilla/play-in excluded. |
| J1 2015 / 2016 | One 34-round season clock across both 17-round stages. 2015 export numbering is already continuous; 2016 restarts at 1. Original second-stage identity receives offset 17, independent of actual kickoff. A packaged 306-fixture phase table was joined to the official J.League table by home team, away team and assigned round, never by date or score. Custom 2016 fixtures require explicit `season_phase` 1 or 2 when their full identity is not in that table. |
| J1 Transition 2026 | 18 regional rounds for each ten-team group. Independent group calendars; playoff round excluded. |

Known provider tournament IDs distinguish regular leagues from playoffs. Scotland allows only tournament IDs 54 (regular), 92066 and 92067 (split); other or missing stages are excluded and cannot shift anchors. Belgium's additional playoff rows produce **NaN**, rather than a restarted clock or false zero; the feature does not silently remove dataset rows. If using automatic imputation, explicitly exclude those playoff fixtures from the modelling population as well. The 128 missing values in the audit are exactly those Belgian playoff fixtures; all 92,855 regular/split league fixtures have finite values in [0,1].

Special source convention: Clausura 2010/11 uses tournament 28 in its publication, whereas later Clausura publications use 16753. Both are handled within their league-season identities.

Formats were checked against the published schedules, including odd-team Serie B/J2 seasons, COVID-truncated seasons and the 2026/27 completed-match snapshots. Future format changes require a reviewed rule update or explicit override.

## Primary format references

- [SPFL split explanation](https://spfl.co.uk/news/post-split-qa-season-202324): rounds 1-33 followed by 34-38.
- [Liga MX official Apertura 2019 calendar](https://lmxwebsite.s3.amazonaws.com/docs/publicaciones/ligamx/CALENDARIO-LIGA-MX-TORNEO-APERTURA-2019.pdf): 19 matchdays with byes.
- [Liga MX official Clausura 2026 calendar](https://cldrsrcs.apilmx.com/v1/docs/publicaciones/ligamx/2025-2026/2/calendario-liga-mx-torneo-clausura-2026.pdf): regular season ends at round 17, with later knockout fixtures separate.
- [J.League 2016 calendar](https://www.jleague.jp/news/article/4675/) and [competition history](https://aboutj.jleague.jp/corporate/about_competitions/tournament_history/): two 17-round stages; second stage starts July 2.
- [J.League 2026 regional schedule](https://www.jleague.jp/news/article/32817/): 18 regional matchdays followed by separate playoffs.
- [Belgian Pro League 2026/27 format](https://www.proleague.be/fr/informations/format-18-clubs-en-jupiler-pro-league-a-partir-de-la-saison-26-27): 18 clubs, 34 regular rounds, no playoffs.

## Verification

Synthetic regressions cover even/odd team counts, endpoints, season isolation, postponements, advanced singleton fixtures, missing timestamps/rounds, invalid denominators, conflicting identities, membership omissions, irrelevant outcome mutations, cutoff invariance, special formats, fixture scope, and JSON/Python recipe preparation. A real Chelsea-Tottenham postponed fixture is checked independently against effective round 35. Native browser verification checks that the new component and format/override controls render and select correctly. No model fits or frozen experiments are needed for the schedule audit.

Final focused regression run: **185 passed**, 4 warnings, 65.26 seconds. See [test receipt](tests.xml) and [environment/source hashes](verification.json). All 289 publication audits completed successfully.


## Review repairs for 7156b381

The original `publication-audit.csv`, `tests.xml` and `verification.json` above are historical evidence. The strengthened audit writes `review-publication-audit.csv`, preserving those receipts. `expected-publication-counts.csv` freezes the reviewed fixture/eligibility counts; the audit independently recomputes stage eligibility from source tournament IDs and requires exact per-fixture agreement with output missingness. All-NaN output, an eligible missing value, a finite excluded value, count drift, infinities and wrong hand-checked rounds now fail. Counts are not recomputed from the new feature output.

`effective-round-checks.json` records 11 literal expectations and explanations across a delayed EPL fixture, a delayed first-stage Japanese fixture, the Japanese stage transition, both Scottish split groups, odd-team Mexico and both Japanese transition groups. These expectations were inspected against the source calendars and raw fixture records; they are not generated by the countdown algorithm. Missingness expectations are reviewed when a publication changes, not silently regenerated.

Rebuild/verify the packaged Japanese phase crosswalk with `python docs/analytics/season-countdown/verify_j1_phases.py --html PATH_TO_OFFICIAL_HTML`. The source URL and capture hash are in the provenance receipt. The tool reads data and compares the packaged CSV; it does not mutate publications.

Final repair validation: **215 passed** (including 54 countdown tests), 4 warnings. See [fresh receipt](review-7156b381-verification.json) and [test XML](review-tests.xml). All **92,983** retrospective published values match the reviewed implementation exactly, including index, dtype and missing positions; [parity receipt](published-value-parity.json). The strengthened audit passes **289 publications**, **92,855 eligible fixtures**, **128 expected playoff exclusions**, and all **11** explicit effective-round checks. Native UI selection and isolated wheel import of the 306-fixture phase resource were verified. No dataset or frozen experiment was modified.
