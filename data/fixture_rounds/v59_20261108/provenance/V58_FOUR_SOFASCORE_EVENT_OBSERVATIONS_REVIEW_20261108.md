# V58 SofaScore event observations

The browser's verification screen cleared without a checkbox action. The following event pages were read directly on 8 November 2026; each displayed the target ordered pair, played date and round. The `v58_sofascore_displayed_round` field records the current provider label separately from the original-draw candidate and all historical alternatives.

| Target | SofaScore displayed round | Original-draw context |
|---|---:|---|
| [73239](https://www.sofascore.com/football/match/western-sydney-wanderers-central-coast-mariners/qucsBiE#id:9209891) | 4 | Same January 19 pair/date in the initial FFA draw; original club narrative says 6 and the league table says 7. |
| [73217](https://www.sofascore.com/football/match/western-sydney-wanderers-melbourne-victory/vucsBiE#id:9324717) | 7 | Target played February 10; initial ordered pair was January 30 with league-table label 10. Continuity is inferred from the unique pair. |
| [73205](https://www.sofascore.com/football/match/western-sydney-wanderers-adelaide-united/WibsBiE#id:9351183) | 10 | Target played February 27; original league table says 4 and the December pre-season notice says match week 3. Continuity is inferred from the unique pair. |
| [73186](https://www.sofascore.com/football/match/western-sydney-wanderers-perth-glory/VibsBiE#id:9382095) | 13 | Target played March 19; original ordered pair was January 24 with league-table label 8. Continuity is inferred from the unique pair. |

The latter three provider labels coincide with the later weeks in which the games were played. The event pages establish the provider's current displayed convention; they do **not** establish the original assigned round. In light of the instruction to prioritize original assignments, the four selected original-round fields remain null while the alternatives are carried for audit. Whether a downstream consumer should use the separately stored provider label as a fallback is a distinct source-precedence decision, not a claim that it is the originally assigned label. Prior conflict gates and V57 identity evidence remain.

All source URLs and observation timestamps are research metadata, not model inputs. `v53_selected_season_round` remains the authoritative selected season field; the V58 provider field must not be automatically coalesced into held original-round nulls. Rebuild using `python v58_sofascore_four_event_observations_REVIEW_20261108.py`. `v58_validation.json` verifies 61,488 fixture rows, 368,928 field records, six-field ledger agreement, four preserved nulls, four directly observed provider labels, and zero integration approvals. No production data, experiment preparation, commit or push was changed.
