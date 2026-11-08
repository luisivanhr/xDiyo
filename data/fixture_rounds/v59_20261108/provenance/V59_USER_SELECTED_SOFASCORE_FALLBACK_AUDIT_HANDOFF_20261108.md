# V59 mapping: user-selected SofaScore fallbacks

**Ready for independent audit; not approved for production.** Iván explicitly chose SofaScore for the four Australian fixtures after being told that three provider labels refer to the later weeks when postponed games were played. V59 applies that choice as a source-priority exception; it does not claim those values are proven original-draw assignments.

| Fixture | V59 selected value | Direct provider event | Interpretation |
|---|---:|---|---|
| 73239 | 4 | [Central Coast–Western Sydney](https://www.sofascore.com/football/match/western-sydney-wanderers-central-coast-mariners/qucsBiE#id:9209891) | Same January 19 pair/date as initial draw; original club and league table respectively say 6 and 7. |
| 73217 | 7 | [Western Sydney–Melbourne Victory](https://www.sofascore.com/football/match/western-sydney-wanderers-melbourne-victory/vucsBiE#id:9324717) | Event played February 10; original ordered pair was January 30 and original league table says 10. |
| 73205 | 10 | [Western Sydney–Adelaide](https://www.sofascore.com/football/match/western-sydney-wanderers-adelaide-united/WibsBiE#id:9351183) | Event played February 27; original league table says 4 and later pre-season notice says match week 3. |
| 73186 | 13 | [Western Sydney–Perth](https://www.sofascore.com/football/match/western-sydney-wanderers-perth-glory/VibsBiE#id:9382095) | Event played March 19; original ordered pair was January 24 and original league table says 8. |

Original schedule alternatives, direct provider URLs and observations, cross-date identity limitations, 1–40 league-table evidence and all conflict gates remain in the field lineage. `v59_original_draw_round_verified` is false for these four; it is null outside their assessed scope, not a global negative. The provider label is a user-selected fallback, not an independently proven original round. The three postponed fixtures' original event continuity is inferred from the unique ordered pair, as documented in V58.

## Authoritative export contract

Use **`v59_selected_season_round`** for V59; do not use `v53_selected_season_round`, whose four historical nulls reflect the superseded pre-decision state. The other selected fields are `v44_selected_phase_round`, `v47_selected_phase_key`, `v44_selected_stage_label`, `v44_selected_knockout_label`, and `v44_selected_leg_number`; the matching ledger value is `v59_selected_value`. `v59_selection_semantics` and `v59_decision_semantics` mark the four explicit fallbacks. Empty text means absent. Preserve any remaining actual nulls and explicit Bulgarian clears; never generic-coalesce historical candidates. Prior version comparison fields such as `v57_round_comparison` are snapshots of that earlier assessment and were not recalculated to imply new source agreement.

The independent V56 audit's 125 league-table comparison differences, 18 unresolved repeated-pair identities after V57's proposed bridges, five named finals and other original source conflicts remain distinct from the four selected fallbacks. All previous field-level gates are preserved, including conservative V56 identity holds pending independent review. The 125 differences are source comparisons, not proven errors. A status-aware consumer must preserve relevant uncertainty, alternatives and namespace; source metadata and match outcomes must never be model inputs.

Rebuild with `python v59_user_selected_sofascore_fallback_REVIEW_20261108.py`. `v59_validation.json` records checksums. The builder checks 61,488 unique targets, 368,928 field rows, six-field selected-value agreement, four observed provider selections, three Bulgarian clears, 60 MLS ordinals and zero integration/model-input approvals. A separate direct comparison confirmed all 368,928 prior field gates are unchanged from V58. Independent Terra audit remains required; no production data, experiment preparation, commit or push was changed.
