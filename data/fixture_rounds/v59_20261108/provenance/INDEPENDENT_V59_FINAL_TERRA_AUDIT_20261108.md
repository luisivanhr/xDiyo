# Independent final Terra audit: V59 user-selected SofaScore exceptions

## Verdict

**Approve V59 for a status-aware Astra integration**, subject to the integration and verification conditions below. The user's latest explicit choice resolves the decision to populate the four Australian fields with the observed SofaScore displayed labels, even though three correspond to the later playing weeks. Do not relabel those values as proven original draw rounds or make a generic coalescing rule for other records. This is a **mixed original/provider-convention mapping**, not a uniformly original-draw dataset. Earlier conflict gates remain uncertainty and provenance, not a reason to disregard the user's choice. No fresh decision is required for these four.

This audit is for the mapping and its faithful integration, not approval to launch or prepare a particular experiment. Prediction-time schedule/countdown behavior and all remaining feature-time issues require their separate review; user's accepted kickoff-time outcome proxy remains a separate modeling assumption. If a target schema cannot carry the choice's semantics and the source/identity caveats in dataset metadata or an associated versioned audit ledger, that implementation is not an acceptable faithful integration.

## Four explicit choices and evidence

| Target | Selected provider fallback | Provider event ID | Qualification |
|---|---:|---:|---|
| 73239 | 4 | 9209891 | Exact Jan 19 pair/date in initial draw; original club says 6 and league table says 7. |
| 73217 | 7 | 9324717 | Feb 10 event; initial same-ordered pairing Jan 30 table label 10. Cross-date identity uses unique pairing. |
| 73205 | 10 | 9351183 | Feb 27 event; initial table says 4, later pre-season notice calls Match Week 3. Cross-date identity inferred. |
| 73186 | 13 | 9382095 | Mar 19 event; initial same-ordered pairing Jan 24 table label 8. Cross-date identity inferred. |

The V58 retained observation CSV and report record a direct browser observation on 8 November 2026 of each provider event, actual match date and displayed round; these equal V59's selections. The source links and event IDs are preserved in the ledger. This independent re-audit did **not** replay the provider pages, bypass a restriction, or obtain independent screenshots, so the provider values are based on that retained direct-observation record, not a second live verification. A separate [Superbru retrospective fixture list](https://www.superbru.com/tournaments/soccer/a-league/2020-21) groups these actual events as 4/7/10/13, corroborating the played-week reading, not independently establishing what SofaScore displayed or proving initial-draw rounds. User's explicit direction authorizes the exception notwithstanding that distinction.

All four have `USER_SELECTED_SOFASCORE_DISPLAYED_ROUND_FALLBACK` and `v59_original_draw_round_verified=False`; outside those four, the verification column is null/unassessed, not false. V59's historic `v53_selected_season_round` remains null for these rows. The correct authoritative field is `v59_selected_season_round`; old V53 is no longer the export selection. Prior original club, league-table and other alternatives, identity evidence and field-level gates are retained. Provider current labels and historical research metadata must not be treated as prediction-time-known without a separate analysis.

## Full population and typed lineage

Independent script `independent_v59_full_population_audit_20261108.py`, plus a separate companion audit, verified 61,488 unique target IDs and matching identity fields against V58, 368,928 unique `(match_id, field)` records, and exact equality of each of the six selected fields to both typed numeric/text and string ledger representations. Exactly the four authorized season-round selections change from V58; no other selected season value changes. All 368,928 base conflict-gate strings match V58 exactly; all integration, feature-as-of and model-input approvals remain false. The 794 historical flags agree with the queue. Three Bulgarian explicit clears and all 60 MLS series game ordinals survive.

Compact SHA-256 `3840a0b53928e89dd67eb1dfffe76826505b258c3e9650f8cc7a52c393ff8f27`; lineage SHA-256 `a89c4ceca4cdbcb5d322d582edaaeb36680705cb155c0186e6532495976aa8c6`; both match the handoff. There remain 13,168 null season-round values, most represented through phase rounds or named stages. Exactly 745 records have neither season nor phase numeric round but have legitimate named knockout/classification stages. Preserve these categories rather than inventing values.

## Scope and remaining source conventions

The V57 comparison is retained as a **historical snapshot**: 125 different, nine equal and four then-null among compared rows, plus 18 identity-ambiguous and five named finals. After the user's four choices, an independent recomputation against the same table claims gives **129 different and nine equal among 138 compared**; this is a cross-convention comparison, not 129 demonstrated errors. The retained V57 `SELECTED_NULL` values describe the older assessment and must not override V59's selected values or be misrepresented as a current computation. The 26 V56 identity gates remain conservatively attached even though V57 supplies eight proposed identity bridges; they are not the count of still-ambiguous pairs. Source-table rounds 1–40, club/weekly labels and SofaScore display values remain distinct and should not be converted with an invented formula. V59 does not newly source-verify every other fixture.

## Integration conditions

1. Whitelist `v59_selected_season_round`, `v44_selected_phase_round`, `v47_selected_phase_key`, `v44_selected_stage_label`, `v44_selected_knockout_label`, and `v44_selected_leg_number`; optionally preserve `v45_series_game_number` separately. Do not generic-coalesce historical columns or fabricate numbers for named stages. Treat empty strings as absent and honor three explicit Bulgarian clears.
2. Preserve the four fallback-selection semantics, provider event IDs/source observations, original-round-not-verified flag, alternatives and field gates in a versioned provenance layer, along with other existing identity and namespace holds. The compact is not a full substitute for the field ledger or source-claims table. Don't treat unassessed/null flags as evidence of correctness.
3. Assert the four actual selected values and semantics, all six selected/ledger value types, 61,488 unique IDs, 794 historical flags, three clears, 60 optional series ordinals and named stages after merge/serialization. Keep review-only source metadata, outcome-bearing text, future information and audit flags out of predictive model features.
4. Retain the distinction between dataset integration and the separate phase-key, schedule/issuance and leakage checks needed before an experiment. Source comparison fields are snapshots, not live provider truths.

Provider source-focused companion report: `INDEPENDENT_V59_FOUR_SOFASCORE_FALLBACK_AUDIT_20261108.md`; V56 source comparison and identity caveats: `INDEPENDENT_V56_TERRA_AUDIT_20261108.md`. No production data, experiment preparation, commit or push was changed by this audit.
