# Independent V59 four Australian provider fallback audit

The V58 retained observation CSV and V59 parquet independently agree for the four target keys:

| Fixture | Target home–away, date | Recorded SofaScore ID | V59 number | Original source comparison |
|---|---|---:|---:|---|
| 73239 | Central Coast–WSW, 19 January | 9209891 | 4 | Original club says 6, league table 7 on same date. |
| 73217 | WSW–Melbourne Victory, 10 February | 9324717 | 7 | Initial league table gives 10 on 30 January; cross-date unique-pair continuity inferred. |
| 73205 | WSW–Adelaide, 27 February | 9351183 | 10 | Initial league table gives 4 on 9 January; 22 December club-hosted notice says matchweek 3; cross-date unique-pair continuity inferred. |
| 73186 | WSW–Perth, 19 March | 9382095 | 13 | Initial league table gives 8 on 24 January; cross-date unique-pair continuity inferred. |

The V58 file `v58_four_sofascore_event_observations_REVIEW.csv` and companion report state these are direct browser observations on 8 November 2026 and provide the four event URLs. This audit checks those retained records and target joins; it does not independently replay SofaScore or certify the captured browser screen. No access restriction or challenge was bypassed. The independent retrospective [Superbru season listing](https://www.superbru.com/tournaments/soccer/a-league/2020-21) groups those same played fixtures into 4, 7, 10, 13 respectively. That corroborates a played-week convention but is not an original assignment or independent SofaScore observation.

All four V59 rows mark `v59_original_draw_round_verified=False` and `USER_SELECTED_SOFASCORE_DISPLAYED_ROUND_FALLBACK`. A separate comparison found zero changes in selected season values outside the four among the other 61,484 rows. All 368,928 ledger rows retained the same four pre-existing hold/gate columns when compared with V58. The user's decision, as relayed in the V59 handoff, therefore has been correctly applied as a source-priority fallback, not as a historical original-round proof. Acceptance remains conditional on a consumer able to preserve mixed semantics, source alternatives and field gates. Do not present this as a uniformly original-round export or use provenance and match outcomes as model features. No production, mapping, or experiment files changed.
