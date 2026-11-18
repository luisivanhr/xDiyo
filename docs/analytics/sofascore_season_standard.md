# SofaScore year-pair standard for new experiments

Policy: `sofascore_year_pair_v1`.

New experiments use the existing two-year-field convention already recorded in
each verified SofaScore manifest's `scope.season_start` and `scope.season_end`.
The loader exposes `experiment_season` as `YYYY/YYYY`, plus both integer fields.
It does not alter observations, provider IDs, raw names or storage aliases.

Examples:
- J1 2024: `2024/2024`, not `2024/2025`.
- MLS 2024: `2024/2024`.
- Premier League 24/25: `2024/2025`.
- Brazil Serie A's delayed 20/21 season: `2020/2021`.
- J1 transition 2026: `2026/2026`, separate from regular `2026/2027`.

A calendar-year league repeats the same year. A cross-year league retains its
actual pair. Neither raw provider `season_label` nor `annual_season` determines
this field. The existing collection helper `season_year_pair` and manifest
validation already enforce this equal-or-consecutive year convention.

## Defaults

`TemporalSplit(unit="seasons", ...)` pools competitions by
`experiment_season`. Provider identity remains `(competition_id, season_id)`;
the pair is an explicit cross-league grouping, not an ID. One provider ID with conflicting pairs fails closed; separate phase IDs may
share a pair without losing their identity. Use native-ID `league_seasons` for separate seasons.
Pooled `league_seasons` also includes competition IDs to avoid collisions.
Ordering remains earliest observed kickoff, never lexical labels or provider ID.
Postponed-match and label-availability safeguards are unchanged.

Every temporal fold records its season policy and actual grouping columns.
Match reports, fold descriptions, transition-year inference and cross-league
round ticket grouping prefer `experiment_season`. Team histories and label
metadata retain it. Existing reviewed catalogue full-year pairs are promoted
without inferring from display names; non-pair catalogue labels are not certified.

## Reproducibility and compatibility

This changes defaults for newly prepared experiments, not archived outputs.
Source keys remain unchanged for joins and exact lineage. The loader adds
metadata in memory; no source manifests or published data files are rewritten.

For an exact historical rerun, pin the original code commit and source manifests.
To request the previous pooled split grouping on this version:
`TemporalSplit(..., unit="seasons", block_by=("source_season",))`.
Custom grouping records `explicit_blocks_v1` in fold metadata.
An explicit report `season_column="source_season"` retains legacy display.
Historical reports/ledgers without the new field keep their old fallback
behavior. New season splits without it raise with migration guidance instead
of guessing a year pair. Reload through the normal loader to obtain it.


## Verification

- 262 focused tests passed (splits, native loading, catalogue, UI, transitions,
  reporting and ticket composition).
- 118 additional UI workflow, preparation parity, loader and multiseason tests
  passed in the preceding validation run.
- All 617 current publication scopes passed metadata-only pair validation:
  148 repeated-year and 469 cross-year scopes; no provider identity has
  contradictory pairs.
- One pre-existing GAT timestamp-resolution assertion was excluded from the
  focused run after reproducing it on unchanged main (nanoseconds vs seconds).
  No GAT behavior was changed.
