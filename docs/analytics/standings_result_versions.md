# Publication-versioned standings results

`build_standings(..., result_versions=journal)` and
`build_phase_context(..., result_versions=journal)` accept explicit initial
results and later administrative revisions. This is a separate, versioned
policy. Existing saved tables and calls without a journal retain their named
legacy behavior, including the explicitly retrospective kickoff-day policy.

The journal is a DataFrame with these required columns:

- `competition_id`, `season_id`, `event_id`, `home_id`, `away_id`, `kickoff_at`:
  exact source-fixture identity and orientation.
- `revision`: consecutive integers starting at zero, including the initial
  sporting result or an explicit initially ineligible state.
- `available_at`: a nonmissing timezone-aware publication/availability bound.
  Revisions of one fixture have strictly increasing bounds. Ambiguous
  simultaneous revisions, naive times and unknown releases are rejected.
- `goals_home`, `goals_away`, `points_home`, `points_away`, `eligible`: the
  complete version. Goals and reviewed 3/1/0 points remain separate.
  An ineligible initial state can represent an abandoned or unplayed fixture;
  it does not enter a form window merely because a later ruling exists.
- `source`: a nonempty evidence reference for each version. Optional
  `time_basis` documents conservative date bounds and other timing limitations.

For each query, only versions with `available_at <= cutoff` and an original
kickoff strictly before the cutoff are eligible. The latest known version
replaces that fixture's contribution at its original kickoff/event position.
It does not consume a second form slot or become the most recent match. A
revision can therefore change cumulative totals without changing a last-one
window whose fixture has already been displaced by a newer match. Previously
emitted snapshots and each historical graph edge's original-cutoff ranks are
not retroactively rewritten.

Supplying a journal enables strict administrative coverage for that source:
every awarded or explicitly reviewed fixture must be covered. Missing
administrative provenance raises instead of falling back to the final score.
Ordinary sporting-result availability still follows the declared `timing`
convention. In particular, daily ordinary results retain the existing
previous-UTC-calendar-day convention; this API does not certify result or
statistic publication timestamps that the source never recorded.

Standings save/load versions 7 and 8 identify timestamp and daily versioned
policies respectively. New phase lookup policies distinguish those outputs
from legacy tables. A separately named mixed phase policy can assemble an
explicitly documented corrected scope with unchanged legacy scopes; its
provenance must identify those scopes and source identities. It must not be
presented as a globally verified as-of reconstruction.

The motivating case is Union Berlin–Bochum, 14 December 2024. The
[DFB announcement of 9 January 2025](https://www.dfb.de/news/dfb-sportgericht-wertet-bundesligaspiel-in-berlin-fuer-bochum)
confirms a played 1–1 and a later 0–2 administrative award. With date-only
publication evidence, a conservative daily preparation makes the award usable
from 10 January 2025 00:00 UTC, while preserving the earlier sporting result.
The library does not guess that bound or contain a match-specific Union
exception: preparation supplies the reviewed journal. This completed sporting
match must not be confused with Bastia–Red Star's abandoned 0–0 and separate
later standings-points ruling.

Regression coverage includes before/at release, venue tables, points distinct
from goals, intervening newer fixtures, form-window eviction, future-revision
poisoning, strict provenance failures, persistence, both graph consumers and
unchanged no-journal retrospective behavior. This is an input-validity repair,
not evidence of improved predictive performance.
