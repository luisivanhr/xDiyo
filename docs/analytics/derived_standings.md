# Score-derived standings and form tables

`DerivedStandings` is an opt-in, season-scoped team feature. Existing provider
`NormalizedStanding`, its missing-value convention, Elo/Glicko, feature presets,
and historical experiments are unchanged. No model is fitted during preparation.

```python
from xdiyo_analytics.features import DerivedStandings, evaluate_features

features = evaluate_features(
    full_history,
    {
        "position": DerivedStandings(None, include_raw=True),
        "form": DerivedStandings([3, 6]),
        "other_windows": DerivedStandings([1, 5, 12, 30]),
    },
    team_seasons=verified_membership,
    cutoffs=native_issuance_times,  # optional; see publication below
    available_at="release",       # optional native retrospective proxy otherwise
    keyed=True,
)
```

The default form window is `(5,)`; it is not automatically added to any feature
bank. Windows are ordered, distinct positive integers, stored as a tuple. Booleans
and fractional/duplicate/nonpositive windows are rejected. None/empty is accepted
only with `include_raw=True`. All home and away matches contribute together.

## Arithmetic and output

Each accepted result contributes 3/1/0 points and exact integer GF, GA, GD=GF-GA,
played/W/D/L. Ordering is points descending, GD descending, GF descending, GA
ascending. The last key is redundant given consistent GF/GD; no fractional ranking
key is used. Exact ties share competition ranks: 1,1,3. IDs only stabilize display.

For every window w, independently take **each team's own last w eligible matches**
in the current competition/season/table, sum its contributions and re-rank the
complete roster. Short histories expand naturally. Missing/unfinished results do
not consume slots. A fixture can remain in one team's window but leave the other's.
This is a form table, not a moving average of prior season-table positions.

* `team::form_rank_w`: `(N-R_w)/(N-1)`, always higher=better, range [0,1].
* `team::season_position`: cumulative R, when `include_raw=True, normalize=False`.
* `team::season_position_over_n`: cumulative R/N, when both switches are True;
  lower=better, range [1/N,1].

The supplied alias prefixes these names even for a single output. `side="against"`
uses opponent columns, and `"both"` emits both perspectives. Default `"for"` gives
one output per window per team; match assembly adds home/away prefixes, so [3,6]
gives four fixture predictors, or six with cumulative position. Toggling raw output
or normalization never changes form values or the underlying state's identity.
No diagnostic count or consistency metric is silently added as a predictor.

## Publication and availability

**Until every scheduled round-1 fixture has a valid available result, all published
ranks remain initial. A postponed opener can hold an entire league at initialization
through later rounds.** Initial R is 1, R/N is 1/N and form rank is 1 for every team.
Those values reflect a shared cold-start convention, not inferred strength.
Buffered eligible counts and pending first-round event IDs remain in diagnostics.

With no explicit `cutoffs`, each original assigned round is frozen at the earliest
kickoff of that round in the complete source schedule. A postponed fixture uses
that original round's issuance time. With explicit `cutoffs`, the supplied native
issuance times are authoritative; callers must supply their real forecast group
cutoffs, not future outcome times. Every scope/cutoff is computed once and shared
by all query fixtures. After the opening gate, no later round-completion barrier
is imposed. Results must have kickoff strictly before the cutoff and availability
no later than it. Target identity/kickoff is validated against the complete schedule.
Equal-time releases are applied as a complete batch before ranking all roster teams,
including idle teams. Late releases insert into kickoff/event chronology and affect
new snapshots only. Every season/table starts fresh, without movement priors.

`availability_times` preserves reviewed `result_available_at` lower bounds. Without
an explicit release policy, the normal earlier-finished-kickoff proxy applies; it is
**not proof of actual result publication time**. The bounded validation uses kickoff
+3 hours, raised by reviewed bounds. Never replace native prediction cutoffs with
provider pregame observation times to improve agreement.

## Complete population and supported formats

A verified `team_seasons` roster is required: N is never estimated from a training
fold. The source must contain the complete independent single- or double-round-robin
schedule (including unfinished fixtures), with reciprocal home/away pairings for
two cycles and an unambiguous round-1 fixture set. N<2, missing fixtures, conflicting
paired scores/releases and malformed identities fail explicitly. Exact integer IDs
are preserved; potentially rounded floating IDs at or above 2**53 are rejected.

Simple competition/season tables work automatically. Multiple tournament stages or
nontrivial stage metadata are rejected rather than combined or reset blindly. For
verified independent groups/phases, supply `standings_table_id` in source and roster
and `standings_rule="independent_3_1_0"` in source. Each scope must independently pass
the schedule checks and use round 1 as its opening round. Carry-over/halved points,
playoffs, partial schedules and irregular formats need an explicit future policy;
this implementation does not invent one. Administrative deductions and official
head-to-head rules are not inferred from provider positions.

Ordinary league results use native current scores only after checking retained
extra-time/shootout evidence. Nonzero extra-time/shootout fields, awarded results or
reviewed retrospective corrections require explicit approved
`standings_goals_for/standings_goals_against` and a known
`standings_result_available_at`. At match-source level these are
`home_standings_goals/away_standings_goals`. They must agree between paired rows.
The approved score becomes eligible no earlier than its known ruling time; absent
original-score versions are not reconstructed retrospectively. Unknown/missing,
abandoned and unfinished outcomes do not accrue points. Legitimate 0-0 does.

Recipes automatically load a separate complete matches/membership source with
awarded rows retained even if the model population excludes them. This can expose
an unresolved awarded-score policy as an error instead of silently dropping the
fixture. Stats/odds/model-row filtering does not define the standings population.
For direct subset queries, pass `standings_history=full_history` and, when needed,
`standings_available_at=full_population_release_times` to `evaluate_features`.

### Reviewed Bundesliga 2021/22 omission

The published season now includes Bochum–Mönchengladbach, round 27, scheduled for
18 March 2022 at 19:30 UTC. The match was abandoned at 0–2 and subsequently awarded
0–2 by the DFB on 24 March. The source keeps `play_status="abandoned"`,
`is_awarded=True`, separate played/awarded scores and explicit standings scores.
Its ID `1000000011784` is in the existing supplemental workbook namespace; it is
not a claimed native provider ID. Names and movement flags come from the season's
existing team records. No missing statistics, lineups, heatmaps or pregame ranks
are invented.

Both release bounds are **2022-03-24 23:00 UTC**, the conservative end of the ruling
day in Europe/Berlin. The exact announcement time is unknown. Before this bound,
the standings continue using the other eligible results. At/after the bound,
Bochum adds one loss and two goals against; Mönchengladbach adds one win, three
points and two goals for. Each adds one played match. The original March 18
chronology determines rolling-window membership, not the March 24 release date.
Earlier snapshots are never revised.

The normal model population still excludes this awarded fixture. The recipe's
separate standings source includes it, so the season builds without changing that
exclusion. Both published Parquet surfaces use ZSTD and updated content hashes;
older pinned recipes should use the existing recipe data-reference refresh helper.
See `data/bundesliga_20220318_review.json`, the idempotent
`examples/repair_bundesliga_standings_20261027.py` and the validation receipts.

## Storage, diagnostics and efficiency

`ratings.build_standings` returns `StandingsRun`; `save()` stores owned rank/totals
snapshots in Zstandard Parquet plus JSON rules, source/query hashes, scope, windows,
release policy and publication provenance. `StandingsRun.load()` supports exact
saved-cutoff lookup only; it is **not an incremental/resumable checkpoint**. Public
properties return copies. A new source, membership, release policy, cutoff set or
window configuration requires a new run; there is no process-global mutable cache.
JSON persistence converts NumPy scalars to native Python values, retaining exact
integer IDs as JSON numbers, including IDs beyond 2^53. This applies to both
artifact versions; nonfinite values and unsupported objects still raise errors.

Within one feature evaluation, all DerivedStandings descendants share one producer
with the union of requested windows and venues. Output-only selections are separate from state.
Canonical paired events are validated once. An availability-ordered replay maintains
integer totals and kickoff-ordered per-team histories; late entries update only
windows containing the entry. Each requested cutoff re-ranks complete tables once.
Only requested cutoff snapshots are materialized. Global contributing event IDs,
pending gate events, effective per-team counts, skipped outcomes, source identity
and late-insertion counts appear in `result.attrs['derived_standings']`. H2H and
rating warm starts cannot narrow or seed this league-wide producer.

The builder exposes the feature and arbitrary window list, raw/normalized switches,
side selection, auxiliary source tables and saved `input.StandingsRun`. JSON and
Python exports use native constructors. `examples/derived_standings.py` demonstrates
both direct evaluation and recipe preparation, without training.

### Home-only and away-only tables

`DerivedStandings([3], venue="home")` ranks the complete league roster using each
team's last three available home results. `venue="away"` does the same for away
results; `venue="same"` selects each requested team's current fixture venue.
`side="against"` selects the opponent's rank, independently of venue: fixed home
means that opponent's home form even if it currently plays away. With same venue,
the opponent's actual venue is opposite the focal team's. The first-round gate,
strict cutoffs, season/table boundaries, approved-score rules and ranking order
remain unchanged. Cumulative position also uses the selected venue.

New names include the venue, e.g. `home_form::team::home::form_rank_3` before
match assembly. Default `venue="all"` keeps existing column names and values.
Extended saved runs use artifact version 2 and record their venue tables in
metadata, snapshots, publication provenance and state identity. All-only saves
retain version 1; both versions load. Requesting an absent saved venue raises an
error. Provider `NormalizedStanding` remains a provider-table feature; it cannot
recover venue tables by filtering provider positions.

See [venue histories](venue_history.md) and `examples/venue_history.py` for the
paper-inspired goals and rank configuration.

## Reference and validation

Berrar, Lopes and Dubitzky (2024), Section 5.5, equation (5), Table 6,
[DOI](https://doi.org/10.1007/s10994-024-06625-9), supplies the normalized-rank formula
and separate form-table example. The paper's wording also discusses averaging ranks;
this implementation follows the user's explicit separate-form-table interpretation,
arbitrary windows (including odd sizes), season resets and short-history rule. It
does not claim to reproduce the paper's full cross-season framework.

The supplied workbook was read statically: 380 scored fixtures / 20 teams give
exact agreement with its cached points/GF/GA/GD arithmetic. No macros ran. Its key
`points + GD/100 + MATCH(name)/10000` is deliberately not used: GD could override
points and the name term arbitrarily separates ties. Its blank-score and fixed-range
behavior is also not copied. See `derived-standings-validation/workbook.json`.

The independent brute-force oracle, provider comparison, repeat-query timing and
memory receipts are in `derived-standings-validation/`. Provider agreement is a
cross-check, not an oracle; official rules, ties, missing results and mismatched
information cutoffs can produce legitimate differences. Protected 2026/27 outcomes
and frozen experiments are not read or modified by this validation.
