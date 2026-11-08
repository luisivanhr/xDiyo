# Original-13 workbook/native round matching

The immutable `data/fixture_rounds/original13_native_20261108` release annotates
all 52,022 original-13 fixtures in the retained international workbook. It is
separate from the unchanged V59 release for the additional 34 leagues. The new
combined loader returns 113,510 disjoint workbook identities. It changes only
the seven round/stage annotation fields; it does not rewrite source match/team
IDs, scores, civil dates, existing native publications or experiment inputs.

There are **51,223 exact native matches** and **799 explicit holds**. In the
2019-07-01 through 2026-07-01 exclusive experiment window, 32,319 of 32,686
original-13 fixtures match and 367 remain held. An unresolved fixture remains in
the dataset with typed null annotations. A hold never falls back to an old value.

## Identity and evidence

The join requires the exact configured workbook league ID, country and league
label; the same full season; ordered canonical team identities resolved through
the reviewed country/name/effective-date directory; and the same calendar date.
Native UTC kickoffs are compared in Europe/Berlin civil time. This convention
does not assert that the workbook supplies a verified timezone. Seventy-three
accepted records have different civil clock times on the same date; both source
times and the discrepancy remain in lineage. Workbook times are never rewritten.

Scores are neither matching keys nor a tie-breaker. The native snapshot contains
only fixture identity and annotation inputs, not outcomes or statistics.
Native event IDs are lineage references for the selected annotation, not a new
general SofaScore ID cross-reference or a replacement for workbook IDs.

Every imported native matches table was checked against its publication's
SHA-256, row count, competition and season. Publications are reread to detect
changes during extraction. The release retains the exact normalized annotation
input, reviewed identity directory, workbook target identities, source
publication fingerprints, selected values, decisions and alternative candidates.
`load_native_fixture_rounds` replays the entire join and compares the selections,
lineage, candidate ledger and counts, after checking every payload hash.

## Unmatched dates and additional stages

The 799 holds comprise 764 records with one same-season ordered-pair candidate on
another date, 23 with multiple such candidates, and 12 with no same-season ordered
pair. A unique pairing on another date is not sufficient: a later playoff can
repeat an earlier league fixture. Reversed opponents are recorded only as review
candidates. No score, nearest-date, unique-pair, chronology or round-number
fallback is implemented. Multiple exact native candidates and multiple workbook
rows claiming one native event are also held rather than silently deduplicated.

The unresolved inventory includes six candidates within three days: Groningen-
Ajax (workbook 2023-05-16/native 2023-05-14), Groningen-Nijmegen (2023-04-25/
2023-04-22), Seraing-Union SG (2022-01-18/2022-01-15), Beerschot-Cercle Brugge
(2021-07-27/2021-07-24), Venezia-Empoli (2020-11-01/2020-10-31), and Gimnastic-
Alcorcon (2018-10-15/2018-10-14). These are retained source disagreements, not
newly adjudicated actual dates. All remain held. Longer differences and repeated
pairings are retained in `provenance/candidates.parquet`; `unresolved.csv` and
`coverage.csv` provide reviewable identities and league-season counts.

## Round semantics

The selected value is the retained normalized native `round`, including existing
corrections. Raw `stage_json` is evidence, never a fallback: the nine Serie B
2015/16 native corrections retain normalized round 4 while raw stage JSON says
0. This import does not newly verify that every inherited number is the original
draw assignment. `original_draw_round_verified` is unassessed/null throughout.

Main-stage labels use `season_round`. Retained Scottish split tournaments 92066
and 92067 keep their existing 34..38 season ordinals. Belgian substages 12656,
15545 and 92614 use `phase_round`, without promoting reset phase numbers into
season-wide rounds. Numeric substage identifiers distinguish their scopes in
`phase_key`; no unobserved official stage name, leg or series number is invented.
Unknown future native stage scopes fail closed. The current release has 51,095
selected season rounds and 128 selected phase rounds.

V59's full provenance, named stages, authoritative nulls, source/conflict gates
and four explicit Australian provider fallbacks remain byte-preserved.
Review flags, native event IDs, source hashes and candidate metadata stay outside
the annotation whitelist and outside model feature columns. Dataset annotation
does not approve a forecast cutoff, schedule-as-of assumption or model feature.

## Reproduce and materialize

From the repository with `PYTHONPATH=src`, use new output directories:

```powershell
python examples/publish_original13_fixture_rounds.py `
  --matches '<retained Matches_Results.parquet>' `
  --native-root '<unchanged data/xDiyo_data>' `
  --output '<new original13 release directory>'

python examples/integrate_fixture_rounds.py `
  --matches '<retained Matches_Results.parquet>' `
  --mapping-root data/fixture_rounds/v59_20261108 `
  --original13-root data/fixture_rounds/original13_native_20261108 `
  --output '<new private annotated dataset directory>'
```

Publication refuses existing output directories and stages all artifacts before
atomic rename. Materialization verifies every original non-annotation column,
all rows/order, typed Parquet round-trips and persisted provenance. The complete
146,000-row private output is not committed. The 32,490 rows outside both mapping
populations retain their existing values. Old loader calls remain valid.

Validation receipts are in `original13-round-validation`. Focused tests cover
exact joins, seasons, ordered teams, dated aliases, ambiguity, duplicate physical
fixtures, clock discrepancies, native corrections, phase scope, authoritative
nulls, immutable input behavior, release replay, provenance and V59 coexistence.
No forecast policy, feature preparation or model training is part of this change.
