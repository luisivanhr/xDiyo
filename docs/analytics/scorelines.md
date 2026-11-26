# Materialized scorelines

`scorelines` is a private, immutable goals-only snapshot. It does not modify or
replace the native SofaScore data, the historical scores inputs, or the existing
catalogue. No adapter or fixture join runs during an experiment.

## Build

From an installed xDiyo checkout:

```sh
python -m xdiyo_analytics.data.scorelines \
  --sofascore-root data/xDiyo_data \
  --isdb /private/ISDBv2.0.txt \
  --archive /private/verified/preferred_standings_checked.tsv \
  --output data/scorelines/2026-11-26 \
  --native-revision <pinned-data-commit>
```

Use the full extended native root at the intended data commit, an unmodified ISDB
v2 TXT, and the reconciled archive TSV, not a candidate-only export. The command
loads and verifies only the matches table of each native publication.
It records exact native publication, manifest, and matches hashes, the
team-directory digest, the two archive hashes and its own code hash. It refuses
to overwrite an existing output or write inside SofaScore. Keep the inputs
and generated files private, outside Git; do not publish licensed provider data.

## Read

```python
from pathlib import Path
import pandas as pd

snapshot = Path("data/scorelines/2026-11-26")
scores = pd.read_parquet(snapshot / "scorelines.parquet")
replay = pd.read_parquet(snapshot / "scorelines_replay.parquet")
```

Pin the snapshot (and its manifest digest) in experiment configuration, not a
mutable `latest` path. The files have the same columns. The broad score file includes unresolved team identities. Use the replay
file for prior-day goals history.

- `league` is the stable xDiyo/annual league key. `season` is always two four-digit
  years, including `2025/2025` for calendar leagues. Provider display names and
  IDs remain separate. Multiple native phases can share an annual league-season
  with different competition or season IDs. A historical ID with no unique
  native match remains null, not synthetic. Annual phase grouping is not a
  claim that two league tables are interchangeable.
- SofaScore wins confirmed fixture overlaps. On reviewed team IDs, an exact
  ordered pair/day with matching score can link to the native observation.
  Conflicts and unlinked rows in already represented league-seasons go to
  `review.parquet` rather than silently filling gaps or duplicating fixtures.
  A native cancelled or awarded fixture does not open a gap for external
  replacement. The same conservative rule applies if two external sources
  occupy an otherwise non-native season, with ISDB before archive.
  `provenance.parquet` records surviving overlap links.
- Finished native rows with a known non-award status and no observed extra-time
  or shootout flags are eligible for the broad table. Historical scored rows
  keep their source score basis: ISDB is reported but not regulation-certified,
  and archival coverage checks do not certify every kickoff or row. Source
  fixture identities and raw source date/clock are kept for traceability.
- Team identity is determined at build time, within source, annual league and
  season. The crosswalk uses dated reviewed aliases, exact names observed in
  the same native league within five seasons, unambiguous native slugs across
  divisions in the same country within ten seasons, and repeated ordered
  fixture matches against known opponents. For fixture evidence the score and ordered
  opponent must agree, the date must be within one day, and a nomination must
  have at least two distinct fixtures and two distinct opponents. No fuzzy
  matching or dropped reserve suffixes. Conflicting nominations stay out.
  `team_crosswalk.json` records each decision and supporting fixture IDs;
  a data-backed match is not a manual review. Unknown teams keep source-scoped
  keys and null SofaScore IDs; they cannot enter replay. Unmapped ISDB leagues,
  predecessor competitions and known calendar transitions stay in the review
  queue. There is no Footiqo data or odds join.

## Prior-day replay

For an ordinary completed fixture whose source reports result day D, its
score is usable starting on analysis day D+1. Use a daily cutoff consistent
with the match-day labels; `replay_available_at` encodes the inclusive UTC
analysis-day boundary, not an observed result-publication timestamp. A recorded
publication timestamp does not gate history. Filter on
`replay_available_at <= prediction_day at 00:00 UTC`. Do not infer exact
kickoff from a date-only row, or mix this daily policy with intraday predictions.
An external source-reported date is still a source assertion, not independent
proof about postponements or local timezones.

Known interrupted or resumed native fixtures without a reviewed completion
day are held out. For exceptions, optionally pass
`--reviewed-results /private/results.json`: a JSON array whose entries contain
`fixture_key`, `completion_date` (ISO date), `timezone` (IANA name),
`regulation_verified` set to true, and `evidence` (citation to the review).
The review uses the end of the documented local completion day, rounded up to a
UTC-day boundary. This can delay admission by an extra day. Archived rows in
which the source bundle already replaced the first kickoff day with the known
completion day keep that completion day. Missing or ambiguous team IDs still
keep records out of the joined replay; score-basis labels are preserved.

## Optional Decisions API review

The build writes `team_slug_index.json` from captured SofaScore team names,
slugs and IDs. The mapping handoff includes `unresolved_teams_ranked.csv`.
Prepare a small offline shortlist first:

```sh
python -m xdiyo_analytics.data.scoreline_decisions   --queue /private/unresolved_teams_ranked.csv   --slug-index /private/team_slug_index.json   --limit 25 --output /private/team_shortlist.json
```

To make one paid request, set `OPENAI_API_KEY` privately and add `--execute`
with a new output filename. This uses OpenAI's [Decisions API](https://developers.openai.com/api/docs/guides/decisions)
(`POST /v1/decisions`, `gpt-6-luna`), not Responses. It submits club names,
country, league, season and a bounded shortlist of SofaScore identifiers and
labels, not fixtures, goals, keys from the source datasets, or raw rows.
Each decision can abstain. A returned ID is only a proposal for review;
no output from this helper is automatically imported into the crosswalk.

## Applying independently reviewed archive team identities

Pass `--reviewed-source-teams /private/reviewed_source_teams.json` when building
a new snapshot. The file is an array of scoped nominations. Each nomination
contains `source`, `league`, the canonical two-year `season`,
`source_team_key`, the verbatim `source_name`, `sofascore_team_id`, and
`evidence_urls` (public HTTPS references on at least two distinct hosts). Use
dated league or fixture evidence as well as the team identity evidence. URLs
are provenance only; building is offline and does not fetch them.

A nomination applies only to that source, league, season and team key. A name
drift, duplicate scope, missing input key or target not observed in the pinned
native league fails the build. Conflicting native, alias or fixture evidence
still holds the identity rather than overriding it. A confirmed decision
includes `reviewed_source_team` in the crosswalk; the manifest records the
SHA-256 of the supplied JSON. Keep the reviewed file with private build inputs.
A Decisions API suggestion is not sufficient evidence to create a nomination.
