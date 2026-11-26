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
mutable `latest` path. The files have the same columns. The broad score file is
for coverage and exploratory work, not trusted point-in-time training.

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
- Only supplied dated, country-scoped reviewed aliases map external team IDs to
  SofaScore. Case/Unicode normalization is not fuzzy matching. Unknown teams
  keep source-scoped team keys and null native IDs; they cannot enter strict
  replay. Unmapped ISDB leagues, predecessor competitions and known calendar
  transitions stay in the review queue. There is no Footiqo data or odds join.

## Point-in-time gate

A date-only source never populates `kickoff_utc`, and an unverified external
clock is retained only in `source_kickoff_utc`. An explicitly recorded native
`result_available_at` takes effect no earlier than the next UTC analysis day.
No completion time is inferred from first kickoff.

For historical matches with independently reviewed completion dates and
regulation scores, optionally pass `--reviewed-results /private/results.json`.
It is a JSON array, each entry containing `fixture_key`, `completion_date`
(ISO date), `timezone` (IANA name), `regulation_verified` set to true, and
`evidence` (citation to the review). The gate rounds the end of the documented
local completion day up to a UTC-day boundary. This is deliberately
conservative for sessions that span time zones. Date-only approved rows can
supply history at or after `replay_available_at`, not a target requiring exact
kickoff. Missing timing or identity evidence leaves rows out.

Do not substitute `scorelines.parquet` if the replay file is small or empty.
The build makes the missing evidence visible rather than turning it into an
optimistic availability assumption.
