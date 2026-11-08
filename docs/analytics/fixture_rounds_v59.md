# Audited V59 fixture rounds

The dataset's versioned annotation layer is `data/fixture_rounds/v59_20261108`.
It assigns the audited round/stage selections to 61,488 workbook fixtures across
34 additional leagues. `fixtures.parquet` contains the fixture identity and seven
selected fields. `provenance/` retains the exact compact, full mapping, 368,928-row
field ledger, historical 794-fixture queue, source-specific claims, original
candidate claims, provider observations and independent audit reports. The
manifest fingerprints every payload. Original review filenames and approval
flags remain unchanged as historical evidence; the final Terra audit authorizes
this status-aware dataset integration.

This is a **mixed original/provider-convention mapping**. It is not a certificate
that every selected round is a proven initial-draw assignment. In particular:

| Workbook fixture | Selected round | SofaScore event |
| --- | ---: | ---: |
| 73239 | 4 | 9209891 |
| 73217 | 7 | 9324717 |
| 73205 | 10 | 9351183 |
| 73186 | 13 | 9382095 |

Iván chose these four displayed SofaScore labels explicitly. Three describe the
weeks in which postponed fixtures were played. All four retain
`USER_SELECTED_SOFASCORE_DISPLAYED_ROUND_FALLBACK` and
`v59_original_draw_round_verified=False`; the verification flag is null outside
those four assessed records. Provider event URLs, retained direct observations,
competing original club/league labels and inferred cross-date identities remain
in provenance. The final audit did not independently replay those provider pages.

## Selected field contract

| Dataset column | Authoritative audited column | Nullable type |
| --- | --- | --- |
| `season_round` | `v59_selected_season_round` | Int64 |
| `phase_round` | `v44_selected_phase_round` | Int64 |
| `phase_key` | `v47_selected_phase_key` | string |
| `stage_label` | `v44_selected_stage_label` | string |
| `knockout_label` | `v44_selected_knockout_label` | string |
| `leg_number` | `v44_selected_leg_number` | Int64 |
| `series_game_number` | `v45_series_game_number` | Int64 |

Empty text becomes absent. Authoritative nulls replace existing selected values;
historical columns never fill them. `v53_selected_season_round` is historical and
retains four superseded nulls. The three explicit Bulgarian clears survive.
There are 13,168 null season rounds and 745 fixtures with neither numeric season
nor phase round but with legitimate named stages. All 60 MLS series game ordinals
remain separate from legs and round numbers. No numeric stage ordinals are
invented and no phase round is promoted into a season round.

## Dataset use

```python
from xdiyo_analytics.data import load_round_enriched_matches

matches = load_round_enriched_matches(
    raw_matches_path,
    "data/fixture_rounds/v59_20261108",
)
```

The loader verifies the full release and joins by **workbook match ID**, checking
country, league, season, ordered team names, league/team IDs, local civil match
datetime and date. It rejects duplicates, missing target IDs and conflicting
identities. It never joins those IDs to native or canonical `event_id` values.
The default requires the full audited target population; an intentional subset
can use `require_all_targets=False` while retaining all identity checks for
covered fixtures. Input row order, index, ID types, kickoff fields, scores and
other columns are preserved. Existing round/stage values outside the audited
population remain unchanged.

Only the seven selected columns are attached. `matches.attrs['fixture_rounds']`
records the release, manifest checksum and separate provenance location, with
feature-as-of and model-input approvals false. Review metadata, alternative
claims, source URLs and outcome-bearing research text are never attached as
feature columns. The `.gz` candidate archive preserves all original CSV bytes;
it is audit material only. Its raw checksum is recorded in the manifest.

To persist a new local dataset version with a verification receipt:

```powershell
$env:PYTHONPATH = 'src'
python examples/integrate_fixture_rounds.py `
  --matches '<unchanged Matches_Results.parquet>' `
  --mapping-root data/fixture_rounds/v59_20261108 `
  --output '<new dataset version directory>'
```

The output contains `Matches_Results.parquet` and `integration.json`. Publication
is staged and renamed atomically; an existing version is never overwritten.
The real integration validated 146,000 source rows, assigned all 61,488 targets,
and preserved the other 84,512 rows and all original columns. The materialized
whole-match source remains local under `scratch/round-integration-v59`; the
audited annotations and their provenance are versioned in Git. The unchanged
native 13-league dataset and existing worktrees remain separate.

This API does not alter schema-2 native loaders or the opt-in canonical catalogue
and does not silently rebind existing datasets, model artifacts or selections.
It does not construct prediction cutoffs, features, folds or training runs.
Phase grouping, source availability and forecast timing still require their
separate analysis before experiment preparation. Null scoped warnings are
unassessed, not clearance. V56/V57 comparisons are historical snapshots: V57's
four `SELECTED_NULL` comparisons do not override V59. Its 125 differences are
cross-convention comparisons, not demonstrated errors; the V59 recomputation in
the final audit is 129 differences and nine equal values. The conservative
26 V56 identity gates remain even after eight proposed V57 bridges.

## Reproduction and checks

`examples/publish_fixture_rounds_v59.py` rebuilds the immutable release from the
audited review directory and original target inventory. It pins the audited
compact, full mapping, ledger and queue checksums, verifies every selected field
against typed and string ledger representations, and compares all 368,928 base
conflict gates with V58 before publication. The original V59 review builder is
retained in provenance as history; it needs its original V58 research inputs.

The tests cover the complete population, four fallbacks, nulls and named stages,
clears, MLS ordinals, identity and join failures, input compatibility, whitelist
isolation, source-claim checksums and Parquet value/type/provenance round trips.
See the adjacent `fixture-rounds-v59-validation` receipts for the executed checks.
