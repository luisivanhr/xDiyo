# Native extension naming and integration audit

Baseline: `fa2014db73336d7ec1a3d3432749ebe4f37e62ee`. This migration updates
one current native dataset in place. It does not keep a second dataset, create
a worktree, run a model, or update an experiment's frozen inputs.

## Names and identities

[name_mapping.csv](name_mapping.csv) is the exact old-key-to-provider-name map
for all 617 publications. [inventory.json](inventory.json) records the baseline
filenames, IDs, scopes, calendar groupings, source team names and hashes before
publication. Its identity evidence comes from saved SofaScore season-list and
event responses, with endpoint/database, capture time and response/payload
hashes. Season labels are not inferred from kickoff years or filenames.

All 328 added publications have captured competition, season and team-name
evidence. Two discovery rows with inconsistent indexed/payload identities were
rejected as evidence. The plan records 36 **older** publication-level exceptions
where some historical teams lack captured name evidence in these discovery
databases. Their source names remain unchanged; no guessed crosswalk is added.
This is an audit of saved provider observations, not a claim of live API access.

Examples of verified display names include A-League Men, Stoiximan Super League,
Liga Portugal Betclic, Liga Profesional de Fútbol, Liga 1 Te Apuesto and
Primera A, Finalización. The provider's historical season name can differ from
its current competition name. Both are retained exactly.

Existing filenames, `source_league`, `source_season` and native IDs remain
stable compatibility keys. This avoids renaming assets, joins or recipe inputs.
The pinned manifests explicitly map those aliases to SofaScore display names.
The loader exposes `competition_name`, `season_name`, `season_label`,
`tournament_name` and `home_display_name`/`away_display_name`. Raw `home_name`
and `away_name` observations remain unchanged. Team histories use the verified
display name and retain the original as `team_source_name`/`opponent_source_name`.
The UI and prepared reporter team catalogue use the verified names.

Annual grouping remains separate from native competition, season and tournament
identity. Brazil's delayed 2020 season retains its old `20_21` storage alias but
displays the captured 2020 season name and annual grouping. Mexico's two short
tournaments retain their native identities within the annual cycle. Colombia's
2026 publication retains both tournament IDs; it is not one independent table.
Japan's 2026 transition groups remain separate from its 2026/27 season. Annual
metadata now survives labels and fixture assembly, so an explicit split may use
`block_by=['annual_season']`. Existing split defaults are not silently changed.

## Tables, countdowns and availability

[integration_status.csv](integration_status.csv) lists every publication's scope
and support status. Among the 328 additions, automatic countdowns are supported
for 196 and explicitly unsupported for 132; independent sporting-result form
tables are supported for 194 and unsupported for 134. Unsupported means a
missing rule/evidence implementation, not a flaw in the scraped observations.

Support requires the reviewed source scope plus complete, consistent membership,
pair counts, rounds and a single independent regular stage. Group schedules,
unequal pair multiplicities, carry/halving rules, partial snapshots without a
verified implemented calendar, and unresolved administrative table treatment
are not guessed. The public countdown/DerivedStandings paths reject those
scopes with the recorded reason. The 289 pre-existing publications retain their
existing explicit membership/format validators (`legacy_verified` in the audit
means those existing validators, not a new certification of every old format).
No countdown clustering, rank formula or model architecture was changed.

These derived standings are independent 3/1/0 sporting-result form ranks, not a
promise to reproduce official administrative standings or all league tiebreaks.
Annual grouping never supplies a table ID or resets a stage implicitly.

Chile 2019 is explicitly `curtailed`, with 193 played rows. The original source
snapshot flag is retained as provenance, not used as its canonical coverage
classification. The reviewed official table includes a separate abandoned-game
administrative draw; no artificial played fixture is inserted. Its planned
countdown is 30 rounds, not the maximum downloaded round. Administrative-table
reconstruction is blocked pending independently timed ruling evidence. The
[ANFP 2019 report](https://www.anfp.cl/?jet_download=e6217e403e029179884da33cb687ec10f8866412)
and the source reconciliation remain linked in the audit.

Native labels already mask unfinished/unlabeled fixtures, and temporal splits
require labels available by the training boundary. The new explicit helper
`data.supervised_native_rows(matches, cutoff=..., available_at=...)` checks
finished finite score pairs, kickoff strictly before cutoff, supplied result
release at or before cutoff, reviewed completion bounds, and awards/annulments.
Unknown supplied release times remain ineligible. It does not manufacture times.

The native exports generally do **not** establish historical publication times
for all results, movements, memberships or schedule revisions. Existing default
earlier-finished-kickoff timing remains an explicitly recorded retrospective
proxy; it is not strict as-of verification. Actual result bounds still take
precedence. Movement flags describe entry into a season, with retrospective
evidence; no end-of-season outcome is shifted backwards. For strict historical
work, supply independently justified availability and membership evidence.
Countdowns still use the supplied revised schedule under their existing policy;
this migration does not claim they reconstruct an archived issuance schedule.

## Awarded fixtures

[awarded_matches.csv](awarded_matches.csv) lists 40 flagged rows in the extension,
up from 37. [awarded_materialization.json](awarded_materialization.json) records
three corrections in the existing normalized and nested Parquet files:

* `14115336`, Atlético Nacional–Atlético Bucaramanga: played 0–1, administrative
  0–3, [DIMAYOR resolution 093](https://dimayor.com.co/wp-content/uploads/2025/09/Resolucion-No.-093-de-2025-COMITE-DISCIPLINARIO-DEL-CAMPEONATO.pdf).
* `15352954` and `15352964`, Unión La Calera's fixtures against Cobresal and
  Everton: administrative 0–3 defeats for La Calera, separately oriented for
  each fixture; [ANFP ruling](https://www.anfp.cl/?jet_download=3de0cfc3a9e951ff916fdd865ab5781df6acb936).

`provider_is_awarded` preserves the original flags. Played scores are unchanged;
administrative scores and annulment details are separate pinned metadata. A
ruling date is not invented as an exact publication timestamp. Annulments are
distinct from awards. Previously excluded unplayed/unfinished fixtures remain
excluded, with their dispositions preserved. The three Greek supplemented
matches retain their source/availability evidence and unknown provider flags.
The audit reconciles available provider/review evidence; it does not certify that
every historical disciplinary decision absent from those sources was found.

The four-file award publication uses a validated recovery journal. A retry
finishes only matching staged changes. Temporary staging files disappear on
success; no additional dataset version is retained. ZSTD is preserved.

## Explicit recipe and manifest-pin refresh

The existing `refresh_recipe_data` helper and UI action are extended. They never
overwrite the input recipe or old selection records. They validate current table
hashes, schemas, identities and overlays before writing, reject missing old pins
or ambiguous identities, and save a new recipe, selection directory and review
report. The report distinguishes table changes from manifest-only changes;
**manifest-only does not imply numerically equivalent**, because a pinned rule
or overlay can change semantics.

From the repository root:

```powershell
python examples/refresh_recipe_data.py path/to/recipe.json --output path/to/recipe_v2.json --previous-revision fa2014db --dry-run
python examples/refresh_recipe_data.py path/to/recipe.json --output path/to/recipe_v2.json --previous-revision fa2014db
```

`--previous-revision` verifies the old manifest hash directly against that Git
commit. For pins matching this migration's baseline, the report can also obtain
the revision from the pinned migration metadata. Other unresolved old revisions
are explicitly reported as unresolved, never invented. The command does not
checkout, duplicate or restore data, migrate trained results, or fit a model.
To reproduce an old recipe later, restore its identified dataset revision when
safe to do so; merely copying old hashes onto current data is invalid.

Only the documentation demonstration selection was refreshed automatically:
`Brazil_Serie_A_20_21`, [demo_v1.json](demo_v1.json) to
[demo_v2.json](demo_v2.json), with [the review report](demo_v2_refresh.json).
The refreshed recipe prepares 380 fixture rows and three folds; no model was
trained. The old recipe fails against the current manifests. Other consumers
must run the explicit helper for their chosen populations. No running experiment
recipe, selection, manifest, prepared data or source was refreshed.

## Verification

[verification.json](verification.json) records the 617-publication audit:
172,274 unique fixture identities, 8,731 checked native table hashes, 8,729
unchanged tables, two changed match tables, and no duplicate identities. It
compares changed observations to Git's original Parquet bytes, checks manifest
links, membership references and the nested publication hashes, exercises all
196 newly supported countdown populations, and records representative calendar,
cross-year, split-stage, transition and curtailed cases.

The active experiment's frozen-input control hashes are recorded separately and
rechecked. Its model training, collector and frozen data were not touched.
Final test counts, commands, environment and source hashes are in
`test_receipt.json`; the XML and JavaScript log are retained beside it.

Reverification (read-only except documentation receipts):

```powershell
python examples/verify_native_integration_20261116.py
```

The initial inventory and plan scripts intentionally require the pre-migration
baseline; do not regenerate them from already changed manifests. Publication is
recoverable through `python examples/integrate_native_20261116.py --apply`.
