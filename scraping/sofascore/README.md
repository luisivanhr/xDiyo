# Shared Sofascore collector

This package implements the initial collection refactor and has passed 55 offline tests plus a bounded three-event live pilot. It uses one plain Chrome session with the external network/VPN route already configured by the user. It does not control a VPN. The three active scraper entry points now delegate to this package. Their original versions are preserved in `scraping/legacy/`; Chromium VPN automation remains outside the supported workflow.

## Setup

From the repository root, using Python 3.11 or later:

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e '.[test]'
./.venv/Scripts/python.exe -m pytest -q
```

The initial offline checks used Python 3.12.14, pandas 3.0.1, PyArrow 25.0.1, Selenium 4.49.0 and pytest 9.1.1. The workspace virtual environment also has CPU PyTorch 2.14.0 for real batch tests. Torch remains optional for collection; the package does not import it until a model dataset is used.

## Commands

### Offline migration of the existing collection

`python -m scraping.sofascore.migrate_legacy` converts the promotion-enriched
match CSVs and the heatmap folder's companion lineup CSVs into the same schema-2
season Parquet layout, under `data/xDiyo_data_legacy`. It makes no API requests,
preserves the source files, joins lineups by match ID, maps saved statistics and
standings, and retains original movement flags with unverified provenance.
Incidents, goal actions, commentary and timing are excluded; injuries and other
discarded fields remain unavailable. It validates original-value parity and
Parquet round trips, checkpoints each season, and quarantines the mislabeled
Eredivisie 2016/17 source. See
[`data/xDiyo_data_legacy/README.md`](../../data/xDiyo_data_legacy/README.md) for
mapping, limitations, loading and resume commands.

### Historical campaign

For manual collection, open [`../Scrape+wait/collect_historical_seasons.ipynb`](../Scrape+wait/collect_historical_seasons.ipynb) with the project's `.venv` kernel. It has a season selector, local checkpoint inspection, foreground progress, explicit access-block retry, and completed Parquet verification. Install notebook support with `python -m pip install -e '.[notebook]'`. The initial notebook run only inspects local state; set `RUN_COLLECTION = True` to collect. Use **Interrupt Kernel** to stop and rerun the collection cell to resume. Proton VPN is managed manually.

The notebook calls the same campaign with `--pause-on-block --resume`. The first option returns control on an access block; the second removes a prior STOP marker after acquiring the collector locks. `--retry-blocked` requests one explicit retry of a saved access block and preserves the server cooldown. A new block pauses again. These controls do not start a background process or recurring supervisor.

Optional request pacing: set `RANDOM_DELAY_MAX_SECONDS = 2` in the notebook, or pass `--random-delay-max-seconds 2` to the campaign, to add a fresh uniform random delay from 0 to 2 seconds between outgoing requests. The default is `None` (disabled); zero is also accepted. The delay is added after other required pacing, including retry backoff; it never shortens server cooldowns. Retries receive the delay, cached components do not, and the first request in a fresh store has no preceding request to wait for. General JSON configs expose the same setting as `rate.random_delay_max_seconds` (`null` disables it); the historical notebook adapters accept `random_delay_max_seconds=`. Negative and non-finite bounds are rejected. Restart the notebook kernel after updating the package.

For a CAPTCHA, set `MANUAL_CAPTCHA = True` in the notebook configuration (default `False`), or pass `--manual-captcha` to the campaign. At the first incomplete season, it opens the normal Sofascore homepage in its visible Chrome session and waits for the user to complete the challenge and press Enter in the notebook/terminal. Only that confirmation clears a saved access block for a retry; cancellation preserves it. `RETRY_BLOCKED_ONCE` is not needed in this mode. No challenge is solved automatically, and confirmation does not guarantee API access: another 403 still pauses the notebook.

The campaign reuses that same browser across seasons, preserving its session, and closes it on completion, interruption or exit. After user confirmation it navigates to `about:blank` to stop homepage polling before fetching API data. Website browsing can make its own API calls outside the collector's pacing control; observed calls to the collector's API origin are recorded as `manual_browser_visit` attempts without payloads, cookies or query strings. Their advertised retry delays remain enforced. Existing cooldowns are honored before the homepage is opened. No browser opens for inspection mode or an already completed selection. CAPTCHA session handling is covered by offline browser and campaign fixtures; acceptance by the live provider still requires the user's manual test.

`python -m scraping.sofascore.campaign --prepare-only` inventories the exact 110 filename stems from `data/xDiyo_data_promotions`. `python -u -m scraping.sofascore.campaign` performs the authorized new collection into `data/xDiyo_data`, newest years first. Supply `--driver-path` when a matching cached ChromeDriver is available. See `data/xDiyo_data/README.md` for the output schema, progress paths and recovery procedure.

The campaign writes a same-stem `.parquet` with one match per row and typed nested component lists, plus a `.manifest.json` and the existing relational tables. The data loader accepts the new season Parquet paths. xG, lineups, heatmaps, injury availability and bundled goal sequences are enabled; commentary is disabled. Scope follows the legacy season's round numbers, with fixture-ID reconciliation and explicit correction of the mislabeled Eredivisie 2016/17 identity. It does not silently claim discovery of unlisted playoff stages.

The user explicitly selected serial navigation at normal browser speed and a 300-second wait between seasons. The campaign therefore sets its interval/minute/hour/run caps to null; regular `RatePolicy` defaults below remain unchanged. Null caps do not disable request accounting, retries, provider cooldowns or access blocks. The cumulative pilot cap still applies to pilot commands. The collector itself pauses on an access block; Proton VPN recovery is performed separately through computer use after a verified country connection. Persistent unattended supervision requires its own approved automation.

Copy `example_config.json` and set the competition, season, actual rounds, output flags and request limits. Its round-one Premier League 2024–25 scope is an example, not automatic full-season discovery.

```powershell
# Inspect existing checkpoints. These commands do not open Chrome.
./.venv/Scripts/python.exe -m scraping.sofascore status
./.venv/Scripts/python.exe -m scraping.sofascore plan --config scraping/sofascore/example_config.json

# Network commands, run only after configuring the scope and external route.
./.venv/Scripts/python.exe -m scraping.sofascore discover --config scraping/sofascore/example_config.json
./.venv/Scripts/python.exe -m scraping.sofascore resume --config scraping/sofascore/example_config.json

# Reparse/export cached successful responses, without network access.
./.venv/Scripts/python.exe -m scraping.sofascore export --config scraping/sofascore/example_config.json

# Retry/refresh only the chosen component of the selected event.
./.venv/Scripts/python.exe -m scraping.sofascore refresh-selected --config scraping/sofascore/example_config.json --event-id 12436870 --component shotmap

# Local inventory and additive migration of an existing prepared season.
./.venv/Scripts/python.exe -m scraping.sofascore audit --input data/xDiyo_data_heatmaps --output data/audits/legacy_inventory.json
./.venv/Scripts/python.exe -m scraping.sofascore import-legacy --config scraping/sofascore/example_config.json --input data/xDiyo_data_heatmaps/Premier_League_24_25.csv --output data/prepared_v2
```

`collect` performs discovery and component collection; `resume` uses cached fixture metadata and retries only due components. Discovery is separately refreshed after its cache interval or with `--refresh-discovery`. Pending fixtures are revisited outside the original round window. Preserve the same `--store` location across seasons so request accounting remains shared. A separate store has a separate ledger and must not be used to multiply the allowance.

The example limits (10 seconds between requests, 6/minute, 120/hour and 40 attempts/run) are conservative operational settings, not a verified Sofascore quota. Retries count. A 429 persists its cooldown and ends the run; an access block latches the collector until access is resolved. `clear-block` clears only the access latch, retaining any server cooldown. No retry rotates routes.

`include_lineups` and `include_injuries` share one lineup request. Both must be false to eliminate that endpoint. Statistics, shotmaps and the two team heatmaps keep their existing request granularity. Commentary defaults to false and has no redundant event lookup. `duration_mode=incidents`, optional match injury incidents and goal sequences share one incidents request. The recommended example now enables goal sequences. Its cold finished-match cost is seven component requests, plus discovery; the original profile without incidents costs six.

## Storage and compatibility

- `data/collection_state/manifest.sqlite`: endpoint checkpoints, request accounting, raw-response versions and fixture metadata.
- `data/collection_state/raw/`: compressed successful JSON payloads, identified by content hash.
- `data/prepared_v2/competition=.../season=.../versions/.../`: versioned typed Parquet tables and a manifest with hashes. `CURRENT.json` changes only after a complete export validates.
- Original CSVs remain in their existing directories. A legacy import records source-file hashes, keeps original ambiguous xG/xGOT separately, and creates no fake API cache entry.

For direct analytical reads, read the required Parquet columns/tables with PyArrow. For existing match-wide consumers, `utils.data_loading.merge_seasons` accepts CSV paths or a Parquet manifest/CURRENT path, joins formations by match ID, supports optional lineups and retains the union of season columns. Numeric unknowns stay missing; availability metadata accompanies the result. The existing `FootballDataset`/`collate_fn` now emit finite placeholders plus `batch["availability"]` boolean tensors matching each numeric feature shape. Shot endpoint availability also has a separate per-match flag. Existing prediction models have not been redesigned to consume the new masks; do that before using missing inputs for training.

`legacy.read_compatibility_views` streams heatmaps into the old nested shape, avoiding a full Python dictionary for every normalized point and every repeated provenance field. `restore_legacy_xg=True` is for forensic parity checks; the ordinary Parquet view leaves suspect defaults unknown. A legitimate source xG of one cannot always be distinguished from the old scraper's fallback.

## Observed capabilities and limitations

The pilot used three events and 23 attempts, including one failed Chrome startup. Raw responses and reports are in `data/collection_state` and `data/audits/capability_pilot.json`. Frozen test payloads record their endpoint, collection time, source hash and removed profile fields. A repeated Belgian match run made zero API calls.

- `lineups` contains both teams, roster statistics and `missingPlayers`. The tested absent players have numeric reason codes; these remain raw/unclassified until their meanings are verified. A later historical fetch is labelled postgame, not pregame knowledge.
- `incidents` contains goal buildup in `footballPassingNetworkAction`. One response provides all supplied sequences for the match. No per-goal or commentary call is needed. `goal_sequences`, `goal_actions` and `passing_edges` are optional companion tables. Pass recipients are not explicit in the observed action objects. `next_action_actor_id` preserves order without asserting a recipient; the strict edge table currently includes only corroborated assist/scorer relationships. Animation coordinate orientation is not silently equated with shotmap coordinates.
- The 2015 Premier League sample has heatmaps, lineups and 19 shots, but no xG and no supplied goal sequence. Both 2024 samples have xG and goal sequences. This is sample coverage, not a guarantee for a whole era or league.
- `injuryTime.length` provides announced added minutes. The tested HT/FT markers use nominal clocks and `addedTime=999`. Exact match/half duration remains null.
- The tested historical pregame-form endpoints return 404; unavailable standings stay missing. A failed refresh cannot replace a previous successful response.

## Awarded-result flag (additive schema-2 field)

The current shared parser exports nullable boolean `matches.is_awarded` from
`event.isAwarded`. A provider `true` remains true even when `status="finished"`
and `status_code=100` (for example Union Berlin–Bochum, event 12499586). An
explicit false or an omitted optional flag on a valid provider event is false.
Unavailable event metadata, explicit nulls and non-boolean flags remain null;
scores and status labels are never used to guess this flag. Reconstructed legacy
rows without the field remain unknown.

The flag is retained in relational exports, same-stem season Parquet files,
and the compatibility loader/CSV view. The normal manifest documents its
`match_awarded_contract`; this nullable addition keeps schema 2 compatible with
existing loaders. The Chrome importer and direct collector share this parser.
The Chrome publisher also fills the field from saved observations when assembling
imports created before this addition. No extra endpoint is needed.

For existing `data/xDiyo_data` publications, the offline command below previews
every season; add `--apply` to publish the revisions, or `--stem Bundesliga_24_25`
to select one season:

```powershell
python -m scraping.sofascore.awarded_backfill
```

It checks capture hashes, timestamps and fixture identities, selects the latest
successful event metadata, and refuses to publish when a flag is unknown. Only
the flag is added/replaced; all other relational files retain identical bytes,
the season's existing fields and order are preserved, and old immutable versions
remain available. Repeating an applied backfill is a verified no-op. Revision
details stay in the ordinary version manifest, with no extra reports or columns.
Raw captures and legacy datasets are unchanged. Restart an already-open Python
collector or notebook kernel before its next run to load the updated parser;
no Chrome extension reload is needed for this Python-only change.

## Match date/time contract (schema 2)

The provider event date/time was already retained in `matches.kickoff_utc`. Schema 2 keeps that one start value and documents its meaning in both Parquet field metadata and the export manifest's `match_time_contract`. No duplicate date-only or purported actual-kickoff column is added.

| Field | Source and meaning | Representation |
|---|---|---|
| `matches.kickoff_utc` | `event.startTimestamp`: the provider's scheduled or revised event start; an actual first-whistle timestamp is **not verified** | Nullable UTC Unix seconds, `float64` |
| `matches.event_metadata_observed_at` | When the collector fetched the event metadata used in this row; not when the match started, finished, or was first announced | Nullable UTC Unix seconds, `float64` |
| Component `observed_at` | When that specific statistics/lineup/etc. response was fetched; may differ from the event metadata's observation | Nullable UTC Unix seconds, `float64` |

All three cached pilot events have provider starts: event 6767929 is `2015-08-08T11:45:00Z`, 12420974 is `2024-07-26T18:45:00Z`, and 12436870 is `2024-08-16T19:00:00Z`. Their metadata was fetched in September 2026. The cached payloads also contain current-period timestamps and update timestamps, which do not establish a separate actual kickoff. Finished status alone does not upgrade the scheduled/revised timestamp to a verified actual start. Original provider evidence remains in the compressed raw responses and timestamped response-version ledger.

Convert explicitly when building chronology:

```python
start_utc = pd.to_datetime(matches["kickoff_utc"], unit="s", utc=True)
# Derive a display date only in the explicitly chosen timezone.
# Example: start_utc.dt.tz_convert("Europe/London").dt.date
```

UTC calendar date can differ from the stadium's or the user's local date. Elapsed-time windows should use UTC instants. No start value is filled from collection time, file time, a period timestamp, round number or season label. Old CSV imports lacking a date remain null; this follow-up did not launch a historical recollection.

Rescheduling preserves the stable event ID. A newer fetched `startTimestamp` replaces the latest match row's start and triggers the existing refresh rules. Older cached observations cannot undo that revision. Earlier observed values remain in raw response versions and previous prepared exports. The current matches table is a latest-view export, not a point-in-time schedule-history table. The observation field identifies this fetch, not the provider's exact revision time or its first publication time. Schedule changes from before the collector began observing are not reconstructed.

For before-match versus before-round cutoffs, the analytics pipeline must choose a schedule snapshot appropriate to the issuance time; today's revised schedule is not proof of what was known then. Round identity also needs competition/season/stage context. No feature-cutoff or CV implementation is added here. Validation: seven focused date/time cases, all 55 project tests passing, and three cached live exports verified without API requests. See `data/audits/match_datetime_validation/report.json` for dates, observation times, source hashes and schema-2 manifests.

## Promotion/relegation flags

```powershell
./.venv/Scripts/python.exe -m scraping.sofascore.movements --input data/xDiyo_data_heatmaps --output data/prepared_v2/team_seasons --legacy-dir data/xDiyo_data_promotions --evidence scraping/sofascore/notebook_movement_seeds.json
```

This command is entirely local. It publishes a compact Parquet table, membership registry, hashes, pinned override IDs and a comparison audit. The original notebook and `utils.promotions_demotions.get_promotions_and_demotions` now use this engine. Slug seeds require explicit full year pairs; adding an older file cannot move a seed to another year. Complete membership confirmations enumerate the exact team IDs and provide an evidence label. The bundled registry maps the existing league labels to Sofascore competition IDs and league systems; it does not certify historical memberships.

The current audit finds 2,120 team-season rows across 109 valid files: 1,505 flag agreements and 615 unknown classifications. `Eredivisie_16_17.csv` duplicates the previous season's 306 match IDs and identity. It is reported and excluded, with its original file preserved. Observed match appearances prove participation, not complete membership or pregame availability. Unknown movements remain nullable.

To join these flags while loading seasons, pass `team_season_flags="data/prepared_v2/team_seasons/CURRENT.json"` to `merge_seasons` or `merge_leagues`. Stable competition, season and team IDs determine both joins. The existing compact entry-reference CSV is not automatically promoted to authoritative override evidence.

## Compatibility and notebooks

```powershell
./.venv/Scripts/python.exe -m scraping.sofascore export-csv --input data/prepared_v2/competition=17/season=61627/CURRENT.json --output data/compatibility_exports

# Bounded pilot: selected event metadata plus requested components.
./.venv/Scripts/python.exe -m scraping.sofascore pilot --config scraping/sofascore/example_config.json --event-id 12436870 --component lineups
```

The pilot cohort and attempts persist across restarts: eight distinct events and 40 attempts maximum, including retries and metadata. `--component` selects components; only `refresh-selected` forces a cached response to be refreshed. A live access block stops the pilot. Keep the same store for every invocation.

The existing `scrape_league_data` and `scrape_league_recent` functions return a run report and manifest path. Extra keyword flags include `with_injuries`, `with_goal_actions`, `duration_mode`, `request_budget`, and `write_csv`. `waiting_time` is a page timeout; pacing comes from the shared rate policy. No CSV is written unless requested. A paused or incomplete run checkpoints and returns `manifest=None`; it does not publish a finished result. The recent adapter revisits the last two requested rounds plus known unresolved fixtures. The old `merge_recent_leagues_columns` name now returns a new output path and never deletes either input. Its `columns` argument controls ordering without discarding later columns.

New CSV nested values use JSON; use the safe compatibility reader rather than a bare `ast.literal_eval` call. An explicit CSV export validates IDs and publishes a versioned manifest. The source CSVs and previous exports remain intact.

## Storage measurements

Premier League 2024–25's original CSV pair uses 23.12 MB; Zstandard tables use 6.52 MB and Snappy 9.41 MB. Full nested compatibility loading measured 1.88 seconds / 480 MB process peak, versus CSV's 1.70 seconds / 294 MB. Scalar-only Parquet loading measured 0.26 seconds / 217 MB. These are single fresh-process measurements with filesystem caches intact, not general speed guarantees.

`python -m scraping.sofascore.heatmap_benchmark --input <heatmap_points.parquet> --output <benchmark-directory>` compares the point table with typed nested list/struct storage and verifies every field and point order. On 573,984 points, nested Zstandard was 4.40 MB versus flat 4.52 MB; full Arrow materialization used 33.4 MB versus 117.4 MB because grouping stores provenance once. Full file read times were approximately 0.05 seconds each. These do not include Python/model view reconstruction. The canonical point table remains the current interface; the compatibility reader already projects point fields and streams them, avoiding repeated provenance dictionaries. Typed nested storage is a useful optional future physical layout if full Arrow materialization becomes a workload requirement.

## Browser setup

Leave `driver_path` unset to let Selenium choose a matching driver, or supply a driver matching Chrome's major version. The successful local pilot used Chrome 152 and ChromeDriver 152. The user-updated standalone driver was version 153 at inspection time. In this Codex execution sandbox Chrome's GPU process crashed; normal desktop execution worked. No browser security switches or VPN changes were required.

See `planning.md` for remaining validation and rollout tasks. Bulk historical re-scraping and model redesign are not performed automatically.
