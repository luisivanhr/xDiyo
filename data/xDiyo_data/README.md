# xDiyo historical collection

This directory contains Sofascore season exports. Recent seasons run first, followed by earlier years. The original CSVs remain available in Git history.

The GitHub copy contains the top-level Parquet files, their manifests, and the relational `_tables/` versions. The `_collection/` directory stays local because it contains personal information.

Each finished season has the original filename stem with `.parquet`, for example `Premier_League_24_25.parquet`. It has one match per row, typed scalar match columns and typed nested lists for statistics, shots (including nullable xG), heatmaps, lineups, player statistics, injury availability, incidents and supplied goal sequences. Commentary is disabled. A same-stem `.manifest.json` contains completeness checks, source inventory provenance and availability counts. Detailed relational Parquet tables and observation provenance are in `_tables/` and linked from each season manifest and Parquet metadata.

Read columns directly with PyArrow, or pass the season Parquet path to `utils.data_loading.merge_seasons` for the existing match-wide view. That loader retains legacy standings normalization; stored positions remain raw. Promotion/relegation flags remain a separate optional join through `data/prepared_v2/team_seasons/CURRENT.json` and are not copied from the old CSVs.

The 122 seasons in the 18 September 2026 offline backfill (44,561 matches,
including the 13 new 2025/26 seasons) have an `is_awarded` boolean in both the
season file and relational `matches` table. The backfill used saved provider event
metadata: 23 matches are flagged true, 44,538 false, and none unknown. Other match
fields, existing statistics repairs and all previous immutable versions were
preserved. The usual version manifest records the backfill; no separate report
or API request was needed. Future captures retain the flag automatically. Its
[source and missing-value contract](../../scraping/sofascore/README.md#awarded-result-flag-additive-schema-2-field)
is independent of the ordinary match status.

The requested scope is the same leagues, years and regular round numbers represented in the legacy files. All returned fixtures in those rounds are retained. Legacy fixture IDs are reconciled, and any missing fixture is fetched individually. The mislabeled `Eredivisie_16_17.csv` must resolve to the provider's actual 2016/17 season; its incorrect 2015/16 IDs are not reused. This campaign does not silently add other competitions or unlisted playoff stages.

The collector uses a single serial Chrome session per season, then waits 300 seconds. The user-provided 8–12 minute season runtime is a reference, not a guarantee. This campaign has no extra per-request, per-minute or hourly throttle. Ordinary scraper defaults are unchanged. Every request and retry is still recorded in the shared `data/collection_state` ledger. Historical successful, empty and unavailable responses are reused on resume. Missing xG remains null and is counted separately from a failed request.

## Progress and control

- `_collection/status.json`: current process, season, phase, counts, pause reason and next resume time.
- `_collection/inventory.json`: all 110 season identities, legacy source hashes and requested profile.
- `_collection/*.out.log` / `*.err.log`: launch-specific process output.
- Creating `_collection/STOP` gracefully stops at the next request boundary or during a wait. Remove only this marker to allow an explicit restart.

Run from the project root using `.venv/Scripts/python.exe -u -m scraping.sofascore.campaign`. A matching driver can be selected with `--driver-path`. Restart uses completed file hashes and endpoint checkpoints. Do not start a second scraper or change the ledger to evade its accounting; OS locks prevent concurrent campaign and HTTP writers.

Provider rate-limit cooldowns are persisted and honored. An access block closes the browser and puts the process into `needs_vpn_recovery`. The user has authorized changing Proton VPN to another country. Recovery requires inspecting the actual block, changing the external Proton VPN connection through the approved computer-use interface, confirming connection success, then clearing the access latch while preserving any cooldown. Internal unexpected-browser-traffic errors require inspection rather than VPN recovery. The script does not itself click Proton VPN or clear access blocks automatically.

Only validated completed seasons receive public season files. An interrupted season remains in the raw cache and resumes from there. A complete request coverage report does not imply that Sofascore supplies xG, lineups or exact duration for every match.
