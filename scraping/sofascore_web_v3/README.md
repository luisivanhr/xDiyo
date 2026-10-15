# Sofascore website capture v3

For the requested 2015/16–2019/20 expansion (21 league-seasons), use
[the prepared historical batch and operating guide](../HISTORICAL_EXPANSION_README.md).
It contains five saved Discovery IDs and one resumable conversion runner.

## Daily history and prediction fixtures (0.4.0)

Use [the daily runner](../run_daily_collection.py) and its
[operating guide](../DAILY_COLLECTION_README.md) for ongoing seasons. It discovers
completed history plus the whole current/next regular round, reuses verified
completed matches by event ID, refreshes unfinished fixtures, and publishes daily
versions automatically. Each league can end independently. The default daily
output is `data/xDiyo_data`, shared with historical seasons. Updates merge by
event ID and retain completed history and previous immutable versions.
Reload the extension and reopen DevTools for version 0.4.1. The complete historical
season workflow below remains available as a separate collection mode.

## Fresh-season discovery (0.3.0)

The extension has **Discover matches** controls. Select **Complete historical season** to resolve an explicit season
(default `25/26`) for the 13 registered leagues, captures the provider's round and
standings inventories, and enumerates numbered round fixtures through the same
Chrome session. Python performs no HTTP requests. Each acknowledged response is
saved under `data/website_capture_v3/discovery/<discovery-id>/discovery.sqlite`.
Reconnecting the same ID resumes only unfinished pages.

Use **Load saved leagues**, select the season/leagues, then **Create / reconnect**
and **Discover / resume**. A prepared discovery can instead be opened by pasting
its ID. Keep an ordinary Sofascore match page open as the session anchor. The
extension reuses the inspected tab and only renews that anchor when session access
expires. Discovery and collection are mutually exclusive; disconnect the old
campaign first. Reload the extension and reopen DevTools after updating it.

Review the per-league preview, then select **Create collection campaign**. This
creates a separate campaign, fills its ID into the existing collection controls,
and connects Python. **Start / resume** begins component collection. The existing
publisher accepts the resulting `<League>_25_26` cohorts; point the publication
watcher at this new campaign. An old watcher's campaign selection does not change
automatically. The notebook provides discovery configuration, saved IDs and a
foreground watcher option.

The selection is strictly the regular league stage: named playoffs and group
stages, including post-split groups, are excluded. This is narrower than some
historical Belgian and Scottish files. The preview reports those exclusions.
The complete pairing schedule must reconcile against teams in the regular-season
standings. Built-in formats are two meetings per pair, or three for Scotland's
pre-split regular season. A new league can be added with a filename name,
Sofascore competition ID and 1–4 meetings per pair. Multi-group or other irregular
formats require an explicit additional validation rule; they do not silently pass.

Belgium and Scotland 2019/20 have explicit curtailment rules: played fixtures
must reconcile with each team's final standings count and the independently
recorded season total. Other seasons retain the complete pairing check. These
rules apply only to historical discovery, not to ongoing daily collection.

Successful enumeration and complete collection are separate states. Missing or
ambiguous season IDs, empty/truncated rounds, unknown stage labels, conflicting
identities, incomplete schedules and unresolved active match statuses prevent campaign
creation. HTTP errors pause discovery; a 404 is not an empty completed round.
HTTP 403/429 discovery responses are not automatically retried. Existing match
collection keeps its own bounded recovery behavior. Discovery's random delay
defaults to `None`; a maximum T adds a uniform [0,T] wait to the usual pacing.

The defaults retain statistics, xG/shotmaps, heatmaps, optional lineups, reported
absences and incidents, with no commentary. Pending/rearranged fixtures remain
in the saved inventory. This mode prepares a completed historical season;
daily updates and upcoming-match collection use the daily workflow above.
Live additions to an already running immutable campaign remain outside both flows.

If a postponed or abandoned fixture has exactly one later finished replacement
with the same league, season, round and ordered teams, collection uses that
replacement. Both provider records remain saved, and the preview lists the
superseded IDs. Ambiguous duplicates remain unresolved and prevent preparation.
Unreplaced abandoned, cancelled or awarded entries without finished status stay
in the discovery inventory and its preview but are excluded from the completed
match collection. They still count when reconciling the league's full schedule.

Validation: offline Python/Node tests exercise identities, resume, stage and
schedule completeness, IPC exclusion, same-tab session use, pacing and failure
handling. The live 2025/26 discovery completed on 17 September 2026 with 517
provider requests and all 13 regular schedules reconciled: 4,502 fixtures,
including 4,501 finished matches. It retained 67 superseded fixture records,
excluded 102 captured post-split group games, and kept the abandoned
Nantes–Toulouse match (14061912) outside the finished-match campaign. Named
playoff rounds were skipped before dispatch. No access denials occurred during
this discovery; the Dutch round-prefix parser repair reused its saved HTTP 200
response without another request.

Discovery ID: `discover-2025-2026-eff840b6`.
Collection campaign: `chrome-2025-2026-82c006eb`.
Collection and publication completed on 18 September 2026: all 4,501 queued
matches are complete and all 13 `<League>_25_26` season files are available under
`data/xDiyo_data`. The publication worker had exited after seven seasons; an
offline recovery verified those seven and published the remaining six without
new provider requests. Final checks matched every match ID, order, row count and
season-file checksum to the campaign and its manifests. The publication watch
status is `finished`, with 13 verified seasons and no error. The original worker
logs are in the campaign directory; recovery events are appended to
`data/xDiyo_data/_collection/chrome_publication_events.jsonl`.

## Awarded-result flag

The current Python importer, season publisher and compatibility loader retain
`is_awarded` from captured `event.isAwarded`. This is a separate nullable boolean:
an awarded fixture may still have ordinary finished/100 status. The full event
response was already captured, so no extension change or additional request is
required. Older imports awaiting publication are upgraded from their saved event
observations. See the shared collector's [flag contract and offline backfill
command](../sofascore/README.md#awarded-result-flag-additive-schema-2-field).

## Planned recovery after the main collection

The [post-collection recovery and republication plan](../../planning.md#post-collection-recovery-and-republication) records the agreed next stage: inventory 404s and suspicious empty components, retry only the missing data after ordinary collection finishes, merge by match ID while preserving order, and publish validated revisions of affected season packages with previous versions retained. Include completed matches with incomplete features, not just deferred queue entries. This workflow is planned; the current publisher does not automatically patch already-published seasons.

## Session access without a captcha header (0.2.21)

The collector now accepts the site's successful request pattern when `X-Requested-With` is present and `X-Captcha` is absent. It requires a recent successful first-party response (200 or 304), retains only the observed headers in Chrome memory, and bounds each dispatch by a 120-second local observation window. This window is not a provider token expiry. Successful 200 component responses refresh the observation. If a captcha header is present, the existing expiry and validation rules still apply; malformed or expired tokens are never silently removed.

The session access test reports `accessMode` and a null `tokenRemainingMs` when no token was observed. Denial recovery, request pacing, match identity and component coverage checks are unchanged. Session renewal also accepts known match section fragments such as `,tab:lineups`, strips only that section, and preserves the exact validated host and match ID.

## Recovery rest timing (0.2.20)

An intentional challenge or expiry rest now excludes its measured elapsed duration from the active match budget, including delayed callbacks. It uses the existing Python timer and continues to check pause controls. Requests and page renewal still consume the normal match budget; the configured cooldown, minimum request spacing, and single challenge retry are preserved.

Reports distinguish requested and actual rest duration, Python pacing time, the native round trip, and local progress time. This makes future local stalls diagnosable without extra provider requests. Version 0.2.20 passed 153 JavaScript checks and 68 Python checks, including regressions that failed on overslept rests before the fix. The underlying cause of the earlier local scheduling delays is still unconfirmed.

## Request pacing and Windows status writes (0.2.19)

Session mode now performs its inter-request wait through the existing Python connection using a high-resolution monotonic clock. The configured minimum request-start spacing, optional random delay, local pause controls, and challenge/season rests are preserved. Reports include the pacing backend and requested/actual wait totals. Only missing components are fetched; the active session route uses neither H2H nor a VPN.

Atomic status replacement retries transient Windows access/sharing conflicts for up to 1.58 seconds. Persistent errors still surface, and previously saved responses remain reusable. The campaign writer and offline publication watcher use the same helper. See the [deployment and validation receipt](../../data/audits/website_capture_v3/reliability/deployment_iteration31.json) and [live pacing evidence](../../data/audits/website_capture_v3/reliability/session_pacing_iteration31.json).

## Bounded session recovery (0.2.18)

With `SESSION_CHALLENGE_RECOVERY = True`, one HTTP 403 whose reason is exactly `challenge` gets a configurable rest (`SESSION_CHALLENGE_COOLDOWN_SECONDS = 180`), one ordinary renewal of the existing anchor, and one retry of the failed component. Previously saved components are reused. The first denial is retained, and the entire match has only one recovery attempt. A second denial, 401, 429, another 403 reason, failed renewal, or visible human challenge pauses. Set the flag to `False` for immediate pauses on all denials. Session values stay inside Chrome, and neither H2H nor a VPN is used.

Renewal requires a successful observation that began after the page was renewed; old Network entries cannot validate it. If a token is nearly expired, the collector waits until it expires before reloading, because the site's initializer otherwise keeps the same token. These fixed rests check local pause controls without component requests, and their measured elapsed duration is excluded from the match budget; renewal and request work still have the normal time budget. Reports retain request counts, renewal counts, recovery attempts, and wait durations.

The active route is **the existing Chrome session → DevTools extension → local Python writer → typed Parquet**. Read [AUTOMATION.md](AUTOMATION.md) for the queue reused from the other scraper, the small pilot, verified feature coverage, limitations, and start/resume instructions. Use [collect_chrome_campaign_v3.ipynb](../Scrape+wait/collect_chrome_campaign_v3.ipynb) to configure and monitor it.

Version **0.2.17** adds `collection_mode="session"`: queued component GETs run sequentially in the existing Sofascore tab using its recently successful site-issued session headers. Only missing components are requested, bodies are acknowledged directly by Python, and session values stay in Chrome. It does not navigate through H2H or open each queued match. The first live metadata test recovered HTTP 200 for blocked match **12448325** in 240 ms on 13 September; sustained collection is still being measured. The original passive collector remains available as `collection_mode="page"`. Python still sends no HTTP requests. No VPN is used.

Set `COLLECTION_MODE = "session"` and `SESSION_REQUEST_INTERVAL_SECONDS = 0.5` in the notebook. The interval is a minimum between request starts, across match boundaries. `RANDOM_ACTION_DELAY_MAX_SECONDS = None` disables jitter; a value T adds a random [0,T] wait before each component request in session mode. Five-minute season rests remain. The saved queue configuration is authoritative: disconnect the writer and apply the policy once with `APPLY_SAVED_POLICY = True`. This changes no fixture IDs, feature flags, captured data, or completed results.

Session mode keeps an ordinary Sofascore match page as its session anchor. It renews that same page between saved matches when the configured memory interval is reached, or when the session has insufficient validity for a match. The request loop never uses H2H. A renewal that needs interaction pauses. Optional bounded recovery for a 403 challenge is described above; other denials pause immediately. Python validates captured metadata against the queued event, teams, competition and season, then requires all selected endpoints to be accounted for before completing the match. The report records `collection_mode="session"`, `identity_source="captured_event_metadata"`, the explicit request count, and renewal evidence. 404 means unavailable data; it never fabricates a populated feature. Pregame positions remain distinct from H2H data.

In **page mode**, the indexed local progress checks and periodic page renewal remain enabled. In the notebook, set `MEMORY_RESET_EVERY_MATCHES = 50` (or `None` to disable). After that many saved matches, the extension briefly opens its local reset page in the same tab, keeps DevTools and the writer connected, starts recording, and opens the next queued fixture. No completed match is revisited for the reset. A failed reset pauses with the checkpoint intact. This is not an exact RAM ceiling: Chrome decides when memory is reclaimed, and reopening DevTools restarts the counter.

For an existing queue, close DevTools and set `APPLY_SAVED_POLICY = True` for the prepare/resume cell once, then restore it to `False`. The helper locks the collector, backs up configuration, and changes only policy. Reload the extension, reopen DevTools, and reconnect the same campaign. Saved matches, raw responses, feature flags, cookies, and browser cache are preserved.

In **page mode**, the same-tab `location.replace` fallback introduced in 0.2.10 remains enabled. Iteration 12 completed 132 more matches with sampled main heap around 99-352 MB; earlier runs reached about 7 GB near 100-120 matches. This is a measured improvement in that run, not proof that all future memory growth is eliminated. See the [recovery report](../../data/audits/website_capture_v3/reliability/REPORT.md) for live reset validation and later measurements.

The first queued season, `Bundesliga_24_25`, has 306 saved imports verified against their Parquet hashes and fixture, competition, season, and team identities: 305 complete and one awarded fixture deferred. Its deferred fixture prevents Chrome season publication, and the pre-existing season output is preserved. Current iteration measurements, coverage checks, memory-recovery evidence, and stop causes are in [the recovery report](../../data/audits/website_capture_v3/reliability/REPORT.md).

After all collection and deferred retries finish, the [offline reconciliation command](AUTOMATION.md#preserve-an-older-output-before-chrome-publication) can preserve the older Bundesliga bundle and install its verified Chrome replacement. It defaults to read-only preview and refuses to apply during collection. It has passed 18 offline tests; the live output remains unchanged.

Completed cohorts can now be assembled offline into original-stem season Parquet files in `data/xDiyo_data`. Set the notebook's `PUBLISH_COMPLETED_SEASONS = True`; `SEASON_EXPORT_STEMS = None` selects all complete cohorts, or provide a list of stems. The publisher verifies component accounting, identities, source and output hashes, and lossless relational tables. It preserves existing conflicting outputs and all original captures. **Sixteen season cohorts** have passed publication and actual legacy-loader standings checks, covering **5,767 matches**. See [publication instructions](AUTOMATION.md#publish-a-completed-season) and [the verification receipt](../../data/audits/website_capture_v3/reliability/published_seasons.json) for the complete list of files and individual verification times. Raw standings stay raw in storage, missing values remain missing, and promotion/relegation processing is separate.

The notebook now defaults to `chrome-from-queue-bc6ae7b3`, containing 40,060 IDs from 109 valid seasons in the original scraper's inventory. The unresolved `Eredivisie_16_17` remains explicitly excluded. The earlier three-match pilot is still available. The original collector and legacy restoration are separate. The manual HAR importer remains available through [explore_website_capture_v3.ipynb](../Scrape+wait/explore_website_capture_v3.ipynb). The earlier Selenium experiment and its failed-access evidence are retained below for context.

The [offline export watcher](AUTOMATION.md#export-automatically-as-seasons-finish) can now publish each fully completed season automatically. It polls the local queue, reuses the verified publisher, and keeps a separate status and event log under `data/xDiyo_data/_collection`. It preserves conflicting outputs and stops independently of Chrome. The publisher and watcher passed 12 offline tests.

## Earlier Selenium experiment

- Opens a visible, ordinary Chrome match page using Selenium. Optional homepage warmup gives you time to complete a challenge manually in the same browser.
- Reads the browser's performance events and already-received response bodies with `Network.getResponseBody`. The collector does not navigate to API URLs, inject `fetch`/XHR, or replay requests.
- Saves selected, validated match responses as compressed JSON with hashes and observation times. Reuses the existing pure parsers to export separate Parquet tables.
- Targets xG/shotmap, statistics, both team heatmaps, lineups, player availability, pregame positions, and incidents containing goal sequences and timing evidence. Source omissions remain missing. Commentary bodies are not retained.
- Records **all observed Sofascore API requests**, including unrelated calls, duplicates, cache responses, and failures. An endpoint count is not a count of TCP connections or a guarantee of server-side load.
- Checks the selected event ID against the actual page URL and captured metadata. Responses for other matches cannot populate this match's components. Query variants are logged but deliberately not normalized into the full-match cache.
- Stops match browsing on 401/403, provider cooldown, a different selected match, or an observed request/time limit. During optional homepage warmup, a 403 leaves the human challenge visible; the same limits still apply.

This is passive **capture**, not request-free collection: the website still makes API calls. It may load several endpoints at once, repeat requests, or fetch data only when you select a tab. Request limits are observational and a browser burst can overshoot. The original scraper's random delay between requests cannot be enforced on website-generated requests by a passive listener.

## Earlier exploration notebook

Open `scraping/Scrape+wait/explore_website_capture_v3.ipynb` with the **Python (xDiyo)** kernel. It defaults to `RUN_EXPERIMENT = False`, so Run All performs a local review without opening Chrome.

For the working Chrome session, keep that flag false and use the notebook's **Import responses** section. A missing HAR response body is a failed capture, not empty source data. Original exported archives are not copied into the prepared output: only selected endpoint bodies, hashes, timestamps, and stripped request metadata are retained.

For one live experiment:

1. Stop any active run of the current collector; the experiment shares its request ledger lock.
2. Select a completed `EVENT_ID` already present in `data/collection_state/manifest.sqlite`.
3. Set `RUN_EXPERIMENT = True`. If a challenge is expected, set `MANUAL_CAPTCHA = True` and choose the warmup duration. Solve the challenge yourself in the visible Chrome window during that interval.
4. Run the collection cell. After warmup, the selected match opens. During the timed browsing window, select its visible statistics/lineups/heatmap/goal views as available. The notebook prints progress while listening.
5. The browser closes at the end of the window and the results cell shows coverage and paths. Do not infer success from the presence of a Parquet file: check `complete`, per-component status, and provenance.

You can independently disable roster export (`INCLUDE_LINEUPS`) and availability (`INCLUDE_INJURIES`). If availability remains enabled, the lineups response is still retained because it contains missing-player information. These flags control retained/normalized data, not what the website chooses to request.

## Output and isolation

Each session gets `data/website_capture_v3/<UTC timestamp>-<id>/` with:

- `reference.json`: cached fixture identity and its original observation time.
- `requests.json`, `report.json`: request accounting, coverage, page identity, actions, stop reason, and export location.
- `captured/`: the experiment's own SQLite component manifest and compressed raw JSON.
- `prepared/competition=…/season=…/`: versioned Parquet files and manifests.
- `screen-*.png/json` for explicit UI snapshots and `stopped-page.png` when a page is stopped.

Existing component data are never copied in as newly captured values. A match row may come from the identity reference even if every live request failed; `event_metadata_source` makes that explicit. A complete capture requires fresh observed event metadata, matching page identity, and terminal selected component coverage. `unavailable` means an observed 404, not a page tab that was never opened. A failed refresh preserves the last good raw payload while marking the latest attempt failed.

The experiment writes request attempts/run summaries and activity timestamps to the existing shared ledger, and can extend its provider cooldown. It leaves the existing access-block latch, event registry, component cache, prepared seasons, scraper code, and notebook untouched. This bounded diagnostic is explicitly allowed to open an ordinary website page while the old direct collector's access latch remains set; it does not reset that latch or resume the campaign.

## First live finding — 12 September 2026 UTC

Selected event **12499375**, Borussia M'gladbach–Bayer Leverkusen, 2024/25 Bundesliga round 1.

| Observation | Result |
|---|---:|
| Observed API requests | 56 |
| API responses with HTTP 403 | 56 |
| Requests for selected component/metadata paths | 1 |
| Other API requests | 55 |
| Repeated paths beyond their first occurrence | 8 |
| All browser requests, including assets and ads | 317 |
| Captured successful selected JSON payloads | 0 |

The screenshot shows a visible Cloudflare “Verify you are human” challenge. The page replaced `#id:12499375` with `#id:16434120`, displaying a future fixture dated 5 December 2026. Thus a rendered page shell did not establish access to the requested historical data. The session was ended without solving the challenge or retrying.

Evidence is under `data/website_capture_v3/20260912T011212Z-160350af/` (`report.json`, `requests.json`, `screen-01.json`, `screen-01.png`). The initial diagnostic preceded the new early-stop and explicit page-identity reporting guards; its original evidence is preserved. Its match/timing rows derive only from the labelled cached identity, not fresh successful responses.

The selected profile has a baseline of **seven direct component requests**, plus fixture discovery/any metadata refresh or retries. This failed website run is not a fair successful-run benchmark, but it demonstrates that normal browsing does not automatically avoid 403 challenges or reduce traffic. No request-saving or full-coverage claim is supported yet.

## Validation and next experiment

Offline browser-event replay checks asynchronous bodies, exact event filtering, no injected requests, duplicate/cache accounting, errors, cooldowns, old-cache isolation, lock release, and normalization/Parquet export. Frozen observed shotmap/lineup/incident fixtures verify xG, roster rows, and goal sequences; statistics/heatmap test payloads are synthetic and do not prove current website coverage.

The notebook is validated top-to-bottom with live collection disabled. The Selenium success path still needs a successful match-page run. The separate existing-Chrome DevTools route has now passed a real HAR-to-Parquet check, described below. A controlled comparison of the same completed match and feature set is still needed to measure request savings before scaling to seasons.

Neither the earlier Selenium experiment nor the new explicit-fixture Chrome pilot performs full-season discovery, normalized seasonal standings assembly, promotion/relegation flag generation, or legacy season-name publication. These remain the next stage after pilot validation. It also does not extract unrequested hydration state, reconstruct heatmaps from images, solve CAPTCHAs, rotate VPNs, retry a blocked session automatically, or fall back to direct API calls.

## Follow-up: existing Chrome session works for the historical page

A later check on 12 September 2026 opened event **12499375** through the user's existing Chrome profile. The selected historical event stayed correct and loaded the 2–3 finished score, lineups, substitutions, and injuries without showing a challenge. Opening the goal animation displayed an app-download demo, not the actual goal sequence. The existing user tab was left unchanged.

This verifies browser UI access for that session. It does not establish the cause of the earlier Selenium failure: the tests occurred at different times and connection conditions may differ. The Codex Chrome plugin exposes no raw JSON network-response capture interface, but Chrome itself does: the new DevTools extension reads existing Network bodies with `getContent()`, and the HAR importer feeds those responses to the existing pure parsers. The v3 Selenium transport has not been made to attach to this working Chrome session. Evidence summary: `data/audits/website_capture_v3/existing_chrome_access.json`.

## First extension HAR: match mismatch, no captured bodies

The user loaded the extension and supplied `data/HAR/sofascore-event-12499375-2026-09-12T02-47-44-551Z.har`. Its first revision contained 38 API records: 10 HTTP 200, 19 HTTP 304, and nine HTTP 404 observations, with no 403s. Every numeric match path was for **16434120**, while the capture filter selected **12499375**. All entries were metadata-only, so the first archive contained no captured match features. The offline import sent zero requests and correctly marked every requested component `not_observed`; any match/timing rows derived only from the labelled identity reference.

The file was replaced while the audit was underway. The revision exported at **02:54:18 UTC** has 138 API records and does include the correct historical match. Its 30 selected component observations comprise **29 HTTP 304** responses and one **pregame-form HTTP 404**. Version 0.1.0 did not attempt body reads for 304s, so this revision also has no captured feature bodies. The updated importer marks the bodyless cached responses as failures and pregame-form as unavailable; it does not fill gaps with old cached features. A frozen snapshot, source hashes, and both revision results are linked from `data/audits/website_capture_v3/har_capture_validation.json`.

Version 0.1.1 now displays observed match IDs and warns explicitly about empty exports. The extension and HAR importer also accept cached bodies attached to HTTP 304 observations only when those bodies are actually present and pass the same schema checks. HTTP status and cached-body provenance are retained. The existing scraper is unchanged.

For the next recording, verify that the page displays the finished **2–3** historical match and the URL ends with `#id:12499375`. Keep DevTools open while selecting that historical fixture. Read the existing Network log first; do not assume a reload preserves the historical selection. Check the panel for matching event IDs and saved bodies before exporting. If the page shows a different fixture, return to the intended historical match rather than changing the filter to the future fixture merely to obtain a nonempty export.

## Successful DevTools capture — 12 September 2026, 03:03 UTC

The user's version 0.1.1 HAR contains 75 observed API entries and 29 saved HTTP 304 cached bodies for seven selected paths. The original bodies survive normalization: 39 shots all with xG, 40 distinct lineup players in two observations, three distinct reported absences, 1,635 heatmap points across both teams, and five goal sequences with 10 actions, including two passes. The importer generated Parquet without sending any API requests. No cached fixture identity was needed for this import; metadata came from the captured event response.

The latest archive alone has pregame-form `not_observed`. A second, explicitly merged import uses the earlier recorded pregame-form 404 and reports endpoint coverage complete with that component `unavailable`. This flag does not establish complete pass networks or exact elapsed durations. Goal actions have coordinates and ordering but no explicit recipient IDs, so `passing_edges` remains empty. The period-end incidents contain `addedTime=999`; exact half/match elapsed duration remains unknown, while the 4- and 10-minute added-time announcements are retained. Availability keeps the source reason code instead of inventing a medical classification.

One source discrepancy is preserved: match statistics report xG **1.58/2.70**, whereas summing the 39 source shot values gives **1.716505/3.310107**. The import keeps both; their disagreement is not caused by dropped or changed shot values. Its cause has not been established.

The validated combined output is `data/website_capture_v3/20260912T030543Z-har-55e6a18a/`. The latest-only output is `data/website_capture_v3/20260912T030422Z-har-7bc9c7a8/`. The reproducible checks and evidence are `data/audits/website_capture_v3/validate_successful_har.py` and `har_capture_validation.json`. All 30 original scraper-file hashes still match. This establishes that passive response-body capture works in the existing Chrome session; it does not establish fewer requests, automatic season navigation, or successful Selenium access.

## CLI and tests

From the repository root with its existing `.venv`:

```powershell
.venv/Scripts/python.exe -m scraping.sofascore_web_v3 --reference data/audits/website_capture_v3/reference_event.json --manual-challenge-seconds 60
.venv/Scripts/python.exe -m pytest tests/scraping/test_website_capture_v3.py -q
node --test tests/scraping/network_capture.test.mjs
.venv/Scripts/python.exe -m scraping.sofascore_web_v3.import_har --har path/to/export.har --event-id 12499375
```

The CLI emits observed UI indexes. Send JSON commands on stdin: `{"command":"snapshot"}`, `{"command":"click","index":0}`, `{"command":"report"}`, or `{"command":"finish"}`. Use an index from the latest snapshot. The CLI continues listening between commands and ends at its session/request limit. `--driver-path` can select an installed matching ChromeDriver; otherwise Selenium Manager resolves it. Run from the repository root: this sibling experiment intentionally does not modify the existing package installation configuration.

The 0.2.12 reset page reports readiness through an internal extension message, checked against the extension origin, tab ID, and per-reset nonce. This fixes 0.2.11's failed tab-URL readiness check without adding permissions.

Version 0.2.13 also recognizes failures that completed before **Start / resume** but were absent from the saved checkpoint because recording had already stopped. Their exact Network-log evidence can receive the same bounded refresh. Unknown completion times, live callbacks, newer failures, and visible challenges still pause immediately. An unresolved older failure still prevents publication after that attempt.
