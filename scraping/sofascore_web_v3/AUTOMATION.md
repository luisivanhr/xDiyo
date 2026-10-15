# Chrome extension + Python collection

## Fresh seasons with extension 0.3.0

The **Discover a season** panel is the entry point for 2025/26 and additional
leagues. See [the discovery workflow](README.md#fresh-season-discovery-030).
Disconnect the previous campaign, then create or reconnect a discovery ID.
Discovery saves its own progress; **Create collection campaign** is enabled only
after all selected regular-season fixture schedules reconcile. It produces a new
campaign ID for the existing collection and publication workflow. Never reuse
the old 40,060-match campaign ID for newly discovered fixtures.

The first prepared 13-league discovery is `discover-2025-2026-eff840b6`.
It was prepared locally with zero provider requests; live enumeration is pending.

## Active component recovery (2026-09-17)

The historical campaign has finished. Its targeted recovery campaign is **`chrome-recovery-c77b81bc`**: 589 matches, 684 component retries, across 35 seasons. Pregame 404s and xG-only absence are excluded; successful saved responses are reused. Resume this ID in the existing xDiyo panel. Current recovery instructions, worker status locations, preservation rules and receipts are in [the recovery run record](../../data/audits/website_capture_v3/recovery_20260917/README.md).

The separate `scraping.sofascore_web_v3.recovery` worker merges usable recovered components into the full published seasons, with staged validation, prior-version preservation and rollback. Do not run the ordinary season publisher on this subset queue; it now rejects that use. A completed retry that again returns 404 remains an unresolved data gap and does not overwrite the prior season.

## Session access without a captcha header (0.2.21)

The collector now accepts the site's successful request pattern when `X-Requested-With` is present and `X-Captcha` is absent. It requires a recent successful first-party response (200 or 304), retains only the observed headers in Chrome memory, and bounds each dispatch by a 120-second local observation window. This window is not a provider token expiry. Successful 200 component responses refresh the observation. If a captcha header is present, the existing expiry and validation rules still apply; malformed or expired tokens are never silently removed.

The session access test reports `accessMode` and a null `tokenRemainingMs` when no token was observed. Denial recovery, request pacing, match identity and component coverage checks are unchanged. Session renewal also accepts known match section fragments such as `,tab:lineups`, strips only that section, and preserves the exact validated host and match ID.

## Recovery rest timing (0.2.20)

An intentional challenge or expiry rest now excludes its measured elapsed duration from the active match budget, including delayed callbacks. It uses the existing Python timer and continues to check pause controls. Requests and page renewal still consume the normal match budget; the configured cooldown, minimum request spacing, and single challenge retry are preserved.

Reports distinguish requested and actual rest duration, Python pacing time, the native round trip, and local progress time. This makes future local stalls diagnosable without extra provider requests. Version 0.2.20 passed 153 JavaScript checks and 68 Python checks, including regressions that failed on overslept rests before the fix. The underlying cause of the earlier local scheduling delays is still unconfirmed.

## Request pacing and Windows status writes (0.2.19)

Reload the extension and reconnect the existing campaign to load this version. Inter-request pacing now uses the Python writer's high-resolution monotonic clock, while all provider requests remain in Chrome. The saved minimum spacing and optional random delay retain their meaning. Local pause checks stay responsive, and challenge/season rests remain unchanged. Each import reports `session_pacing_backend`, `session_pacing_requested_wait_ms`, and `session_pacing_actual_wait_ms`.

The campaign writer and offline export watcher now retry atomic status replacement on transient Windows sharing conflicts. They reuse the same fully written temporary file, preserve the previous status if replacement remains blocked, and surface persistent errors. The interrupted match 12467555 resumed using just its two remaining requests and passed hashes and fixture identity. See [the access report](../../data/audits/website_capture_v3/reliability/SESSION_ACCESS.md).

## Bounded session recovery (0.2.18)

With `SESSION_CHALLENGE_RECOVERY = True`, one HTTP 403 whose reason is exactly `challenge` gets a configurable rest (`SESSION_CHALLENGE_COOLDOWN_SECONDS = 180`), one ordinary renewal of the existing anchor, and one retry of the failed component. Previously saved components are reused. The first denial is retained, and the entire match has only one recovery attempt. A second denial, 401, 429, another 403 reason, failed renewal, or visible human challenge pauses. Set the flag to `False` for immediate pauses on all denials. Session values stay inside Chrome, and neither H2H nor a VPN is used.

Renewal requires a successful observation that began after the page was renewed; old Network entries cannot validate it. If a token is nearly expired, the collector waits until it expires before reloading, because the site's initializer otherwise keeps the same token. These fixed rests check local pause controls without component requests, and their measured elapsed duration is excluded from the match budget; renewal and request work still have the normal time budget. Reports retain request counts, renewal counts, recovery attempts, and wait durations.

The first live automatic recovery succeeded on La Liga match **12437540**: a lineup challenge triggered the 180-second rest and same-tab renewal, after which only lineups and incidents were requested. The export passed its hashes and fixture identity checks, and collection continued without operator input. See [the access and recovery report](../../data/audits/website_capture_v3/reliability/SESSION_ACCESS.md) for the comparison and its limits.

The active queue now uses **0.2.21 session mode**, introduced in 0.2.17. The extension requests only missing queued components through the existing Sofascore tab, using the site's recently successful session headers. Python validates and saves the responses. It keeps one ordinary match page open as its session anchor; it does not navigate through H2H or visit each queued match. H2H and commentary are excluded from these requests. Pregame positions remain selected for historical standings.

The first live session run recovered blocked match **12448325**, then completed **93 consecutive matches** before a shotmap challenge on match 12449872. An ordinary renewal and resume recovered that match using only its five remaining components; 12 more completions finished Eredivisie 2024/25. All **306 queued Eredivisie matches** passed the season export audit. The first La Liga match under 0.2.18 then passed identity and Parquet hash checks using eight requests. See [iteration 28](../../data/audits/website_capture_v3/reliability/session_route_iteration28.json), [iteration 29](../../data/audits/website_capture_v3/reliability/session_route_iteration29.json), and [the Eredivisie audit](../../data/audits/website_capture_v3/reliability/season_37_61666.json). Ordinary anchor-page traffic is additional; these results are not a provider-side total or a guarantee against future 403s.

The notebook defaults to `COLLECTION_MODE = "session"`, `SESSION_REQUEST_INTERVAL_SECONDS = 0.5`, `RANDOM_ACTION_DELAY_MAX_SECONDS = None`, and `MEMORY_RESET_EVERY_MATCHES = 50`. The saved full queue has those settings applied. Normal starts need only **Connect to Python → Start / resume**; do not recreate the queue or reapply policy on every start. Five-minute rests between seasons remain enabled. Session values stay in Chrome; no VPN is used.

The queue reached **112 completed matches** before the September 13 recovery experiments. The stopped Sofascore tab later displayed **Out of Memory**; reopening the tab and DevTools alone did not recover three missing map responses. Version **0.2.4** adds one ordinary same-tab refresh with recording active when the first section pass leaves requested data missing. Both passes share the original limits, and unresolved gaps still pause. Current iteration counts, timings and stop causes are in [the recovery report](../../data/audits/website_capture_v3/reliability/REPORT.md); that report is updated from read-only queue observations. Completed cohorts can now be published offline under their original season filenames, as described below.

Use [collect_chrome_campaign_v3.ipynb](../Scrape+wait/collect_chrome_campaign_v3.ipynb). Its default queue is now **`chrome-from-queue-bc6ae7b3`**, prepared directly from the original scraper's `data/xDiyo_data/_collection/inventory.json`: **40,060 fixture IDs across 109 valid seasons**, in the saved season/ID order. No new fixture-discovery requests were needed.

The source queue includes one additional season, `Eredivisie_16_17`, marked invalid because its filename and stored identity disagree. It is explicitly excluded and reported, not silently mapped to another season.

The original three-match pilot **`chrome-pilot-16d25576`** remains available:

| Event | Fixture | Season |
|---|---|---|
| 12499375 | Borussia M'gladbach – Bayer 04 Leverkusen | Bundesliga 2024/25 |
| 12499386 | RB Leipzig – VfL Bochum 1848 | Bundesliga 2024/25 |
| 12499406 | TSG Hoffenheim – Holstein Kiel | Bundesliga 2024/25 |

The native host `com.xdiyo.sofascore` is registered in the current user's Chrome NativeMessagingHosts key, bound only to `chrome-extension://bhkjpmoblllmoheemejdcemnhmdcdnbg/`. No local HTTP server is used.

## Start

1. Reload **xDiyo Network Capture** at `chrome://extensions`, then close/reopen the Sofascore tab's DevTools.
2. In the **xDiyo** panel, check **Version 0.2.21**, enter `chrome-from-queue-bc6ae7b3` (or the small pilot ID), and select **Connect to Python**. It must report **Python 0.2.21 connected**.
3. Select **Start / resume**. Keep DevTools open and leave that tab available to the collector. Other tabs can be used normally.

The controller lives in the DevTools page, so changing DevTools panels does not destroy it. Closing DevTools closes its one persistent native port; Python pauses the active match, retaining acknowledged responses. Reloading extension source also requires reopening DevTools.

The cumulative observation limit is disabled by default (`MAX_API_OBSERVATIONS_PER_RUN = None` in the notebook, `null` in saved JSON). The prepared full queue `chrome-from-queue-bc6ae7b3` also has this limit disabled. An existing Python connection keeps its loaded policy: close/reopen DevTools, connect the same campaign ID, and select **Start / resume** to apply the change. The per-match limit applies to selected observations. The recorder retains a separate 5,000-entry memory guard for all metadata; its existing body/memory limits remain. Match timeouts, manual pauses, and requested-data provider pauses remain active. A positive integer can still set an optional cumulative limit for a new queue.

In **page mode**, you can start from any page on `www.sofascore.com` in the inspected tab. **Start / resume** uses the queued match ID and URL, then searches up to six pages of the visible H2H history if the site opens a newer fixture. It opens the missing lineup/statistics/heatmap sections automatically. On a resumed provider failure, if a newer fixture for the same team pair is already loaded, version 0.2.15 introduced an attempt to use that page's H2H links before another document navigation. This first search is limited to six pages. If it finds no link, one ordinary navigation follows without repeating the same search; the original match deadline still applies. A different team pair uses the usual queued navigation first. The report records `historical_link_recovery` when this route succeeds. When the first resumed match has saved 401/403/429 responses, that click grants one bounded navigation/history/view attempt. Those exact saved failures, plus selected Network-log failures with known completion times at or before the Start click, are deferred during the first resumed match only; a new denial on a requested endpoint or a visible challenge stops immediately. If the correct match is already displayed, newer responses in its Network log are read first; one ordinary refresh is allowed if a saved failure remains. There is no refresh per endpoint or progress check. At the end of the view pass, an unresolved saved denial still pauses and prevents publication. Old response evidence is retained, and only newer observed responses supersede it. Reload extension **0.2.21** and reopen DevTools to activate this behavior. Connect the same campaign ID and select **Start / resume**; finding the match manually is not required.

## Periodic memory renewal

In **session mode**, the same-tab reset returns to the existing anchor page. It occurs between saved matches, after the configured completion count or when the session has insufficient validity for another match. The bounded challenge recovery described above may also renew that page during a partially saved match. A visible challenge, unusable renewed session, or denial outside that one permitted recovery pauses with acknowledged responses saved. The next resume requests only remaining components. The sections below describing H2H, fixture navigation, view waits, and passive observation limits apply to **page mode**.

In page mode, version **0.2.15** improves an explicit resume after a saved provider failure: if Sofascore has left a newer fixture for the same teams open, the collector first searches that page's real H2H links for the queued historical match. It keeps recording during this bounded search and falls back to one normal navigation if the link is absent. Fresh requested-data denials still pause immediately; this is not a guarantee against 403 responses.

The indexed local progress checks and periodic page renewal remain enabled. In the notebook, set `MEMORY_RESET_EVERY_MATCHES = 50` (or `None` to disable). After that many saved matches, the extension briefly opens its local reset page in the same tab, keeps DevTools and the writer connected, starts recording, and opens the next queued fixture. No completed match is revisited for the reset. A failed reset pauses with the checkpoint intact. This is not an exact RAM ceiling: Chrome decides when memory is reclaimed, and reopening DevTools restarts the counter.

For an existing queue, close DevTools and set `APPLY_SAVED_POLICY = True` for the prepare/resume cell once, then restore it to `False`. The helper locks the collector, backs up configuration, and changes only policy. Reload the extension, reopen DevTools, and reconnect the same campaign. Saved matches, raw responses, feature flags, cookies, and browser cache are preserved.

The same-tab `location.replace` fallback introduced in 0.2.10 remains enabled. Iteration 12 completed 132 more matches with sampled main heap around 99-352 MB; earlier runs reached about 7 GB near 100-120 matches. This is a measured improvement in that run, not proof that all future memory growth is eliminated. See the [recovery report](../../data/audits/website_capture_v3/reliability/REPORT.md) for live reset validation and later measurements.

## What changed

- **Awarded fixtures (0.2.5):** a fresh event response must explicitly set `isAwarded=true`, and a recovery refresh must confirm that the requested website tabs are absent while other match tabs remain available. Only otherwise-unobserved statistics/shotmap/heatmaps/lineups can be deferred. The queue records `deferred`, retains the report with `complete=false` and the original `not_observed` statuses, and continues. Deferrals remain outside the completed count and require later review; a finished queue with deferrals is `finished_with_gaps`. Missing event/pregame/incidents, failed bodies, and provider denials still pause. The original API scraper and shared normalizer are unchanged.
- **Recovery of preloaded data:** an incomplete section pass can refresh the current fixture once while recording remains active. This is useful after DevTools is reopened or page data is already cached in the application. A fixture that already used its saved-denial recovery refresh cannot refresh again. Complete or explicitly unavailable data never triggers this extra refresh. Requested-endpoint denials, wrong identities, visible challenges and deadlines still stop the run.
- **Automatic local writes:** each response is validated, stripped of cookies/headers/query values, inserted into SQLite, and acknowledged. The controller waits for acknowledgements before advancing. HAR export is still available for manual recordings.
- **A resumable queue:** IDs and season order now come from the other scraper's saved inventory. Its CSV identity columns supply team IDs/slugs, matchup IDs and season/round identity; its cache supplies exact provider routes for 615 fixtures. The remaining 39,445 routes are constructed from saved team slugs and matchup IDs and must pass the live page/new-response identity checks. Scores, timestamps, match status and components are not imported from legacy routing metadata. Completed matches are skipped on resume. An interrupted match keeps its observations and receives a new lease, rejecting stale writes.
- **One existing tab:** the controller first uses visible fixture links, otherwise navigates the same tab once. It opens only the missing lineup/statistics/heatmap views. Historical fixtures that resolve to a future match get a bounded H2H search through visible site controls. Failure to verify the requested ID pauses the run.
- **Cached content:** HTTP 304 remains labelled as 304. Its body is accepted only when DevTools supplies readable cached content. No endpoint is replayed. Initial backfill uses only the latest selected-match response for each requested path, excluding unrelated old logs.
- **Honest completion:** raw response status and feature coverage are separate. `complete` means selected endpoint accounting passed, including explicitly unavailable 404s. `partial` means core accounting passed but some optional pregame coverage is missing. `needs_attention` pauses on missing core data by default. A wrong match, requested-endpoint 401/403/429, visible challenge, time limit, or selected-observation limit pauses future actions. Errors from unselected endpoints remain diagnostic metadata and do not stop the queue.
- **Pacing and flags:** xG, heatmaps, statistics, lineup export, reported absences, goal actions, and pregame data remain independently configurable. Commentary is disabled. Lineup response capture is still necessary when absence export is enabled. Random action delay defaults to `None`, with uniform `[0,T]` when configured. The website controls its individual HTTP requests; this is an action delay, not a per-request delay. Default rest between queued seasons is 300 seconds. `VIEW_WAIT_SECONDS` is an upper bound: navigation/section waits end early once the matching data is saved and the relevant controls are rendered. Jitter and season rests keep their full duration; H2H pager changes retain their bounded settling wait.
- **Versioned handshake:** the extension and writer reject mismatched protocol or application versions. Only one collector holds the shared OS lock. Native status messages show at most 100 active/recent fixtures, while counts cover the entire queue, keeping large inventories below Chrome's message-size limit. The native interface exposes local queue operations, not arbitrary commands or filesystem destinations.

## Measured collection pace

The 30 additional matches completed between 23:46:03 and 23:54:17 JST on 12 September. Most completion intervals were about 18 seconds. Selected-response timestamps show data arriving before the fixed waits ended. The saved observations contained 377 selected and 2,705 unselected entries (about 88% unselected); this is recorder evidence, not an exact provider request count.

Version 0.2.3 removed the unrelated-endpoint pauses and fixed waits. It subsequently reached 112 completions, with a timeout and missing-map stop along the way. Those measurements describe the early page-mode pilot. Later session-mode iterations 28 and 29 sustained median completion intervals near eight seconds and completed the remaining Eredivisie fixtures. Version 0.2.18 later completed La Liga and recovered two selected challenges automatically. Version 0.2.19 later added 3,921 matches before a recovery-rest timeout. Version 0.2.20 addresses that timing gap; see the recovery report for current throughput, interruptions, and feature checks.

## Verified feature coverage

Evidence comes from the frozen real capture for match **12499375**, exported at **2026-09-12 03:03:20 UTC**. The 29 saved responses span seven unique selected paths. Combining those responses with an earlier observed pregame-form 404 completes endpoint accounting, while pregame remains unavailable.

| Requested information | Observed result | Remaining limitation |
|---|---|---|
| xG and shots | All 39 shots have xG; source xGOT is preserved where supplied | Team totals 1.58/2.70 differ from shot sums 1.716505/3.310107. Both values are retained; the cause is unknown. |
| Team heatmaps | 1,635 points across both teams, including goalkeeper points | These are team heatmaps, not a verified full set of individual player heatmaps. |
| Lineups | 40 distinct players across two observations | Snapshot rows are intentionally separate; 80 rows do not mean 80 players. |
| Injuries / absences | Three distinct reported missing players | Provider reason codes are preserved without inventing an injury/suspension classification or medical history. |
| Goal passing actions | Five goal sequences, 10 actions, including two passes | No explicit pass recipient IDs were supplied. Sequence order does not prove a directed pass network. |
| Match and half duration | Added-time announcements of four and ten minutes | Exact elapsed duration is absent. HT/FT `addedTime=999` is a sentinel, not an elapsed-time value. |
| Pregame positions | Observed 404 in the earlier capture | Not supplied for this fixture. A currently displayed league position is not a historical pregame position. |
| Normalized standings | Not established by this pilot | Existing `utils/data_loading.py` divides each side's standing by its maximum observed standing across the loaded season. Raw positions plus that complete-season denominator are needed for parity; a single-match import cannot establish it. |
| Promotion / relegation flags | Separate season-membership processing | These are derived locally after verified season collection. The other thread owns the legacy restoration/processing work. |
| Commentary | Not retained | The site may still request it; passive capture cannot prevent that traffic. |

On 12 September, the historical match page's About text displayed **current** positions while the header showed the 2024 fixture. That text is not used to fill historical standings.

Machine-readable coverage: `data/audits/website_capture_v3/feature_coverage.json`. Frozen source hashes and exact source-to-Parquet checks: `data/audits/website_capture_v3/har_capture_validation.json` and `validate_successful_har.py`.

## Storage and verification

Collection evidence is below `data/website_capture_v3/campaigns/<campaign-id>/`: configuration, a WAL SQLite queue, pause marker, response observations, per-match HAR archives, and isolated imports with compressed source JSON plus typed Parquet tables. Each import includes `report.json`, `feature_coverage.json`, and a hashed table manifest. The optional offline publisher assembles complete queue cohorts into `data/xDiyo_data` while retaining these original captures.

### Publish a completed season

The notebook exposes `PUBLISH_COMPLETED_SEASONS = False` and `SEASON_EXPORT_STEMS = None`. Enable publication and run its **Publish completed seasons** cell after collection has completed a season. `None` selects all fully completed cohorts; a list such as `["Eredivisie_24_25"]` selects named seasons. This local operation can run while Chrome collects another season. It makes no network requests and does not change the queue or collection policy.

The equivalent command is:

```powershell
.venv/Scripts/python.exe -B -m scraping.sofascore_web_v3.publish --campaign data/website_capture_v3/campaigns/chrome-from-queue-bc6ae7b3 --stem Eredivisie_24_25
```

Each output retains the original stem, for example `data/xDiyo_data/Eredivisie_24_25.parquet`, with one match per row and typed nested feature columns. The adjacent manifest points to the lossless relational tables under `_tables`; `_collection/chrome_exports` records the exact source reports, manifests, feature coverage, and hashes. Publication verifies the complete saved cohort, event/team/season identities, selected component accounting, every source table hash, and the published table round trip. Pending, running, paused, or deferred entries prevent publication. Matching existing outputs are verified and reused; conflicting or damaged outputs are rejected rather than replaced.

**Sixteen season cohorts** have been published and independently checked, covering **5,767 matches**. [The publication receipt](../../data/audits/website_capture_v3/reliability/published_seasons.json) lists every season file and records its hashes, table counts, repeat-publication checks, actual legacy-loader standings parity, and verification time. Raw positions remain raw in storage; the existing loader derives normalized standings over the loaded season. Missing source values stay missing. Promotion/relegation processing remains separate. The Bundesliga Chrome cohort still has one deferred awarded fixture, and its pre-existing season output is preserved.

The publisher and export watcher passed 12 targeted offline tests, including refusal of incomplete cohorts, conflicts, source corruption, incorrect identities, and missing component accounting. All eight notebook code cells executed with collection and publication disabled; [the executed copy](../../data/audits/website_capture_v3/reliability/publication_notebook_executed.ipynb) is retained. Publication establishes completeness against the saved queue cohort; it does not rediscover or prove the provider's full fixture inventory.

### Export automatically as seasons finish

Run the offline watcher from a terminal at the repository root:

```powershell
.venv/Scripts/python.exe -B -m scraping.sofascore_web_v3.publish_watch --campaign data/website_capture_v3/campaigns/chrome-from-queue-bc6ae7b3 --watch --interval-seconds 60
```

It checks the local queue once a minute, verifies each already-published complete season once at startup, and publishes newly completed cohorts with the same exporter. It does not send provider requests, navigate Chrome, or alter the collection queue. Only one export watcher can hold its separate lock. It stops on a conflicting output or changed campaign configuration, preserving the collector's progress. Ctrl+C stops only the export watcher. Omit `--watch` for one pass over currently complete cohorts.

Status is saved to `data/xDiyo_data/_collection/chrome_publication_watch.json`, and publication events are appended to `chrome_publication_events.jsonl` in the same directory. Each entry records when that season was verified; finished files are not repeatedly rehashed during idle polls. A final queue with deferred fixtures is reported as `finished_with_gaps`; an incomplete cohort is never published. The tests cover a fixture finishing between queue reads so the last season is not missed.

The watcher was started for the live campaign at 14:29 UTC on 13 September and reverified all four existing Chrome season files. The separate [season audit watcher](../../data/audits/website_capture_v3/reliability/watch_completed_season_audits.py) subsequently automated the full-season feature, standings, and published-file loader checks. La Liga 2 finished at 15:33:53 UTC, was published at 15:35:18 UTC, and passed all three independent checks at 15:36:26 UTC. Ligue 1 subsequently finished at 16:04:28 UTC, was published at 16:04:48 UTC, and passed the same three checks at 16:05:56 UTC. Ligue 2 finished at 16:35:12 UTC, was published at 16:36:20 UTC, and passed all three checks at 16:37:24 UTC. The audit watcher reads only local files and the queue; a change to its configuration, audited outputs, or audit/loader sources stops that watcher independently of collection. Its [status](../../data/audits/website_capture_v3/reliability/season_audit_watch.json) records verified seasons and any error.

Source responses are durable after the Python acknowledgement. A crashed or closed browser cannot prove that unacknowledged responses were saved. Repeated exact observations are deduplicated, while distinct lineup observation times remain separate. Observation counts cover the configured Sofascore API hosts during recording, not all browser resources or provider-side request counts. The website may keep polling after the extension pauses.

Offline checks cover transport framing, Unicode and partial reads, cached bodies, source sanitization, wrong identities, stale leases, disconnect/resume, writer failure, source omissions, optional fields, and Parquet round trips. The compiled Windows launcher also passed two framed round trips while its input pipe remained open. This caught and fixed a pipe-buffering issue that a test ending stdin would have missed.

```powershell
.venv/Scripts/python.exe -m scraping.sofascore_web_v3.check_native_host
.venv/Scripts/python.exe -m pytest tests/scraping/test_chrome_queue_source.py tests/scraping/test_chrome_campaign.py tests/scraping/test_network_archive.py tests/scraping/test_website_capture_v3.py -q
node --test tests/scraping/network_capture.test.mjs tests/scraping/chrome_campaign.test.mjs tests/scraping/session_probe.test.mjs tests/scraping/session_collection.test.mjs
```

The queue-source adapter additionally checks season selection, source CSV hashes and ID parity, cached identity conflicts, source-data isolation, mandatory fresh event metadata, and bounded status messages for large queues. Campaign tests also cover unlimited cumulative observations, optional numeric limits, and manual/provider pauses when the cumulative limit is disabled.

Queue provenance and counts: `data/audits/website_capture_v3/prepared_chrome_source_queue.json`. The original queue, stop markers, cache and data files are read only by this adapter. To prepare a different selection, use `QUEUE_SOURCE = "existing_scraper"`, optional `SOURCE_SEASON_STEMS`, optional `SOURCE_MAX_MATCHES`, and `CREATE_NEW_CAMPAIGN = True` once. Manual `EVENT_IDS` remain supported with `QUEUE_SOURCE = "manual"`.

The earlier 14-cell notebook validation is retained at `data/audits/website_capture_v3/collect_chrome_campaign_v3.executed.ipynb`. The current notebook has 16 cells, including eight code cells, and its latest executed validation is linked in the publication section. Browser collection requires the extension reload and panel start above only when installing a changed extension.

## Retry a deferred fixture after ordinary collection

The offline `retry_deferred` command reviews awarded fixtures that were deferred because the earlier page collector never observed selected views. Its default is a read-only preview, which can be run during collection:

```powershell
.venv/Scripts/python.exe -B -m scraping.sofascore_web_v3.retry_deferred --campaign-id chrome-from-queue-bc6ae7b3 --event-id 12499586
```

Once all ordinary queue work has finished, disconnect the collector's Python writer and run the same command with `--apply`. The command takes the existing collector lock and refuses to apply while any ordinary fixture is pending, running, paused, or needs attention. It verifies the selected deferred report, profile, table hashes, and fixture identities, preserves the old report and observations, and returns that queue row to pending. Reconnect the existing writer and use **Start / resume**. Chrome then re-evaluates the saved component outcomes and requests what remains missing. Requeueing itself does not complete the fixture or supply missing features. Repeat `--event-id` to select more than one eligible deferred fixture in one transaction.

A retry receipt under the campaign's `retries` directory records the previous row and source hashes. A `prepared` receipt alone does not establish that the queue update committed. If the command explicitly reports that the queue was requeued but its receipt update failed, inspect the queue and resume that pending fixture rather than applying again.

## Preserve an older output before Chrome publication

The separate offline `reconcile_publication` command handles a completed Chrome cohort whose original season filename already belongs to an older direct export. Its preview validates the old nested Parquet, relational table hashes and row counts, fixture identities, manifest, and `CURRENT.json` pointer without changing the output or taking the live collector lock:

```powershell
.venv/Scripts/python.exe -B -m scraping.sofascore_web_v3.reconcile_publication --campaign data/website_capture_v3/campaigns/chrome-from-queue-bc6ae7b3 --stem Bundesliga_24_25
```

After the **entire queue is complete, including deferred fixtures**, disconnect the collector, let the publisher finish, and keep dataset readers idle for the switch. Run the preview command with `--apply --expected-existing-sha256` followed by the old season hash from that preview. The command stages and verifies the complete replacement, copies and verifies a self-contained prior bundle under `_collection/chrome_reconciliations/<stem>/<run>/prior`, and preserves the original relational versions and capture stores. It then installs the new version and switches the nested Parquet, adjacent manifest, and `CURRENT.json`. The old direct profile is preserved in its backup; old feature values are not mixed into the Chrome capture.

Each entry file is replaced atomically, but the three-file switch is not a filesystem transaction. A failed switch restores the old entry files. If a process interruption or persistent file lock prevents completion, both bundles and a durable `receipt.json` remain. Run `--recover-receipt <receipt-path>` with the same `--output` if using a nondefault directory: recovery validates both bundles, finalizes a fully installed replacement, or restores the prior bundle. It refuses unexplained edits. Repeating a successful reconciliation verifies and reuses the existing Chrome publication.

The [18 offline tests](../../tests/scraping/test_publication_reconciliation.py) passed, including active locks, unfinished work, changed hashes, path containment, preservation, rollback, idempotence, and recovery after a failed receipt update. The [real Bundesliga preview](../../data/audits/website_capture_v3/reliability/publication_reconciliation_preview_iteration31.json) verified all 19 files of the 306-match prior bundle and identified fixture 12499586 as still deferred. The live output has **not** been reconciled. Its existing Parquet hash remains `5dea1b2daefd56a797c0c70f9e1329d26dd1bb0f669faa9b2d28fc5850409978`.

The helper passed **14 offline tests**, including reuse of five saved components followed by three new unavailable outcomes, active-writer exclusion, unfinished-queue refusal, corrupt-capture rejection, and rollback/receipt failures. A [live read-only preview](../../data/audits/website_capture_v3/reliability/deferred_retry_validation_iteration31.json) confirms three unobserved paths for Bundesliga fixture 12499586, with its deferred row unchanged. The new helper has not been applied to the live campaign. Its eventual Chrome export must be reconciled with the existing Bundesliga file's different provenance before publication; the helper preserves that existing output.

## Remaining campaign work

1. Continue the saved queue and measure session renewal, challenge recovery, throughput, and memory across seasons. Completed-season audits exist; collection of the whole inventory remains unfinished.
2. Compare observed traffic with the same requested data from the original scraper. Session mode removes H2H traversal and repeated per-match page loads, but ordinary anchor traffic remains additional to the saved component counts.
3. Reconcile the reused fixture inventory against season membership and any missing/postponed fixtures. The existing 40,060 valid IDs already remove the need to rediscover those fixtures. Resolve the excluded season and deferred awarded fixture separately. Publish further completed cohorts with the offline publisher, retaining raw captures and source gaps.
4. Continue complete-season pregame coverage and legacy normalized-standing parity checks; sixteen completed cohorts have passed. Guard incomplete denominators and unsupported competition stages. Join verified promotion/relegation flags after season assembly.
5. If the extension route remains unreliable, assess a Playwright/MCP controller against the same coverage contract; changing the controller cannot create data that the site does not supply.
6. **Deferred extension work:** integrate fresh-season discovery for 2025/26 across the same 13 leagues, then incremental discovery and retrieval of the latest season fixtures for daily predictions. Coordinate queue changes with the Python writer and both conversion watchers. The detailed scope and proposed handling of pre-match observations versus later results are in [Phase 5 of the collection plan](../../planning.md#phase-5-make-discovery-complete-and-incremental). This is planned work, not a capability of the current extension.
7. **Deferred VPS design:** make the collection/discovery logic reusable by a future Hostinger prediction service, without requiring the server to run this extension. Evaluate a server browser controller and target-host access, preserve checkpoints and output parity, and define unattended recovery and data-freshness behavior before migration. See the [Hostinger VPS collection plan](../../planning.md#deferred-hostinger-vps-collection); no server setup or deployment is part of this planning update.

References: [Chrome Native Messaging](https://developer.chrome.com/docs/extensions/develop/concepts/native-messaging), [DevTools Network API](https://developer.chrome.com/docs/extensions/reference/api/devtools/network), [DevTools extension lifecycle](https://developer.chrome.com/docs/extensions/how-to/devtools/extend-devtools), [Scripting API](https://developer.chrome.com/docs/extensions/reference/api/scripting).

The 0.2.12 reset page reports readiness through an internal extension message, checked against the extension origin, tab ID, and per-reset nonce. This fixes 0.2.11's failed tab-URL readiness check without adding permissions.

Version 0.2.13 also recognizes failures that completed before **Start / resume** but were absent from the saved checkpoint because recording had already stopped. Their exact Network-log evidence can receive the same bounded refresh. Unknown completion times, live callbacks, newer failures, and visible challenges still pause immediately. An unresolved older failure still prevents publication after that attempt.


## Chrome session collection

Version **0.2.17** adds `collection_mode="session"`: queued component GETs run sequentially in the existing Sofascore tab using its recently successful site-issued session headers. Only missing components are requested, bodies are acknowledged directly by Python, and session values stay in Chrome. It does not navigate through H2H or open each queued match. The first live metadata test recovered HTTP 200 for blocked match **12448325** in 240 ms on 13 September; sustained collection is still being measured. The original passive collector remains available as `collection_mode="page"`. Python still sends no HTTP requests. No VPN is used.

Set `COLLECTION_MODE = "session"` and `SESSION_REQUEST_INTERVAL_SECONDS = 0.5` in the notebook. The interval is a minimum between request starts, across match boundaries. `RANDOM_ACTION_DELAY_MAX_SECONDS = None` disables jitter; a value T adds a random [0,T] wait before each component request in session mode. Five-minute season rests remain. The saved queue configuration is authoritative: disconnect the writer and apply the policy once with `APPLY_SAVED_POLICY = True`. This changes no fixture IDs, feature flags, captured data, or completed results.

Session mode keeps an ordinary Sofascore match page as its session anchor. It renews that same page between saved matches when the configured memory interval is reached, or when the session has insufficient validity for a match. The request loop never uses H2H. A renewal that needs interaction pauses. Optional bounded recovery for a 403 challenge is described above; other denials pause immediately. Python validates captured metadata against the queued event, teams, competition and season, then requires all selected endpoints to be accounted for before completing the match. The report records `collection_mode="session"`, `identity_source="captured_event_metadata"`, the explicit request count, and renewal evidence. 404 means unavailable data; it never fabricates a populated feature. Pregame positions remain distinct from H2H data.

## Session access experiment

Version **0.2.16** adds an explicit **Session access test** for one match metadata request. It uses fresh, recently successful X-Captcha and X-Requested-With headers from the existing Sofascore tab, keeps those values inside Chrome, and returns only redacted status and match-identity evidence. The test does not navigate, open H2H, change the campaign queue, retry a denial, or use a VPN. The one-request experiment is separate from queue collection. Its successful result is saved in `data/audits/website_capture_v3/reliability/session_probe_iteration28.json`.

Pause collection, keep an ordinary Sofascore page open, enter the target ID in **Session access test**, and select **Test session access**. A recently successful API observation and at least 30 seconds of token validity are required; otherwise it sends no request. It validates the returned event ID and displays the provider response. **Export test result** saves diagnostics without session headers or cookies. It never marks a match collected.

The older H2H-first recovery uses exact route matching and can miss canonical URL aliases. It is not being extended while session access is investigated.
