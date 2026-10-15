# xDiyo Network Capture 0.3.0

## Season discovery

**Discover a season** resolves 2025/26 (or another explicit season), saves its
regular-season fixture inventory, validates team pairings, and creates a new
campaign for the existing collector. The 13 saved leagues are selectable and
additional single-table leagues can be added. See the [discovery workflow](../README.md#fresh-season-discovery-030)
and the updated notebook for resume IDs, optional lineups, random pacing and
the conversion watcher. Reload the extension and reopen DevTools to load 0.3.0.
The same inspected Sofascore tab is reused; the first live discovery run remains
to be validated. Historical collection and recovery campaign IDs are unchanged.

## Session access without a captcha header (0.2.21)

The collector now accepts the site's successful request pattern when `X-Requested-With` is present and `X-Captcha` is absent. It requires a recent successful first-party response (200 or 304), retains only the observed headers in Chrome memory, and bounds each dispatch by a 120-second local observation window. This window is not a provider token expiry. Successful 200 component responses refresh the observation. If a captcha header is present, the existing expiry and validation rules still apply; malformed or expired tokens are never silently removed.

The session access test reports `accessMode` and a null `tokenRemainingMs` when no token was observed. Denial recovery, request pacing, match identity and component coverage checks are unchanged. Session renewal also accepts known match section fragments such as `,tab:lineups`, strips only that section, and preserves the exact validated host and match ID.

## Recovery rest timing (0.2.20)

An intentional challenge or expiry rest now excludes its measured elapsed duration from the active match budget, including delayed callbacks. It uses the existing Python timer and continues to check pause controls. Requests and page renewal still consume the normal match budget; the configured cooldown, minimum request spacing, and single challenge retry are preserved.

Reports distinguish requested and actual rest duration, Python pacing time, the native round trip, and local progress time. This makes future local stalls diagnosable without extra provider requests. Version 0.2.20 passed 153 JavaScript checks and 68 Python checks, including regressions that failed on overslept rests before the fix. The underlying cause of the earlier local scheduling delays is still unconfirmed.

Version 0.2.19 moves only inter-request pacing to the existing Python connection, using a high-resolution monotonic clock and checking local pause controls. Minimum spacing, optional jitter, and challenge/season rests retain their configured values. Reports record the pacing backend and wait totals. Python status writes retry transient Windows sharing conflicts. Reload the extension and close/reopen DevTools so both sides use this version.

## Bounded session recovery (0.2.18)

With `SESSION_CHALLENGE_RECOVERY = True`, one HTTP 403 whose reason is exactly `challenge` gets a configurable rest (`SESSION_CHALLENGE_COOLDOWN_SECONDS = 180`), one ordinary renewal of the existing anchor, and one retry of the failed component. Previously saved components are reused. The first denial is retained, and the entire match has only one recovery attempt. A second denial, 401, 429, another 403 reason, failed renewal, or visible human challenge pauses. Set the flag to `False` for immediate pauses on all denials. Session values stay inside Chrome, and neither H2H nor a VPN is used.

Renewal requires a successful observation that began after the page was renewed; old Network entries cannot validate it. If a token is nearly expired, the collector waits until it expires before reloading, because the site's initializer otherwise keeps the same token. These fixed rests check local pause controls without component requests, and their measured elapsed duration is excluded from the match budget; renewal and request work still have the normal time budget. Reports retain request counts, renewal counts, recovery attempts, and wait durations.

Version **0.2.17** adds `collection_mode="session"`: queued component GETs run sequentially in the existing Sofascore tab using its recently successful site-issued session headers. Only missing components are requested, bodies are acknowledged directly by Python, and session values stay in Chrome. It does not navigate through H2H or open each queued match. The first live metadata test recovered HTTP 200 for blocked match **12448325** in 240 ms on 13 September; sustained collection is still being measured. The original passive collector remains available as `collection_mode="page"`. Python still sends no HTTP requests. No VPN is used.

Set `COLLECTION_MODE = "session"` and `SESSION_REQUEST_INTERVAL_SECONDS = 0.5` in the notebook. The interval is a minimum between request starts, across match boundaries. `RANDOM_ACTION_DELAY_MAX_SECONDS = None` disables jitter; a value T adds a random [0,T] wait before each component request in session mode. Five-minute season rests remain. The saved queue configuration is authoritative: disconnect the writer and apply the policy once with `APPLY_SAVED_POLICY = True`. This changes no fixture IDs, feature flags, captured data, or completed results.

Session mode keeps an ordinary Sofascore match page as its session anchor. It renews that same page between saved matches when the configured memory interval is reached, or when the session has insufficient validity for a match. The request loop never uses H2H. A renewal that needs interaction pauses. Optional bounded recovery for a 403 challenge is described above; other denials pause immediately. Python validates captured metadata against the queued event, teams, competition and season, then requires all selected endpoints to be accounted for before completing the match. The report records `collection_mode="session"`, `identity_source="captured_event_metadata"`, the explicit request count, and renewal evidence. 404 means unavailable data; it never fabricates a populated feature. Pregame positions remain distinct from H2H data.

In **page mode**, the indexed local progress checks and periodic page renewal remain enabled. In the notebook, set `MEMORY_RESET_EVERY_MATCHES = 50` (or `None` to disable). After that many saved matches, the extension briefly opens its local reset page in the same tab, keeps DevTools and the writer connected, starts recording, and opens the next queued fixture. No completed match is revisited for the reset. A failed reset pauses with the checkpoint intact. This is not an exact RAM ceiling: Chrome decides when memory is reclaimed, and reopening DevTools restarts the counter.

For an existing queue, close DevTools and set `APPLY_SAVED_POLICY = True` for the prepare/resume cell once, then restore it to `False`. The helper locks the collector, backs up configuration, and changes only policy. Reload the extension, reopen DevTools, and reconnect the same campaign. Saved matches, raw responses, feature flags, cookies, and browser cache are preserved.

In **page mode**, the same-tab `location.replace` fallback introduced in 0.2.10 remains enabled. Iteration 12 completed 132 more matches with sampled main heap around 99-352 MB; earlier runs reached about 7 GB near 100-120 matches. This is a measured improvement in that run, not proof that all future memory growth is eliminated. See the [recovery report](../../../data/audits/website_capture_v3/reliability/REPORT.md) for live reset validation and later measurements.

In **page mode**, the main navigation is identified from its H2H tab, so commentary and nested filters cannot confuse active-view readiness. Read-only page checks wait at most five seconds for Chrome replacing frame 0 during navigation. Persistent failures retain the operation name, and cancellation checks run before each retry. A link whose frame disappears is accepted only if a subsequent read verifies the queued match, without replaying the click. Reports retain evidence from the main match tabs. An awarded fixture can be deferred when fresh provider metadata confirms its awarded status and a bounded refresh confirms the requested website views remain absent. Its report stays incomplete, and it is excluded from the completed count.

Version 0.2.4 gives an incomplete match one ordinary refresh with recording active. This recovers responses that were loaded before recording began. It retains the original deadline, requires the correct fixture, stops on fresh denials, and leaves unresolved data missing. Recovery iterations are measured in `data/audits/website_capture_v3/reliability/REPORT.md`.

A DevTools recorder for the user's existing Sofascore Chrome session. It reads response bodies already obtained by the website, including available HTTP 304 cached content. Manual HAR recording remains available; automatic collection reuses the same inspected tab and saves through a persistent native connection to Python.

Use [AUTOMATION.md](../AUTOMATION.md) and [collect_chrome_campaign_v3.ipynb](../../Scrape+wait/collect_chrome_campaign_v3.ipynb). The notebook now defaults to `chrome-from-queue-bc6ae7b3`, reusing the original scraper's 40,060 valid fixture IDs. The small pilot `chrome-pilot-16d25576` remains available. Reload the extension at `chrome://extensions`, close/reopen DevTools, check Version 0.2.21, enter that ID, select **Connect to Python**, then **Start / resume**.

## Access and lifecycle

The extension uses `nativeMessaging` and `scripting`, restricted to `https://www.sofascore.com/*`. Session collection makes first-party component GETs inside the existing tab. Page mode remains passive. Neither uses debugger permission, a background worker, cookie changes, or a new browser window. The separate Session access test makes one explicit first-party metadata GET in the existing page and requires collection to be paused. Ordinary site navigation can cause website requests. The content-security policy forbids extension-origin network connections; native IPC is local.

The DevTools page owns collection, so switching panels does not discard it. Closing DevTools disconnects the writer and pauses an active match; acknowledged campaign responses survive. Manual recordings still need **Export HAR** before closing DevTools. Save a manual recording before reloading this extension.

The Python host is bound to `chrome-extension://bhkjpmoblllmoheemejdcemnhmdcdnbg/`. Preparation and registration are provided by `install_native_host.py`; the notebook does not silently register or start anything with its default settings.

## Recording scope

- Exact full-match GET paths for the selected event: metadata, pregame form, statistics, shotmap, team heatmaps, lineups, and incidents, filtered by the saved profile.
- No commentary bodies, other-match bodies, query values, cookies, or authorization headers. Python independently reapplies these restrictions.
- HTTP 304 bodies are saved only if DevTools supplies them. Missing, oversized, timed-out, or invalid content stays explicit.
- Campaign startup backfills only the latest selected-match response per requested path when already on that fixture. Live callbacks are deduplicated by URL, timing, method, status, and size. Distinct live observations remain separate.
- A missing view does not prove source unavailability. An observed 404 does; feature coverage still records the omitted values. Exact duration and explicit pass recipients are not inferred from incomplete evidence.
- Manual limits remain 5,000 API entries, 8 MiB per body, and 100 MiB response text. Campaigns count selected observations for their per-match cap, with a separate 5,000-entry guard for all metadata and a match timeout; the cumulative connection cap defaults to `None` (unlimited). These limit recording/actions, not the site's individual requests.
- A visible challenge, 401/403/429 on requested data, wrong match, missing core data, or a budget stops further campaign actions. Unselected endpoint errors are retained as diagnostics and do not stop collection. The site may still issue background requests until its page is closed or left.

## Manual recording

Enter the selected match ID, select **Start recording**, optionally **Read existing Network log**, browse the desired match sections, then **Export HAR**. Import that file using `HAR_FILES` in the exploration notebook. The status shows saved-body counts and observed match IDs; a nonempty archive alone does not establish coverage.

## Verification and historical captures

The adapter/controller have 21 offline Node checks. The relevant Python campaign, archive-import, and v3 checks total 54 passing tests including the queue-source adapter. The compiled local launcher passed two framed responses while its input remained open; installed-Chrome automation still requires a live pilot. The notebook has also executed with collection disabled.

The user loaded version 0.1.0 and supplied a real export on 12 September 2026: `data/HAR/sofascore-event-12499375-2026-09-12T02-47-44-551Z.har`. Its first revision contained 38 API records (10 HTTP 200, 19 HTTP 304, nine HTTP 404) and zero response bodies. The selected ID was 12499375, but numeric match paths belonged to 16434120.

The same file was then replaced with an export timestamped 02:54:18 UTC. This revision contains 138 API records, including 30 selected component responses: 29 HTTP 304 and one pregame-form HTTP 404. The match ID was correct, but version 0.1.0 skipped all 304 bodies, leaving every entry metadata-only. A subsequent 02:59:38 export still used version 0.1.0 and also had no bodies.

**Successful capture:** the 03:03:20 UTC export uses version 0.1.1 and contains 29 saved cached bodies across seven selected component paths, among 75 observed API entries. Source-to-Parquet checks passed for 39 shots with xG, 40 distinct lineup players, three reported absences, both team heatmaps (1,635 points), and five goal sequences with 10 actions. The two lineup observations intentionally produce 80 roster rows and six absence rows; these are not distinct-player counts. The importer sent no requests.

The latest archive omits pregame-form. Combining it with the earlier archived 404 completes endpoint accounting, with pregame-form marked unavailable. Goal-action data includes two passes but no explicit recipient IDs; exact elapsed duration remains unknown. Match-level xG (1.58/2.70) differs from shot-level sums (about 1.7165/3.3101), so both source values are retained separately. The validation summary, hashes, prepared manifest, and frozen revisions are recorded in `data/audits/website_capture_v3/har_capture_validation.json`.

References: [DevTools Network API](https://developer.chrome.com/docs/extensions/reference/api/devtools/network), [DevTools extensions](https://developer.chrome.com/docs/extensions/how-to/devtools/extend-devtools), [Network export](https://developer.chrome.com/docs/devtools/network/reference#save-as-har).

Version 0.2.3 handles match navigation on Start / resume: it uses the queued URL and ID, searches visible historical fixtures when needed, and opens missing sections. An explicit resume permits one bounded recovery pass for exact saved denials. An already displayed failed match may be refreshed once after checking its existing Network log. Fresh denials on requested endpoints, visible challenges, user pauses, and time limits still stop it; unresolved saved denials prevent completion. No manual match lookup is required.

Navigation and section waits now end early when the matching data is saved and controls are ready, up to `VIEW_WAIT_SECONDS`. Requested-data failures remain explicit. This does not suppress or replay the website's requests; configured action jitter and season rests remain intact.

The 0.2.12 reset page reports readiness through an internal extension message, checked against the extension origin, tab ID, and per-reset nonce. This fixes 0.2.11's failed tab-URL readiness check without adding permissions.

Version 0.2.13 also recognizes failures that completed before **Start / resume** but were absent from the saved checkpoint because recording had already stopped. Their exact Network-log evidence can receive the same bounded refresh. Unknown completion times, live callbacks, newer failures, and visible challenges still pause immediately. An unresolved older failure still prevents publication after that attempt.
