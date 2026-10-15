# Proposed data-collection changes

Review scope: the scrape-and-wait workflow with an external VPN, the prepared data, and the immediate loading boundary. This is the design proposal list. Implementation has now started; current progress and remaining work are recorded in [the collection section of planning.md](C:/Users/luisi/Documents/Programming/Python/xDiyo/planning.md#data-collection-refactor). Original datasets remain intact.

1. **Keep one scraper implementation and one browser session per run.** Consolidate the duplicate modules around `Scrape+wait`, retain the external-VPN workflow, and retire the Chromium-VPN entry point from active use. This reduces maintenance and browser startup overhead; it does not itself reduce required API calls.
2. **Add real request pacing and a shared hourly budget.** `waiting_time` currently controls page-load timeout, with no sleep between requests. Give navigation timeout and request spacing different settings. Count retries too, and pause the whole run on rate limiting or blocking.
3. **Cache each successful endpoint response and checkpoint immediately.** Reuse completed match components across restarts and later runs. A failed away heatmap should require retrying only that heatmap. Reprocessing stored JSON should make zero API requests.
4. **Plan requests by match ID, status, requested components, and freshness.** Deduplicate before fetching. Skip post-match statistics/shotmaps/heatmaps for unplayed fixtures in historical collection. Refresh new or changed matches instead of scraping two complete rounds every time.
5. **Preserve the existing efficient endpoint grouping.** Statistics already covers both teams and multiple periods; one shotmap and one lineup response each cover both teams. Keep the two team heatmap calls. A smaller equivalent endpoint set has not been verified.
6. **Evaluate cheaper fixture discovery, subject to complete coverage.** Compare any verified paginated season listing against round discovery, including playoffs and rearranged games. Adopt it only if it preserves all required fields and match IDs and uses fewer total calls. Otherwise retain round discovery. Fresh-season discovery is deferred for integration into the current Chrome extension, initially for 2025/26 across the same 13 leagues; it must obtain a new fixture inventory rather than depend on legacy files.
7. **Keep commentary off, with an independent flag for future use.** It is already disabled in the active notebook, so this is not a new saving. When enabled, reuse team names and round metadata from fixture discovery and eliminate the redundant event-detail request.
8. **Keep shot-level xG, but correct missing-value handling.** Missing xG/xGOT must remain unknown; do not substitute `1.0` or assume an own goal. Preserve every shot, its identity, own-goal metadata, and available timing/coordinates. Team aggregate xG already arrives with statistics. An optional aggregate-only profile could save one call per match, but loses shot detail and is not the default.
9. **Make heatmap availability explicit.** Preserve both player and goalkeeper point arrays and distinguish valid empty data, unavailable coverage, an unrequested component, and a failed request. Do not turn a transport error into an apparently genuine empty heatmap.
10. **Fix lineups without adding player-by-player calls.** Keep an optional lineup flag. Parse the roster even when formation is absent, handle each side independently, retain player IDs, and remove the blanket second-fetch repair pass. Represent predicted and confirmed lineups separately; never present the previous fixture's roster as the actual lineup.
11. **Add match timing with explicit meaning and quality.** First retain timing metadata already returned by discovery. Investigate structured match incidents independently of commentary, at up to one extra request per eligible match. Store exact/recorded durations only when supported; store announced added time separately and leave unknown duration null.
12. **Correct period, score, and historical-standing semantics.** Use the API's period labels instead of list order. Preserve first half, second half, extra time, and shootout scores separately. Replace artificial middle-of-table standings with missingness plus provenance, while retaining the match-specific pregame-form request.
13. **Make incremental updates safe and chronological.** Upsert by match ID, preserve good cached values after failed refreshes, retain tournament stage and kickoff time, and handle postponed fixtures outside the most recent round. Use atomic writes and deterministic ordering. For the future daily prediction workflow, integrate discovery of the latest season fixtures and retrieval of new or due data into the same extension. This work is deferred; coordinate queue changes with the collector, converter and auditor.
14. **Use typed Parquet for prepared data and retain raw JSON.** Replace nested Python objects serialized into CSV cells with ID-linked tables for matches, shots, heatmap points, lineups and the new enrichments. Keep compressed source JSON and a SQLite collection manifest. Preserve current CSV consumers through a versioned adapter; join lineups by match ID and allow missing lineup files. New fields must survive cross-season loading instead of being discarded by the first file's column list. Benchmark a representative season before migration; existing CSVs remain intact.
15. **Audit the historical xG and add focused verification.** Flag suspect legacy xG without rewriting originals or inventing recoverable values. Test the request planner, missingness, ID joins, resume behavior, period parsing, and duration interpretation. Validate a small paced live sample before a full collection run.
16. **Collect injury/availability reports from shared responses.** Inspect each side's `missingPlayers` in the lineup payload and retain reported injuries, doubtful status, suspensions and other absence reasons separately. With lineups already fetched this should add no requests, subject to current payload validation. Injuries can stay enabled when lineup output is disabled; the planner then fetches the shared endpoint once and explains that dependency. Match-time injury incidents are a separate optional scope, sharing the incidents request with duration.
17. **Collect the passing sequence leading to each goal.** Interpret the requested network as Sofascore's Goal Action Animation / goal buildup sequence. Preserve its ordered actions, players and supplied coordinates, then derive a directed passing graph locally where the payload supports it. The implementation pilot found ordered actions bundled in the incidents response, so no per-goal call is needed for this observed payload. Historical coverage varies; the 2015 sample has no sequence. Keep this separately configurable and leave commentary disabled.
18. **Refactor promotion/relegation flags within this plan.** Use stable team IDs and explicitly aligned consecutive seasons, validate membership completeness, and record uncertain or exceptional entries separately. Bind initial seeds to a specific league and season instead of whichever file happens to be earliest. Store one row per team/competition/season and join the four home/away flags when needed. This is local derived-data work and needs no API requests when membership evidence is sufficient; it belongs after the canonical match schema is established.
19. **Plan portable collection for a future Hostinger prediction service.** Keep discovery, job planning, saved state, validation and exports reusable across the desktop extension and a future VPS runner. A server browser controller is a candidate; an extension is not a deployment requirement. Validate provider access from the actual VPS, unattended recovery, resource use and prediction-data freshness before choosing a transport or migrating. This is deferred design work, not an instruction to provision, install or deploy now; see the [VPS design section](../planning.md#deferred-hostinger-vps-collection).

## What the request savings actually look like

For a completed season, with `M` matches, `R` round-list requests, shotmaps, both team heatmaps and lineups enabled, and commentary disabled, the current source makes approximately:

`R + 6M + F`

Here `F` is the extra lineup repair requests; this excludes browser auxiliary traffic and navigation failures. A 380-match, 38-round season costs **2,318 requests plus repairs**. This is a source-derived count, not a measured live run.

| Change | Effect on API requests | Same data? |
|---|---|---|
| Persistent cache and targeted resume | Saves every still-valid component request already completed | Yes, within the stated freshness policy |
| Parse missing-formation lineups from their first response | Removes the corresponding repair calls | Yes; often retains more data |
| Skip post-match endpoints on unplayed fixtures | Avoids premature calls; collect them when the fixture finishes | Yes for the final historical dataset |
| Reuse metadata when commentary is enabled | Saves one event-detail call per match | Yes |
| Validated alternative discovery | Saves `R - P` if complete discovery takes `P < R` calls | Only after coverage and field parity |
| Disable lineups and injury/availability collection | Saves the shared lineup call per match, plus related repairs | Reduced optional output |
| Aggregate-only xG profile | Saves one shotmap call per match | Loses individual shots |
| Add structured incidents for duration | Adds up to one call per selected match | Adds new data; exact duration still needs validation |
| Add reported injuries while fetching lineups | No additional call if the availability fields are present | Adds new data from the same response |
| Add match-time injury incidents | No additional call if incidents are already fetched; otherwise one per selected match | Adds reported incidents, not a complete injury history |
| Add goal buildup sequences | One shared incidents call per uncached match in the observed payload; zero additional calls if incidents are already fetched | Adds new data; reuse bundled sequences if available |
| Refactor promotion/relegation flags and prepared storage | No API calls for local transformation; any missing membership evidence is reported separately | Preserves supported facts, with explicit uncertainty and compatibility checks |

Reducing concurrency or increasing waits reduces request pressure, not request count. Conditional HTTP responses also still count as requests. There is no verified large reduction for a first scrape of fully covered matches while preserving all existing detail.

The six-request baseline above describes the original output profile. Goal-action enrichment and any new shared dependencies must be added explicitly; they are not included in that baseline.

## Findings that affect the plan

- The prepared heatmap folder has **110 match CSVs and 110 lineup CSVs**, covering **40,366 matches across 13 league labels**. Every pair has matching match IDs and row order; no duplicate match IDs were found within those files. This does not establish that every season is complete.
- No duration, timestamp, injury-time, or added-time columns were found in the match CSV headers. The collection currently discards the raw fixture payload after constructing its export.
- In the complete Premier League 2015–16 file, **all 9,818 stored shots have xG = 1.0**; the corresponding Bundesliga file has **7,914 such shots**. These include ordinary missed shots. The source assigns these values whenever `xg` is absent, so those legacy values must not be treated as genuine measured xG. The prepared files alone cannot identify the original missingness of every individual value.
- Seven inspected season files contained readable player/goalkeeper heatmaps for both sides of every match. Some recent files contain empty shot lists, whose original reason is no longer recoverable from the export alone.
- `Scrape+wait/football_scrape_xdiyo.py` and `football_scrape_xdiyo_novpn.py` are byte-identical. The parent `football_scrape_xdiyo.py` differs principally in manual startup pauses and one statistics guard.

## Match-duration conclusion

Sofascore's [Vissel Kobe–Kashima Antlers match page](https://www.sofascore.com/football/match/vissel-kobe-kashima-antlers/JmbsWmb#id:16348595), inspected in Chrome, displayed one minute of first-half added time and seven in the second half. It did not expose an exact whistle timestamp in the inspected text. Announced added time is a minimum that can be increased, so `90 + added_time_1 + added_time_2` is not an exact duration. See [IFAB Law 7](https://www.theifab.com/laws/latest/the-duration-of-the-match/).

Direct API navigation was blocked by the browser client, and the web tool could not open that API URL. Exact current timing fields and any alternative discovery endpoint remain validation items. The plan therefore includes a bounded duration investigation with a useful outcome even when exact duration is unavailable: preserve the timing evidence, its meaning, and an explicit availability flag.

## Injury and goal-sequence feasibility

The wrapper author's [lineup implementation](https://tommhe14.github.io/sofascore-wrapper/_modules/sofascore_wrapper/match.html) reads `home.missingPlayers` and `away.missingPlayers` from the existing lineup endpoint. This is an implementation lead, not official confirmation of current coverage or reason-code meanings. The local scraper only exports formation and roster and drops those lists. Store the report as it was observed for that match; a missing list does not establish that every player was healthy, and a retrospective fetch cannot establish what was known before kickoff.

Sofascore explicitly documents [Goal Action Animation](https://www.sofascore.com/news/sofascores-goal-action-animation-bringing-goals-to-life), including passes, assists and the final shot leading to a goal. The feature exists, but a current machine-readable endpoint, its request granularity and availability for older seasons have not been verified here. The proposed artifact is the observed sequence for each goal and a graph derived from its pass actions. A full-match passing network would require additional data beyond these goal sequences.

## Promotion/relegation review

The earlier plan only preserved these derived files. The review now explicitly includes [utils/promotions_demotions.py](C:/Users/luisi/Documents/Programming/Python/xDiyo/utils/promotions_demotions.py) and [its notebook](C:/Users/luisi/Documents/Programming/Python/xDiyo/promotions_demotions.ipynb). The flag means that a team **entered the current season through promotion or relegation**. It does not describe the team's outcome at the end of that season.

The code compares team slugs found in match rows and treats new upper-division names as promoted. Lower-division newcomers are split according to whether their name appeared in the previous upper division. This works under assumptions about complete seasons, stable names and ordinary league movement, but those assumptions are not checked. Adjacent available files need not represent consecutive seasons, and the upper/lower successor years are not explicitly checked for alignment. Initial seeds are attached to each division's earliest available file; locally Serie A starts in 2015–16 and Serie B in 2019–20. These are concrete fragilities, not evidence that every existing flag is wrong.

The replacement should retain known flags, mark unsupported cases unknown, and accept explicit sourced overrides for league expansion, administrative entries and other exceptional movement. An unseen team alone is insufficient proof of promotion. A team-season companion table also avoids rewriting complete heatmap-bearing match files whenever the flags change.

## Storage assessment

The CSVs are usable for recovery and compatibility, but I recommend changing the primary prepared format. The 220 match/lineup CSVs occupy approximately **2.35 GB**, and the 110 promotion-enriched CSVs add **1.80 GB**. In the inspected Premier League 2024–25 pair, the enriched file adds just four flag columns to the original 302 columns. A single home heatmap cell in that season contains 19,615 characters. This layout requires parsing serialized dictionaries/lists, repeats bulky content in derived copies, and provides no explicit contract for types or missingness.

Recommended roles:

| Layer | Format and purpose |
|---|---|
| Source archive | Compressed JSON per response, with source and observation metadata, for offline reparsing |
| Collection state | SQLite manifest for request accounting, cache keys, checkpoints and component status |
| Prepared analytical data | Typed, nullable Parquet tables linked by stable IDs, grouped by competition and season |
| Promotion/relegation | Small team-season Parquet table, joined into a match view when required |
| Legacy compatibility | Explicit CSV exports using the existing shapes and a safe reader |

Parquet supports compressed storage and reading selected columns, which fits loading match statistics without also decoding every heatmap. See the [Apache Arrow Parquet documentation](https://arrow.apache.org/docs/python/parquet.html). The proposed design separates repeated records such as shots and goal actions instead of merely putting the existing text blobs into a Parquet container. No size or speed improvement has been measured yet: benchmark disk size, load time and memory, and verify values, point/action order, IDs and nulls before switching consumers. A format conversion cannot recover fields already discarded or fix invented historical xG on its own.

## Implementation evidence

See the current status in [planning.md](C:/Users/luisi/Documents/Programming/Python/xDiyo/planning.md). The three-event pilot used 23 attempts including one startup failure; repeat cached collection used zero requests. The goal sequences, announced added time and absence reports are retrievable, with the interpretation limits recorded there. The ID-based flag audit also identified a duplicated/mislabelled Eredivisie season file, which remains preserved and excluded from derivation.
