# Daily history and prediction-fixture collection

Use [run_daily_collection.py](run_daily_collection.py) with the xDiyo Network
Capture extension **0.4.1**. Chrome discovers and captures the provider responses;
the script waits locally and publishes the finished collection automatically.
Python makes no Sofascore HTTP requests.

## Source checkout setup

From the project root, install the collector dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[collector]"
```

For the offline Python tests also install pytest (`pytest>=8,<10`). Run the
collector scripts from this checkout. The collector extra supplies the Selenium
bindings used by the browser client and its simulated browser tests.

## Start a run

1. Let any active historical collection finish first. Keep its Chrome session,
   pacing and converter running. For a new run, verify that the DevTools panel and native host
   both report version **0.4.1**. When ready to begin a new daily run, disconnect the completed campaign
   and reconnect the prepared Discovery ID to load a fresh Python native host.
2. Edit the short settings block in `scraping/run_daily_collection.py`. The default
   split-year season is **26/27**, and `CALENDAR_SEASON` is **2026**. These are
   independent explicit selectors; an advertised future season is never chosen
   accidentally. `LEAGUES = None` selects the original 13 plus all **32 new
   leagues** in the November coverage expansion, for **45 leagues** in total.
3. From the xDiyo project folder, run:

   ```powershell
   .\.venv\Scripts\python.exe -u -B .\scraping\run_daily_collection.py
   ```

4. Leave the script running. It prints **Discovery ID: `discover-...`**. In Chrome,
   keep an ordinary Sofascore football match open, open DevTools → **xDiyo**,
   paste that ID into **Discovery ID**, select **Create / reconnect**, then
   **Discover / resume**. Disconnect any previous campaign first. Reconnecting a
   prepared ID uses its saved settings, regardless of the panel's season selector.
5. Once the inventory is ready, select **Create collection campaign**. The extension
   fills in the campaign ID and connects it. The script also prints **Campaign ID:
   `daily-...`**, the number of matches reused, and the number to collect.
6. Under **Automatic collection**, select **Start / resume**. Keep DevTools, Chrome
   and the Python script open. When every queued match has been saved, the script
   publishes the update, prints its location and exits. If everything was already
   collected, the zero-match queue publishes without needing Start.

The extension continues using its existing inspected tab/session; it does not
open a browser or revisit the match page for each request. Its existing session
renewal, pacing, periodic memory reset and pause controls still apply. Other Chrome
tabs can be used; leave the collection tab on Sofascore with its DevTools open.

The older historical converter is not needed for this run. **The daily script is
the converter watcher.** Keep it running, or restart it after collection finishes.
The historical converter deliberately excludes daily campaigns because it cannot
merge reused history with unfinished fixtures.

## Resume, check once, or run on another day

Running the same script with unchanged settings automatically reopens its latest
unfinished run. Its Discovery ID stays the same, and acknowledged requests remain
saved. Once that run has published successfully, running the script again creates
a fresh discovery and checks the latest schedule.

To resume a specific run, including restarting its converter:

```powershell
.\.venv\Scripts\python.exe -u -B .\scraping\run_daily_collection.py --resume discover-PASTE-THE-SAVED-ID
```

Use the actual printed ID. It is also saved in
`data/xDiyo_data/_collection/last_run.json`. The campaign ID is associated
with that discovery and appears automatically once prepared; it is not a permanent
ID shared by all days. A new discovery gets a new campaign, while resuming uses
the existing IDs.

For a single local status check, adding `--once` prints current progress and exits;
it also publishes if capture has finished. `--prepare-only` prints the discovery
ID and exits before watching. These checks send no provider requests.

If Chrome pauses, the script reports its saved reason. Resolve the displayed
issue and press **Discover / resume** or **Start / resume**, as appropriate. If
Chrome disconnected, reconnect the printed ID first. The script does not click
resume automatically. Ctrl+C stops the watcher; it does not stop a running Chrome
collector. Stop collection with the extension's Pause button.

## Which matches are selected

New prepared daily runs use schema 2. Each league has its own saved season years:

- Enumerate the advertised regular numbered rounds as compact inventories. This
  covers rescheduled earlier fixtures and ordinary stages that reuse round
  numbers. Finished components are reused after identity/hash checks.
- Keep completed history, live/interrupted and postponed matches, plus scheduled
  fixtures in the next **14 days**. The interval is fixed at preparation and saved
  in the discovery and publication evidence. Change `FIXTURE_WINDOW_DAYS` to
  select 1–60 days. Metadata is refreshed when each selected match is captured.
- Revisit unresolved older fixtures. An old postponement does not prevent moving
  to the next scheduled round. A unique later replacement with the same league,
  season, round and ordered teams supersedes the obsolete postponed/abandoned ID.
  Ambiguous identities or incomplete team coverage require review.
- Exclude playoffs and post-split stages. Retain the ordinary Apertura/Clausura
  stages, Argentina's regular zones and MLS conferences; never renumber rounds.
  Cancelled/abandoned entries without a finished result remain in the
  discovery inventory and its preview, outside prediction/history selection.
- An ended regular season has history and **no next round**. A season absent from
  the requested inventory is reported as `season_not_listed` and blocks creation
  until the selection is reviewed; it is never silently dropped or replaced by
  the latest season. `outside_fixture_window` means fixtures are advertised but
  later than the selected interval. An access error or malformed
  response remains an error; it is not treated as proof that a season ended.

Saved schema-1 daily IDs retain their original current/next-round selection and
missing-season behavior. `--resume` never upgrades or rewrites an existing run.
Calling the helper with plain legacy league dictionaries also preserves schema 1;
catalog names or dictionaries with `season_kind` select schema 2.

Prediction candidates are fixtures still marked `notstarted` whose scheduled
kickoff is in the future. Postponed, live and finished matches are not prediction
candidates. The loader below checks the kickoff again when reading the output.

## Duplicate prevention and refresh

The join key is **Sofascore's event/match ID**. A published season contains one
match row per ID, sorted by scheduled kickoff and then ID.

Saved finished matches are reused only after checking the stored files, fixture
identity, requested feature profile and component coverage. Changes visible in
the round listing—such as a corrected score, round or kickoff—cause a refresh.
The shared settling policy also permits a refresh of recent results after six
hours, during the first 48 hours after scheduled kickoff. Confirmed unavailable
components, such as an unsupported xG endpoint, are valid terminal outcomes.

Unfinished matches are refreshed on a new run. Metadata is collected first; its
new status determines the needed endpoints. A fixture that has since finished is
collected with the final-data profile and replaces the existing match row under
the same ID. It is never appended as a second match.

Distinct lineup/player/availability observations retain their timestamps and
`pregame`, `inprogress` or `postgame` phase. Identical saved snapshots are deduplicated.
These snapshots are intentional history, not duplicate match rows. A newer 404
does not erase previously captured statistics/heatmap data; coverage records its
latest unavailable attempt alongside the retained last-good values. Missing data
is never filled with invented zeros.

## Settings and collection scope

| Setting | Meaning |
| --- | --- |
| `SEASON` | Split-year cycle, for example `26/27` |
| `CALENDAR_SEASON` | Calendar year, for example `2026`; change independently at New Year |
| `LEAGUES` | `None` for all 45 leagues, or a list of catalog names |
| `SEASON_OVERRIDES` | Selected league name → explicit calendar year or split cycle |
| `FIXTURE_WINDOW_DAYS` | Scheduled future fixture window; default 14 days |
| `INCLUDE_LINEUPS` | Save lineups/player statistics when available; defaults to `True` |
| `RANDOM_DELAY_MAX_SECONDS` | `None` by default, or an extra random wait from 0 to T seconds between requests |
| `LINEUP_WINDOW_HOURS` | Request upcoming pregame/lineup data inside this window; defaults to 24 hours |

Completed matches retain the existing xG/shotmap, heatmap, statistics, reported
absences, incidents/goal-action and timing settings. `is_awarded` is populated
from metadata. Commentary and H2H are excluded.

Upcoming fixtures outside the lineup window receive metadata only; run again
closer to kickoff for pregame information. Inside the window, one lineup response
supplies both lineups and reported absences. Turning off lineup export can therefore
still require that endpoint for absences. Unconfirmed or unavailable lineups remain
explicitly so. No final statistics/xG/heatmap requests are made for unfinished matches.

To select existing leagues in the script:

```python
LEAGUES = ["Premier_League", "Brazil_Serie_A", "Eliteserien", "MLS"]
SEASON = "26/27"
CALENDAR_SEASON = "2026"
# At New Year, keep SEASON="26/27" and set CALENDAR_SEASON="2027".
# SEASON_OVERRIDES = {"Allsvenskan": "2026"}  # Only if selected above.
```

The complete catalog is `daily_league_catalog()` in
[`daily_catalog.py`](sofascore_web_v3/daily_catalog.py), with the canonical archive
output names. Calendar leagues are Liga Profesional, Brazil A/B, Chile, China,
Colombia, Ecuador, Norway, Peru, Sweden and MLS. All other new leagues use a
split-year selector. Calendar outputs use `26_26`; split-year outputs use `26_27`.
The manual extension form retains its original small historical catalog. Prepare
expanded daily IDs with this script/helper and reconnect them in the same panel.

Schema 2 requires exact competition/season identity, all advertised regular round
responses, unique fixture identities, known statuses and standings reconciliation
by ordinary stage. Standard round robins also require complete, nonconflicting
team coverage per round. Uneven schedules and conferences are checked against
played counts without assuming every team plays in every numbered round. Unknown
group/round labels, missing rounds, ambiguous inventories and count discrepancies
block collection. Raw provider responses are retained for review. A new provider
format must be reviewed; these rules do not authorize guessing or bypassing gaps.

Format references used for this extension:

- [MLS competition guidelines](https://www.mlssoccer.com/about/competition-guidelines)
  distinguish its conference schedule from a full pairwise round robin.
- [Argentina's 2026 fixture](https://www.ligaprofesional.ar/notas/primera/2025/12/11/fixtures-de-la-temporada-2026/)
  identifies the ordinary Apertura/Clausura zones and interzonal fixtures.
- [Peru's provider competition](https://www.sofascore.com/football/tournament/peru/liga-1/406)
  supplies distinct ordinary stage tables within a calendar year.
- [Colombia's current competition](https://www.sofascore.com/football/tournament/colombia/primera-a-apertura/11539)
  is 11539 for the combined 2026 inventory. ID 241 shows the obsolete 2011
  inventory. This correction is confined to new daily selection; existing archive
  configurations are untouched. Exact returned identities remain mandatory.

Pre-split selection uses the regular standings participants and the reviewed
number of meetings (three in Scotland/Switzerland, one in Egypt, two in the other
selected split leagues). Championship/relegation/placement groups remain excluded.

## Saved data and loading

Updates go to **`data/xDiyo_data`**, alongside the historical seasons, using the
same `<League>_26_27.parquet` or `<League>_26_26.parquet` naming convention and typed season/relational format.
The converter reuses completed matches and updates the selected season by event
ID, retaining completed history, repaired fields and previous immutable versions.
Other seasons are left unchanged. This folder is both the default history source
and the output destination.
The Python helper accepts `history_root` and `output_root` for another location.

`LATEST.json` points to the successfully published run, including its per-league
history IDs, prediction IDs, unfinished IDs and version manifests. The pointer is
updated only after all selected leagues have published. A completed collection
means that the requested observations have terminal outcomes; it does **not**
mean that the season is over or that all optional data exists.

Use the current selection instead of globbing every version or every season file:

```python
from scraping.sofascore_web_v3.daily import OUTPUT
from scraping.sofascore_web_v3.daily_storage import read_daily_tables

history = read_daily_tables(OUTPUT, view="history")
to_predict = read_daily_tables(OUTPUT, view="prediction")
# Each maps a season stem to tables: tables["matches"], tables["lineup_players"], etc.
```

`view="unfinished"` includes live/postponed fixtures; `view="all"` returns the whole
current snapshot. All reads verify saved hashes. Both historical and ongoing
seasons are under `data/xDiyo_data`; load each season's current publication once
instead of concatenating its older versions. Use finished history
for labels/training, and respect observation timestamps and lineup phases when
building features at a prediction cutoff. This workflow prepares observations;
it does not run the predictive model or upload data to Hostinger/Drive.

For a small notebook, run the same two helpers from a kernel opened in the project:

```python
from scraping.sofascore_web_v3.daily import prepare_daily_run, watch_daily_run
run = prepare_daily_run(season="26/27", calendar_season="2026", include_lineups=True)
watch_daily_run(run["discovery_id"])
```

If a notebook cell is interrupted, call `watch_daily_run("discover-...")` with the
printed ID. Preparing/watching never launches another browser.

## Validation status

Offline tests cover all 32 new catalog entries, mixed calendar/split publication
and prediction loading, ordinary stages, conferences, uneven schedules, and
pre-split exclusions, as well as mixed and ended seasons, incomplete inventories, partial
rounds, postponed replacements, fresh-status request selection, repeated runs,
profile changes, same-ID result updates, source preservation, snapshot phases,
publication rollback and resume. The saved Premier League 2025/26 compatibility
check reused all 380 completed matches without provider requests. A saved original
13-league daily run has already published successfully. The expanded route was
also replayed against saved Chrome responses for all nine current-season cohorts
in coverage wave 1; all passed, without new provider requests. Live daily capture
across all 45 leagues is still pending the current historical collection's finish.
