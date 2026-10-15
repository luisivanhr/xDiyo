# Daily history and prediction-fixture collection

Use [run_daily_collection.py](run_daily_collection.py) with the xDiyo Network
Capture extension **0.4.0**. Chrome discovers and captures the provider responses;
the script waits locally and publishes the finished collection automatically.
Python makes no Sofascore HTTP requests.

## Start a run

1. Reload **xDiyo Network Capture** at `chrome://extensions`, then close and reopen
   DevTools. The panel must show **Version 0.4.0**. This also reconnects the updated
   Python writer; the existing native-host registration is sufficient.
   If the panel reports that a reload is required, close DevTools and reopen it
   after reloading the extension. An old open session can still hold earlier code.
2. Edit the short settings block in `scraping/run_daily_collection.py`. The default
   season is **26/27**; change it to **25/26** to update the previously collected
   season. The season is explicit, so an advertised future season is never chosen
   accidentally. `LEAGUES = None` selects all 13 registered leagues.
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

Each league is handled independently within the selected season:

- Discover the regular numbered rounds through Sofascore's current round. When
  that round is settled, continue to the first round with unfinished fixtures.
  These compact round inventories are refreshed on a new run, so rescheduled
  fixtures and changed results are seen.
- Keep the completed history and **the whole current/next round**. A partly played
  round includes its finished, in-progress and unplayed matches. Completed matches
  become history immediately. Live/interrupted matches remain unfinished until a
  subsequent run captures their final result.
- Revisit unresolved older fixtures. An old postponement does not prevent moving
  to the next scheduled round. A unique later replacement with the same league,
  season, round and ordered teams supersedes the obsolete postponed/abandoned ID.
  Ambiguous identities or incomplete team coverage require review.
- Exclude playoffs and post-split/group stages, matching the recent regular-season
  collection. Cancelled/abandoned entries without a finished result remain in the
  discovery inventory and its preview, outside prediction/history selection.
- An ended regular season has history and **no next round**. A season absent from
  a league's season list is reported as `season_not_listed`. Neither condition
  prevents the other leagues from being collected. An access error or malformed
  response remains an error; it is not treated as proof that a season ended.

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
| `SEASON` | Explicit provider year pair, for example `26/27` |
| `LEAGUES` | `None` for the 13 saved leagues, or a list of league specifications |
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
from scraping.sofascore_web_v3.discovery import league_catalog
LEAGUES = [league for league in league_catalog()
           if league["name"] in {"Premier_League", "La_Liga"}]
```

To add another regular league, include a specification such as
`{"name": "League_Name", "competition_id": THE_PROVIDER_ID, "meetings_per_pair": 2}`
using its actual Sofascore competition ID. Daily mode validates each selected
round's team coverage. Multi-group or nonstandard formats need their own validation;
do not turn off the checks. The extension also exposes a daily/whole-season selector
for manual creation; reconnect its daily Discovery ID with the script to publish it.

## Saved data and loading

Updates go to **`data/xDiyo_data`**, alongside the historical seasons, using the
same `<League>_26_27.parquet` naming convention and typed season/relational format.
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
run = prepare_daily_run(season="26/27", include_lineups=True)
watch_daily_run(run["discovery_id"])
```

If a notebook cell is interrupted, call `watch_daily_run("discover-...")` with the
printed ID. Preparing/watching never launches another browser.

## Validation status

Offline tests cover mixed and ended seasons, incomplete inventories, partial
rounds, postponed replacements, fresh-status request selection, repeated runs,
profile changes, same-ID result updates, source preservation, snapshot phases,
publication rollback and resume. The saved Premier League 2025/26 compatibility
check reused all 380 completed matches without provider requests. The daily flow
has not yet had a live Chrome collection run; the existing historical route was
already exercised live.
