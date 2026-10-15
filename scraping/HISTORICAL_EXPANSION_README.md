# Historical collection: 21 additional league-seasons

The requested selection is already prepared as five discovery runs. It uses the
existing xDiyo Network Capture extension, its usual Sofascore tab/session, and
the existing converter.

The later Europe 2010/11-2014/15, Japan and Mexico requests have separate setup in
[ARCHIVE_EXPANSION_README.md](ARCHIVE_EXPANSION_README.md). They do not alter this
batch or its pending runs.

## Start

1. Reload xDiyo Network Capture at `chrome://extensions` for version 0.4.1. If
   DevTools is already open, close and reopen it so the writer and panel load the
   updated discovery code. Open an ordinary Sofascore football match in Chrome,
   then open DevTools → **xDiyo**. Disconnect any previous collection/discovery.
2. From the xDiyo project folder, run this command and leave it running:

   ```powershell
   .\.venv\Scripts\python.exe -u -B .\scraping\run_historical_expansion.py
   ```

3. Paste the first ID below into **Discovery ID**, select **Create / reconnect**,
   then **Discover / resume**. The ID already contains the correct season and
   league selection; there is no need to enter competition IDs or change the
   panel's season field.
4. When discovery is ready, select **Create collection campaign**, then select
   **Start / resume** under Automatic collection. The campaign ID is assigned at
   that point and filled in automatically.
5. Keep Chrome, the collection tab's DevTools, and the Python script open. The
   script converts each completed league into `data/xDiyo_data` automatically.
6. After the script confirms that all files for that year are verified, disconnect
   the collection in the extension and repeat steps 3–5 with the next Discovery ID.
   The script moves its conversion watcher to that next year automatically.

The five discovery/collection transitions in Chrome are manual. The script
handles their saved IDs, progress checks and conversion; it does not click Chrome
or send Sofascore requests. Other Chrome tabs can be used normally.

## Prepared Discovery IDs

| Season | Discovery ID | Leagues |
| --- | --- | --- |
| 2015/16 | `discover-2015-2016-a687bca1` | Pro League, Premiership, Championship, La Liga 2, Bundesliga 2, Ligue 2, Serie B |
| 2016/17 | `discover-2016-2017-e6436181` | Pro League, Premiership, Eredivisie, Bundesliga 2, Ligue 2, Serie B |
| 2017/18 | `discover-2017-2018-85bbdb10` | Pro League, Premiership, Serie B |
| 2018/19 | `discover-2018-2019-b8e48ee7` | Pro League, Premiership, Serie B |
| 2019/20 | `discover-2019-2020-113e7943` | Pro League, Premiership |

Pro League means Belgium (38); Premiership means Scotland (36). The other IDs are
Eredivisie 37, Championship 18, La Liga 2 54, Bundesliga 2 44, Ligue 2 182 and Serie B 53.
The editable selection is [historical_expansion_2015_2020.json](historical_expansion_2015_2020.json).
The saved run list is [batch.json](../data/website_capture_v3/batches/historical-expansion-2015-2020/batch.json).

## Scope and saved files

- Regular league stage only, following the existing collection scope. Playoffs,
  promotion/relegation ties and post-split groups are excluded. For Scotland,
  regular-season teams normally meet three times before the split. Discovery
  starts at round 1 and skips round 0 when older inventories advertise it.
  The requested Scottish seasons stop at round 33 even when old provider
  responses do not label their post-split games as a separate group.
  [SPFL format rules](https://spfl.co.uk/admin/filemanager/files/shares/SPFL%20Rules%20and%20Regulations%2018-May-20%20%28MASTER%20COPY%20B%29%20CLEAN.pdf).
- The first 2015/16 campaign is `chrome-2015-2016-15c4e0d6`, with 2,599 finished
  matches. Its inventory retains one Scottish fixture marked cancelled by the
  provider (6773232); that unfinished record is not in the capture queue.
- Nine Serie B 2015/16 round-4 games were found in the provider's round-0 bucket
  during an explicitly approved inspection. Their identities, ordered pairings,
  dates and complete round/team coverage were verified before admission. The
  normalized `round` is 4; original responses and `stage_json` retain the provider
  label. This is a correction for those exact fixtures, not general round-0
  collection. See [event_round.py](sofascore/event_round.py).
- Statistics, xG/shotmaps, heatmaps, lineups/player statistics, reported absences,
  incidents/goal actions, timing and `is_awarded` use the existing profile.
  Commentary and H2H remain off. Unavailable features remain missing; they are
  never represented by invented values.
- Filenames remain `Pro_League_15_16.parquet`, `Premiership_15_16.parquet`, etc.,
  with the usual tables and manifests in **`data/xDiyo_data`**.
- The preparer found no existing publication for any of these 21 targets.
  Restarting reuses the saved IDs rather than creating duplicate campaigns. The
  converter verifies existing output from the same campaign and refuses a
  conflicting output instead of replacing it.
- The existing request pacing, five-minute rests between league-season cohorts,
  periodic memory reset, checkpointing and pause controls remain in effect.
  `RANDOM_DELAY_MAX_SECONDS` is `None` and `INCLUDE_LINEUPS` is `True` in the script.
  Saved batch settings are authoritative; changing them after preparation requires
  a separate batch rather than silently rewriting a live queue.

## Reviewed Serie B 2018/19 discovery exceptions

The complete regular-season inventory contains 342 fixtures, of which 341 were
played. Lecce-Ascoli (8168679, 23 March 2019) is absent from the numbered round
responses because its event metadata has no `roundInfo`. Its exact Chrome-captured
event record was recovered separately; `match_round` assigns round 22 only when
its event, competition, season, ordered teams and kickoff all match. The original
metadata and round responses remain unchanged.
[Lecce's round-22 announcement](https://uslecce.it/il-presidente-balata-consegna-due-targhe-al-lecce-e-allascoli/).

Cosenza-Verona (7930529, 1 September 2018) was never played and was awarded 0-3.
Sofascore retains `postponed` together with `isAwarded=true`. An exact-identity
rule recognizes this verified unplayed award as a terminal exclusion, keeping it
in the discovery inventory and out of the played-match capture queue. It does not
alter the provider status or classify other postponed games as complete.
[FIGC decision](https://www.figc.it/media/65942/com-uff-047-sez-unite-testi-delle-decisioni-riunione-del-18102018-cu.pdf).

All three 2018/19 inventories still pass the full pairing-schedule validation.
The resulting campaign has 779 played matches: Belgium 240, Scotland 198 and
Serie B 341. No round-0 bucket was requested for this repair.

## The two shortened 2019/20 seasons

Historical discovery now recognizes these two exact league-season identities:

- Belgium: 16 teams, 29 played rounds, **232 played matches**.
- Scotland: 12 teams, **179 played matches**, with final team totals of 29 or 30.

For these seasons, discovery still enumerates every advertised numbered round.
Before collection can begin, finished fixtures must match the independently
recorded total and every team's played-match count in the captured final standings.
Team identities, pairing limits, round participation, kickoff dates and fixture
statuses are checked. Only finished matches enter the queue; unplayed fixtures
remain in the discovery inventory without fabricated results.

An empty acknowledged round can be legitimate for these curtailed seasons. A
404, access error, missing played match, absent standings count or other mismatch
still blocks collection. Every other season keeps the full pairing-schedule check.
The new rules have passed offline tests; the provider's live historical responses
will be checked when Chrome performs discovery.

Sources: [SPFL final table and curtailment announcement](https://spfl.co.uk/news/ladbrokes-premiership-and-spfl-season-201920-cur)
and [Belgian Competition Authority record of the 2019/20 Pro League decision](https://www.belgiancompetition.be/sites/default/files/content/download/files/abc-2020-vm-26_pub.pdf).

## Resume and check progress

Re-running the start command uses the same five IDs. To watch one year directly,
for example after restarting the computer:

```powershell
.\.venv\Scripts\python.exe -u -B .\scraping\run_historical_expansion.py --season 16/17
```

Reconnect that year's Discovery ID in Chrome. If its campaign already exists,
**Create collection campaign** retrieves the same saved campaign; alternatively,
paste the Campaign ID printed by the script into Automatic collection and connect.
Press **Start / resume** to continue the saved queue.

For a single local check, without leaving a watcher running:

```powershell
.\.venv\Scripts\python.exe -B .\scraping\run_historical_expansion.py --once
```

This prints the saved discovery/queue state and converts any complete cohorts.
`--prepare-only` just prints the saved IDs. Neither command starts Chrome or
requests provider data.

Ctrl+C stops the Python watcher; the extension's **Pause** stops collection.
If the converter reports `finished_with_gaps`, it stops advancing the batch so
the affected year can be reviewed. Completed captures remain saved. Use
`--season` to work on a different year independently when appropriate.

If discovery pauses or says it needs review, retain the displayed error and ID.
Acknowledged responses are saved locally for diagnosis; do not create replacement
IDs or disable completeness checks to get past a missing fixture or season.
