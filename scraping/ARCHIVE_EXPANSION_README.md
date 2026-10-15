# Future archive campaigns

Prepared on 2026-10-04, separately from the pending 2015/16-2019/20 historical
expansion. Each group has **one Discovery ID and, after validated discovery,
one collection campaign**. Its queue contains multiple league-season cohorts.
The existing converter publishes each completed cohort into `data/xDiyo_data`.

Preparation sends no provider requests, starts no browser or worker, and does
not attach anything to the current collection. Start these after the pending
historical expansion has finished and its outputs have been verified.

## Requested selections

| Group | Selection | Discovery ID |
| --- | --- | --- |
| `europe-2010-2015` | 13 leagues × 5 seasons = **65 league-seasons**, 2010/11-2014/15 | `discover-europe-2010-2015-3f08453b` |
| `japan-2010-2027` | **35 league-seasons**, J1/J2 2010-2025; J1 spring 2026; 2026/27 completed matches | `discover-japan-2010-2027-d49ec4f4` |
| `mexico-2010-2026` | **33 requested tournaments**, Apertura 2010 through Apertura 2026 | `discover-mexico-2010-2026-d1028227` |

The European selection is frozen to the original thirteen leagues, so adding a
league to the main registry later cannot expand this request accidentally:

| League | Sofascore competition ID |
| --- | --- |
| Bundesliga | 35 |
| Bundesliga 2 | 44 |
| Championship | 18 |
| Premier League | 17 |
| La Liga | 8 |
| La Liga 2 | 54 |
| Ligue 1 | 34 |
| Ligue 2 | 182 |
| Serie A | 23 |
| Serie B | 53 |
| Eredivisie | 37 |
| Scottish Premiership | 36 |
| Belgian Pro League | 38 |

Regular-season rounds begin at 1. Playoffs, finals, promotion/relegation ties,
play-ins and placement matches are excluded. Scotland stops at round 33, before
the split. The old approval for nine misplaced Serie B **2015/16** games does
not authorize other round-0 requests in these campaigns.

## Japan: separate calendar and transition seasons

- J1 uses competition **196**; J2 uses **402**. Both Sofascore selectors expose
  2010-2026 and 26/27. Individual inventories and match coverage still require
  live discovery through Chrome.
- 2010-2025 are separate calendar-year cohorts. J1 2015 and 2016 include both
  ordinary league stages, excluding the championship playoffs. Sofascore's 2015
  UI already lists regular rounds 1-34; the scraper retains provider rounds.
- Spring 2026 J1 has two regional groups of ten. Only their 180 league games are
  eligible; placement playoffs are excluded.
  The captured season-87931 inventory labels its third- and fifth-place ties
  `Match for 3rd place` (round 50) and `Match for 5th place` (round 51).
  These numbered placement labels and their slugs are excluded along with
  `Final` and `Placement match`; the selected league rounds remain 1-18.
- Spring 2026's second-tier event combines **J2 and J3**: four groups of ten,
  360 group-stage games. This is not a normal J2 season. **The user chose to skip
  this special J2/J3 tournament**, so J2 goes directly from 2025 to 2026/27 in
  this campaign. No J3 matches are requested.
- 2026/27 is a distinct season for each league. Discovery inventories its
  advertised regular rounds, reconciles completed matches against current
  standings, and queues completed matches only. It is a dated snapshot, not a
  claim that the whole ongoing season has finished.

The explicit format rules cover changing team counts and odd-team byes, including
J2's 19-team 2010 schedule. Regional validation requires complete within-group
home/away pairings and the expected group sizes. Spring 2026 penalty shootouts
are left in the existing score/incident fields; the collector does not invent
extra-time durations or convert shootout wins into regulation-time goals.

Sources: [J.League format history](https://www.jleague.jp/corporate/about_competitions/tournament_history/),
[spring 2026 format](https://www.jleague.jp/news/article/29712/?mode=pc),
[2026/27 season](https://www.jleague.jp/corporate/about_competitions/season/),
[SPFL explanation of the split](https://clublogin.spfl.co.uk/news/202526-post-split-fixtures-qa).

### Reviewed Japanese provider inconsistencies

The 8 October 2026 discovery retains every original response and obsolete stub.
These corrections require the exact competition, season, event, ordered teams
and kickoff; they do not authorize general relabelling or omission.

- J1 2016 (provider season `11307`) reuses rounds 1-17 in each ordinary stage.
  Validation checks two complete 153-game round robins, with all 18 teams,
  complete home/away pairings across the season and no duplicate appearance in
  a stage-round. The official stage windows are 27 February-25 June and
  2 July-3 November. Stage-local provider rounds remain in captures and outputs;
  only validation distinguishes the two stages.
  [First-stage completion](https://www.jleague.jp/special/meijiyasudaj1_1st_winner/2016/),
  [Vissel Kobe's stage-by-stage schedule](https://www.vissel-kobe.co.jp/match/?y=2016).
- The postponed Tosu-Kobe game `7071662` (2 June 2016, 0-0) is misfiled as
  round 1 and normalizes to first-stage round 7. This also resolves the retained
  unplayed stub `7003252`; raw source round 1 remains intact.
  [Official rescheduled-match report](https://www.jleague.jp/news/article/6012/?mode=pc).
- Yokohama FM-Kashima `9117396` (3 November 2020, 2-3) is misfiled as round 25
  and normalizes to round 31.
  [Official match record](https://www.jleague.jp/sp/match/j1/2020/110321/live/).
- Four J1 2021 stubs describe games played earlier: `9309246` -> `9486403`
  (Kawasaki-Sendai, round 20, 12 May, 2-2), `9309261` -> `9486404`
  (Gamba-Hiroshima, round 20, 12 May, 1-2), `9309348` -> `9486368`
  (Nagoya-Cerezo, round 13, 8 May, 1-0), and `9309349` -> `9486384`
  (Gamba-Kawasaki, round 13, 8 May, 0-2). Both exact identities and final scores
  must match before a stub is superseded.
  [Round 20 report](https://www.jleague.jp/news/article/19907/),
  [Nagoya match](https://www.jleague.jp/match/j1/2021/050803/livetxt/),
  [Kawasaki match](https://www.frontale.co.jp/goto_game/2021/j_league1/13.html).
- J2 2013 Tokushima-Yamagata `3506406` (16 September, round 33, 2-2) retains
  stale postponed status/code 60. Its exact identity, kickoff and score permit
  completed-match collection; raw status/code are retained.
  [Tokushima archive](https://www.vortis.jp/game/match/?year=2013),
  [Yamagata archive](https://www.montedioyamagata.jp/match/40/).
- J2 2021 contains fifteen obsolete Yamagata stubs with incorrect dates/rounds.
  Each mapping below requires both exact provider identities and the verified
  replacement score; the original records remain in discovery. No general
  cross-round deduplication is permitted.

  | Retained stub | Played record | Official record |
  | --- | --- | --- |
  | 9309687 | 9309823 | [28 March, round 5, Yamagata-Okayama 0-1](https://www.jleague.jp/match/j2/2021/032803/summary/) |
  | 9309763 | 9309827 | [3 April, round 6, Kitakyushu-Yamagata 0-0](https://www.montedioyamagata.jp/match/p/2021040309/) |
  | 9309829 | 9309720 | [20 June, round 19, Mito-Yamagata 0-1](https://www.montedioyamagata.jp/match/p/2021062004/) |
  | 9309840 | 9309979 | [10 October, round 33, Yamagata-Akita 2-1](https://www.montedioyamagata.jp/match/p/2021101001/) |
  | 9309863 | 9309950 | [25 September, round 31, Ehime-Yamagata 0-4](https://www.montedioyamagata.jp/match/p/2021092515/) |
  | 9309864 | 9310055 | [3 November, round 37, Yamagata-Tokyo V 1-2](https://www.montedioyamagata.jp/match/p/2021110301/) |
  | 9309903 | 9309844 | [11 April, round 7, Yamagata-Niigata 0-2](https://www.jleague.jp/match/j2/2021/041110/review/) |
  | 9309913 | 9309875 | [3 July, round 21, Omiya-Yamagata 1-3](https://www.montedioyamagata.jp/match/p/2021070312/) |
  | 9309941 | 9309884 | [7 March, round 2, Tokyo V-Yamagata 0-2](https://www.jleague.jp/match/j2/2021/030707/review/) |
  | 9309957 | 9632202 | [5 December, round 42, Yamagata-Kitakyushu 5-1](https://www.montedioyamagata.jp/match/p/2021120501/) |
  | 9309976 | 9309819 | [21 March, round 4, Yamagata-Tochigi 1-2](https://www.montedioyamagata.jp/match/p/2021032112/) |
  | 9310005 | 9309894 | [7 November, round 38, Okayama-Yamagata 1-0](https://www.montedioyamagata.jp/match/p/2021110708/) |
  | 9310053 | 9309949 | [18 September, round 30, Yamagata-Kyoto 0-2](https://www.montedioyamagata.jp/match/p/2021091810/) |
  | 9310077 | 9309797 | [16 May, round 14, Yamagata-Ehime 1-0](https://www.montedioyamagata.jp/match/p/2021051603/) |
  | 9310099 | 9309959 | [2 October, round 32, Kofu-Yamagata 2-0](https://www.montedioyamagata.jp/match/p/2021100211/) |

- J2 2022 Yamagata-Okayama round 8 was officially resumed on 31 August after
  a refereeing rules error. The superseded 3 April record `10094765` (0-1)
  is retained; collect `10648806` with the final 0-2 result.
  [League decision](https://www.jleague.jp/news/article/22666/),
  [final result](https://www.jleague.jp/match/j2/2022/083101/summary/).
- J2 2023 Gunma-Kanazawa round 31 was resumed on 27 September after lightning
  stopped the 20 August game. Both provider records contain the final 1-1;
  retain `11019080` as superseded and collect `11680527` once.
  [League decision](https://www.jleague.jp/news/article/26000/),
  [final result](https://www.jleague.jp/match/j2/2023/092702/summary/).
  Both replay rules require exact original and replacement scores and identities.

- J2 2018 round 39 omits two played matches from the provider round response.
  Exact event responses captured in the existing Chrome session identify
  Omiya-Kyoto `8036367` (1-2) and Kofu-Sanuki `8036368` (1-0), both at
  16:00 JST on 28 October. Both raw responses omit `season` and `roundInfo`.
  [Omiya's official match report](https://www.rbomiya.com/match/368/) and
  [Kofu's official 2018 schedule](https://www.ventforet.jp/games/pdf/game2018.pdf)
  establish J2 2018 round 39. Only these exact IDs, competition/tournament,
  ordered teams, kickoff, finished status/code and scores, with both fields
  absent, receive normalized season `16005` / `2018` and round 39.
  Original HAR bodies, discovery payloads and stage metadata remain unchanged.
  The two HAR exports and their hashes are retained in the Japan batch directory.
  Reconciliation then verifies all 462 matches and every home/away pairing.

For a fixture genuinely omitted from an already enumerated round, the existing
`admit_recovered_event` workflow also supports archive discovery. Use only an
original exact-route Chrome HAR response, under the collector lock with the
extension disconnected, after enumeration and before campaign creation. It
preserves every raw round response and applies the same cohort identity, stage,
round, kickoff and complete schedule validation. It cannot fill missing data.

## Mexico: collect each short tournament separately

Use the separate provider competitions: **Apertura 11621**, **Clausura 11620**.
The chronological sequence is Apertura 2010 (second half of 2010), Clausura 2011
(first half of 2011), Apertura 2011, Clausura 2012, and so on through Apertura
2026. The first two belong to the 2010/11 football cycle. There is no request for
Clausura 2010 or the future Clausura 2027.

**Reviewed 9 October 2026:** the captured Clausura selector starts at 2011/12.
The Apertura source season `2770` / `10/11` contains both short tournaments:
its round-1 response has nine July 2010 and nine January 2011 fixtures.
The January Puebla-Chivas fixture `1588049` agrees with the club's official
[9 January, Clausura round-1 report](https://www.chivasdecorazon.com.mx/es/noticias/14641/Anuario-Chivas-2011).
Only this exact missing Clausura 2010/11 selection may reuse the captured
Apertura source identity `11621 / 2770`, provided that exact source season was
resolved from its inventory. Its requested competition `11620` remains explicit
in cohort metadata. Raw competition/season IDs and every response are retained.

Mexican round responses are partitioned by the already saved tournament windows:
July-December for Apertura and January-June for Clausura. Fixtures outside the
entire selected football cycle still fail validation. An opposite-half fixture
remains in the original response; it is selected only into its intended cohort.
Both 2010/11 halves must independently pass all 18-team, 17-round, 153-pairing
checks. Other missing or ambiguous source seasons remain blocking errors.
The two outputs that share the provider identity have separate canonical
`_tables/cohorts/<stem>/competition=11621/season=2770` partitions and CURRENT
pointers; no provider ID is invented or rewritten. Other outputs retain their
existing partition layout.

Ordinary short tournaments have one meeting per pair. Apertura 2019 uses 19
teams and 19 rounds with byes. Clausura 2020 is handled as the shortened played
schedule: 18 teams, ten games per team, 90 matches. A missing match within those
90 remains an error. The latest Apertura 2026 is a completed-match snapshot
reconciled against current standings.

Provider season spelling is preserved. Historical Mexican selectors generally
use `YY/YY`; the current Apertura selector uses `2026`. Discovery permits those
two explicit aliases for the current tournament but rejects an ambiguous result.

Sources: [Sofascore Apertura](https://www.sofascore.com/football/tournament/mexico/liga-mx-apertura/11621),
[Sofascore Clausura](https://www.sofascore.com/football/tournament/mexico/liga-mx-clausura/11620),
[official Apertura 2019 calendar](https://lmxwebsite.s3.amazonaws.com/docs/publicaciones/ligamx/CALENDARIO-LIGA-MX-TORNEO-APERTURA-2019.pdf),
[Chivas' Clausura 2020 table](https://www.chivasdecorazon.com.mx/es/estadisticas/tabla_general/Clausura-2020).

### Reviewed Mexican duplicate identities

The 9 October 2026 review retains these original response bodies and records
the exact replacement IDs in the superseded-fixture inventory. Each mapping
requires both IDs, competition, season, year, round, ordered teams, timestamps,
statuses and the reviewed scores. Missing or changed evidence remains an error.

- Apertura 2014, round 8: Monterrey-Chivas began on 13 September and resumed on
  15 November, finishing 1-0. The provider stores that result on `5559934`, with
  the original kickoff, and keeps a cancelled November stub `6133946`. Only
  `6133946 -> 5559934` is resolved; neither timestamp nor score is rewritten.
  [Chivas' official account](https://www.chivasdecorazon.com.mx/es/noticias/24585/las-particularidades-alrededor-del-chivas-vs-rayados).
- Apertura 2017, round 11: Monterrey-Santos resumed in November and ended 1-1.
  Both `7450393` and `7636813` carry that final result; the exact September
  duplicate is superseded by the November record, with both scores checked.
  [Rayados' match report](https://www.rayados.com/es/noticias/12247/rayados-empata-en-reanudacion).
- Apertura 2023, round 4: Atlas-America moved to the Azteca and finished 1-1
  on 20 August. The postponed `11341785` and cancelled `11528701` (Atlas home)
  are superseded only by `11557348` (America home), with the exact 1-1 result.
  [Broadcast match report and embedded club updates](https://www.tvazteca.com/aztecadeportes/america-vs-atlas-ver-en-vivo-gratis-jornada-4-apertura-2023-liga-mx),
  [league announcement quoted contemporaneously](https://mexico.as.com/futbol/futbol-mexicano/atlas-vs-america-cambia-de-horario-n/).
- Clausura 2024: the cancelled April round-15 stubs `11911065` (Pumas-San Luis)
  and `11911068` (Atlas-Leon) are superseded by `11911187` (San Luis-Pumas,
  19 January, round 2, 3-1) and `11911139` (Atlas-Leon, 18 February, round 7,
  0-1). Raw rounds and venues stay intact; only those exact duplicate pairs
  are recognized. [Pumas' round-2 fixture announcement](https://pumas.mx/noticias/2224/antecedentes-num%C3%A9ricos-vs-san-luis),
  [San Luis' record of the 3-1 result](https://www.atleticodesanluis.mx/noticias/1489/Vamos-Potosinos),
  [Atlas' round-7 match report](https://atlasfc.com.mx/post/https-www-atlasfc-com-mx-articles-atlas-fc-recibi-c3-b3-a-la-fiera-20882).
- Apertura 2026, round 7: the postponed America-Tijuana `16317921` and
  Queretaro-Monterrey `16317922` have exact scheduled replacements `17025169`
  and `17020784`. The clubs announced 28 October and 14 November respectively.
  These mappings require both records to be scoreless and the replacement to
  remain `notstarted`; they resolve inventory duplicates only. Neither future
  match enters the 88-match completed snapshot, which must reconcile against
  every club's captured standings count. Other future duplicates stay blocking.
  [Xolos' announcement listing](https://www.xolos.com.mx/sitio/index.php/),
  [Rayados' date announcement](https://www.rayados.com/es/noticias/21586/rayados-disputara-la-jornada-7-en-noviembre).

### Deferred postprocessing of Mexican rounds

After publication, combine Apertura Y and Clausura Y+1 in that order, keeping
both provider identities and the original rounds in the source files. In the
combined view, offset Clausura by the validated regular-round span of Apertura:
normally 1-17 followed by 18-34; after Apertura 2019 the offset is 19. Use each
tournament's actual validated span rather than assuming every tournament lasts
17 rounds. Clausura 2020 contributes only its ten played rounds.

Keep match ordering by actual kickoff timestamp, with event ID as a stable tie
breaker, so postponed games remain chronologically placed. A nominal round
number is not a replacement for time ordering. Deduplicate by provider event ID
and refuse contradictory identities. Leave incomplete/latest cycles explicit.
If a later experiment needs one global round sequence across all cycles, apply
the same cumulative-offset rule to the ordered tournament list.

This join/renumbering is deferred as requested. No source rounds or existing
datasets have been rewritten, and no extra match columns have been introduced.

## Operation

From the repository directory, print/reuse prepared IDs:

```powershell
.\.venv\Scripts\python.exe -B scraping/run_archive_expansion.py
```

This is the safe default while another campaign is running. The saved selections
are in `data/website_capture_v3/batches/archive-*/batch.json`; their corresponding
discovery configurations and response databases are in the normal discovery
directory. Re-running preparation reuses IDs and rejects changed settings.

After the pending historical batch is finished:

1. Finish/stop its Python watcher. Use one conversion watcher at a time.
2. Start the watcher for the desired new group, for example:

   ```powershell
   .\.venv\Scripts\python.exe -u -B scraping/run_archive_expansion.py --watch europe-2010-2015
   ```

3. In the existing Sofascore Chrome tab's DevTools → **xDiyo**, disconnect the
   completed collection. Paste the group's **Discovery ID**, select
   **Create / reconnect**, then **Discover / resume**. Do not use Create new
   discovery with the panel's single-season form for these multi-season groups.
4. Once discovery reports ready, inspect its per-season counts and select
   **Create collection campaign**, then **Start / resume**. This creates one
   campaign for the entire group. Repeating the control retrieves the same ID.
5. Leave the Sofascore tab, its DevTools and the watcher running. Conversion
   occurs automatically as each league-season/tournament finishes. The watcher
   prints the campaign ID and checks that every intended cohort was published.
6. Repeat with the other group IDs after completion. There are no manual
   year-by-year campaign transitions within one group.

No extension UI/JavaScript update is needed: opening the saved Discovery ID uses
the updated Python native writer. If an old writer remains connected, close and
reopen DevTools at the transition between campaigns so it loads the new Python
code. Do not reload or disconnect an active collection to prepare these files.

For a read-only progress report:

```powershell
.\.venv\Scripts\python.exe -B scraping/run_archive_expansion.py --status
```

The default and `--status` create missing local preparation files, if necessary,
but do not start discovery, collection or conversion. Ctrl+C stops a watcher;
the extension's Pause controls collection. Reconnect the same ID to resume.

Japan's confirmed choice is saved in `archive_expansion_selection.json` as
`japan_2026_j2_j3: "skip"`. Once prepared, do not edit saved configuration to
alter scope; re-running the script verifies it against the saved batch.

## Reviewed European provider inconsistencies

The 2026-10-04 discovery found the following exceptions. They are resolved only for the
exact competition, season, event IDs, teams, rounds and kickoff timestamps below;
raw responses and provider stage labels remain unchanged. Every season still
has to pass the complete pairing and round validation.

- Championship 2013/14: Wigan-Doncaster (`3967161`, 20 August 2013) is stored in
  the provider's round-3 bucket but fills the only missing round-2 pairing. Its
  normalized round is 2. [The FA's fixture list](https://www.thefa.com/~/media/Files/PDF/Leagues/FLC-fixtures1314)
  confirms the scheduled opponents and kickoff; the [season schedule](https://www.transfermarkt.co.uk/wigan-athletic/spielplan/verein/1071/saison_id/2013)
  identifies matchday 2. The original round bucket is retained for replay.
- Premier League 2010/11: the unplayed Blackpool-Wigan stub `1379596` is
  superseded only when the exact reversed-venue match `1380091` is present with
  the verified 0-4 result. Both are dated 14 August 2010.
  [Premier League match account](https://www.premierleague.com/ar/news/698090).
- Premier League 2013/14: Chelsea-Villa was moved from 31 August to 21 August
  2013. The earlier completed 2-1 match `4166893` supersedes the obsolete
  postponed stub `3964835`, with both identities checked.
  [Contemporary report quoting the Premier League announcement](https://www.standard.co.uk/sport/football/chelseas-premier-league-fixture-with-aston-villa-brought-forward-8732292.html).
- Ligue 1 2014/15: Montpellier-Lyon on 19 October 2014 was reversed to Lyon
  after flooding. The cancelled original `8060497` is superseded only when the
  exact played event `5509696` is present with the verified 5-1 result.
  [Montpellier's season account](https://mhscfoot.com/articles/2015-2016/equipe-pro/retrospective-2014-2015-octobre-2014/).
- Serie B 2013/14: Carpi-Siena `4167965` was played on 24 November 2013 after
  the previous day's postponement and finished 0-1. Its exact identity, kickoff,
  stale status code and score must all match before collection and normalized
  status treat it as completed. Raw status and status code are retained.
  [Contemporary match report](https://www.repubblica.it/sport/calcio/serie-b/2013/11/24/news/palermo_latina-71816673/).
- Premiership 2010/11 and 2013/14: the postponed St Johnstone-Celtic stub
  `1380321` is superseded by the earlier 0-1 match `1694223` (12 April 2011).
  Hearts-St Mirren `3971519` is superseded by the completed 2-1 record `4934258`
  with the same kickoff (29 January 2014). Both identities and scores must match.
  [SPFL archive](https://spfl.co.uk/league/premiership/archive/298?page=2),
  [Hearts match report](https://www.skysports.com/football/hearts-vs-st-mirren/report/289467).
- Premiership 2014/15: Celtic-Partick `5587698` (3 December 2014, 1-0) is the
  delayed round-1 fixture, mislabelled round 32. The separate April round-32
  meeting remains unchanged. [Celtic match report](https://www.celticfc.com/news/7222),
  [season round listing](https://www.topscorersfootball.com/scottish-premiership/standing-and-matches-2014/2015).
- Pro League 2011/12, 2012/13 and 2013/14: captured standings use Mons ID `2906`,
  while all 30 Mons fixtures per season use `549294`. Both identify RAEC Mons.
  Validation reconciles these IDs only for season IDs `3414`, `4676`, `6241`,
  with 16 teams, 240 fixtures and every team's 30 appearances verified. The
  original fixture IDs and standings are retained; no global club alias is added.
- Pro League 2014/15: Standard-Zulte `5555478` (19 October 2014) was abandoned
  in the 89th minute; its 1-2 score was subsequently upheld. The exact stale
  postponed record is classified as abandoned for schedule validation and is
  excluded from completed-match collection, as other abandoned matches are.
  [Arbitration decision, pages 18 and 22](https://www.bas-cbas.be/cms/resources/141210-1-sentence-arbitral.pdf).

The obsolete stubs stay in discovery storage and the existing superseded-fixture
inventory. They are not added to the played-match queue. These rules do not
permit arbitrary reversed opponents, earlier replacements or round relabelling.

Premiership 2014/15 also has a misplaced game in hand: Inverness-Dundee United
`6726416` (5 May 2015, 2-1) is stored in round 37, but replaces the postponed
round-29 stub `5587666`. Discovery reads that one additional source bucket;
only the exact competition, season, match ID, teams and kickoff are mapped back
to regular round 29. The separate post-split match `6740413` (16 May, 3-0) and
every other post-split fixture remain excluded. Both raw round labels and the
obsolete stub are retained. This broader inspection was explicitly approved
on 4 October 2026.
[Contemporary match report](https://www.skysports.com/football/inverness-ct-vs-dundee-united/report/315594).

## Publication and validation

- Filenames keep the existing `League_YY_YY` convention. For calendar years the
  suffix is repeated: `J1_10_10.parquet` means calendar 2010, not 2010/11.
  Examples: `J1_Transition_26_26.parquet`, `J1_26_27.parquet`,
  `Liga_MX_Apertura_10_11.parquet`, `Liga_MX_Clausura_10_11.parquet`.
  The current Apertura file follows its resolved provider spelling, likely
  `Liga_MX_Apertura_26_26.parquet`.
- Raw provider `season_year`, round and stage payloads remain intact. The usual
  manifest records archive scope and distinguishes current snapshots. Outputs
  with conflicting provenance are not overwritten.
- Lineups, xG/shotmaps, heatmaps and the existing other capture fields retain
  their current profile. Commentary and H2H stay off. Missing provider features
  remain unavailable, not invented. Existing pacing and between-cohort rests
  are retained; optional random request delay remains `None`.
- Season lists are requested once per competition for the whole group. Format
  validation uses either captured standings or an explicit known team count,
  avoiding unnecessary standings requests for the historical Japanese and
  Mexican cohorts. All selected numbered round pages must still be captured.
- Missing seasons, unknown named stages, HTTP errors, duplicate/conflicting
  fixtures, missing round pages and schedule mismatches remain review issues.
  None is silently converted into a successful empty season.

The preparation and calendar-year conversion have been exercised with offline
tests. This does not certify live API coverage for all requested old seasons:
that check happens when the existing Chrome extension runs discovery.
