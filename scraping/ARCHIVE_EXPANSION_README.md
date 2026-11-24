# Future archive campaigns

## Coverage-table expansion, 9 November 2026

The two supplied division-coverage CSVs contain 47 leagues. Comparison with
the 289 outputs published at setup identifies 15 represented leagues and **32
new leagues** (28 first divisions and four second divisions). The CSV league
IDs are not Sofascore unique-tournament IDs. The saved comparison and provider
links are in `data/website_capture_v3/batches/coverage_expansion_2026-11-09.json`.
Existing published leagues are excluded from this expansion.

The user revised the requested start to **2017/18**, or **calendar 2017**,
through completed matches in 2026/27 or calendar 2026. Source-table coverage
dates do not narrow that request. Missing provider seasons remain explicit;
they cannot be silently omitted. Regular-season exclusions and capture settings
below still apply.

| Batch | Selection | Discovery ID | Preparation |
| --- | --- | --- | --- |
| `coverage-wave1-2017-2027` | Brazil A/B, Croatia, Norway, Portugal, Russia, Saudi Arabia, Slovenia, Sweden; 90 cohorts | `discover-coverage-wave1-2017-2027-8bf569e6` | All 23,184 captures and 90 outputs validated on 11 November |
| `coverage-europe2-2017-2027` | Austria, Bulgaria, Czech Republic, Denmark, Greece, Poland, Romania, Serbia, Switzerland, Turkey; 100 cohorts | `discover-coverage-europe2-2017-2027-ae490c01` | All 20,885 native captures and 100 outputs independently validated on 13 November; 20,888 published matches include three reviewed Greek supplements |
| `coverage-europe3-2017-2027` | Ukraine, Dutch Eerste Divisie, Portuguese Liga Portugal 2, Scottish Championship; 40 cohorts | `discover-coverage-europe3-2017-2027-49c2d446` | All 9,666 captures and 40 exact intended outputs independently validated on 13 November; collection and conversion finished |

Argentina, Australia, Chile, China, Colombia, Ecuador, Egypt, Peru and USA
remain in the saved request. An initial public-source format review is recorded
below and in the coverage plan; historical and provider checks remain open.
Do not describe these nine leagues as collecting or silently drop their
curtailed seasons, ordinary stages or separate provider tournament identities.
Each batch must finish discovery and pass every schedule check before
its collection campaign can be created. Preparation is not validation.

The new runner option watches a saved batch without preparing any old group:

```powershell
.\.venv\Scripts\python.exe -u -B scraping/run_archive_expansion.py --watch-batch data/website_capture_v3/batches/archive-coverage-wave1-2017-2027/batch.json
```

Adding `--status` reads that batch without starting a watcher. Keep one
publication watcher at a time. Use the saved Discovery ID in the existing
xDiyo panel, then Discover / resume. The previous three archive selections
remain unchanged.

Brazil's calendar 2020 Serie A edition is advertised by Sofascore as `20/21`,
season **27591**. Its explicit selector aliases are `2020` and `2020/2021`;
the resolved output retains the provider identity, `Brazil_Serie_A_20_21`.
This edition is distinct from calendar 2021, season 36166. The alias was
reviewed before collection, with discovery paused and disconnected. All 258
already captured response hashes were preserved while the original saved
season-inventory response was reprocessed locally. Evidence and configuration
hashes are in that batch's `reviewed_season_alias.json`.
[CBF's 2020 fixture document](https://www.fpf-pe.com.br/assets/uploads/159543778029.pdf?v=16108122997)
and [Flamengo's 2021 annual report](https://images.flamengo.com.br/public/arquivos/transparencia/17/1648765726084.pdf)
document the edition and its extension into February 2021.

The Brazil 2026 round-21 Chapecoense-Vasco duplicate is an unplayed postponement,
not two completed matches with competing statistical coverage. The exact
scoreless postponed record `15235440` is superseded only by scoreless scheduled
record `17449066`, with competition 325, season 87678, round 21, ordered teams
21845/1974 and both captured kickoff timestamps checked. The separate finished
Vasco-Chapecoense round-2 match `15237895` remains selected. Neither unplayed
round-21 entry enters the completed-match snapshot; raw responses are unchanged.
[CBF's official match and alteration notice](https://www.cbf.com.br/futebol-brasileiro/jogos/campeonato-brasileiro/serie-a/2026/chapecoense-x-vasco-da-gama-saf/832099)
and [Chapecoense's match announcement](https://chapecoense.com/2026/11/06/servico-de-jogo-para-chapecoense-vs-vasco-serie-a-do-campeonato-brasileiro-2026/)
confirm 14 November 2026 at 20:30 Brazil time. The batch's
`reviewed_brazil_fixture_2026-11-09.json` records the evidence and whether the
running worker has loaded the correction.

Discovery summary updates now invalidate only cohorts backed by the changed
provider season, including both views of the reviewed Mexican shared season.
Season-selector changes still refresh all identities. Completion and collection
creation force a fresh full validation. The events table has a competition/season
lookup index, and the watcher polls queue progress during enumeration instead of
rebuilding every season summary. After enumeration it revalidates on database
changes, even when job counts stay the same. Read transactions end before each
wait. The batch's `performance_recovery_2026-11-09.json` records before/after
timings, exact saved-row/configuration hash checks and live restart evidence.

### First-wave fixture review, 9 November 2026

The next wave is on hold until every fixture issue in this wave is resolved.
The original 3,221 discovery responses, event rows and selected configuration
are retained. The following exceptions match exact IDs, competition, season,
year, round, ordered teams, kickoff, status and scores. An obsolete entry is
excluded only when its exact reviewed replacement is also present. Raw payloads
are never rewritten. Regression evidence is in
`tests/fixtures/sofascore/coverage_reviewed_fixtures.json`; deployment evidence
is in the batch's `fixture_recovery_2026-11-09.json`.

| Obsolete event | Retained event | Verified reason |
| --- | --- | --- |
| 7649803 | 7457453 | Rudar-Gorica 2017/18 round 16 resumed after fog; original ID holds final 2-0 |
| 9764637 | 9614355 | Mura-Radomlje 2021/22 round 5 resumed after rain; original ID holds final 4-2 |
| 7885037 | 7885113 | Saudi 2018/19 round 4 actually Ahli 2-0 Batin |
| 7885066 | 7885186 | Saudi 2018/19 round 10 actually Fayha 1-1 Hazem |
| 9688833 | 10000039 | Saudi 2021/22 round 4 actually Hilal 0-0 Fayha |
| 10913432 | 10665958 | Saudi 2022/23 round 14 Nassr 1-0 Ettifaq played 22 January |
| 11347303 | 11459684 | HNL 2023/24 round 4 moved venue; Dinamo 0-3 Lokomotiva |
| 16275179 | 17184945 | HNL 2026/27 round 2 Rijeka-Dinamo scheduled 3 December |
| 15260931 | 17505158 | Norway 2026 round 26 Kristiansund-Valerenga rescheduled 9 November |
| 16653115 | 17356156 | Saudi 2026/27 round 10 Abha-Kholood scheduled 7 December |
| 16653146 | 17413886 | Saudi round 13 Ahli-Taawoun retains its postponed 7 November listing; new date unknown |
| 16653103 | 17329607 | Saudi round 9 Ittihad-Shabab scheduled 19 December |
| 16653113 | 17328039 | Saudi round 9 Riyadh-Hilal scheduled 19 December |
| 16653114 | 17328047 | Saudi round 9 Ahli-Nassr scheduled 6 March 2027 |
| 16653119 | 17291748 | Saudi round 10 Ettifaq 1-4 Ittihad completed at the same listed kickoff |
| 16378721 | 17391680 | Slovenia 2026/27 round 6 Celje-Radomlje scheduled 2 December |
| 16378725, 16890867 | 17488629 | Slovenia round 16 Koper-Aluminij scheduled 23 November |

Sources: NZS reports for [Rudar-Gorica](https://www.nzs.si/podrocja/novinarji/novice/rudar-do-zmage-proti-gorici-236862)
and [Mura-Radomlje](https://www.nzs.si/podrocja/novinarji/novice/mura-v-nadaljevanju-tekme-strla-odpor-radomelj-244016);
SPL results for [Ahli-Batin](https://www.spl.com.sa/ar/match/b447021dcb1e4f108a6fb9e1d25c72fd/%D8%A7%D9%84%D8%A8%D8%A7%D8%B7%D9%86-vs-%D8%A7%D9%84%D8%A7%D9%87%D9%84%D9%8A)
and [Fayha-Hazem](https://www.spl.com.sa/en/match/70b0c934973048659e4cad27391c2030/al-fayha-vs-al-hazem);
[Hilal's 2021 archive](https://alhilal.com/ar/matches/mens-team/results/2021/saudi-arabian-league),
[SAFF's 2022/23 results](https://www.saff.com.sa/championship.php?id=122&type=all),
[HNL's Dinamo-Lokomotiva report](https://hnl.hr/matches/78093923/dinamo-lokomotiva-0-3/),
[HNL's current Dinamo schedule](https://hnl.hr/klubovi/dinamo/),
[NFF's postponement announcement](https://www.fotball.no/turneringer/eliteserien/2026/kristiansund---valerenga-utsatt/),
[SPL's round-10 report](https://www.spl.com.sa/ar/news/rsl-matchweek-10-highlights),
[Ahli's round-13 postponement](https://en.alahlifc.sa/news/postponement-of-al-ahli-vs-al-taawoun),
[SPL's three round-9 changes](https://www.spl.com.sa/ar/news/spl-reschedules-three-fixtures-mw9),
and NZS schedules for [Celje-Radomlje](https://www.nzs.si/klubi/moski/prva-liga-telemach/tekme/nk-celje-nk-radomlje-1snl2627-2026-12-02-173000)
and [Koper-Aluminij](https://www.nzs.si/klubi/moski/prva-liga-telemach/tekme/fc-koper-nk-aluminij-kidricevo-1snl2627-2026-11-23-173000).
Scheduled and postponed entries validate the inventory but are never promoted
to completed captures. No new date is inferred for Ahli-Taawoun.

Russian IDs `8680226`, `8680347`, `8680361` and `8809808` are exact unplayed
awards, despite stale postponed statuses. Their score, `isAwarded=true` and
full fixture identity must match. They remain in schedule validation and are
excluded from completed collection, consistent with the existing Cosenza rule.
RFU decisions: [Orenburg-Ural](https://www.rfs.ru/news/211886),
[Tambov-Sochi](https://www.rfs.ru/news/211964),
[Krylya Sovetov-Sochi](https://www.rfs.ru/news/211977),
[Rostov-Rotor](https://www.rfs.ru/news/212338).

Saudi 2018/19 IDs `7884024` and `7885025` have reversed provider side labels.
The official results are Hazem 0-0 Wehda on 30 August and Ettifaq 3-2 Batin on
13 September. The return fixtures `8096128` and `8101297` need no correction.
Exact guarded normalization swaps match home/away columns and component side
labels while preserving team IDs, player IDs, scores attributed to each team,
statistics, coordinates and source JSON. Goal-action flags receive the same
correction. Schedule, publication and retained-output identity checks use the
same canonical team order; capture references retain the original provider
identity. All other events continue to require ordinary home/away balance.
Sources: official SPL head-to-head results for
[Hazem-Wehda](https://www.spl.com.sa/en/match/8be3d4d9ac2a452d9091acd697c8f462/al-hazem-vs-al-wehda)
and [Batin-Ettifaq](https://www.spl.com.sa/ar/match/2f8848b94eaa4b0d856a18c4bc5f068f/%D8%A7%D9%84%D8%A8%D8%A7%D8%B7%D9%86-vs-%D8%A7%D9%84%D8%A7%D8%AA%D9%81%D8%A7%D9%82).

Austria changes from four meetings in 2017/18 to a 22-round double round robin
before the split from 2018/19. Switzerland changes from four meetings to a
33-round triple round robin before the split from 2023/24. Source payloads
and round numbers remain unchanged; later split rounds are excluded.
[Austrian format reform](https://www.bundesliga.at/de/news/artikel/die-details-der-ligareform-so-wird-ab-2018-19-gespielt),
[Swiss 2023/24 season report](https://report.sfl.ch/report/sportlicher-rueckblick/2023-24/yb-jubelt-die-rueckkehr-des-strichkampfs).

### Second-wave standings review, 11 November 2026

Czech 2018/19 season 17475 returns the 16-team, 30-match regular table beside
six-team championship and relegation tables. The championship table is named
`Championship Round`, but its tournament label is abbreviated to
`Czech First League, Championship`. Selection now reads the table's own stage
labels as well as its tournament labels. It also recognizes a tournament label
formed from its exact parent competition name plus `, Championship`, or the
equivalent parent slug plus `-championship`, in archive and daily selection.
An ordinary league named `Championship` or `Scottish Championship` remains
eligible. Multiple regular tables and conflicting competition IDs still fail.
The [LFA's official format](https://www.lfafotbal.cz/clanek/34-herni-format-i-ligy-od-sezony-2018-19)
confirms the ordinary double round robin followed by separate split stages.
Raw responses and saved selections remain unchanged; retained-response recovery
is recorded in the second batch's `standings_recovery_2026-11-11.json`.

### Second-wave fixture review, 11 November 2026

The completed inventory retains its original 3,330 captured response bodies and
saved selection. Eight obsolete entries are superseded only when both exact
captured identities match, including competition, season, year, round, ordered
teams, kickoff, status and scores. Future replacements remain unplayed and do
not enter the completed-match snapshot. Full provider records and source links
are retained in `coverage_europe2_reviewed_fixtures.json` under the test fixtures
and `fixture_review_2026-11-11.json` in the second batch.

| Obsolete event | Retained event | Reviewed reason and source |
| --- | --- | --- |
| 10539414 | 11074941 | Bulgaria 2022/23 round 22 used the wrong CSKA club; [Lokomotiv Sofia 0-6 CSKA 1948](https://cska1948.bg/cska-razgromi-loko-sf) is the played fixture |
| 16295834 | 17285568 | Bulgaria 2026/27 round 10 Levski-Ludogorets has a [3 December listing](https://levski.bg/match/?matchId=24721); the provider replacement is still postponed and scoreless |
| 16295875 | 17485700 | Bulgaria 2026/27 round 16 Dunav-CSKA is [scheduled for 1 December](https://fcdunav.bg/programa/) |
| 16559945 | 17486940 | Greece 2026/27 round 7 Atromitos-AEK was [moved to 16 December](https://www.atromitosfc.gr/news/oristike-gia-tis-16-dekemvriou-to-atromitos-aek/) |
| 11351404 | 11351555 | Poland 2023/24 round 3 reversed venue; [Pogon-Slask was rescheduled](https://fantasy.ekstraklasa.org/news/nowy-termin-meczu-pogon-szczecin-slask-wroclaw), with captured result 0-2 on 30 August |
| 9723710 | 9625801 | Romania 2021/22 round 6 used the wrong Craiova club; the [official round-six appointments](https://lpf.ro/noutati/casa-pariurilor-liga-1-observatorii-etapei-a-6-a/2967) and [later head-to-head](https://lpf.ro/noutati/fc-u-1948-cs-mioveni-palmaresul-e-al-oaspetilor/4146) identify Mioveni 1-0 FCU 1948 |
| 16326495 | 17479954 | Serbia 2026/27 round 14 Radnicki Nis-Zeleznicar was [postponed to 25 November after a power failure](https://www.superliga.rs/odlozen-mec-14-kola-radnicki-nis-zeleznicar-novi-termin-25-novembar/); both entries are scoreless, and the replacement kickoff time remains provider supplied because the official notice did not fix a time |
| 16357129 | 17526778 | Switzerland 2026/27 round 14 canceled the reversed-venue listing; [Vaduz's schedule](https://www.fcvaduz.li/news/spielplan-brack-super-league-runde-11-bis-22) and [Lausanne's announcement](https://www.lausanne-sport.ch/pages/news/gc-ls3996) confirm Lausanne at home on 22 November |

Piast-Gornik event `7433450`, Poland 2017/18 round 26, was abandoned before
completion and awarded 0-3. The provider retains `postponed`, code 60, and no
`isAwarded` flag. Only that exact identity and score resolve the schedule;
the match remains outside completed captures. The [league decision](https://piast-gliwice.eu/aktualnosci/newsy/komunikat-komisji-ligi)
was [upheld by the PZPN appeal body](https://piast-gliwice.eu/aktualnosci/z-klubu/komunikat-najwyzszej-komisji-odwolawczej-pzpn).

Turkey 2022/23 contains **313 played matches**, plus **29 unplayed awards**
omitted from Sofascore's inventory. The [TFF withdrawal decision](https://www.tff.org/default.aspx?ftxtID=39572&pageID=285)
retained earlier results for Hatayspor and Gaziantep and awarded their remaining
matches from 6 February 2023. The exact round/home/away pairings were checked
against the [TFF fixture archive](https://www.tff.org/default.aspx?pageID=1628).
Validation reconciles these 29 official omissions in schedule counters only.
It requires competition 52, season 42632, all 19 exact clubs, all 313 completed
provider fixtures, final 36-game standings, the exact missing directed pairings,
and no withdrawn-club fixture after the cutoff or round 23. Other missing
fixtures still fail. No awarded event, result, capture or queue item is invented;
the collection selection remains exactly the 313 actual provider fixtures.
`coverage_europe2_withdrawals.json` retains the captured test fields and official
pairings, with regression coverage through collection-campaign preparation.

Greek 2020/21 season 32553 contains **179 of 182** ordinary Sofascore fixtures.
Round 18 lacks Panathinaikos-OFI (23 January, 2-0), PAOK-AEK (24 January, 2-2),
and PAS Giannina-Volos (24 January, 0-1). The [official Panathinaikos programme](https://www.pao.gr/wp-content/uploads/2021/01/22/my_match-program-pao-ofi2_compressed.pdf),
printed page 38, establishes the round and schedule. These are played matches,
not awards or stage exclusions. Calendar and head-to-head searches did not
recover their Sofascore event IDs. Do not substitute post-split meetings or
invent provider IDs.

On 11 November the user explicitly approved collecting the 179 available
fixtures while marking the three missing Sofascore captures and supplementing
their observed statistics from Flashscore. The exact reviewed sources are
[Panathinaikos-OFI](https://www.flashscore.com/match/football/ofi-crete-bTObIPJB/panathinaikos-SENMmweG/summary/stats/overall/?mid=z99UJF7d),
[PAS Giannina-Volos](https://www.flashscore.com/match/football/giannina-rN9JmpKO/volos-nDZbFBqG/summary/stats/overall/?mid=2F1GsJa3),
and [PAOK-AEK](https://www.flashscore.es/partido/futbol/aek-ANpZncAM/paok-Kry6HxA9/resumen/estadisticas/general/?mid=dIAQKehj).
`sofascore_web_v3/reviewed_sources/greek_2020_21_flashscore.json` retains the
visible DOM observations, final and half scores, original statistic labels,
timestamps, source URLs and independent fixture references. `source_gaps.py`
pins its hash. Source observations are not HARs or Sofascore responses.

Only these three reviewed identities may complete the schedule counters for
the exact 179-event, 14-team, 26-round season. All other validation failures
remain blocking. Campaign metadata explicitly records `source_capture_complete:
false`, three missing Sofascore fixtures, 179 selected captures and 182 expected
dataset matches. All 100 cohorts remain selected; Wave 2 captured **20,885 real
Sofascore events** and published **20,888 matches**, including the supplements.

The three supplements preserve known Sofascore competition/season/team IDs but
use separate internal event identities: `10**15 + base62(Flashscore match ID)`
with alphabet `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz`.
They never enter browser queues or pretend to have Sofascore event/custom IDs.
The normal publisher appends their **174 observed team-statistic rows** across
full match and both halves, preserving every captured native row. Missing
statistics, shots/xG, heatmaps and player data remain empty; no zeros or player
identities are imputed. Free kicks and throw-ins retain provider-specific keys.
The publication manifest and lineage retain the exact review, source gaps and
source hashes. Restart, reconciliation and retained-publication validation
check the 179 native identities and all three supplemental identities/values.

### Third-wave fixture review, 13 November 2026

Wave 2 finished cleanly, and its converter exited before the next watcher was
started. Its `completion_validation_2026-11-13.json` independently checks all
100 intended outputs, exact native and supplemental match identities, capture
coverage, normalization/schema checks, provenance, Parquet counts and publication
hashes. The 40-cohort third wave keeps the same capture profile and pacing.
Ukraine 2017/18 through 2019/20 selects first-stage rounds 1-22; Scotland
2020/21 selects its ordinary 27-round, three-meeting schedule. Other playoffs
and post-split fixtures remain excluded.

The following four seasons ended early. Their exact official club identities
and played totals, completed fixture identities, pairings, rounds and cutoff
dates are validated instead of pretending that the unplayed schedule finished.
Only these explicitly selected historical reviews use this validation path.

| Season | Played matches | Final played-table constraint | Evidence |
| --- | ---: | --- | --- |
| Eerste Divisie 2019/20 | 290 | 20 clubs, 29 played each; through round 29 | [KNVB season termination](https://www.knvb.nl/nieuws/betaald-voetbal/competitiezaken/59905/afwikkeling-voetbalseizoen-2019%E2%80%9920-een-feit) |
| Liga Portugal 2 2019/20 | 216 | 18 clubs, 24 played each; through round 24 | [Portuguese sports-arbitration record](https://www.tribunalarbitraldesporto.pt/files/decisoes/TAD_23-2020-2.pdf) |
| Scottish Championship 2019/20 | 137 | Exact ten-club table, 26-28 played; through round 28 | [SPFL final-season decision](https://spfl.co.uk/news/ladbrokes-premiership-and-spfl-season-201920-cur), [final rules and table](https://spfl.co.uk/admin/filemanager/files/shares/SPFL%20Rules%20and%20Regulations%2018-May-20%20%28MASTER%20COPY%20B%29%20CLEAN.pdf) |
| Ukrainian Premier League 2021/22 | 143 | 16 clubs: Rukh and Inhulets 17 played, the others 18; through round 18 | [UPL final table](https://upl.ua/ua/news/view/7075), [season termination](https://upl.ua/en/news/view/6244?id=6244) |

Nineteen obsolete entries are excluded only when their exact reviewed
replacement is also present. Guards cover both IDs, competition, season, year,
round, ordered clubs, kickoff, status codes and scores. Unplayed replacements
remain outside the completed-match snapshot.

| Obsolete event | Retained event | Reviewed reason and evidence |
| --- | --- | --- |
| 9348229 | 8853426 | Inhulets-Vorskla 2020/21 round 15, 2-2; [formal host retained despite venue change](https://upl.ua/ua/news/view/4991) |
| 8853492 | 8853543 | Olimpik-Desna 2020/21 round 7, 0-2; [played fixture](https://uaf.ua/article/40537) |
| 7441786 | 7775419 | Jong Ajax-NEC 2017/18 round 34; [ineligible-player review](https://www.knvb.nl/nieuws/betaald-voetbal/jupiler-league/47614/knvb-onderzoekt-speelgerechtigdheid-speler-jong-ajax-nec); invalid 2-0 replaced by played replay 1-0 |
| 7528258 | 7620751 | Oss-Jong AZ 2017/18 round 5; [KNVB ordered replay](https://www.knvb.nl/nieuws/betaald-voetbal/jupiler-league/39427/wedstrijd-fc-oss-jong-az-wordt-overgespeeld); invalid 0-2 replaced by replay 2-1 |
| 8246895 | 8593372 | Jong AZ-Almere 2019/20 round 23; [KNVB ordered replay](https://www.knvb.nl/nieuws/betaald-voetbal/keuken-kampioen-divisie/59388/systeemfout-leidt-tot-overspelen-jong-az-almere); invalid 3-3 replaced by replay 3-1 |
| 9567709 | 9985017 | MVV-Roda 2021/22 round 13, abandoned at 0-0; [resumed on 29 November](https://www.knvb.nl/nieuws/betaald-voetbal/keuken-kampioen-divisie/65474/mvv-roda-jc-op-29-november-uitgespeeld) and [finished 1-1](https://nos.nl/artikel/2407530-assehnoun-helpt-emmen-aan-zege-roda-niet-voorbij-mvv-in-restant-limburgse-derby) |
| 12196211 | 11408637 | LNZ-Minaj 2023/24 round 17, 1-2; retain the correctly timed identity with the most complete provider coverage: [15:30 schedule](https://uaf.ua/article/50844), [result](https://uaf.ua/article/50917) |
| 16412900 | 16693841 | Kudrivka-Shakhtar 2026/27 round 16; [reviewed venue arrangement](https://fckudrivka.com/2026/07/27/poiedynok-shakhtar-kudrivka-vidbudetsia-u-lvovi/); replacement remains unplayed |
| 16412938 | 17524909 | Kryvbas-Kharkiv 2026/27 round 6; [25 November reschedule](https://upl.ua/ua/news/view/12045?id=12045) |
| 16412944 | 17085331 | Livyi Bereh-Polissya 2026/27 round 21; [reviewed venue swap](https://www.upl.ua/ua/news/view/11948); future kickoff remains provider supplied |
| 16412985 | 17400310 | Obolon-Shakhtar 2026/27 round 26; [reviewed venue swap](https://upl.ua/ua/news/view/12063); future kickoff remains provider supplied |
| 16413004 | 16413128 | Shakhtar-Livyi Bereh 2026/27 round 13; [20 November home fixture](https://upl.ua/ua/news/view/12224) |
| 16317481 | 17540733 | Vitesse-De Graafschap 2026/27 round 14; [23 November schedule](https://www.degraafschap.nl/wedstrijdschema) |
| 16472617 | 17525221 | Leiria-Sporting B 2026/27 round 11, 2 December; [reported league reschedule](https://www.zerozero.pt/noticias/liga-portugal-anuncia-novas-datas-e-horarios-para-as-jornadas-11-e-12-da-segunda-liga/1249326?redird=1) |
| 16472618 | 17525218 | Benfica B-AVS 2026/27 round 11, 2 December; same league notice |
| 16472623 | 17510693 | Vizela-Feirense 2026/27 round 11, 22 November; same league notice |
| 16472627 | 17532374 | Porto B-Torreense 2026/27 round 11, 3 January; same league notice |
| 16362250 | 17517605 | Morton-Ayr 2026/27 round 12; [SPFL 1 December reschedule](https://clublogin.spfl.co.uk/news/rearranged-fixtures-51502) |
| 16362265 | 17502226 | Dunfermline-Arbroath 2026/27 round 15; [SPFL 1 December reschedule](https://spfl.co.uk/news/william-hill-championship-fixture-change-51496) |

The resumed MVV-Roda event `9985017` still has an abandoned status in the raw
provider record. Only its exact reviewed identity and final 1-1 score normalize
to finished. Original status and source payload remain retained. Missing
statistics or heatmaps stay unavailable; no component values are invented.

Mariupol-Dynamo `7431119`, Ukraine 2017/18 round 7, is an exact unplayed 3-0
award with a stale postponed status and no provider award flag. The
[UAF decision](https://uaf.ua/article/31997) resolves its schedule identity;
it is excluded from capture, leaving 131 played fixtures in the 132-match
regular stage. Shakhtar-Vorskla `8853483`, 2020/21 round 7, has reversed raw
home/away labels. The [official 1-1 result](https://upl.ua/ua/news/view/4609)
supports exact normalization through the existing match/component side logic.
Team/player attribution and coordinates are preserved; the correct return
fixture `9130460` is unchanged.

All 1,431 original response hashes and actual saved bodies, the discovery
configuration and batch selection were checked unchanged after these repairs.
`discovery_validation_2026-11-13.json` records all 40 clean cohort validations
and the exact 9,666 selected IDs. Regression coverage includes malformed
identities/statuses/scores, all four curtailed seasons through queue creation,
side correction through publication and resumed-match publication with genuinely
unavailable components: **345 tests passed**. The corresponding retained fixture
evidence is in `tests/fixtures/sofascore/coverage_europe3_reviewed_fixtures.json`
and `coverage_europe3_curtailed_seasons.json`. The third batch's
`handoff_from_wave2_2026-11-13.json` records actual collection advancement from
eight to 32 completed matches, live collector/converter commands and a fresh
waiting publication heartbeat for the new campaign. Publication is waiting for
the first complete cohort; no output is claimed published at this startup check.

### Wave 3 completion and restart hold, 13 November 2026

The completed campaign `chrome-coverage-europe3-2017-2027-0c2e695d` contains
**9,666 captured matches** and **40 published outputs** across Ukraine, Eerste
Divisie, Liga Portugal 2 and the Scottish Championship. Independent verification
is recorded in
`data/website_capture_v3/batches/archive-coverage-europe3-2017-2027/completion_validation_2026-11-13.json`.

The audit refreshed all 40 discovery schedule checks and verified all 1,431
stored discovery-response hashes; the exact selected fixture list/order; the
completed queue without gaps; capture component accounting, profile, schema and
table hashes; publication schemas/counts/identities; source lineage; canonical
pointers; and the finished watcher's exact output set. No source capture,
queue, discovery database, published output or live configuration was changed.
The four 2026/27 outputs remain completed-match snapshots at discovery time.
Provider-unavailable fields retain their missing/unavailable status.

**The user requested a computer restart before the next wave, then returned
and authorized continuation.** The coverage plan now records
`continuation_hold.state = released_by_user_after_restart`. Preserve all
completed wave evidence and reviewed historical exceptions; resolve each
next batch's outstanding format/fixture checks before creating collection.

## Remaining nine leagues: offline format preflight, 13 November 2026

The public-source review has started for all nine remaining leagues. Its
findings, source URLs and outstanding checks are stored in the existing
coverage plan's `remaining_leagues_preflight` record. This is a first pass,
not completed fixture validation. Eight remain `format_review_pending`.
Egypt's ten seasons are now prepared as the next batch, as described below.

### Post-restart metadata review and Egypt batch

`discover-inventory-937d9ffd589c` captured 195 season-list, round-list and
standings responses for the remaining leagues, preserving their original
bodies and hashes. It stopped at the MLS 2022 round-list endpoint
`/unique-tournament/242/season/40071/rounds`, which returned HTTP 404; eleven
later metadata jobs remain pending. That unresolved MLS evidence is retained.
Metadata inventory cannot create a collection campaign or mark a schedule as
validated. The new inventory route and existing archive regressions passed
266 tests, with no Python provider HTTP.

All ten Egyptian season mappings were captured before the MLS gap. The next
batch is `archive-coverage-egypt4-2017-2027`, with saved discovery ID
`discover-coverage-egypt4-2017-2027-052aff93`. Its
`format_preflight_2026-11-13.json` records the provider season identities,
source-response hashes, team counts and regular-round limits. The selection
uses 18 teams and 34 rounds through 2023/24, 18 teams and 17 initial rounds in
2024/25, 21 teams and 21 rounds including byes in 2025/26, and a completed-match
snapshot of the 20-team, 19-round initial stage in 2026/27. Championship and
relegation stages remain excluded. Broader historical date windows accommodate
postponed finishes; exact provider competition/season identity remains required.

The ordinary-stage rules follow the league's
[2024/25 announcement](https://egyptianproleague.com/news/details/167),
[2025/26 published schedule](https://media.egyptianproleague.com/api/file/354bbe81-471e-4132-b5d5-0e65ac28646e.pdf),
and [2026/27 announcement](https://www.egyptianproleague.com/news/details/331),
reconciled against the captured season-specific tables. Full fixture validation
is still required for all ten cohorts before collection can start.

### Egypt fixture review, 14 November 2026 JST

The same saved discovery completed all 307 enumerated requests and eight exact
recovered event requests. All ten seasons now pass schedule validation, selecting
2,585 completed matches. Campaign `chrome-coverage-egypt4-2017-2027-7fa312ce`
started on 14 November at 04:40 UTC; fresh completions and the converter heartbeat
were verified. See `discovery_validation_2026-11-14.json` and
`collection_start_2026-11-14.json` in the Egypt batch directory.
The exact evidence, source links and unresolved records are retained in the
Egypt batch's `fixture_review_2026-11-14.json`; the original 307 response hashes,
raw event records and saved configuration still match the pre-repair baseline.

Reviewed normalization covers six home/away presentation errors, ten misplaced
2020/21 rounds, and six obsolete fixture identities with exact played
replacements. Corrections require the reviewed competition, season, teams,
kickoff, status and score; raw provider payloads are unchanged. The regression
fixture is `tests/fixtures/sofascore/coverage_egypt4_reviewed_fixtures.json`.

Eight missing 2019/20 events were found in original Chrome responses:
8816749 (round 14), 8822047 (round 15), 8822046 (round 18), and
8823198, 8831404, 8831405, 8835330, 8835333 (round 19). Their raw responses lack
`roundInfo`. Contemporary match reports establish only their original rounds;
no statistics are borrowed or generated. The original Chrome HAR was saved and
its eight exact successful event responses were admitted under the collector
lock while disconnected. Every original round response, original event record
and saved setting was preserved. `missing_fixture_recovery_2026-11-14.json`
records the original HAR and response hashes. Do not repeat discovery or recovery.

The 2017/18 Raja-Ittihad matches 7675984 and 7779213 retain Raja as home.
The EFA's [round 17 sheet](https://img.youm7.com/ArticleImgs/2017/8/30/94461-9.jpg)
(fixture 145) and [round 34 sheet](https://media.dotmsr.com/ArticleImgs/2018/4/15/111126-جدول-الدوري.jpg)
(fixture 303) both list Raja first. These official documents resolve conflicting
secondary reports. Their images and hashes are retained in the batch review.
`reviewed_home_allocation` recognizes only these exact two finished identities,
rounds, kickoffs, teams and scores in the complete 306-match, 18-team season.
It changes only their expected directed pairing counts; all other pairing,
round, team and status checks remain required. Neither raw nor normalized sides
or statistics are reversed for these two matches. The discovery summary records
the exception and sources. The targeted discovery, recovery, side-attribution,
publication and campaign regressions passed 507 tests. The converter was
restarted with the reviewed code and is waiting normally for the first complete
captured season. Original collection settings and pacing remain unchanged.

The collector is **not needed for public rules/history research, local code
review or preparing validation requirements**. The existing native connection
does reject opening discovery while a collection campaign is attached
(`sofascore_web_v3/native_host.py`). This inventory review began after the
completed European wave was disconnected. The original offline preflight found
no retained event inventory for these nine leagues, including Colombia's
alternative identities; Egypt's subsequent discovery is recorded above.

The following table preserves the initial 13 November review. The 14 November
preparation below supersedes its resolved source and identity gaps, including
the recovered Chile 2019 official table; remaining fixture gates still apply.

| League | Established preflight finding | Work still needed before collection |
| --- | --- | --- |
| Argentina | Historical [Superliga editions](https://www.afa.com.ar/es/posts/reglamento-de-la-superliga-20172018) use split-year labels. The [2025 format](https://www.afa.com.ar/site/posts/formato-de-los-torneos-de-lpf-para-2025) has two ordinary zonal tournaments, including inter-zone games, before knockouts. | Map each historical edition and league/cup identity, including disrupted 2020; verify 2026 and historical counts. The old calendar-only plan label is not an approved selector. |
| Australia | [Current rules](https://aleagues.com.au/more/official-documents/) require 26 games per club with mixed two/three meetings per opponent. [2024/25](https://aleagues.com.au/news/when-does-the-a-league-season-start-dates-finals-series/) has 13 clubs and byes. Finals are separate. | Verify the fixture matrix and club totals year by year, plus the COVID-delayed 2019/20 end date. Do not impose uniform pair meetings. |
| Chile | The [2017 club report](https://cruzados.cl/wp-content/uploads/2022/02/Memoria-Cruzados-SADP-2017.pdf) records two short tournaments. The [2019 championship ended early](https://cruzados.cl/news/detail/efemeride-cruzada-catolica-campeon-2019-y-consigue-un-nuevo-bicampeonato/). | Resolve 2017 Clausura/Transicion identities, obtain readable official final 2019 counts and verify later club-count changes. The ANFP annual-report download timed out; no exact curtailed fixture set is approved yet. |
| China | CFA's revised [2020 first stage](https://www.thecfa.cn/zyls1/20200713/28732.html) has 14 rounds and 112 scheduled matches. The [revised 2021 format](https://www.sport.gov.cn/n20001280/n20067662/n20067613/c23432765/content.html) also separates initial groups from a later championship/relegation split. | Resolve those provider stages; reconcile 2022 awards/cancellations and all historical completed counts. Superseded full-season drafts cannot validate the pandemic editions. |
| Colombia | The [2025 rules](https://dimayor.com.co/wp-content/uploads/2025/04/REGLAMENTO-LIGA-V1-2025-04-03.pdf) have an extra derby round; the [2026 rules](https://dimayor.com.co/wp-content/uploads/2026/01/REGLAMENTO-LIGA-2026-V12.pdf) have 19 ordinary rounds without it. | Reconcile IDs 241, 11539 and 11536 without duplicate fixtures. Verify each tournament's extra pairings and the revised single 2020 ordinary phase. Semifinal quadrangular groups remain excluded. |
| Ecuador | [2024 rules](https://ligapro.ec/wp-content/uploads/2023/12/Reglamento-Competiciones-ed.-2024-1.pdf) include two ordinary full-field legs. [2026](https://ligapro.ec/se-termino-la-espera-volvio-nuestra-liga-ecuabet/) has a 30-game ordinary phase followed by rank-based groups. | Verify each older format, especially 2017-2019. Include an ordinary second stage; exclude championship/relegation/qualification splits. |
| Egypt | [2024/25](https://egyptianproleague.com/news/details/167) starts with one league round before a nine/nine split; the [2026/27 announcement](https://www.egyptianproleague.com/news/details/331) describes a later top-six split. | Verify the senior-league club and first-phase round counts for each edition, especially 2025/26 and 2026/27, and historical delayed dates. Do not confuse youth or cup rules with the league. |
| Peru | [2026 rules](https://liga1.pe/wp-content/uploads/2026/03/Reglamento-Liga1-Te-Apuesto-2026.pdf) explicitly distinguish Apertura and Clausura from Play-Offs and a separate cup. | Obtain authoritative 2017/18 and 2020/21 stage rules, then reconcile withdrawals, annulments and provider identities. Historical group and crossover assumptions are not yet approved. |
| MLS | [2026 rules](https://bracket.mlssoccer.com/league-reports/competition-guidelines/) combine conference and cross-conference opponents. [2020 rules](https://www.mlssoccer.com/news/2020-mls-competition-guidelines) count MLS is Back group fixtures as regular-season games, and [final totals were unequal](https://www.mlssoccer.com/news/mls-use-points-game-determine-final-2020-regular-season-standings). | Verify each year's opponent matrix and actual club totals; resolve/deduplicate 2020 group fixtures. MLS Cup playoffs and separate cups remain excluded. |

The existing schedule validator assumes constant pair multiplicities and, for
groups, disconnected equal-sized round-robin components. The findings above
identify **format-specific work**, not grounds to lower validation standards:

- Select ordinary stages using exact competition/season/stage identities. Keep
  round resets separate; a numerical round or group flag alone is insufficient.
- Check uneven schedules against their approved opponent matrix, per-club
  played totals, duplicate identities and round occupancy. An aggregate match
  count alone is insufficient.
- Reconcile curtailments, withdrawals and administrative results against
  official final records and exact event identities. Nominal scheduled games
  are not proof of actually played matches.
- Verify historical date windows and current completed-match cutoffs. A late
  played date must not move a fixture into another provider season.
- Add and test any necessary rules against retained discovery evidence before
  queue creation. Do not change the running wave's scraper code, settings or
  connection during this preflight.

Next steps are to close the historical source gaps, prepare the explicit
per-season rules, and obtain the missing provider inventories after the current
wave has finished and passed validation. The number of remaining batches and
outputs is deliberately unset: short tournaments and multiple provider
identities must be resolved first. No requested league or year is dropped.

### Remaining eight leagues: combined archive, 14 November 2026

After Egypt completed and all ten outputs validated, the remaining Argentina,
Australia, Chile, China, Colombia, Ecuador, Peru and MLS inventories were
reconciled together: **88 exact provider-season identities, 112 ordinary
phases and 22,968 capture targets**. Use the prepared
`archive-coverage-remaining-2017-2027/batch.json` and saved discovery
`discover-coverage-remaining-2017-2027-4a304569`; never recreate per-league
campaigns or rediscover the completed inventory.

The durable review is
`data/website_capture_v3/batches/coverage_remaining_preparation_2026-11-14/PREPARATION.md`.
`combined_archive_preparation.json` and `fixture_format_validation_current.json`
record the all-cohort gate. `fixture_decisions_2026-11-14.json` names every exact
exception and source. The raw evidence and 195 initial metadata hashes are
unchanged. Sixty explicit dispositions distinguish played administrative
results, annulled matches, abandoned games, unplayed awards and one misplaced
friendly. Unplayed/unfinished fixtures are not completed-match targets.
Awarded played outcomes retain their provider flag and must not be treated as
ordinary on-pitch scores. Cusco-Cienciano 9679732's later TAS-restored 2-2 is
explicitly distinguished from the captured provider's historical 3-0 award.

The phase-aware import rechecks complete page chains, frozen response hashes,
exact season and table identities, every club's played/W/D/L/goals, ordinary
round occupancy and schedule pair/home counts before queue creation. Eight
historical Australian seasons additionally match the official guide's full
result matrices. Two 2021/22 home assignments and the 2024/25 matrix axes were
resolved against dated match evidence; only the exact 2023 Grand Final is
removed from the derived regular-season reference. Uneven derby/crossover
pairs are frozen from captured fixtures and constrained by official format
rules, without claiming an external per-fixture oracle for every season.

Exact normalization guards cover rounds for 9927907, 9652499, 12014145 and
10829731; reversed Chinese sides for 10536153, 10536163 and 10943287; and
Cuenca-Catolica 11031965's away score (periods/incidents 3, erroneous header 4).
Raw data remains intact and the publication cohort includes these reviews.
Original HAR entries supply missing Colombia event 16390806 and fresh 2026
standings. Missing ordinary MLS rounds remain null under an exact fixture-ID
scope. The four unavailable MLS round routes remain unavailable evidence.

`prepare_remaining_archive.py` defaults to audit-only and requires `--prepare`
to import the reviewed evidence into the immutable archive. It never starts
collection. The existing Chrome xDiyo panel creates/resumes the one campaign;
`run_archive_expansion.py --watch-batch <batch.json>` runs its normal converter.
Keep lineups, shots/xG, heatmaps, five-minute cohort rests and session pacing;
comments and H2H stay off. **652 offline regression tests passed**. Consolidation
and verified duplicate cleanup freed 252,657,902 bytes while retaining original
source evidence and all datasets. Per-cohort publication still requires full
capture, schema, count, provenance and hash validation.

## Original archive selections

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
