# Odds database

`xdiyo_analytics.odds` extracts fixture metadata and prices from the supplied
imported workbooks into Parquet. It filters by **country + league**, and keeps only
league-season pairs present in the native data directory's manifest filenames.
The caller chooses the seasons explicitly. The current extraction includes all
matching seasons through **2026/27**, including the partial current season.

The purchased data lives under `data/odds/database/`, which is excluded from Git.
The extraction code and documentation can be shared without sharing the data.
The original workbooks remain unchanged.

## Extract and load

```python
from pathlib import Path
from xdiyo_analytics.odds import extract_odds, load_odds

downloads = Path.home() / "Downloads"
snapshot = extract_odds(
    [
        downloads / "corners_cards_odds.xlsx",
        downloads / "football_odds.xlsx",
    ],
    data_root="data/xDiyo_data",
    output_root="data/odds/database",
    seasons=[f"{year:02}_{year + 1:02}" for year in range(15, 27)],
)

corners = load_odds(
    snapshot,
    league="Premier_League",
    season="26_27",
    sheet="Corners_Closing_Odds",
)
fixtures = load_odds(
    snapshot,
    league="Premier_League",
    season="26_27",
    sheet="Corners_Closing_Odds",
    kind="fixtures",
)
over_95 = corners.loc[
    (corners["selection"] == "over")
    & (corners["line"] == "9.5")
    & (corners["status"] == "valid")
]
```

`snapshot` is an ordinary directory path, returned automatically; users do not
need to provide hashes. Identical source files, league-season scope and mapping
reuse the same snapshot after checking its files. Updated inputs create a new
directory. An incomplete extraction raises an error rather than reusing partial
data. Both workbooks must have distinct filenames.

## Available sources

| `sheet` | Prices retained |
| --- | --- |
| `Corners_Closing_Odds` | Closing over/under 7.5, 8.5, 9.5 and 10.5 total corners |
| `Cards_Closing_Odds` | Closing over/under 2.5, 3.5, 4.5 and 5.5 total yellow cards |
| `Odds` | Opening/closing 1X2, goals O/U 0.5 through 4.5, BTTS; closing Asian handicap columns |

The main workbook's result/statistics sheets are never opened. The observed
corner/card counts adjacent to the prices are never decoded or exported.
Formula cells in selected fields are rejected. Price cells outside the selected
league-season population are not decoded.

## File layout and columns

```text
data/odds/database/<snapshot>/
    manifest.json
    league=Premier_League/season=26_27/Corners_Closing_Odds/
        fixtures-0000.parquet
        quotes-0000.parquet
        ...
```

Each quote row represents one source fixture, market, selection, line and timing.
`decimal_odds` contains a finite numeric price greater than 1 when valid;
`raw_price` retains its original text. `status` distinguishes `valid`, `missing`
and `invalid_decimal_odds`. Missing opening prices are never replaced by closing
prices. Missing or invalid prices remain recorded with a null numeric price.

`source_league` and `source_season` use native names such as `Premier_League` and
`26_27`. The fixture table also retains the provider's original names, season
text and available team/league IDs. IDs are exact strings, not floating-point
numbers. `line` is an exact decimal string; absent lines are null.

Both tables retain the source filename, worksheet, row and snapshot. Quotes also
retain their source column. The manifest records source file SHA-256 hashes,
sizes, headers, mappings, requested native pairs, partition hashes, coverage
counts and excluded populations. Export chunks keep memory bounded; the loader
reads only the requested league-season-sheet partition and verifies its hashes.

## Interpretation and integration boundary

- These are **provider fixture IDs**, not native event IDs. Duplicate IDs within
  a selected worksheet are rejected. The same ID can legitimately occur in
  each of the three source sheets; join using provider identity, never row order.
- This is a league-season extraction, not a native match intersection. Provider
  playoffs or other extra fixtures may remain in a matching league-season.
- `start_datetime` retains the provider's original Excel serial/text. Its timezone
  has not been established. `bookmaker` and actual `quote_timestamp` are unknown.
  Opening/closing labels alone do not establish availability at a prediction cutoff.
- `period="FT"` identifies the source's full-time columns. Settlement metadata
  explicitly remains unmapped. In particular, the sign of the vendor's away
  handicap must be verified before native handicap settlement is enabled.
- The extract does not modify feature preparation or training. The native
  crosswalk and `OddsSeries` integration below connect it to the betting
  reporters and UI. Never equate provider IDs with native event IDs.
- Coverage is the availability in these files, not a claim that every fixture
  has every price. Inspect `status` and the snapshot's coverage counts.

League aliases are explicit in the adapter. An optional `league_aliases` mapping
replaces that mapping, keyed by `(provider_country, provider_league)` and valued
by the native league name. Unknown leagues are excluded and listed in the manifest.

## Match fixtures once, then reuse the mapping

```python
from xdiyo_analytics.odds import build_fixture_crosswalk, save_crosswalk
from xdiyo_analytics.odds.crosswalk import native_fixture_metadata

seasons = ["25_26"]  # Explicit; may include other imported seasons.
native = native_fixture_metadata("data/xDiyo_data", seasons=seasons)
mapping = build_fixture_crosswalk(
    snapshot, native, seasons=seasons,
    team_aliases={
        # (native league, exact vendor team name): native team ID
        # ("Premier_League", "Reviewed vendor name"): 123,
    },
    date_tolerance_days=0,
    native_timezone="UTC",
)
print(mapping["status"].value_counts())
review = mapping.loc[mapping["status"] != "matched"]
crosswalk = save_crosswalk(
    mapping, snapshot=snapshot, output_root="data/odds/mappings"
)
```

Only fixture identity/time columns are read from native tables. No vendor results,
native scores, statistics or prices are used in matching. Automatic acceptance
requires a unique ordered home/away team pair in the same mapped league and
season, within the selected date tolerance. Exact unique team names ignore case
and surrounding whitespace. Different names need an explicit alias to a native
team ID. There is no automatic fuzzy acceptance.

The tolerance is an integer from 0 to 7 **calendar days**. Vendor timezone remains
unknown; the comparison uses the native kickoff's date in `native_timezone`
(UTC in the UI). This is recorded evidence, not proof of equal instants. Provider
Excel timestamps are rounded to seconds to reconcile serialization noise; the raw
values remain in `source_evidence`. Substantive disagreements between provider
sheets are quarantined. Ordered sides remain nominal home/away even on neutral
grounds. Postponements outside the chosen tolerance remain unmatched; repeated
fixtures/playoff stages yielding multiple candidates remain ambiguous. Multiple
provider IDs cannot silently claim the same native fixture.

The saved mapping keeps `matched`, `unmatched`, `ambiguous` and `conflict` rows,
candidate counts, reasons, raw provider evidence, resolved native team IDs and
alias evidence. Its accepted key is
`source_league, source_season, competition_id, season_id, event_id`.
Only accepted rows supply prices. Correct team aliases or metadata and rebuild;
the content-addressed mapping creates a new version. Crosswalk manifests pin the
exact odds snapshot manifest and hash their Parquet payload. Relocate the whole
snapshot/crosswalk directories and update their path fields together.

## Select quotes for either betting reporter

```python
from xdiyo_analytics.odds import OddsSeries, paired_quotes
from xdiyo_analytics.evaluation import BetOffer, BetSpec
from xdiyo_analytics.labels import BetOption, MatchTotal
from xdiyo_analytics.features import Stat

option = BetOption(
    MatchTotal(Stat("ALL", "Match overview", "cornerKicks")),
    selection="over", line=9.5,
)
prices = OddsSeries(
    snapshot="data/odds", crosswalk=str(crosswalk),
    seasons=("25_26",), leagues=("Premier_League",),
    market="corners", selection="over", line=9.5, quote_type="closing",
    settlement_confirmed=True,  # Only after verifying native/vendor equivalence.
)
offer = BetOffer(option, odds=prices)  # BetOutcomeReporter offers
manual = BetSpec(option, odds=prices, take=True)  # BetPerformanceReporter bets
```

`OddsSeries` is callable and returns a Series aligned to the reporter's prediction
occurrences using the full native identity. It also exposes `quotes()` for provider
rows and `resolve(context)` for aligned prices plus provenance. This integration
supports match layout totals and 1X2, not team markets. Snapshot and crosswalk
manifest hashes are pinned automatically at construction, persisted in reporter
configuration/provenance, and checked on use. Users need not type hashes.

Every quote change is a reporting input: market, selection, line, timing, seasons,
league subset, snapshot and crosswalk. No rows are removed from the forecast
population. Missing/invalid quotes and unmatched fixtures yield no bet, even if
the reporter has `default_odds` set. That fallback still works for ordinary offers
with no odds configured. Fixed scalar odds and identity-indexed tables remain
supported. There is no implicit provider, snapshot, price or bookmaker mixing.

For a 1X2 source, use `market="1x2"`, `selection="home"`, `"draw"` or `"away"`,
and `line=None`. `Outcome(perspective="home")` win maps to home and loss to away;
away perspective reverses those two. Draw remains draw. Choose
`HighestExpectedProfit`, since `TightestLine` is only for over/under. A corners
classifier cannot supply 1X2 probabilities.

Paired availability for goals or 1X2 retains the union of offered fixtures:

```python
from dataclasses import replace
home = replace(prices, market="1x2", selection="home", line=None)
availability = paired_quotes(home)
# opening, closing, opening_status, closing_status,
# missing_opening, missing_closing, paired; indexed by provider match_id.
```

There is no automatic paired-only restriction. Missing opening stays missing;
no price is chosen using a realized outcome. Corners/cards have no opening side,
so asking for opening or paired prices raises a clear error.

### Settlement checks

| Market | Required native target / behavior |
| --- | --- |
| Corners | `MatchTotal(Stat("ALL", ..., "cornerKicks"))`, numeric `value` |
| Yellow cards | `MatchTotal(Stat("ALL", ..., "yellowCards"))`, numeric `value`; verify second yellows/bench cards and exclude reds |
| Goals totals | `MatchTotal(Stat("ALL", ..., "goals"))`, numeric `value`; verify regulation-time definition |
| 1X2 | Home/away goals `Outcome`, higher-is-better; regulation-time result; no draw-no-bet reinterpretation |
| BTTS / Asian handicap | Prices can be inspected, but native selection/settlement integration is explicitly disabled |

`settlement_confirmed=False` is the default and blocks betting evaluation until
the caller confirms native target and void-rule equivalence to the vendor's
regulation-time convention. This is especially relevant to `Outcome(source=None)`,
which uses native current scores and does not itself separate extra time.
Integer totals require push on equality; quarter lines are rejected. Existing
synthetic settlement/ticket tests cover win, loss, push and void accounting.
No bookmaker-specific void rules are inferred from the spreadsheets.

## Experiment builder

In **Post-training analysis**, add a Bet Outcome Reporter (or manual Bet Performance
Reporter). For each offer/bet:

1. Enable **Odds**, then choose **Odds database** under **Odds source**.
2. Select seasons and optionally leagues. The current odds database is selected automatically; there is no version picker.
3. Choose a saved **Fixture mapping**, or use **Build fixture mapping**. This links provider match IDs to native match IDs. The most comprehensive mappings appear first. Its default
   tolerance is zero calendar days. The result gives matched/quarantine counts
   and a local Parquet review path.
4. Saved aliases in `data/odds/team_aliases.csv` are applied automatically. For additional aliases, supply a CSV with `source_league,vendor_team,native_team_id`.
   IDs are read as strings. Rebuild after reviewing aliases. Duplicate alias keys fail.
5. Choose market, selection, line where relevant and quote timing. Corners/cards
   expose closing only. 1X2 hides the line control and offers home/draw/away.
6. Confirm settlement equivalence after reviewing the target definition.

**Refresh odds sources** discovers manifests under `data/odds/database/` and
`data/odds/mappings/` in the builder workspace. The UI builds mappings;
initial workbook import uses `extract_odds` in Python. External snapshot paths
are supported by imported recipes/Python. Version-1 recipes remain compatible;
`input.OddsSeries` is a registered constructor inside existing reporter fields,
not a new top-level recipe section. Python/notebook exports use the same node.
The `input.Table` identity picker now includes `source_league` and `source_season`.

Both reporters retain quote provenance in the ledger and a downloadable
`odds_provenance` table. Bet Outcome alternatives distinguish missing prices,
unmatched fixtures and policy rejection. Performance can reuse the exact earlier
decision ledger. Reports carry the unknown-bookmaker/quote-time caveat. Prices do
not establish executable T-minus-24-hour entries, CLV or common-time tickets.

### Report-only reuse

With `reuse=True`, changing only the post-training odds source or quote timing
refreshes reports from retained predictions and restores saved fitted models.
It does not fit or predict again. Post-analysis signatures include pinned odds
configuration, and ledger artifacts retain its provenance. Feature preparation,
pre-training selectors and selection evidence still determine fitting identity.
The adapter's own source files are excluded from the generic training-source hash;
reporters used as selection evidence still participate via their signatures.
This integration changes shared library code and can invalidate artifacts made
under an older code revision. Cross-revision compatibility is not promised.

## Standalone synthetic example and verification

Run `python examples/odds_synthetic.py --output <fresh-directory>` to generate
an entirely synthetic source workbook, native data, snapshot, crosswalk, importable
`recipe.json`, Python export and notebook export. The generator does not train.
Import that recipe in the builder to exercise the full flow without paid data.

Automatic tests use synthetic data for exact large IDs, ambiguous/reversed team
matches, explicit aliases, bounded dates, one-sided opening missingness, identity
alignment, market guards, serialization, fixed-price compatibility and report-only
reuse with fit/predict forbidden. The original six-league sample has been
superseded by the complete reconciliation below, using fixture identities and
schedules across all overlapping exports. This establishes mapping coverage,
not profitability or valid prices for every market.

## Automatic database selection

The builder uses the most recently imported database under `data/odds`. A new run resolves it automatically and retains its identity with the result; already running jobs keep their original data. Explicit paths in older Python recipes continue to work. Fixture mappings must belong to the current database: rebuild a mapping after updating the import. Existing private directories remain readable without moving or rewriting their contents.

## Complete fixture reconciliation

The current local mapping covers all 135 overlapping league-seasons in 13 leagues,
from 2015/16 through the available 2026/27 exports. It maps **45,271 distinct
provider fixtures to 45,271 distinct native fixtures**. That is 99.916% of the
45,309 native fixtures. The remaining 38 native fixtures have no corresponding
fixture in the supplied odds export; 424 provider fixtures have no corresponding
fixture in the native export. These gaps remain explicit rather than being
assigned another meeting between the same teams. Fixture coverage does not imply
every market has a valid price.

All 473 league-specific team-name aliases are resolved. Aliases were established
from unique normalized names or at least three consistent fixtures against known
opponents, then checked against the complete ordered-team schedule. Detailed
fixture evidence is retained privately under `data/odds/reconciliation/`.
Different names on different source sheets may agree after alias resolution;
genuine disagreements still fail matching.

Nine individually reviewed fixtures use different original/resumption dates.
`data/odds/fixture_overrides.json` records the two exact dates, both identities,
league-season, reason and supporting source for each exception. Applying an
exception still requires the same ordered team IDs and unique native fixture.
It never widens the normal date window or substitutes a playoff for a league match.
Original kickoff times in the native dataset are unchanged. A mapped resumed
fixture does not establish that its quoted price was available before the original
kickoff; both betting reporters retain the schedule evidence with quote provenance.

Both the UI and `build_fixture_crosswalk()` reuse these saved rules automatically.
Python discovers the rules beside the versioned imports; callers can also use
`load_mapping_rules(directory)` and pass `team_aliases` and `fixture_overrides`
explicitly. Custom aliases supplement the saved aliases. Passing an explicit
`fixture_overrides=[]` disables saved schedule exceptions for that call.

Local audit outputs:

- `data/odds/reconciliation/summary.json`: totals and active complete mapping path.
- `data/odds/reconciliation/coverage.csv`: coverage per league-season.
- `data/odds/reconciliation/native_without_provider_fixture.csv`: all 38 native gaps.
- `data/odds/reconciliation/provider_without_native_fixture.csv`: all 424 provider gaps.
- `data/odds/reconciliation/team_alias_fixture_evidence.parquet`: alias corroboration.
