# Published odds data

These are the existing local outputs, published without changing prices,
missingness, fixture identities or source manifests. `publication.json` records
the exact published files, SHA-256 hashes, sizes and Parquet row counts.

## Footiqo

`footiqo/29a84259c0b8a53578d8b6bc/` contains the reviewed workbook extraction:
45,695 distinct provider fixtures across 135 league-season pairs in 13 leagues.
There are 2,121,796 valid prices, 116,754 missing prices and two invalid prices.
The fixture rows repeat across source worksheets; they are not additional
distinct matches. See the original `manifest.json` and `verification.json`.

The completed fixture crosswalk is
`mappings/2d830f97d6799add94b50663/`: 45,271 matches and 424 unmatched provider
fixtures. The other mapping versions remain available for existing pinned
consumers. `team_aliases.csv` and `fixture_overrides.json` preserve the matching
evidence. This is the scope of the original extraction, not a claim of odds
coverage for every league or season in the expanded native library.

```python
from xdiyo_analytics.odds import load_odds

prices = load_odds(
    "data/odds/footiqo/29a84259c0b8a53578d8b6bc",
    league="Premier_League", season="25_26", sheet="Odds",
)
```

Bookmaker identities and publication timestamps are unavailable for these
workbook prices. Opening/closing are vendor labels; publishing them establishes
neither historical availability at a prediction cutoff nor executable prices.
The original manifests' statement that the data was excluded from Git describes
its import-time state. Those immutable manifests are retained byte for byte so
their dependent crosswalk hashes and existing recipes remain valid.

## OddsPapi

`oddspapi/20261028/historical_odds.parquet` is a **partial archive checkpoint**,
with 413,244,086 price-change rows. Its recorded consolidation covers 7,258 of
442,738 planned fixture/bookmaker-batch requests, including 320 empty responses;
6,938 requests produced files. It is not a completed download. The frozen scope
contains 3,629 fixtures. `consolidation.json`, `download_scope.json`, fixture and
bookmaker plans and `team_crosswalk.json` retain that scope and identity evidence.

The archive includes in-play and inactive quotes. Consumers must apply the
existing quote-availability and market contracts before using it for betting.
Do not load the entire archive into pandas: select columns and apply filters or
read Arrow record batches.

The 1.81 GB archive uses Git LFS. After cloning, install Git LFS and run:

```sh
git lfs install
git lfs pull --include="data/odds/oddspapi/20261028/historical_odds.parquet"
python examples/verify_published_odds.py
```

For a smaller checkout, set `GIT_LFS_SKIP_SMUDGE=1` when cloning and pull the
archive later. The verification command intentionally rejects an unhydrated
LFS pointer. Footiqo tables are ordinary Git files and require no LFS download.

Account details, worker locks, request logs, SQLite recovery state, duplicate
download parts and intermediate reconciliation files remain ignored locally.
No collector or experiment was restarted to produce this publication.
