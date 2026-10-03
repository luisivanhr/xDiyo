# Standalone result-sharing formats

These templates display **already-exported evidence**. They are independent of
FootballExperiment, recipes, training, model selection and pre/post reporters.
They do not load model pickles, fit anything, select bets, multiply probabilities,
settle tickets, aggregate portfolios, or calculate new P&L/risk statistics.

## Build and share

```python
from xdiyo_analytics.reporting.templates import (
    build_tickets, build_curves, build_portfolio,
)

page = build_tickets("tickets.json", "shared/tickets")
page = build_curves("curves.json", "shared/comparisons")
page = build_portfolio("portfolio.json", "shared/portfolio")
```

CLI, with the repository `src` on PYTHONPATH or an installed package:

```text
python -m xdiyo_analytics.reporting.templates tickets tickets.json -o shared/tickets
xdiyo-report curves curves.json -o shared/comparisons
xdiyo-report portfolio portfolio.json -o shared/portfolio
```

Output directories must be empty. Validation runs before writing. Open
`INDEX.html` directly, or zip and share the **whole output directory**. Recipients
need a modern browser; no Python, server, CDN, login or notebook is needed.
Ticket chunks use local script files, so `file://` works without fetch/CORS.
Do not send only the HTML. All exported rows are included, including evidence
outside the currently visible filter. This is not a redaction mechanism.

Each bundle includes a portable `manifest.json`, exact source CSV/figure copies
in `evidence/`, and `BUILD.json` containing input hashes and checks. Its manifest
can rebuild another empty destination. Ticket CSV bytes are also embedded in the
policy chunk for browser downloads. Supplied asset attribution files are copied.
The `BUILD.json` provenance may retain original input paths; review the bundle
before distributing outside your intended audience.

## Ticket manifest: version 1.0

```json
{
  "schema_version": "1.0",
  "title": "Saved ticket evidence",
  "subtitle": "Selected policies",
  "notice": "Historical simulated selections; not actual purchases.",
  "footer": "Policies are separate alternatives; do not sum them.",
  "accounting_tolerance": "0",
  "policies": [{
    "id": "model-a/policy-1",
    "label": "Model A · policy 1",
    "group": "Corner models",
    "csv": "tickets.csv",
    "fields": {
      "ticket_id": "ticket_id",
      "league": "league",
      "result": "result",
      "first_kickoff_utc": "first_kickoff_utc",
      "legs": "legs_json",
      "stake_units": "stake_units",
      "payout_units": "payout_units",
      "profit_units": "profit_units"
    },
    "summary": {
      "rows": 1,
      "stake_units": "1.00",
      "payout_units": "2.25",
      "profit_units": "1.25",
      "roi_percent": "125.00"
    }
  }]
}
```

Paths are relative to the manifest; absolute input paths also work. Policy IDs
are unique text identifiers, never output paths. Ticket IDs must be unique
within their policy. All IDs in JSON leg records stay strings, including large
integers and leading zeroes. CSV encoding is UTF-8, with an optional BOM.

The eight mappings above are required. Optional mappings are `season`, `round`,
`stage`, `combined_odds`, `settlement_time`, `timing_note`, `details`.
Unknown mapping keys and absent mapped columns fail clearly. Every original
column, including unmapped ones, survives both download paths.
Optional `links` entries (`{"label":"Reading guide","file":"README.md"}`)
copy local TXT/Markdown/JSON/CSV/PDF attachments and link them from the header.

The mapped `legs` cell contains a JSON array with one or more objects:

```json
[
  {
    "event_id": "14035825",
    "competition_id": "38",
    "season_id": "77040",
    "kickoff_utc": "2025-07-25T18:45:00Z",
    "market": "Total corners",
    "selection": "Under 7.5",
    "home_id": "2889",
    "home_name": "Home team",
    "away_id": "4860",
    "away_name": "Away team",
    "odds": "2.25",
    "probability": "0.80",
    "result": "win",
    "home_score": "1",
    "away_score": "1"
  }
]
```

`event_id`, `kickoff_utc`, `market`, `selection` are required. Other fields are
optional; non-team events can supply `label` instead of team fields. Missing
results/scores remain unknown. All probabilities and numeric leg fields must
be finite decimal strings. Probabilities must lie in [0,1], decimal odds must
be at least 1. Supplied probabilities, combined odds, settlements and outcomes
are shown without reconstruction. One, two, and three-or-more legs use the same
card structure. Dates require explicit UTC (`Z` or `+00:00`); neither ordering
nor settlement proxies are invented.

Accounting is validated using exact Decimal arithmetic:

\[
\text{payout}_i-\text{stake}_i=\text{profit}_i,
\qquad
\sum_i\text{amount}_i=\text{supplied policy total}.
\]

Stakes/payouts must be nonnegative. Policy row counts and all three totals are
required; ROI is optional and displayed as supplied. Default tolerance is zero.
An explicit nonnegative `accounting_tolerance` permits upstream rounded evidence;
it never rewrites values. Unsettled tickets without supplied accounting are not
accepted by this accounting-complete v1 format; unknown **leg** outcomes are
allowed when the upstream ticket accounting is supplied.

The browser sums filtered profit with integer mantissas and decimal scales,
using BigInt. Formatting rounds only the display. Full-policy downloads reproduce
the **original CSV bytes**, including BOM/newlines/decimal spelling. Filtered
downloads preserve original header order and every literal cell string with
RFC 4180 quoting and CRLF records. They intentionally do not promise identical
bytes to a slice of the original file.

### Badges and provenance

Supply `badges` keyed by exact team ID with `name` and embedded `badge` image URI,
or `team_catalog` pointing to a native TeamCatalog JSON. PNG, JPEG, WebP and
inactive SVG are supported with the existing 2 MB size convention. Embedded
PNG/JPEG tiles in SVG are permitted. No remote images are loaded. Names fall back
to the catalog, then the ID; a visible ◇ marks missing badges.

When badges are included, `attribution_files` is required. Supply all applicable
source/license notices and asset provenance files. They are copied into the
bundle and linked from the footer. Neither a catalog nor this renderer establishes
additional rights to club artwork. Plotly's local runtime retains its original
copyright/license header.

### Explicit legacy adapter

`adapter="legacy_draw_pairs"` reads Ayre's **already audited flat CSV**, with
`leg1_` and `leg2_` fields, `opening_draw_odds`, `draw_probability`,
`draw_settlement`, and `home_score_current`/`away_score_current` suffixes.
Omit `fields.legs` and map the other ticket fields normally. It labels only this
explicit format as Draw. No gzip membership reconstruction, uint16 index reuse,
probability multiplication, EV derivation or historical builder execution occurs.
Original source CSV downloads remain byte-exact.

### Loading and interaction

The index holds policy summaries and the chosen badge catalog. Only the active
policy chunk is loaded; the previous payload/script is released on a switch.
Late script completions are ignored. Missing chunks show a visible error and
disable downloads until another selection succeeds. Pagination renders 50 cards
by default, with 20/100 choices. It bounds the DOM, **not** the selected policy's
decoded memory. Browser/file caching is outside the viewer's control.

Filters include league, result, first-kickoff UTC dates and text search over
ticket ID, round, leg event IDs, team names, labels, markets and selections.
`window.TICKET_VIEWER_QA` exposes loaded status, current policy, index, source
records and filtered records for verification. No UI operation retrains anything.

## Saved curves and portfolio manifest

Both use the same contract with `kind="curves"` or `kind="portfolio"`:

```json
{
  "schema_version": "1.0",
  "kind": "curves",
  "title": "Saved P&L comparisons",
  "scope_label": "2025/26 retained policy exports",
  "timing_note": "Settlement proxy: latest leg kickoff +3 hours UTC.",
  "units": "One unit per ticket; supplied cumulative profit in units.",
  "aggregation_note": "Alternative policies; no portfolio sum.",
  "plotly_js": "plotly.min.js",
  "plotly_js_version": "3.1.0",
  "numeric_tolerance": "0.00000001",
  "sections": [{
    "title": "Policy paths",
    "notes": ["Supplied daily-unit Sharpe is distinct from fixed-capital Sharpe."],
    "figures": [{
      "title": "Preserved native curves",
      "json": "paths.plotly.json",
      "producer_revision": "original-producing-commit-sha",
      "plotly_js_version": "3.1.0",
      "endpoints": ["12.5", "-3.25"]
    }],
    "tables": [{"title": "Supplied metrics", "csv": "metrics.csv"}]
  }]
}
```

The runtime version above is illustrative; use the matching **actual saved
Plotly JavaScript** and version. Every figure declares its producer revision and
runtime version. Mixed original revisions are retained per figure. Incompatible
runtime versions require separate bundles. Do not relabel the older portfolio's
`+120 minutes` proxy as the newer draw report's `+3 hours`.

Native public `Artifact`, `StudyResult`, `StudyRun`, `AnalysisReport` provide
the Study Explorer shell, section navigation, collapse, and lazy Plotly display.
No private `_style`/`_plotting` dependency or pipeline registry is added.
The supplied Plotly JSON retains every trace, point, tie, name, line color/dash,
hover template, legend and dropdown mask. Output includes exact original JSON
bytes for download. The renderer supports scatter/scattergl/bar curve figures;
animated or remote-resource figures are outside this format.
Fixed-width original figures scroll horizontally within their own viewport on
narrow screens; their saved dimensions and trace layout are not rewritten.

`endpoints` must have one supplied final y value per trace, in original order;
use null only for empty traces. Each endpoint is checked. Hidden losing policies
are retained. The builder does not combine policies or reconstruct a curve from
tickets. Portfolio inputs must already contain the audited portfolio paths;
`aggregation_note` must state the stake/overlap treatment. Supplied metrics and
tables are displayed as strings with byte-exact CSV downloads.

Optional `curve_tables` on each figure validates **every point** against a CSV:

```json
{
  "trace": 0,
  "csv": "daily.csv",
  "x": "date_utc",
  "y": "cumulative_profit",
  "where": {"policy": "baseline"},
  "calendar": {
    "start": "2025-01-01T00:00:00Z",
    "end": "2025-12-31T00:00:00Z",
    "initial_zero": true
  }
}
```

Rows retain source order, including equal timestamps. `calendar` is optional;
when supplied, every UTC day must be present and the declared initial zero must
already exist. The renderer does not fill missing dates or insert anchors.
Flat dates and unresolved outcomes must be distinguished upstream. All mapped
curve CSV files are copied too. Risk definitions, sample windows and uncertainty
caveats belong in supplied notes/tables, not calculations hidden in the template.

Standalone StudyRun records carry no synthetic match membership. The native
shell shows empty row/match scopes alongside the explicit supplied scope label;
actual ticket/curve counts are in their evidence and summaries. These records
are presentation objects, not training/evaluation populations.

## Example and verification

```text
python examples/standalone_reports.py --output /path/to/new/example
```

This creates synthetic ticket, curve and portfolio inputs and bundles. It does
not start a football study. `tests/analytics/test_standalone_templates.py` covers
strict input validation, exact decimals/CSV/IDs, general leg counts, portable
rebuilds, offline browsing, paging/filter/downloads, selectors and source styling.
Reference styling is pinned in `reporting/templates/assets/SOURCES.md`.

Browser checks use Chromium from `file://`, at 1440×1000 and 390×1000. The
Browser plugin is not installed; the existing Playwright runtime is used.
Source/reference archives remain untouched. No model/data/recipe changes are
required to create or view these sharing bundles.

### Validation on 2026-10-03

- 34 standalone tests passed, including real `file://` browser interactions.
- 177 standalone + existing reporting/scope/statistics/ticket checks passed.
  One existing sklearn deprecation warning arose in a compatibility fixture.
- Ayre's unmodified 60-ticket sample rendered with local badges; downloading
  the complete CSV reproduced the original file bytes. The sample is not a full
  strategy result or independent statistical validation of that strategy.
- Four original figure JSON files copied byte-for-byte: 25 Glicko traces
  (331,053 points), ten strategy paths (11,390), one combined path (1,139),
  and three winter-comparison paths (3,420): **347,002 points retained**.
- All original Glicko dropdown choices were exercised; their masks collectively
  expose all 25 traces. The original has nine choices, not 25 separate buttons.
- Ticket/reference and native portfolio/reference screenshots were compared
  at desktop and 390px widths. CSS and card/badge hierarchy are preserved.
  Intentional differences are caller-supplied headings, generic market/leg text,
  exact downloads, and accounting/scope notes from the new manifest. Display
  numeric rounding remains separate from exported precision.
- Clean runtime/console checks, nonblank page/title checks, empty-filter/reset,
  20/50/100 pagination, rapid switching, missing-chunk recovery, and full/filtered
  downloads passed in Chromium. Firefox/Safari and million-row memory use were
  not benchmarked. Pagination does not eliminate selected-policy payload memory.
