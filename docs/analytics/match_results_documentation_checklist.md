# Match-result documentation and verification coverage

This batch covers the match-result reporter, team catalog, offline renderer and
their shared viewer integration. Source ownership remains with the primary
implementation task. The verification worker changes tests, documentation,
notebook and evidence only. The final source and artifact hashes are recorded in
[match_results_check.json](match_results_check.json).

## Documents and examples

| Surface | Coverage |
|---|---|
| [Guide](match_results.md) | Six executable blocks: retained synthetic data, numeric/custom categorical reports, initial filters, paired labels, probabilities/decisions and catalog helpers. |
| [Reference](match_results_reference.md) | All 20 constructor fields; reporter run/supported types; TeamCatalog construction/load/derive/display; exact-ID helper; tables, artifacts, rendering and browser helpers. |
| [Equations](match_results_equations.md) | Scope, signed error, inclusive tolerance, categorical agreement, probability validity, tie/threshold decisions, paired markers and full/visible populations. |
| [Notebook 12](../../notebooks/12_match_results_quickstart.ipynb) | Separate minimal walkthrough, fresh top-to-bottom execution and saved native iframe interactions. Eleven preceding notebooks stay unchanged. |
| [Asset notes](team_assets/README.md) | Existing optional catalog provenance and name-only entries. The worker independently reconciles IDs/names and local bytes without editing asset files. |

## Independent verification matrix

| Area | Evidence and expectation |
|---|---|
| Retained calculation | Both layouts, targets, exact large IDs, numeric signs/boundary, custom numeric classes, built-in categorical definitions, missing values and settlements. |
| Scope and identity | Test/score partitions, selected folds, repeated occurrences and pooling, initial/unknown/zero filters, duplicate/conflicting pairs, partial sides, full table retention. |
| Probability schema | Distinct class columns, alignment, finite/range/sum validation, missing pooled classes, companion output, no implicit decision, argmax tie and inclusive binary threshold. Empty/all-invalid scopes still validate threshold class schema. |
| Names and badges | ID-based lookup, catalog precedence, nameless catalog metadata fallback, copy behavior, relative paths, one resolution per team, name-only/missing/rejected files, SVG subset and embedded PNG/JPEG tiles. |
| Compact browser view | Result/Prediction plus optional Error/Probabilities; no Status or within-tolerance text cells. Header/body/group colspans align for both layouts and all optional column combinations. Colors still use retained status. |
| Interaction | Independent folds/targets, either-side team filter, cascading/reset/unknown filters, numeric sorting, pagination, full filtered CSV including status, exact IDs, escaping, desktop/mobile and zero network requests. |
| Shared viewer | Prior training-controls examples executed into new scratch paths; existing controls browser checks exercise curves, tables, downloads and navigation. Earlier outputs are preserved. |
| Packaging | Isolated wheel build/extraction/import/probe; all analytics modules included, optional repository catalog separate. No installation. |
| Presentation | All displayed LaTeX rendered with local KaTeX; documentation and report/notebook screenshots reviewed, links and API signatures checked. |

## Repairs and evidence boundaries

Two actionable findings were sent to the primary and repaired there: nameless
catalog entries now use available metadata names; binary-threshold class schema
is validated even without a valid row. Original failed tests/logs and source
snapshots remain preserved. A later user-requested presentation revision removed
fixture status text while preserving exported status and colors. The final
verification binds that revised source.

The existing badge-agent sample HTML was not visually checked because its CUA
file URL was rejected. A separate independently generated fixture report verifies
four actual local PNG/SVG badges, including SVGs with raster tiles. This does not
claim the original sample was inspected. All 286 active catalog badge files were
independently checked against recorded bytes and accepted by the helper.

Synthetic fixture calculations are demonstrations, not predictive evaluation.
Real prepared-source reads were limited to identity/name reconciliation. Fresh
kernel execution and saved HTML iframe interactions are verified separately
from live Jupyter frontend behavior, which remains unverified. Earlier source
freezes and evidence keep their original scope. No success callback is sent.
