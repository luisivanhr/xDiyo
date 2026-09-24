# Label documentation and verification coverage

Status: verified. Focused/full tests, prepared-source reconciliation,
eight executable guide examples, the fresh notebook, isolated wheel import,
LaTeX/Markdown rendering and stable-source preservation passed. Evidence is recorded in
[labels_check.json](labels_check.json).

## Complete API coverage

| API | Import/signature and contract | Equations or practical use |
| --- | --- | --- |
| `LabelExpr` | [Namespace and marker base](labels_reference.md#expression-signatures-and-settings) | [Independent feature/label entry points](labels.md) |
| `LabelData` | [Every field, copying and metadata](labels_reference.md#entry-point-and-result) | [Unit, perspective and identity](labels.md#unit-perspective-and-identity) |
| `create_labels` | [Arguments, validation and output mapping](labels_reference.md#entry-point-and-result) | [Complete setup and request](labels.md#create-several-target-definitions) |
| `TeamValue` | [Source and all sides](labels_reference.md#expression-signatures-and-settings) | [Team values](labels.md#team-values-and-match-totals) |
| `MatchTotal` | [Finite paired sum and match layout](labels_reference.md#expression-signatures-and-settings) | [Match total equation](labels.md#team-values-and-match-totals) |
| `Outcome` | [Source, perspective and direction](labels_reference.md#expression-signatures-and-settings) | [W/D/L and statistic comparison](labels.md#outcomes) |
| `Above` | [Source, strict finite threshold and restrictions](labels_reference.md#expression-signatures-and-settings) | [Strict threshold equation](labels.md#strict-thresholds) |
| `BetOption` | [All selection, tie, encoding and status settings](labels_reference.md#expression-signatures-and-settings) | [Settlement rules](labels.md#explicit-settlement-and-numeric-encoding) |
| Shared `Stat` | [Metadata selection, period/group/key/field and counterpart](labels_reference.md#input-schema-and-pairing) | [Independent period expansion](labels.md#team-values-and-match-totals) |

## Behavioral coverage

- [x] Explain complete paired input, status agreement, reversed/distinct IDs and optional partition keys.
- [x] Explain exact unsigned IDs, duplicate indices, shuffled rows and home-order away outcomes.
- [x] List every output field, identity tuple, metadata column and expanded name rule.
- [x] Distinguish observation unit from perspective and explain mixed-unit dictionaries.
- [x] Document all defaults, domain restrictions, supported immediate children and validation timing.
- [x] Document finite/missing observations, single missing side, overflow and retained unfinished rows.
- [x] Describe object, nullable and Arrow missing observations; preserve source history and independent outputs.
- [x] Give LaTeX equations for team values, paired totals, signed/statistic and W/D/L outcomes, strict thresholds, settlement and encodings.
- [x] Explain actual-outcome semantics, score-current/shootout caveats and absence of target cutoffs.
- [x] Keep draw, push, void and missing distinct; document status precedence and finite override mappings.
- [x] Explain tuple/list void statuses, exact matching, literal fractional lines and excluded split-line/parlay behavior.
- [x] Explain parent-child caching, shared expression reuse and separate feature/label namespaces.
- [x] Provide executable examples of all five expressions, multiple periods, perspectives and settlement rules.
- [x] Preserve four existing notebooks and use a separate minimal fifth notebook.
- [x] Explain later assembler/reporting use without implying a final X/y assembly, training split or fitted model.
- [x] Complete LaTeX/Markdown rendering, navigation, wheel and stable-snapshot preservation audit.

## Independent evidence and limits

The focused suite has 119 label/settlement cases; the complete analytics suite has
394 passing cases, retaining the prior 275. The first focused run found an
object-dtype missing-sentinel conversion failure; the primary task repaired it,
and the original regression plus mixed nullable/Arrow/object cases pass.

One pinned Premier League 2024/25 publication provides 380 matches, 760 team rows
and 98,224 statistics rows. Direct prepared Parquet references agree exactly on
19,380 numeric cells and 5,320 settlement cells across 16 label definitions and
three independent periods. Synthetic tests supply cancellation, missingness,
partition collisions and other boundary cases absent from this real season.

Eight Python examples across the two guides and the separate nine-cell notebook
(five code cells) execute in the existing environment. The real check does not
establish any bookmaker settlement rule or model benefit. Existing source exports,
prior evidence, tests and notebooks are unchanged; the final audit records their
hashes and all 27 source files used for the stable verification run.
