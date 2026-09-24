# Reporting documentation and verification coverage

Status: runtime, real-data, standalone/iframe, saved-notebook, package and rendered
documentation and preservation checks pass. Live Jupyter frontend/trust behavior
is unverified because no frontend/server is installed in the checked runtimes.
Final evidence: [reporting_check.json](reporting_check.json).

## Public API coverage

| API | Complete contract | Practical use |
| --- | --- | --- |
| `PreTrainingAnalysis` | [Scope and orchestration](reporting_reference.md#orchestration-and-scope) | [Compose studies](reporting.md#compose-named-studies-and-a-training-selection) |
| `Reporter` | [Structural protocol](reporting_reference.md#shared-records-and-reporter-protocol) | [Custom reporter](reporting.md#add-a-reporter-with-its-own-artifact-arrangement) |
| `AnalysisContext` | [All fields, copies and catalogues](reporting_reference.md#analysiscontext) | [Scope choices](reporting.md#choose-execution-and-partition-separately) |
| `Artifact` | [Fields](reporting_reference.md#shared-records-and-reporter-protocol), [renderers](reporting_reference.md#rendering-and-notebook-display) | [Flexible layout](reporting.md#add-a-reporter-with-its-own-artifact-arrangement) |
| `StudyResult` | [Tables, artifacts, notes, selection](reporting_reference.md#shared-records-and-reporter-protocol) | [Custom reporter](reporting.md#add-a-reporter-with-its-own-artifact-arrangement) |
| `StudyRun` | [Exact scope provenance](reporting_reference.md#shared-records-and-reporter-protocol) | [Scope choices](reporting.md#choose-execution-and-partition-separately) |
| `AnalysisReport` | [HTML/native notebook display and selections](reporting_reference.md#rendering-and-notebook-display) | [Display and export](reporting.md#display-once-navigate-without-recalculation) |
| `FeatureSelection` | [Columns and ranking](reporting_reference.md#shared-records-and-reporter-protocol) | [Retained selections](reporting.md#compose-named-studies-and-a-training-selection) |
| `FeatureSelector` | [Common three-mode contract and hooks](reporting_reference.md#common-featureselector-and-voting) | [Non-correlation selector](reporting.md#three-selection-modes-share-one-contract) |
| `vote_selections` | [Generic votes, scores and ties](reporting_reference.md#common-featureselector-and-voting) | [Consensus equations](reporting.md#three-selection-modes-share-one-contract) |
| `FeatureDistributionReporter` | [Settings, status and tables](reporting_reference.md#featuredistributionreporter) | [Histogram/KDE formulas](reporting.md#interpret-distributions) |
| `CorrelationAnalysis` | [All metrics, settings and schemas](reporting_reference.md#correlationanalysis) | [Metric/rank formulas](reporting.md#interpret-association-columns) |
| `FeatureTimeline` | [Grouping, aggregation and display](reporting_reference.md#featuretimeline) | [Team/league examples](reporting.md#follow-team-or-league-timelines) |
| `TopKCorrelationSelector` | [Reuse, ranking and consensus](reporting_reference.md#topkcorrelationselector) | [Composition](reporting.md#compose-named-studies-and-a-training-selection), [formulas](reporting.md#three-selection-modes-share-one-contract) |

The [helper inventory](reporting_reference.md#internal-helpers-and-error-boundaries)
also covers shared validation, numerical conversion, rendering and errors.

## Acceptance checks

- [x] Explicit reporter mapping, type/partition, all execution scopes and original positions.
- [x] Both layouts, shuffled/nonpositional indexes, exact IDs and row-versus-match counts.
- [x] Unique overall unions, empty reports/partitions and explicit dependency order.
- [x] Context isolation, nested definitions, fold metadata and earlier result copies.
- [x] Histogram densities/counts, Gaussian formula and SciPy bandwidth parity.
- [x] Missing/nonfinite values, constant/empty scopes and rejected invalid bandwidths.
- [x] Pearson, average-tie Spearman, Kendall tau-b and declared/thresholded MCC references.
- [x] Multiclass correspondence, zero denominators, pair counts and status reporting.
- [x] Feature percentile ranks, multicolumn absolute sums and missing coverage.
- [x] Training-only selections, exact score reuse, explicit train-to-test names and no source mutation.
- [x] Top-k ties, fewer eligible features, zero selections and downstream no-feature notes.
- [x] Independent non-correlation selector in local, pooled and consensus modes.
- [x] Hand-calculated votes, mean-score/input-order tie-breaks, selected-fold subsets and saved rankings.
- [x] Cross-season team IDs, explicit league aggregation, missing gaps and sampled displays.
- [x] Browser fold/navigation/search controls, deferred plots, entity selection, hover and zoom.
- [x] Exact integer/float sorting, signed bars, missing-last, full CSV and SVG exports.
- [x] Escaped ordinary content, trusted custom renderer extension and iframe isolation.
- [x] Execute all ten guide examples and the separate minimal eighth notebook in a fresh kernel.
- [x] Verify the saved notebook through nbconvert's Lab HTML template and browser interaction.
- [x] Verify isolated wheel imports and package inclusion without installation.
- [x] Review rendered LaTeX, tables, 63 local links and API/helper coverage.
- [x] Confirm 41 stable source files, 298 protected earlier files, seven prior notebooks and 165 archived reporting files; publish evidence/manifests.

## Numerical and real-data evidence

The repaired source passes all 824 analytics tests: 192 new reporting cases plus
the previous 632. Independent references use SciPy/sklearn, direct Gaussian sums,
closed-form histogram normalization, pairwise concordance counts and hand-specified
selection votes. Generic custom selectors verify that the shared contract does
not depend on correlation-specific scoring.

Four pinned Premier League/Bundesliga 2023/24–2024/25 publications supply 1,372
matches and 2,744 team rows. Two layouts produce 26 bounded descriptive views.
Independent checks reconcile 22,638 scope positions, 192 coefficient cells,
192 percentile cells, 256 histogram bins, 868 KDE values, 171 full timeline values
and 48 selection-ranking rows. Twenty prepared-source/selection files are unchanged.
MCC thresholds are explicit verification choices; no pilot feature recipe or model
was selected. Statistical weights follow the declared row layout.

## Defects and preserved revisions

Four failure callbacks covered six implementation defects, repaired by the
primary: nonpositive KDE bandwidth could produce negative density; consensus
children lost fold metadata; child definitions were shared; closed study panels
initialized plots early; exact integer browser sorting rounded IDs through Number;
sidebar fragment navigation replaced a srcdoc report with the parent notebook.
The original failing regressions and source revisions remain recorded. Authorized
common-selector and notebook-display additions were source changes, not defects.
Expected empty-index dtype, CSV newline and scroll-margin assumptions were corrected
in verification fixtures.

The final standalone/native-iframe browser exercise passes 31 checks with no
browser errors or remote requests. Ten additional saved-notebook checks verify
the iframe, sidebar retention/scrolling, fold controls, association bars and KDE
plots in nbconvert's Lab HTML export. The nine notebook cells (five code cells)
execute top to bottom; refreshing outputs preserved their sources and IDs.
All 32 LaTeX expressions render without errors or overflow; 17 sections were
visually reviewed. The final wheel copy covers 85 source files and all eight new
package files. Synthetic browser outputs are verification
fixtures, not experiment results. Earlier seven notebooks, tests, numerical
evidence and source exports are preserved. The primary's archived pyproject copy
already contains package registration and is explicitly not a pre-source snapshot.

The real-data numerical record retains its earlier source hashes: only the final
sidebar JavaScript changed afterward. The final full suite, examples, notebook,
wheel and browser checks use the final source. Prior failing and passing revisions
remain archived rather than being relabelled as final-source results.

## Environment limit

JupyterLab, Notebook, nbclassic and jupyter_server are absent from the project,
base and bundled Python runtimes checked. No installation or server launch took
place. The saved nbconvert export and standalone native iframe are browser-tested;
live Jupyter trust handling and frontend integration remain unverified.
