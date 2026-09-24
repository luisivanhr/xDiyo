# Post-training documentation and verification coverage

Status: verified on synthetic data, source revision 2. All 164 new cases and
1,078 analytics tests pass, together with 12 guide examples, notebook 10,
33 offline browser checks and isolated imports/numerical use from a 101-file
wheel staging copy. All 54 LaTeX expressions render without errors or overflow;
27 documentation segments and 11 report/notebook screenshots were reviewed.
The final audit preserves 493 earlier artifacts, nine earlier notebooks,
20 data/selection files and four archived continuity documents against 55 tested
source fingerprints.

Final evidence: [post_training_check.json](post_training_check.json).

## Public API coverage

| API | Contract | Practical example |
| --- | --- | --- |
| `PostTrainingAnalysis` | [Execution](post_training_reference.md#posttraininganalysis) | [Named studies](post_training.md#choose-studies-and-their-populations) |
| `PostTrainingContext` | [Context fields](post_training_reference.md#posttrainingcontext-and-shared-result-scope) | [Scopes](post_training.md#declare-how-repeated-held-out-rows-are-pooled) |
| `PredictionReporter` | [Shared reporter fields](post_training_reference.md#prediction-reporters) | [Studies](post_training.md#choose-studies-and-their-populations) |
| `PerformanceReporter` | [Reporter contracts](post_training_reference.md#prediction-reporters) | [Metrics](post_training.md#choose-studies-and-their-populations) |
| `ResidualAnalysisReporter` | [Reporter contracts](post_training_reference.md#prediction-reporters) | [Residuals](post_training.md#choose-studies-and-their-populations) |
| `PredictionDistributionReporter` | [Reporter contracts](post_training_reference.md#prediction-reporters) | [KDE overlay](post_training.md#choose-studies-and-their-populations) |
| `LabelPredictionDistributionReporter` | [Alias](post_training_reference.md#prediction-reporters) | [Overlay modes](post_training.md#choose-studies-and-their-populations) |
| `CalibrationReporter` | [Reporter contracts](post_training_reference.md#prediction-reporters) | [Probabilities](post_training.md#inspect-classification-probabilities) |
| `PredictionTimelineReporter` | [Reporter contracts](post_training_reference.md#prediction-reporters) | [Timeline](post_training.md#choose-studies-and-their-populations) |
| `Metric` | [Requests](post_training_reference.md#metric-requests-and-registry) | [Numerical metrics](post_training.md#use-numerical-metrics-without-rendering) |
| `MetricDefinition` | [Definition fields](post_training_reference.md#metric-requests-and-registry) | [Custom registration](post_training.md#use-numerical-metrics-without-rendering) |
| `register_metric` | [Registry](post_training_reference.md#metric-requests-and-registry) | [Custom metric](post_training.md#use-numerical-metrics-without-rendering) |
| `list_metrics` | [Discovery](post_training_reference.md#metric-requests-and-registry) | [Custom metric](post_training.md#use-numerical-metrics-without-rendering) |
| `evaluate_metrics` | [Inputs/output schema](post_training_reference.md#input-checks-and-metric-table) | [Direct calculation](post_training.md#use-numerical-metrics-without-rendering) |
| `BetSpec` | [Options and decisions](post_training_reference.md#betspec-and-evaluate_bets) | [Synthetic labels](post_training.md#supply-explicit-bet-options-quotes-and-decisions) |
| `evaluate_bets` | [Ledger and metrics](post_training_reference.md#betspec-and-evaluate_bets) | [Bet report](post_training.md#supply-explicit-bet-options-quotes-and-decisions) |
| `BetPerformanceReporter` | [Timeline and scope](post_training_reference.md#betperformancereporter) | [Bet report](post_training.md#supply-explicit-bet-options-quotes-and-decisions) |
| `ExperimentStore` | [Persistence](post_training_reference.md#experimentstore-and-configuration_hash) | [Saved runs](post_training.md#save-explicit-configurations-and-compare-runs) |
| `configuration_hash` | [Supported configuration](post_training_reference.md#experimentstore-and-configuration_hash) | [Automatic hashes](post_training.md#save-explicit-configurations-and-compare-runs) |
| `rank_runs` | [Comparison contract](post_training_reference.md#rank_runs-and-experimentleaderboardreporter) | [Weighted comparison](post_training.md#save-explicit-configurations-and-compare-runs) |
| `ExperimentLeaderboardReporter` | [Experiment scope](post_training_reference.md#rank_runs-and-experimentleaderboardreporter) | [Viewer](post_training.md#save-explicit-configurations-and-compare-runs) |

The [helper inventory](post_training_reference.md#internal-helper-inventory) also
covers internal calculations and persistence. Shared `StudyRun` scope fields and
`AnalysisReport` rendering are documented without changing earlier guides.
The [equation catalogue](post_training_equations.md) covers populations, pooling,
regression/classification/probability metrics, diagnostics, bet settlement and
weighted comparison.

## Acceptance checks

- [x] Independent metric arithmetic, aliases, custom registration, complete-case coverage and sample binding.
- [x] Reversed probability class order, entropy without observed labels, invalid/empty/constant cases.
- [x] Occurrence scopes, explicit pooling, conflicting duplicates, missing class cells and copied study inputs.
- [x] Independent KDE Gaussian mixtures, ECDF/frequencies, residual tables, calibration bins and timeline gaps.
- [x] Both betting layouts, exact large-ID joins, selected/unselected bets, push/void, missing inputs and partial ROI.
- [x] Persisted configurations, UUIDs, Parquet outputs, metric identities, explicit failures and pending-directory invisibility.
- [x] Weighted directions/scales, ties, selectors, incompatible comparison groups, missing metrics and experiment-only analysis.
- [x] Bounded Windows publication retry, preserved exceptions, both save paths and successful repeated filesystem publication.
- [x] Full prior analytics regression suite plus the new cases.
- [x] Execute every guide example and a new minimal tenth notebook.
- [x] Shared viewer/saved-notebook navigation, fold controls, figures and CSV/SVG downloads.
- [x] Package metadata, isolated wheel imports and numerical use without installation.
- [x] Render all LaTeX/tables and inspect the resulting pages and notebook.
- [x] Final source/data/artifact preservation audit, archive manifest and completion evidence.

## Corrections and scope limits

One callback clarified an error in the primary handoff: bet accounting supports
both layouts, with matching settlement/prediction units; a match-only rejection
test was replaced by independent team-level accounting and mismatch rejection.
The original source and failed test/log remain preserved. A fixture also needed
five identity-index names rather than three.

The initial browser check assumed every figure panel was open and Plotly's input
arrays were already decoded. The corrected check opens the collapsed panels and
checks the rendered arrays; all 33 checks pass without changing the viewer.

A subsequent callback reproduced intermittent Windows directory-publication
errors. The primary added bounded retries without a copy fallback or permission
changes. The lock's root cause is not established. Deterministic retry/error tests
and all twelve repeated filesystem saves pass on the repaired source.

All fits and accounting are synthetic. No real pilot, source-data rewrite, search,
scheduled refit, strategy execution or installation occurs. Live Jupyter frontend
trust behavior remains outside this increment; execution and saved HTML are
separate checks. Custom metric code, supplied decision eligibility and matching
training/report provenance remain caller responsibilities.
