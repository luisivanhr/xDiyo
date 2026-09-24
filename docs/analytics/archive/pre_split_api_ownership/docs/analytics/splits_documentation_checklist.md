# Splits and CV documentation and verification coverage

Status: verified. Runtime checks, all examples, the fresh notebook, wheel imports,
rendering and preservation audit pass. Evidence: [splits_check.json](splits_check.json).

## Complete public API coverage

| API | Signature, settings and schema | Practical use or equations |
| --- | --- | --- |
| `Fold` | [Every field and original positions](splits_reference.md#fold-and-splitplan-records) | [Slice with iloc](splits.md#slice-with-original-positions) |
| `SplitPlan` | [Fields and iterator methods](splits_reference.md#fold-and-splitplan-records) | [Future fitting boundary](splits.md#connect-a-future-fitting-layer) |
| `create_split_plan` | [Options and stored objects](splits_reference.md#fold-and-splitplan-records) | [Prepare chronological folds](splits.md#inspect-expanding-chronological-folds) |
| `TemporalSplit` | [All settings and metadata](splits_reference.md#temporalsplit-settings-and-eligibility) | [Eligibility](splits.md#state-prediction-and-availability-times), [score membership](splits.md#separate-held-out-membership-from-scoring) |
| `MatchKFold` | [Allocation, seed and assumptions](splits_reference.md#matchkfold-and-groupkfold) | [Retrospective K-fold](splits.md#retrospective-match-and-group-k-fold) |
| `GroupKFold` | [Metadata/external groups and greedy balancing](splits_reference.md#matchkfold-and-groupkfold) | [Group example](splits.md#retrospective-match-and-group-k-fold) |
| `CPCV` | [Intervals, paths, defaults and metadata](splits_reference.md#cpcv-configuration-intervals-and-paths) | [Combinatorics, overlap and embargo](splits.md#combinatorial-purging-and-embargo) |
| `reconstruct_paths` | [Every prediction/input/output constraint](splits_reference.md#prediction-path-contract) | [Multiple outputs and missing values](splits.md#reconstruct-held-out-values-without-fitting) |

The [internal helper inventory](splits_reference.md#internal-helpers-and-error-boundaries)
also covers shared validation, time conversion, exact grouping and position expansion.

## Behavioral coverage

- [x] Both layouts, exact large IDs, reused partition IDs, shuffled rows and indivisible matches.
- [x] Temporal expanding/sliding multi-season spans, step, block/time gaps and partial horizons.
- [x] Observed block ordering, postponed fixtures, kickoff ties and explicit stage keys.
- [x] Separate league calendars, deliberately synchronized rounds and pooled kickoff batches.
- [x] Two-day prediction lead, minimum pair cutoff and maximum pair availability.
- [x] Missing explicit availability including all-NaT timezone conversion; missing training targets.
- [x] Score-only block/round ranges retain every unscored test row outside training.
- [x] Match/group K-fold allocation, reproducibility, greedy matching counts and external tuples.
- [x] Group disagreement inside a match, missing groups and alignment errors.
- [x] CPCV complete combinations, all incidence uses, equal-time batches and path coverage.
- [x] Closed endpoint overlap, enclosing intervals, disjoint intervals and run-based embargo.
- [x] Empty training errors, wrong/missing/extra prediction positions and numeric output contracts.
- [x] Chronological paths, stable ties, multiple outputs and unchanged NaNs/nullable values.
- [x] Selector/refit options default to None and are never invoked; inputs remain unchanged.
- [x] Distinguish outer folds, future inner selection, initial fits and additional refit schedules.
- [x] Explain retrospective state/feature reconstruction and nonindependent CPCV paths.
- [x] Preserve the read-only reference notebook and exclude its finance/statistical calculations.
- [x] Execute all practical examples and the separate minimal seventh notebook.
- [x] Verify packaged imports, rendered LaTeX/tables, links and stable-source preservation.

## Independent runtime evidence

All 632 analytics tests pass: 142 new splits/CV cases and the prior 490 regressions.
The first focused core/temporal and CPCV runs passed. No implementation defect,
source repair or failure callback was required. The all-missing availability
regression checks the primary's pre-dispatch UTC normalization fix.

The bounded real check loads four pinned Premier League/Bundesliga 2023/24 and
2024/25 publications: 1,372 matches and 2,744 team rows. Fourteen scheme/layout
combinations reconcile 192,240 membership positions and 41,160 reconstructed
numeric cells. References use independent match dictionaries, block membership,
pairwise interval overlap and contiguous-run embargo rules.

The explicit availability scenario is kickoff plus three hours, a synthetic
sensitivity input rather than observed result-publication evidence. Two-day lead
times test prediction boundaries. Path values are alignment sentinels; no model
was trained and no predictive/statistical conclusion follows from their values.
Previous notebooks, tests, source data and evidence remain outside the edit scope.

## Documentation and preservation evidence

All ten practical examples execute. The separate seventh notebook has nine cells,
five containing code, and executed top-to-bottom in a fresh existing kernel.
A bounded 77-file wheel copy imports every splitter and reconstructs paths in an
isolated interpreter, without sklearn imports or package installation.

All 27 LaTeX expressions, including six display equations, render without errors
or overflow. Thirteen sections were visually inspected; table column counts and
58 local links pass. The source review checks the scoring/eligibility rules,
closed overlap, duration embargo and combinatorial identity; the latter also
passes exact integer checks for 66 count pairs.

The final audit confirms 33 unchanged source files across runtime, real-data,
notebook and example execution; 164 protected earlier files; 20 prepared-source
and selection files; and all six earlier notebooks. Six original documents are
archived before replacement, including the two archived by the primary task.
The user reference notebook's hash is unchanged and it was never executed.
The worker made no library edits and sent no success or failure callbacks.
