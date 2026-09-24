# Splits and CV documentation and verification coverage

Status: verified. The corrected API passes focused verification and the final
preservation audit. The preceding algorithm results retain their original provenance.
Final evidence: [splits_check.json](splits_check.json).

## Complete public API coverage

| API | Signature, settings and schema | Practical use or equations |
| --- | --- | --- |
| `Fold` | [Every field and original positions](splits_reference.md#fold-and-splitplan-records) | [Slice with iloc](splits.md#slice-with-original-positions) |
| `SplitPlan` | [Fields and iterator methods](splits_reference.md#fold-and-splitplan-records) | [Future fitting boundary](splits.md#connect-a-future-fitting-layer) |
| `create_split_plan` | [Split options and membership-only ownership](splits_reference.md#fold-and-splitplan-records) | [Prepare chronological folds](splits.md#inspect-expanding-chronological-folds) |
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
- [x] Verify SplitPlan/factory own no fitting policies and reports can inspect membership independently.
- [x] Distinguish outer folds, future inner selection, initial fits and additional refit schedules.
- [x] Explain retrospective state/feature reconstruction and nonindependent CPCV paths.
- [x] Preserve the read-only reference notebook and exclude its finance/statistical calculations.
- [x] Refresh all practical examples and the separate minimal seventh notebook for the corrected API.
- [x] Verify current packaged imports, rendered LaTeX/tables, links and stable-source preservation.

## Independent runtime evidence

The preceding API revision passed all 632 analytics tests: 142 new splits/CV
cases and the prior 490 regressions. Those results do not certify the corrected
API. The ownership correction changes only SplitPlan's fields and the factory's
parameters/return construction; an AST comparison confirms the splitting algorithms
and path assembly are unchanged. The first focused runs found no implementation
defect. The all-missing availability regression checks the primary's earlier UTC
normalization fix.

The corrected API passes all 142 focused split/CV checks, including exact plan
fields/signature, rejection of misplaced fitting options, independent fold
reporting, temporal boundaries and CPCV path assembly. The unrelated regression
suite and real-data numerics were not repeated. Their original source hashes,
tests and results are preserved in the [prior API archive](archive/pre_split_api_ownership/initial_revision_evidence.json).

The bounded real check on the preceding API loaded four pinned Premier League/Bundesliga 2023/24 and
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

All ten practical examples execute against the corrected API. The separate
nine-cell/five-code-cell seventh notebook ran in a fresh kernel after changing
only the two affected cell sources. Its previous executed version is archived.
A new 77-file wheel copy imports every splitter, checks the corrected plan fields
and reconstructs paths in isolation without sklearn imports or installation.

Fresh rendering validates all 27 LaTeX expressions, including six display
equations, without errors or overflow. Thirteen sections were visually inspected
initially; the three affected API sections were inspected again after the
correction. Table column counts and 59 local links pass. The source review checks the scoring/eligibility rules,
closed overlap, duration embargo and combinatorial identity; the latter also
passes exact integer checks for 66 count pairs.

The pre-correction audit passed on 33 source files, 164 protected earlier files,
20 prepared-source/selection files and six earlier notebooks. The source guard
then detected the approved API correction before any final evidence was published.
The final audit distinguishes both source snapshots and confirms 33 stable
current source files, 164 unchanged earlier files, 108 unchanged archived
revision artifacts, 20 unchanged prepared-source/selection files and six unchanged
earlier notebooks. Original batch docs and pre-correction tests/docs/notebook/
evidence are archived. The user reference
notebook remains read-only and unexecuted. The worker makes no library edits and
sends callbacks only for genuine defects; this approved contract change is not one.
