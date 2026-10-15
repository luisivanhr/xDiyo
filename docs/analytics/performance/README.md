# Output-preserving library performance work

## Scope and delivery

- Branch: `codex/output-preserving-performance`.
- Frozen source reference: `01617da9f414d8929f4a2afd5d6ccd35d2d78512`.
- Isolated checkout: `C:/Users/luisi/.codex/worktrees/output-preserving-performance/xDiyo`.
- [Audit](https://chatgpt.com/s/cx_6ad06fb28f3481918b7fdc69937345c6) and `library_performance_handoff.md` supplied by the user.
- Main's local edits, the separate auxiliary-target branch, frozen experiment files and active local/remote jobs were not changed. No deployment, merge or push is part of this delivery.

This is a partial implementation of the handoff with explicit deferrals below. Public model architecture, losses, solvers, batching, seeds, reduction formulas and validation cadence are unchanged. Internal optimizations are enabled by default; no new recipe fields, UI resource controls or user-visible retention policy were introduced.

## Implementation and verification matrix

| Item | Status | Implementation and evidence / reason for deferral |
| --- | --- | --- |
| 1. Narrow copies | Implemented | Labels and pre-training analysis project the required columns before copying rows. Normal pandas indexing retains independent output copies. Label, pre-training, feature and assembly regressions passed. |
| 2. Losing candidates | Implemented | Weak identity registry for weak-referenceable objects, strong identity fallback for other objects, guarded callback removal against recycled IDs. Restart and selection guards stop owning discarded models. Explicitly release backend, checkpoint, state and checkpoint-metric bound-method locals. Tests cover live reuse detection, unhashable objects, fallback, ID turnover and collection before the next factory call, including metric hooks. |
| 3. Composition temporaries | Implemented | Release private inner contexts/models, prediction chunks and meta inputs at stage boundaries. Public OOF data, final/deployment models and audit fields remain. Composition recovery/adversarial/fold-isolation suites passed. |
| 4. Neural finalization | Deferred | Native graph/CNN serialization saves model/config/preprocessing/history but is not a verified optimizer-and-RNG resume checkpoint. A prediction-only lifecycle needs an explicit supported state machine and fresh-process resume tests before releasing training state. No live optimizer, gradient or resumable state is discarded by this patch. |
| 5. Repeated deterministic work | Implemented, deliberately narrow | Native NB/Poisson responses reuse a mean and deterministic preprocessing once; exact-type whitelist excludes custom/subclass/instance overrides, callable imputation and in-place `copy=False` transforms. Other paths retain their original calls. Reuse builtin validated O/U Brier, extract selected class probabilities once per fold (legacy path for duplicate source indices), hoist invariant NB log-factorial. Count/metric/classifier suites passed; new exact output and stateful/in-place fallback tests passed. |
| 6. Bankroll replay | Implemented | One `math.fsum` reserve per recorded event state; timestamp-indexed settlements retain reference ordering. No persistent reserve cache over publicly mutable tickets. Bankroll, settlement, void/refund, ticket and recovery regressions passed. |
| 7. Ingestion/report duplication | Partial | Shared-string XML removes completed children, preserving rich text and whitespace. CSV download links share content-keyed base64 within one render under an 8 MiB budget; CSV is regenerated to detect mutation by custom renderers. Native report/XML/error and mutation tests passed. Eager UI HTML removal deferred: that call currently participates in job completion/error handling, so deleting it changes when invalid reports fail. |
| 8. Rolling/EMA history | Implemented | Invocation-local sorted eligible indices supply bounded match-slot windows. Missing feature values still consume slots. EMA uses the same scalar recurrence only across extending prefixes; late releases, reordered prefixes and extended-precision alpha fall back/reset. Warm-start paths remain. Randomized H2H/venue/late-release oracles, exact recurrence, warm-start, spatial lineage and assembly suites passed. Full-history audits still enumerate their original eligible positions. |
| 9. CPCV | Implemented | Closed interval lookup uses binary search over sorted merged intervals. Randomized independent scan oracle and split/purge/embargo tests passed. Small vector workspace can be larger than the old temporary mask, despite faster execution. |
| 10. Bayesian lookup | Implemented, invocation-local | Build one validated snapshot index for all competitions during a feature call, preserving each competition's initial state. Fixture home-state lookup searches release times, then checks strict contributing-kickoff eligibility. Custom non-native datetime tables retain legacy comparisons. No posterior update or calibration changes. Exact frozen-reference lookup benchmark, Bayesian suites, mutation and duplicate-snapshot tests passed. No persistent cache assumes mutable snapshots are immutable. |
| 11. Spatial/table memory | Partial | Recovery private DataFrame traversal drops inherited attrs while retaining separately encoded public attrs. Preallocate observed spatial rows in original block/row order; PCA/KMeans and precision unchanged. Spatial/PCA/KMeans/recovery suites passed. Graph serializer copy removal is deferred because public/private alias and concurrent mutation ownership is not enforced. Cross-expression grid sharing and once-only KL-target validation likewise require immutable/versioned ownership; current mutable tensor validation must remain. No blanket float64 removal. |
| 12. GPU input cache | Deferred | Requires aggregate per-process/device budgeting, alias-safe immutable ownership, invalidation, low-budget/OOM fallback and deterministic GAT/CNN trajectory tests. A GPU process is active on the host; no GPU workload was started. Existing GPU policies/cache behavior are untouched. |
| 13. Graph inspection | Partial | Full graph embeddings, priors and attention are computed as before; requested rows are selected before CPU transfer, preserving duplicate query/order behavior and full-graph normalization. Existing frozen-reference CPU tests cover outputs, losses, gradients, complete AdamW/model state, prediction, embeddings, attention and save/load. No graph pruning, projection regeneration or preparation redesign. CUDA/profile-specific acceptance is still pending. |
| 14. Recovery serialization | Implemented for tables/containers | Stream the existing tagged encoding, Series boxing in chunks of 1,024. Public `pack` and fingerprint encoding remain; exact file-byte comparisons and legacy dtype/metadata/error/recovery tests passed. Arrays/index objects still use materialized legacy packing. Exclusive-create and ExperimentStore pending-directory/final-manifest publication stay unchanged. `dump_bundle` itself was not and is not claimed to be an atomic standalone replacement operation. |
| 15. Search fingerprints/run index | Deferred | Current keys depend on actual source/data bytes; skipping rehash based on identity or mtime misses in-place edits. A reliable run index additionally requires authoritative manifest revisions and concurrent-writer/interrupted-write handling. No validated ownership/revision protocol exists here; original checks retained. |
| 16. Odds/UI retention | Partial | Odds/crosswalk parsing reuse is operation-scoped and LRU byte-bounded (64 MiB retained DataFrame memory estimate), replacing global eight-entry partition retention. File hashes are checked on hits; crosswalk logical version validation remains. Copy isolation, >8 partitions, budget fallback, invalidation and exception release tested. Heavy UI payload eviction deferred: prepared/unsaved inspection state lacks a universally verified complete reload path. Existing UI retention is unchanged. |

## Lifecycle and cache contracts

- **Identity registry:** holds only weak references where possible. Non-weak-referenceable custom objects remain strongly held to preserve actual-reuse detection; those custom cases do not receive the same memory reduction. Public winners/fitted results still own their models.
- **History:** one invocation owns group indices and transient EMA prefixes. Groups with at most 64 candidate rows memoize short windows; large histories never retain every full prefix. Audit generation can still require full histories; this work does not promise linear runtime for all spatial audit output.
- **Ratings:** snapshot index lifetime is one feature call. The next call rebuilds after appended/revised values, with no mtime or object-identity shortcut.
- **CSV:** 8 MiB retained key-plus-payload budget per HTML render. Oversized entries remain uncached. Both existing download links are retained. CSV construction remains necessary to detect mutation.
- **Odds:** nested `prepare_bets`/`paired_quotes` calls share a context-local 64 MiB table cache; it is released in `finally`, including cancellation/errors. No reuse survives the outer operation. Locks protect a deliberately propagated context used by multiple threads. Oversized tables are read without retention; the budget does not cap temporary reads, returned copies or all process RAM. File validation is not bypassed.
- **Recovery:** chunks are transient. Unsupported/new public values are still handled by the existing data-only encoder. Store transaction and corruption checks remain unchanged. Concurrent mutation of a payload while saving is not a newly supported use case.
- **Neural models:** no finalization flag, optimizer cleanup or cross-process resume promise was added. Training remains usable under its existing contract; prediction and inspection retain the model. Resume validation is still a prerequisite to future lifecycle/cache work.

## Tests

- Broad selected regression run: **2,103 passed, 8 skipped, 1 failed**, in 194.16 seconds.
- After the final checkpoint-hook and in-place-transform guards: **192 passed** in 12.80 seconds. This overlaps the broad suite; do not add counts as unique tests.
- The failure is `test_ui_bayesian.py::test_bayesian_inventory_distinguishes_team_fixture_and_training_fields`. It expects a serializer list that omits two serializers already present at `01617da`. The same failure was reproduced using `git archive 01617da` source. It was not hidden or changed by this branch. See `baseline-test-status.json`.
- Three initial collection failures are unrelated missing `examples` modules for historical-result/Japan/Mexico enrichment. Relevant suites were selected by filename instead; a completely clean full-repository run is not claimed.
- Three CUDA cases were skipped with `CUDA_VISIBLE_DEVICES=-1`; five array-API cases require `SCIPY_ARRAY_API`. The host has CUDA-capable hardware. These skips are not evidence of absent GPU support.
- Complete test names, durations, failures and skip reasons: `regressions.xml`, `final-lifetime-checks.xml`.

Relevant suites were selected with this PowerShell filter:

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUDA_VISIBLE_DEVICES='-1'
$perfTests = @(Get-ChildItem tests/analytics/test_*.py | Where-Object {
    $_.Name -match 'performance_(lifetimes|history|io)|history|rolling|warm_start|warmstart|spatial|bayesian|count_brier|recovery|negative_binomial|labels|classifier_diagnostics|pre_training|reporting|composition|training_controls|model_selection|stake_bankroll|cpcv|odds_adapter|ui_odds|post_training_metrics|count_parametric|graph_preparation_memory|graph_integration|graph_determinism' -and
    $_.Name -notmatch 'historical_result_reviews'
} | ForEach-Object FullName)
& 'C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe' -m pytest @perfTests -q --tb=short --junitxml=docs/analytics/performance/regressions.xml
```

The final targeted command selects `test_performance_lifetimes.py`, `test_training_controls_*.py` and `test_count_parametric*.py`, with the same environment and `--junitxml=docs/analytics/performance/final-lifetime-checks.xml`.

## Benchmark method and limits

`benchmarks/library_performance.py` loads the relevant frozen modules with `git show 01617da`; the reference feature evaluator is explicitly bound to the frozen history implementation. It checks exact output arrays/DataFrames/attrs or saved bytes before writing a successful receipt. Unchanged dependencies and fixture construction are shared in the same interpreter. This is a bounded before/after module benchmark, not a completely isolated installed reference runtime or a production experiment.

Each case warms imports/allocators, then runs three timed repeats and reports the median. Python allocation tracing and sampled RSS runs are separate from the timings. RSS samples share a process and its allocator arenas: they are **not isolated process peak RAM**, and file cache is not attributed separately. No Linux container/cgroup applies to this native Windows run. CUDA was masked, so no VRAM/transfer-cache statistics or synchronized GPU timing is claimed. No active experiment was opened or run.

Hardware: AMD Ryzen 7 7800X3D (8 cores, 16 logical processors). The host RTX 4070 SUPER (driver 610.88, 12,282 MiB reported) was not used. The exact Python/package versions, source hashes, fixture hashes and timing samples are in `library-performance-01617da.json`.

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUDA_VISIBLE_DEVICES='-1'
& 'C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe' benchmarks/library_performance.py --output docs/analytics/performance/library-performance-01617da.json
```

| Case | Reference median | Candidate median | Reference/candidate | Traced Python allocation peak, before → after |
| --- | ---: | ---: | ---: | ---: |
| bayesian_four_league_lookup | 0.046883 s | 0.014946 s | 3.14× | 0.243 → 0.176 MB |
| bounded_history_3200_rows | 0.411034 s | 0.058397 s | 7.04× | 21.476 → 1.378 MB |
| long_ema_3200_rows | 0.928690 s | 0.453447 s | 2.05× | 21.488 → 0.733 MB |
| real_premier_2010_11_mean_ema | 0.024233 s | 0.017408 s | 1.39× | 0.363 → 0.439 MB |
| recovery_20000_by_10 | 0.394996 s | 0.431416 s | 0.92× | 7.012 → 0.375 MB |
| cpcv_2000_by_2000 | 0.064775 s | 0.001237 s | 52.37× | 0.011 → 0.087 MB |

All six probes passed exact output comparisons. Recovery is a memory improvement with a measured 9.2% time regression in this run; earlier exploratory runs varied, and no faster serialization claim is made. The short real-season case and CPCV use more small temporary Python allocations in the final measurement. These are allocation peaks, not total RSS.


The real fixture is the complete 2010/11 Premier League season (760 team rows). Other cases are synthetic probes, including the four-league Bayesian lookup. Reported improvements are not training speedups. The real short-history feature workload is small and sensitive to timing noise; the larger history and interval microbenchmarks must not be extrapolated to end-to-end pipeline throughput.

## Before merging

1. Review the independent commits and the fallback/validation contracts above.
2. Resolve or separately acknowledge the baseline inventory failure and missing example-module collection problems; run required CI on the integrated candidate.
3. Validate the supported pandas-version matrix and interactive browser report/export behavior. This delivery ran the installed version and native report tests, not every supported version or a fresh interactive browser session.
4. Schedule deterministic CUDA checks when the GPU is free for the inspection transfer change. Do not claim the CPU evidence as CUDA trajectory evidence.
5. Obtain a separately isolated realistic full-fold/profile run before claiming production RAM or training/epoch speedups. Cold preparation, update/post-update validation, save/load, multi-candidate workloads and OS file-cache effects are not all measured by these probes.
6. Before implementing deferred finalization/GPU residency, establish and test fresh-process optimizer/RNG resume plus immutable ownership and device budgets. Do not enable those policies by inference from this patch.
