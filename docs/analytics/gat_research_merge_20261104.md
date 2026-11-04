# GAT integration: review candidate

The retained GAT architectures and optional acceleration are integrated on `codex/gat-main-integration`, based on main `64861849d8b7044e0b209b9ac9db28080a92b1e4`. The reference and acceleration commits were pushed for review; nothing has been merged. The original research checkout and its uncommitted work remain unchanged.

The reference foundation is commit `e8210cb8237e71c3b1b3ad2f5169d1f62d83cb0d`; acceleration is commit `06d6ad6d0e7bd6fba6cbc5d3b2862afe25f0c3e1`. The original integration gates, 822-case record, full replay and timings below refer to that source. An independent audit subsequently found a missing historical statistic-release boundary; the focused correction and its separate verification are recorded next. The authority is `GAT_RECONCILIATION_DECISIONS.md`, which supersedes `GAT_MERGE_PLAN.md`, plus the corrected handoff index dated 2026-11-04.

## PR #2 corrections: 2026-11-05

Both findings in `GAT_PR2_FINDINGS_INVESTIGATION.md` are repaired in code commit `b8ffda6e35245149348cd77015922efddf43b7a9`, compared with pre-repair source `2079118b409cff7d67be21d5af68aac360c12968`. The preceding independent audit does not cover this new code; the following checks are supplemental verification for review.

**Release-aware scaler fitting:** the backend passes its maximum **fitting** history cutoff to preprocessing. Selected statistics released later or with missing release times are masked before fitting the scaler, using the same inclusive release rule as edge encoding. Direct fitting with release columns requires a nonmissing timezone-aware cutoff. Goals, statistics without release metadata, legacy arithmetic and explicitly restored preprocessing retain their behavior. The original array layout is preserved during legacy reductions to avoid even last-bit rounding changes.

**Observed gaps with boundary memory:** bounded-gap metadata now uses full result-eligible history at the current cutoff. The encoder keeps its segment-limited edges, support and carried state. An indexed reduction preserves exact timestamps and arbitrary team IDs without constructing another full graph. Legacy standardized gaps retain their previous segment semantics. Retained factories all disable boundary memory.

| Verification | Result |
| --- | --- |
| Pre-repair reproduction | Eight scaler invariance cases and two memory-boundary cases fail for the reported reasons; two in-segment controls pass |
| New CPU/CUDA regressions | 48 passed; included in the broader suite below |
| Affected regression suites | 329 passed, no failures or skips |
| Fresh-process native continuation | 16 passed, including reference/accelerated retained profiles on CPU and CUDA |
| Pre-repair reference comparisons | Six cases on CPU and six on CUDA, three updates each, exact |
| Retained saved exports | 18/18 exact full original-fixture forward comparisons and round trips |
| Available legacy exports | 54/54 exact four-fixture probes and round trips |
| Frozen retained training | B/N, C/E1 and B/CD_DW, fold2/seed17, three updates each, exact |
| Preservation | 5,931 distinct files hashed with no mismatches; original artifacts and previous receipts preserved |

There are **345 unique passing pytest cases**, including 67 explicitly CUDA-named cases. Reference comparisons use `rtol=atol=0` for tensors and exact array/input/identity comparisons. The existing recursive-oracle and cross-device tolerances were retained. Source hashes are identical across every completed verification command. Python 3.14.0, pandas 2.3.3, NumPy 2.4.6, Torch 2.9.0+cu130, pytest 9.1.1 and the RTX 4070 SUPER were used; no dependency changes were made.

The focused cases cover future/missing releases, release at cutoff and +/-1ns, all-unavailable and empty columns, mixed release metadata, all temporal modes, public retained fitting, frozen preprocessing, training-only scaler boundaries, both memory policies, late results, future exclusion, new/large-ID teams, nanosecond boundaries, reversed/repeated queries and ordinary/resident replay. Source-sensitive continuation rejection also passes.

Fresh release-aware fitting intentionally changes when unavailable statistics previously contaminated its scaler. Saved preprocessing is restored without refitting; forward/replay parity does not certify the information policy that originally produced it. Saved models using the affected bounded-gap plus boundary-memory combination intentionally receive corrected gaps. The 18 retained exports and generic legacy controls remain exact within the tested scopes.

The original **103-update replay was not repeated**. Its frozen preprocessing bypasses fitting, its retained configuration has no boundary memory, and the new bounded training/inference checks are exact. Its original receipt remains attached to the earlier source revision. The **17 CUDA exclusions from the independent CPU audit** remain exclusions of that audit; the new listed CUDA checks do not retroactively extend it. Unavailable legacy families and cross-source optimizer continuation remain uncertified. Historical pandas-3 failures remain confined to unchanged reference runtimes.

Initial test-development attempts are preserved: the synthetic statistic fixture first violated paired-perspective consistency and two control calls used the new keyword before implementation. Correcting the fixture and comparing the untouched archived source produced the valid ten failures. An initial candidate check then detected last-bit legacy scaler rounding from array layout; the final implementation preserves that layout. No historical source or numerical tolerance was changed. Archive source comparison permits only CRLF/LF normalization.

Preflight found 36 additional missing tracked files in an existing temporary worktree. The user explicitly accepted that investigated drift and instructed that the worktree remain untouched. No files were restored, reset or cleaned; other worktree states and captured uncommitted hashes match the accepted baseline.

[Compact repair evidence](data/gat_pr2_repairs_20261105/verification.json) records exact commands, versions, source hashes, test names, tolerances and receipt hashes. Full attempts and trajectories remain local under `experiments/gat_pr2_repairs_20261104_01`. The recommendation is renewed scoped PR review. No merge is authorized; the isolated repair commit can be reverted if necessary without resetting worktrees or deleting evidence.

## Independent audit correction: historical statistic releases

The audit's P2 finding was reproduced locally before changing production code. With `temporal_mode='historical'`, `statistic_availability_policy='per_stat_release_v1'`, and a selected statistic's release column, input preparation raised `AttributeError` because a synthetic snapshot lacked `cutoff`. Four pre-existing control configurations passed.

Historical batches can contain the same source row at several query times. `TimeIndexedPlan` now encodes contiguous query-time blocks using each query's configured local-midnight history cutoff. It preserves row order and duplicates and delegates masking to the existing scalar encoder. An empty batch receives a valid cutoff but carries no observations. The original source-row cache path remains in use when encoding is independent of releases. Only `graphs/temporal.py` changed in production code.

| Follow-up check | Result |
| --- | --- |
| New historical-release regressions | 22 passed, including CPU/CUDA coverage |
| Temporal attention, reconciliation, preparation/cache and integration-gate suites | 140 passed |
| Four pre-existing control configurations | Exact before/after inputs, initial parameters and encoder outputs |
| Original worktrees, research artifacts and prior integration receipts | 6,023 distinct files hashed; no mismatches |

The new cases cover release just before, exactly at, and just after the cutoff; missing release timestamps; repeated source rows at different historical times; later queries followed by earlier queries; empty history; and agreement with direct recursive snapshots across one to three layers and both time encodings. The recursive test oracle now supplies its own snapshot cutoff explicitly.

All **162 focused cases passed**, with no failures or skips in the completed runs. The first broad invocation encountered two pytest temporary-directory setup errors after 46 passes; the unchanged suite passed all 140 cases using a fresh short native Windows temporary directory. Both attempts and the original failing regression are retained. [Follow-up verification](data/gat_historical_release_fix_20261104/verification.json) records source hashes, exact control comparisons and receipt hashes; full local evidence is under `experiments/gat_historical_release_fix_20261104_01`.

No real-data fit, retained-profile replay, benchmark, dependency change, or research-grid expansion was performed for this correction. Retained profiles use `time_features`, so they do not enter the corrected historical path. Their original results remain anchored to `06d6ad6d`; the new focused checks supplement that evidence. Source-specific native resume continues to reject cross-source continuation.

## Remaining historical-reference verification: 2026-11-04

For this earlier follow-up, the tested code revision was `361e205f8890d5bfa45a87c4f1920f83ca1ab953`. The evidence-bearing branch tip examined in that review was `cf105e0e296aeb9bc412b8f80c3dd8960bed3988`, which added documentation and verification metadata only. The clarification at `2079118b409cff7d67be21d5af68aac360c12968` also changed documentation only. Main remains `64861849d8b7044e0b209b9ac9db28080a92b1e4`. At that review, production source, existing tests and comparison workers were unchanged from the tested code revision; the subsequent PR corrections are documented above.

Both historical-reference failures were reproduced under pandas 3.0.1, before candidate execution: `graphs/statistics.py:29` attempts an in-place update of the read-only array returned by `loc.notna().to_numpy()`. The unchanged references run successfully under pandas 2.3.3 in a disposable environment. Both environments use Python 3.14.0, NumPy 2.4.6 and Torch 2.9.0+cu130, with CUDA disabled. Existing shared dependencies were reused without modification; only the disposable reproduction environment received pandas 3.0.1. No production workaround or numerical test relaxation was needed.

| Comparison | Standalone statistics | Combined statistics, memory and heatmaps |
| --- | --- | --- |
| Historical `5ee87b57`, two updates plus legacy reload | Passed, exact | Passed, exact |
| Pinned main `64861849`, three updates | Passed, exact | Passed, exact |

The two pytest cases are `test_pinned_source_two_update_parity_and_legacy_reload[5ee87b5791f142f0c80432f1e0647d01f137a599-gat_statistics]` and its `gat_combined` counterpart. Initial graph inputs, preprocessing and parameters, each update's losses/gradients/AdamW state/weights, predictions, embeddings, attention and model identity agree. All 1,840 saved tensor comparisons (466,352 elements) are byte-exact; maximum absolute difference is zero, with `atol=0` and `rtol=0`. Historical model save/load checks also pass. These are four small synthetic comparisons, not new research fits.

The first disposable-environment attempt lacked the repository-level `utils.glicko_rating` import. The environment was corrected by adding the repository root to its import paths; that utility was verified identical in both reference commits and the candidate. Initial setup failures, the subsequent exact pandas failures and all successful attempts are preserved separately. Historical package sources were compared against Git archives, allowing only documented CRLF/LF differences.

That preservation check rehashed 6,041 distinct files, including all 412 prior integration receipts and 17 historical-fix receipts, with zero mismatches. The research checkout retained its 20 tracked modifications, 40 untracked entries and zero staged changes; the other worktrees retained their previous state. That follow-up added no production changes affecting the prior 18-export retained inference, 54 legacy probes or 103-update replay. Those receipts remain evidence at their recorded revisions; no new retained inference, CUDA run, benchmark or full replay was performed in that follow-up. The current PR corrections and their new verification are documented above.

The **17 CUDA cases excluded from the independent CPU follow-up remain excluded**, not newly passed. Unavailable legacy families and cross-source optimizer continuation remain uncertified. The successful pandas-2 comparisons do not claim that the unmodified historical references support pandas 3. The requested CPU comparison gates are closed; a scoped merge review is reasonable with these limits retained.

[Verification summary](data/gat_remaining_verification_20261104/verification.json) records the results and evidence hashes. The local [full verification report](../../experiments/gat_remaining_verification_20261104_01/REPORT.md) contains exact commands, runtime details, original failure traces, artifact locations and the rollback plan. Its sibling `*.command.json` receipts contain complete argument vectors, environment overrides, exit codes and elapsed times. No merge or push was performed in this follow-up.

## What is available

```mermaid
flowchart LR
    C[TemporalGraphConfig] --> M[Shared GAT implementation]
    P[Three explicit retained profiles] --> M
    M --> G[Generic backend]
    M --> R[Retained raw-CE backend]
    R --> S[Optional accelerated backend]
    G --> L[One native training and checkpoint loop]
    R --> L
    S --> L
    L --> E[Versioned model export]
```

| Profile | Phase context | Goal loss weight | Parameters |
| --- | --- | ---: | ---: |
| B/N | None | 0; goal head remains materialized | 38,041 |
| C/E1 | Corrected v2 E1, explicit source ID | 0.1 | 39,065 |
| B/CD_DW | Corrected v2, explicit source ID and W10 or W20 | 0; goal head remains materialized | 38,437 |

All three retain d32, two layers, two attention heads, the original ordered shots/corners features, and the original H/D/A head. Factory settings are explicit; a factory does not select a fitted checkpoint or winning seed. Six [archived replay recipes](../../examples/gat_retained/README.md) expose the two folds and require seed 17, 42, or 73. CD_DW explicitly uses W10 in fold1 and W20 in fold2.

Generic constructors retain main's legacy gaps and statistic-availability semantics. Retained profiles explicitly select bounded gaps, rational time features and per-statistic release masking. The loader binds known original exports to a hash-backed 18-entry compatibility manifest; missing or ambiguous metadata cannot silently acquire research semantics. Old phase-v1 tables load with their original policy and source identity. New phase production uses main's current standings/countdown implementation.

Public JSON/Python recipes support the new profile, training and execution specifications. Named-data, dated-alias, membership, cutoff, current standings, venue, reporting and prediction-boundary behavior remain covered. New source-stage mappings require an explicit complete `round_stage_column`. Model loading leaves acceleration disabled. To opt in for an eligible retained model, supply `research_execution=RetainedGATExecution()` alongside `RetainedGATTraining` and its declared raw-validation-CE control; omitting the execution specification selects reference execution.

## Evidence and compatibility scope

| Gate | Result | Scope |
| --- | --- | --- |
| Pinned-main behavior | Exact on CPU and CUDA | Six cases, three complete updates each: default, released statistics under legacy semantics, memory, statistics, combined extensions, CNN. Inputs, initialization, gradients, AdamW, outputs and identities compared. |
| Retained original inference | 18/18 exact, including round trips | All original outer fixtures: 4,288 in fold1 and 4,185 in fold2, for each family and seed. Independent frozen-source forward comparison; inputs, logits, probabilities, goal rates and identities. |
| Older model exports | 54/54 exact probes and round trips | 53 representative schemas plus recovered expanded-totals fold02; four fixed original fixtures per export. This is bounded inference evidence. |
| Original profile/preprocessing contracts | Six family/fold variants | All 18 canonical initial states, ordered features, dimensions and original configuration projections; added semantic policy asserted separately. |
| Public integration | Passed | Six sealed replay recipes prepare. Fourteen public cases cover JSON/Python, fit, reuse, save/load, data-only recovery, phase-v2 production and rejection boundaries. |
| Retained training | Three families exact | Fold2, seed17, three full fitting-population updates, complete inner validation, gradients including None, AdamW, RNG, selection and restoration. |
| Native continuation | Passed in fresh processes | Three reference profiles and three accelerated profiles on CPU and CUDA, plus the generic fixture-feature route. Disposable intermediate checkpoints match uninterrupted controls; completed reopen performs zero updates. These use declared miniature populations. |
| Accelerated execution | 19/19 short checks and full replay passed | Five independent flags and combined flags for each family, plus a 1 MiB cache-budget case. |
| Representative stopping replay | All 103 updates exact | C/E1, fold2, seed17. Selected update **88**, completed update **103**, best inner raw CE **1.0002489273849902**. Selected weights and fitting/inner inference also exact. |
| Focused regression suite | **822 unique cases passed** | Latest result per case, including the CUDA follow-up; no unresolved test failures or skips. |
| Inventory generation | Deterministic | Two identical regenerations; seven graph additions, no removed components, and unchanged non-graph components/stages/metrics versus independently regenerated main. |
| Original-source preservation | Passed | 5,613 distinct files hashed, original five worktree registrations/statuses unchanged, all supplied source/artifact inventory entries match. |

The full replay took 19.60 minutes, including its verification/checkpoint overhead. This is an engineering acceptance replay. No research grid, architecture search, beta selection, residual sweep, or new outer performance evaluation was performed. The speed checks produce fitting/inner outputs only. Retained outer forward execution is confined to the explicit original-model parity gate.

The original retained backend oracle uses the archived class and exact numerical helper definitions, with file hashes and selected definitions recorded. Explicit original path bindings avoid an unused reporting import absent from that frozen closure. It does not substitute the integrated implementation for the independent numerical oracle.

## Runtime and memory

![Matched training and inner-inference timings](data/gat_research_merge_20261104/speed_timing.png)

| Measurement | Reference median | Accelerated median | Reference / accelerated |
| --- | ---: | ---: | ---: |
| Training update / cold cache | 38.276 s | 33.997 s | 1.13x |
| Training update / warm cache | 14.831 s | 10.794 s | 1.37x |
| Inner inference / cold cache | 6.449 s | 6.041 s | 1.07x |
| Inner inference / warm cache | 1.139 s | 1.021 s | 1.11x |

Fixed ABBA/BAAB order, four constructed backends per arm, three complete updates per trial: four cold-cache and eight warm-update measurements per arm. Inner inference is measured separately after restoring identical selected weights, with four cold-cache and four warm-cache passes. Outputs are copied to CPU and checked. Construction is excluded from those timings and reported below. Cold refers to input caches within one process, not a fresh Python process. These are C/E1 fold2 measurements on this machine, not general throughput guarantees or public end-to-end prediction latency.

| Arm | Construction median | CUDA allocated peak, GiB | CUDA reserved peak, GiB | Sampled process RSS peak, GiB |
| --- | ---: | ---: | ---: | ---: |
| Accelerated | 12.29 s | 1.489 | 1.508 | 1.825 |
| Reference | 12.55 s | 1.518 | 1.549 | 1.839 |

Memory columns are medians of four per-trial peaks. CUDA peaks include construction and three training updates. RSS is sampled every 20 ms after construction; it can miss short spikes and is distinct from private memory. No other verification job used the GPU during timing. Parallel short-check runtimes are excluded from speed conclusions.

Acceleration preserves parameter registration, initialization, loss reduction and selection. Its borrowed cache owns one train/validation device-input budget; native duplicate residency is disabled, indices/transient transfers are counted, public access keeps copying, and completion/error releases ownership. Full-run cache statistics are recorded before the final cleanup; ownership tests verify cleanup behavior. The low-budget run retains no inputs at completion; its largest required transient was 4,948,724 bytes, explicitly accounted for despite the 1 MiB residency budget.

Skipping diagnostic replay publishes `pre_update_train_ce`, `pre_update_train_goal_loss`, `pre_update_train_regularization` and `pre_update_train_objective`. It removes the post-update replay and its replay-only failure check. Raw validation CE still selects checkpoints. The runtime benefit therefore includes the explicitly authorized reduction in diagnostic work; it is not solely faster arithmetic.

## Saved report arithmetic

| Retained family | Raw CE | RPS | Argmax accuracy |
| --- | ---: | ---: | ---: |
| time_B/CD_DW | 1.007471848261 | 0.204497066447 | 49.974429% |
| time_B/N | 1.007665945797 | 0.204524841155 | 50.005901% |
| time_C/E1 | 1.007575891381 | 0.204473362908 | 50.076714% |

These reproduce the supplied saved-output audit to at most 2.7e-15 in CE and 2.3e-16 in RPS. Each seed is scored separately over its 8,473 pooled outer fixtures, then the three scores are averaged. Probabilities use H/D/A order and float64 normalization. No probability ensemble is evaluated. RPS is half the sum of the two cumulative squared errors. This arithmetic check supports report fidelity; independent forward comparisons establish inference parity.

Original reports and calibration metadata remain untouched. The speed/residual study used raw probabilities; the corners study also retains calibration outputs. The actual residual selection remains fold1 C/E1 NONE and fold2 B/N EDGE. Keeping the later R_NONE families is a separate simplicity decision.

## Recovered evidence and limits

Both Space ZIPs were downloaded and SHA256-verified against the corrected index. The expanded-fold archive's eight model/data files match the original isolated-copy receipt. Its 213-file runtime source closure was recovered from Git `11ffa446`, with recorded line-ending normalization. The older four-study ZIP has one reporting member inconsistent with its source manifest; a separate runtime copy uses the untouched research file that matches the manifest exactly. Seven missing rest/propagation source members were recovered from `9499d4af` with exact hashes. Original archives and manifests were preserved.

The 54 older-export checks cover the stated probes; they do not certify every historical population or optimizer continuation. Other unavailable historical model families remain unverified. October25 temporal and October26 goal-auxiliary fits lack the relevant native intermediate states. Retained selected exports can pair restored weights with terminal optimizer state, and their old terminal checkpoints are stopped. Neither establishes clean continuation from the selected update. Cross-source native resume continues to reject; extending a phase table requires a new identity and is outside this integration.

Earlier failed checks remain in the evidence folder. Corrected implementation issues include legacy identity projection, JSON-stable competition identities, data-only public phase recovery, and default cache-policy normalization. Source-sensitive resume was rerun after code stopped changing. Windows long-path dataset publication failures passed in a fresh shorter temporary directory without changing application behavior. Initial collection errors are resolved by subsequently collected passing cases.

The first full speed attempt matched update1, then failed because its verification harness omitted the caller-owned directory required by the native checkpoint writer. The corrected harness creates a fresh unique directory. All model/training source hashes and every worker AST node outside `full()` were unchanged; 19 verified short receipts were carried into `speed_acceptance_v2` with explicit provenance. The failed queue and its source are retained. No numerical failure was waived. The final GPU test invocation initially skipped 15 cases requiring a second idle-GPU opt-in flag; the report gate rejected those skips, and all 15 subsequently passed with both flags.

## Review and rollback

The original research branch remains at `9499d4afd85933304a47034d05baad0913c43cab`, with 20 modified tracked and 40 untracked status entries, and no staged changes. The other original worktrees, including pre-existing damaged temporary checkouts, were preserved. Twenty-one designated main files remain byte-equivalent after line-ending normalization. The complete `src` snapshot and numerical queue seal record `06d6ad6d`; the historical-release correction has its own source hashes and verifies that it changes only the declared production file.

The candidate is ready for review within these verified scopes. Generic execution is the default; disabling the execution spec returns retained training to reference execution. The reference, acceleration and historical-release correction can be reviewed separately. New-schema exports require the integrated loader. Original exports, original frozen runtimes and all prior artifacts remain the rollback authority. A merge remains pending user review.

[Compact acceptance data](data/gat_research_merge_20261104/acceptance.json) contains gate summaries and receipt hashes. [Timing data](data/gat_research_merge_20261104/speed_measurements.json) accompanies the plot. The complete local [artifact index](../../experiments/gat_integration_20261104_01/ARTIFACT_INDEX.json) links the preserved attempts, raw receipts and source seals; the full evidence folder is intentionally outside Git.
