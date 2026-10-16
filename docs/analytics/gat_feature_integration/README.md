# GAT feature integration readiness

## Result

**CPU integration gates passed: 300 passed, 33 CUDA-only skips, zero failures.**
The 33 focused interaction cases are included in that total. Native renderer
checks: **31 passed**, zero failures. Final CPU duration: **619.02 seconds**.
The source and test hashes remained unchanged throughout continuation checks.
One existing PyTorch TF32 API deprecation warning was emitted.

The authorized bounded CUDA gate also passed: **56 passed, zero failures or
skips** (23 integration cases and 33 existing regressions). See
[`cuda-verification.md`](cuda-verification.md) for commands, exact comparisons,
continuation and reload coverage. Runtime source is unchanged from `c4e26d8`.
The candidate remains local, unmerged and unpushed. Live-browser QA is not claimed.

## Scope and pinned sources

- Integration branch: `codex/gat-feature-integration`, isolated worktree.
- Current main/base: `861b00461943be7ff589c9bd7deace909e96c9c1`.
- Feature tip: `5ee87b5791f142f0c80432f1e0647d01f137a599`.
- Merge base: `302aaa1efe3573ad742a2c5bad943e933889d291`.
- The refreshed remote refs matched the supplied handoff. Main and the retained
  research worktree were not used for implementation. No experiment selection,
  refit, push, or merge to main is part of this work. The user subsequently
  authorized the bounded idle-GPU verification recorded below.

## Integration changes

1. Resolved seven conflicted files semantically: adapter, network, persistence,
   inspection, reporter notes, GAT documentation and `.gitattributes`. Retained
   both the feature changes and current-main league labels, selective exports,
   owned reconstruction validation, native serialization and resumable training.
2. Added a common native group-lease path for ordinary and statistics losses.
   Fixed labels/observation selections remain separate from learned forecasts.
   Loss terms and their sequential reduction order are unchanged.
3. Boundary replay uses native optimized tensor/map preparation. Optional
   residency caches only deterministic segment inputs, keyed by issuance index,
   segment boundary and required team identities within an owned generation.
   The carry is attached to the transient consumer copy after leasing. Boundary
   transitions and learned detached states are recomputed each pass; they are
   never retained in an immutable input cache or across weight updates.
4. Embedding and attention inspection selects requested teams/edges on-device
   before conversion, retaining the correct local indices and boundary metadata.
5. Inference exports preserve both statistics scales/schema and input-cache policy.
   Native training checkpoints explicitly capture the statistics scale and reject
   a mismatch with the scale rebuilt from fitting labels. Observation caches are
   reconstructed lazily after resume, not serialized as learned state.
6. Prediction-only finalization releases training/validation statistics labels and
   observation selections along with main's existing training buffers. Scales
   needed for original-unit inference remain available.
7. The merged curated UI inventory retains both sets of controls. Full regeneration
   under Python 3.14 introduced unrelated annotation/schema churn, so its output
   was discarded. Actual JSON/Python round trips and native renderer events are
   tested. Existing frozen references retain their exact bytes; small test-only
   adapters accommodate new internal keywords without changing reference math.

The options remain disabled by default. Input residency also remains disabled by
default. No predictive benefit or new large-fold speedup is claimed.

## Evidence and acceptance matrix

The final CPU and renderer receipts are `cpu-regression.log`,
`cpu-regression.xml`, and `ui-renderers.log`. `environment.json` records runtime
versions and SHA-256 hashes of the source and focused tests. The earlier failing
regression log is retained as `initial-regression.log`; its failures were frozen
reference call signatures and regenerated inventory annotations, subsequently
corrected without weakening numerical assertions.

| Gate | Evidence |
| --- | --- |
| Disabled default, scalar and CNN | Separate processes import complete pinned main source; initial prepared groups/scaling/parameters and two updates compare exactly, including gradients, full AdamW, post-update loss, predictions, embeddings, attention and model identity; original inference exports reload in the candidate |
| Feature-source equivalence | Separate processes import pinned feature source for memory, numeric-statistics and memory+numeric-statistics+KL reconstruction; same exact trajectory and output comparison |
| Boundary memory | Both season/fixed-day policies, retention 0/0.4/1, membership movement, new teams, ties, late release, empty history/segments, query ordering, update/restore replay freshness |
| Statistics | All legacy families, numeric support/transforms, missing/zero-weight labels, training-only scale, future-label invariance, all three head routes, native 74-target/222-output ordering and reports |
| Combined residency | 12 combinations of heatmaps, memory and head route; actual residency-pool logic exercised on CPU with a simulated memory probe, exact cached/uncached updates, cache budget/hits, no initial-state cache; not a CUDA residency claim |
| Inference and cleanup | Exact CPU save/load, selected-team exports, legacy artifacts, rejected schema/scaler changes, finalization then reload/resume |
| Native continuation | Independent processes for memory/statistics/combined, regular update and early-stop interruption; exact gradient/model/AdamW/scaler/RNG/metric/winner state, plus completed-checkpoint zero-update check; existing default GAT/CNN continuation suite also included |
| UI | JSON and Python recipes, catalog/inventory and renderer controls/events including residency/finalization; bounded live-browser preparation, execution, export and report interaction verified; see browser-verification.md |

## Reproduction

Use the recorded Python environment, with the integration checkout as the working
directory. Explicitly disable CUDA and both opt-in GPU test gates:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:XDIYO_CUDA_CHECKS='0'
$env:XDIYO_RUN_GPU_AUDIT='0'
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe' -m pytest tests/analytics/test_temporal_graph.py tests/analytics/test_graph_extensions.py tests/analytics/test_graph_statistics_repair.py tests/analytics/test_graph_numeric_targets.py tests/analytics/test_graph_integration.py tests/analytics/test_graph_preparation_memory.py tests/analytics/test_graph_speed.py tests/analytics/test_graph_goal_features.py tests/analytics/test_heatmap_cnn.py tests/analytics/test_input_residency.py tests/analytics/test_performance_inspection.py tests/analytics/test_ui_inventory.py tests/analytics/test_ui_catalog_compatibility.py tests/analytics/test_ui_retention_audit.py tests/analytics/test_native_training_resume.py tests/analytics/test_graph_integration_gates.py -q --tb=short --junitxml=docs/analytics/gat_feature_integration/cpu-regression.xml
node --experimental-vm-modules --test tests/analytics/ui_cnn.mjs tests/analytics/ui_composition.mjs tests/analytics/ui_feature_bundles.mjs tests/analytics/ui_feature_discovery.mjs tests/analytics/ui_svm.mjs
```

The parity worker extracts only the pinned Python package into test temporary
directories. It never modifies the protected study source or artifacts. Runtime
 source remains fixed throughout checkpoint continuation tests.

## Remaining scope

Bounded deterministic CUDA parity/continuation, same-device export/reload and
CPU-reload tolerance checks now pass for this integrated candidate. These are
fresh receipts, separate from the original feature branch receipts. Bounded live-browser checks also pass, with the numeric-target form repair and limitations documented in browser-verification.md. No hosted
CI or integrated large-fold benchmark claim is made. A new research selection/refit is unnecessary
for these integration gates. Keep the integration branch unmerged and unpushed.
