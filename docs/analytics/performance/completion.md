# Performance completion pass

Branch: `codex/performance-owned-caches-resume`. No merge or push. The frozen
local and Hostinger runtimes, datasets and production fits are unchanged.

## Status

The user explicitly requested waiting for the active frozen GPU fit to finish.
CUDA checks and saved-fold GPU benchmarks must not run while that fit or its
successor workers occupy the GPU. Pending CUDA results are not represented as
passed, and CPU-emulated cache tests are not CUDA evidence.

| Area | Implementation and verification |
| --- | --- |
| Checkpoints and cleanup | Native full-state checkpoints and prediction-only finalization remain; fresh-process CPU continuation covers four native model forms. CUDA continuation is an explicit pending gate. |
| Optional GPU inputs | Process/device shared accounting, private immutable snapshots, isolated consumer leases, idle LRU eviction, headroom checks, owner cleanup and disabled/CPU fallback. Settings are native ExecutionPolicy UI/JSON/Python fields and persist with models. CPU emulation checks full model/AdamW parity; real CUDA verification is pending. |
| Spatial/export/inspection | Deterministic source-row grids are shared within the native preprocessor, with store/config/orientation/grid identity and causal lazy population. Native GAT KL targets now validate on the CPU when each source-row grid is first cached; native updates reuse that validation. Bounded-cache fallback validates newly reconstructed inputs again. Public arbitrary snapshots and caller-supplied groups retain validation. Complete-transfer frozen inspection reference matches optimized inspection on CPU with and without CNN maps. CUDA variants are pending. |
| Frozen search | Owned isolated dataset/fold inputs, cached shared signatures, fresh per-candidate copies and source-content verification. Existing recovery-key bytes match exactly. Mutable callers keep the original full-content path. |
| Run index | Authoritative manifest bytes are reread. Decoding and lookup reuse are in memory; the atomic on-disk hint is disposable. Mutation, corrupt/stale hints, corrupt manifests, concurrent instances and interrupted hint publication are tested. |
| UI retention | Native completion rendering is reused. Optional `launch_ui(retention_bytes=...)` / `--retention-bytes` eviction requires verified result/model/reporter reload; active, unsaved and unsupported custom results stay pinned. Documents and spatial payloads use verified disk spools. Reopen and native model export are tested after eviction. |
| Compatibility | pandas 2.2.3 and 3.0.0: 407 tests passed in each environment. Current-runtime full-suite result is recorded below when complete. |

### Memory accounting boundaries

The input-cache budget includes retained tensors and active isolated input leases,
shared across native owners on the process/device. It excludes model parameters,
optimizer state and activations, for which configurable free-memory headroom is
reserved before retaining more inputs. When an individual required input exceeds
the budget, residency is discarded and ordinary single-group streaming is used;
the required streaming group itself can exceed the optional cache budget. This is
not an absolute cap on CUDA allocator reserved memory or on the entire process.
No learned activations are cached, no optimizer update is retried, and no global
pinned-memory duplicate is created. Derived CPU sources remain reconstructible.

UI retention bytes estimate the tracked heavy result and rendered-document
payloads, not total process RSS. Pinned jobs can exceed the target. Spools remain
on disk; this change does not introduce automatic disk-history deletion.

## Current CPU benchmark

Final source receipt: `completion-cpu-benchmark-final.json` (implementation `592a6ff`); earlier exploratory receipt: `completion-cpu-benchmark.json`; command and exact source hashes are in
the receipt. Host timings were collected while the unrelated frozen GPU fit was
running, so shared-host contention remains a limitation.

- 20,000 rows x 24 columns, six candidate keys: reference 7.9062 s; frozen
  snapshot construction 0.5778 s plus six keys 4.5185 s. About 1.55x including
  construction, 1.75x for the already-frozen keys. All key bytes matched.
- 600 authoritative run manifests: reference lookup 0.1431 s versus indexed
  lookup 0.1470 s; effectively unchanged. Cold index creation 0.1442 s. Content
  reads are intentionally preserved; no meaningful lookup speedup is claimed.
- Synthetic native lifecycle: three candidates and two inner folds, 16 fixtures.
  Fit 0.6179 s, reload 0.0604 s, reuse 0.0634 s, verified eviction 0.0794 s,
  report reopening 0.1147 s, native model export 0.0433 s. This small lifecycle
  benchmark is not a production-scale training speed claim.
- The tracked heavy result/HTML payload estimate fell from 236,861 bytes to 57
  bookkeeping bytes after verified eviction. This excludes other job metadata,
  previews and shared process allocations. Process RSS rose from about 102 MB
  to 247 MB during the whole benchmark; process peak was 247 MB and the final
  GC-tracked object count was 201,849. OS file-cache attribution is unavailable.
  CUDA allocated/reserved metrics are null in this CPU-only receipt.

## Interactive native UI

Tested the actual native builder served on localhost with an isolated synthetic
workspace, not a blocked local-file preview. The browser completed an experiment,
opened the native metrics report after eviction, expanded/closed it, and exported
the retained model successfully. Execution controls exposed the cache budget and
headroom defaults. No browser console errors or warnings were observed.

CSV download links retained the expected filename and exact data payload. The
in-app browser did not produce a download event or file during the interactive
attempt, so successful browser CSV downloading is not claimed. Automated report
and download-link checks remain separate from this browser limitation.

## Test receipts and repair history

- `completion-integrated-final.xml`: **5,710 passed, 33 skipped**, 812.28 s.
  This full run preceded the final native GAT KL-validation and frozen-dataclass refinements. The affected areas were rerun below on the final source.
- `completion-neural-final.xml`: **218 passed, 28 skipped**, 129.19 s; final graph/CNN/preparation/inspection/persistence and cold-process CPU resume coverage.
- `completion-selection-final.xml`: **397 passed**, 35.15 s; final search, recovery, ownership, index and retention coverage. Frozen snapshots preserve non-constructor dataclass state, and custom experiment stores retain the original path.
- `completion-owned-kl.xml`: **45 passed, 5 CUDA-gated skips**, including exact
  three-update KL losses, gradients, full AdamW state, selection metrics, mutable
  validation fallback and causal future-map rejection.
- `completion-repairs.xml`: 64 passed, 6 explicitly skipped.
- `completion-inspection-cpu.xml`: 3 passed, 2 CUDA-gated skips.
- `completion-pandas22-final.xml`: 407 passed in 36.03 s.
- `completion-pandas30-final.xml`: 407 passed in 38.60 s.
- `completion-integrated-rerun.xml`: earlier 5 failed, 5,701 passed, 31 skipped.
  Four failures exposed weak-reference registry serialization; the fifth was a
  legacy renderer object without `_renderers`. Both causes were repaired, and
  all five formerly failing cases passed in the focused repair run.
- Earlier pandas completion attempts had one test-only cloudpickle import error
  with joblib 1.6; the test now uses the same vendored/external fallback as the
  production serializer. Final complete matrix reruns passed.
- The original collection errors came from an installed `examples` package
  shadowing the repository examples; an explicit local package fixes resolution.

Counts overlap; do not sum them as unique tests. No repository GitHub workflow is
tracked, and no remote CI pass is claimed. The current local integrated run is the
available full-suite gate. The prior EMA qualification remains: recurrence reuse
still examines full eligible-history prefixes and is not fully linear-time.

## Remaining GPU procedure

1. Confirm through full process command lines and GPU telemetry that the frozen
   worker and any successor workers have released the GPU. Never pause them.
2. Use the same Python 3.14 environment, `XDIYO_CUDA_CHECKS=1`, deterministic
   execution and `CUBLAS_WORKSPACE_CONFIG=:4096:8`; remove the CPU mask.
3. Run CUDA input-residency parity, frozen inspection parity and fresh-process
   resume tests. Preserve exact comparison expectations and validation cadence.
4. Run `benchmarks/gpu_residency_saved_fold.py` on isolated copies of the saved
   plain and Bayesian expanded-totals fold 02, with cache disabled and enabled.
   Verify manifest hashes, prepared tensor hashes, losses and complete training
   state hashes. This compares candidate residency policies, not an older model.
5. Run `benchmarks/owned_cnn_inputs.py` in separate frozen-reference and candidate
   processes for CNN preparation and updates including post-update loss evaluation.
   Record RSS, object counts and CUDA allocated/reserved peaks separately.
6. Update this document and fresh receipts. Any parity failure blocks merge;
   never relax tolerances or claim an unexecuted check passed.

## Repeated CNN CPU comparison

`completion-cnn-summary.json` summarizes three alternating-order independent
process pairs, each comparing the frozen `2e3b967` CNN adapter/network to current
source on 2,000 synthetic fixtures. All five losses and model/gradient/full-AdamW
hashes matched in every pair. Reference versus candidate medians: cold preparation
0.92295 s / 1.03207 s, warm update including post-update evaluation 0.07883 s /
0.07112 s (1.108x observed), peak working set 723.55 MB / 723.12 MB. GC-tracked
objects 346,859 / 347,054. Slightly slower preparation and effectively unchanged
memory accompany this small observed warm difference; it is not a production
throughput claim. File-cache attribution is unavailable and CUDA peaks are null.

A quiet 30-minute heartbeat will resume the CUDA gate after the user-requested
wait. Exact continuation instructions are in [completion-handoff.md](completion-handoff.md).
