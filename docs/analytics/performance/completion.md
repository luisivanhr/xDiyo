# Performance completion pass

Branch: `codex/performance-owned-caches-resume`. Keep unmerged; the user authorized
pushing the verified repairs. The frozen
local and Hostinger runtimes, datasets and production fits are unchanged.

## Status

Follow-up review repairs: see [audit-8e1b37f.md](audit-8e1b37f.md). The
completion evidence below describes the previously verified 8e1b37f revision;
the repaired source has now passed its own final gate: 56 tests, zero skips or
failures. See the fresh audit CUDA receipts for exact source/runtime hashes.

Completed on 2026-10-15 after full process-command and GPU telemetry checks
confirmed that the frozen worker and its successors had exited. No production
worker was paused or modified. The remaining CUDA gates passed on implementation
`e26e58a`; this final evidence update changes no runtime model source.

| Area | Implementation and verification |
| --- | --- |
| Checkpoints and cleanup | Native full-state checkpoints and prediction-only finalization remain; fresh-process CPU continuation covers four native model forms. Fresh-process deterministic CUDA continuation also passed for all four native model forms. |
| Optional GPU inputs | Process/device shared accounting, private immutable snapshots, isolated consumer leases, idle LRU eviction, headroom checks, owner cleanup and disabled/CPU fallback. Settings are native ExecutionPolicy UI/JSON/Python fields and persist with models. CPU emulation checks full model/AdamW parity; real deterministic CUDA parity passed, including full AdamW state. |
| Spatial/export/inspection | Deterministic source-row grids are shared within the native preprocessor, with store/config/orientation/grid identity and causal lazy population. Native GAT KL targets now validate on the CPU when each source-row grid is first cached; native updates reuse that validation. Bounded-cache fallback validates newly reconstructed inputs again. Public arbitrary snapshots and caller-supplied groups retain validation. Complete-transfer frozen inspection reference matches optimized inspection on CPU with and without CNN maps. Both deterministic CUDA inspection variants and owned-KL update parity passed. |
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

Host ownership qualification: disabled residency shares the fitted preprocessor's
source-row caches across streams. Enabled residency instead creates an isolated
fixed plan and preprocessor per PreparedGroups stream, including copied fixture
metadata/positions and its own lazily populated edge/map caches. Each cache is
bounded separately at 128 MiB (up to 256 MiB per stream, plus plan/transient
storage); these CPU copies are outside the GPU input budget. They are included
in process RSS measurements. Training and validation can therefore duplicate
CPU encodings; there is no claim that all CPU caches are shared.

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

## Completed CUDA verification and benchmarks

`completion-cuda.xml` and `.log`: **37 passed, zero skipped**, 182.11 s.
This includes 11 GPU-specific cases: four residency cases, one owned-KL case,
two frozen-inspection cases, and four fresh-process continuation cases. Exact
checks cover losses, gradients, model/full-AdamW state, selection/history, RNG,
prediction and persistence where applicable. Continuation workers independently
exercise uninterrupted, intentionally interrupted, resumed and completed-no-op
runs. This establishes same-source, same-runtime deterministic continuation;
it does not establish cross-runtime or nondeterministic bit-exact continuation.
One PyTorch TF32 API deprecation warning remains; there were no failures.

`completion-cuda-environment.json` records test/source hashes and the environment.
Python 3.14.0, torch 2.9.0+cu130, pandas 2.3.3, numpy 2.4.6, RTX 4070 SUPER,
driver 610.88; deterministic CUDA with `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
`completion-cuda-summary.json` records exact comparisons and benchmark hashes;
individual receipts contain commands, source hashes, model manifests and raw times.

### Saved expanded-totals fold 02: optional residency

Each arm uses an isolated copy of its saved model and train metadata, verifies
all manifest hashes, and freshly initializes matching candidate models. Original
saved-model hashes were reverified after benchmarking. This compares residency
policies on current source, not current source against an older GAT implementation.
Both plain and Bayesian populations contain 18,888 rows and 1,463 issuances.
Enabled budget is 6 GiB with 2 GiB headroom; disabled budget is zero.

| Population / cache | Cold native prep (s) | Warm update + post-update evaluation (s) | Peak RAM (MiB) | Peak CUDA allocated / reserved (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Plain / disabled | 11.896 | 41.292 | 1807.95 | 216.14 / 248.00 |
| Plain / enabled | 11.693 | 44.748 | 1832.08 | 6386.56 / 8076.00 |
| Bayesian / disabled | 12.878 | 41.838 | 1780.70 | 216.16 / 248.00 |
| Bayesian / enabled | 12.517 | 41.712 | 1807.70 | 6389.10 / 8098.00 |

Prepared-input hashes, all three update metrics and model/gradient/full-AdamW
state hashes match **exactly** in each pair. Training-loss evaluation remains
at every update. Warm time is the median of updates two and three. Cold native
preparation excludes file/model loading. The prepared-hash verification pass is
outside timings and warms CPU inputs, but does not warm GPU residency.

Both enabled arms recorded zero hits and 8,778 misses, with no retained or active
input bytes at completion. Eviction churn provides no demonstrated benefit on
these folds: plain was 8.37% slower; Bayesian was effectively unchanged (1.003x).
**Keep residency disabled for these large folds.** Its default remains disabled.
The much larger allocated/reserved peaks illustrate its memory cost. A single
pair per GAT population and desktop graphics activity limit timing conclusions;
these results do not establish a statistically reliable speedup or universal
cache sizing rule. OS file-cache attribution is unavailable.

### CNN: three alternating-order process pairs

The frozen reference loads the `2e3b967` CNN adapter/network into the same current
support environment; it is not an entire frozen library checkout. The candidate
uses owned prepared inputs and 256 MiB optional residency. All five update losses
and model/gradient/full-AdamW hashes match exactly in every pair, on the same
2,000 synthetic fixtures, including missing maps and KL reconstruction.

Median across three independent runs per arm:

| Measure | Frozen adapter/network | Candidate |
| --- | ---: | ---: |
| Cold preparation (s) | 1.13244 | 1.17592 |
| Warm update + post-update evaluation (s) | 0.091814 | 0.075073 |
| Peak host working set (MiB) | 1371.79 | 1328.24 |
| Peak CUDA allocated (MiB) | 67.58 | 69.17 |
| Peak CUDA reserved (MiB) | 70.00 | 70.00 |
| GC-tracked objects | 349975 | 350329 |

Observed warm speed ratio is **1.223x**, with about 3.8% slower preparation.
This measures the combined owned-input/residency path on a small synthetic
population, not isolated cache contribution or production throughput. CUDA
metrics are allocator peaks, host memory is process peak working set, and neither
includes a separately attributed OS file cache. The earlier CPU results below
remain a distinct experiment.

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

All scheduled CUDA verification is complete. Pause the heartbeat; retain this
branch unmerged and unpushed. Reproduction commands remain in
[completion-handoff.md](completion-handoff.md).
