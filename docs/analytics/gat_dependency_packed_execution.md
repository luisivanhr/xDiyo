# Optional GAT dependency execution

Status: experimental, disabled by default. Full-training predictive acceptance
is pending. Numerical interchangeability with native execution has not been established.

`TemporalGraphConfig.execution=None` retains the native path. Opt in with
`GraphExecution(mode='dependency_packed')`; the UI exposes the same optional
component. `native`, `dependency` (native snapshot then dependency selection),
and `dependency_lean` (selection before input construction, no packing) are
available for controlled ablation. Existing recipes omit the component.

Supported: scalar full-history GAT, with or without a quantile statistics head,
including 74 targets / 222 outputs at quantiles 0.25, 0.5, 0.75. Embeddings and
predicted statistics can jointly enter the outcome head. Bayesian rating
features remain ordinary timestamp-checked fixture inputs; neural weights are
not Bayesian. CNNs, boundary memory, nonquantile statistics families and enabled
input residency explicitly retain native execution.

For L layers, start from the scored fixture teams at layer L. Work backwards:
keep every incoming edge of each required receiver, and retain the receiver
itself plus all senders for the preceding layer. This preserves softmax
denominators, source edge order and residual inputs. Lean preparation validates
the complete native eligible history on cold caches, preserves source-cache
identity/invalidation, full eligible incoming support for every retained node,
membership and timestamp-overflow
checks. It does not sample or truncate history. Disjoint node offsets preserve
separate states for the same team at distinct issuances. No encoder or fixture
head pooling or normalization spans components.

Optimizer cadence, per-target observed-weight denominators, global outcome
weight denominator, training-only scales, prior penalty and CE/auxiliary
weights are unchanged. `train_loss`/`validation_loss` mean outcome CE;
`*_statistics_loss` mean auxiliary loss; `*_objective` mean combined loss
(the training objective also includes the prior penalty). Supported optimized
fits with validation select checkpoints and patience only by minimizing
`validation_loss`. Native fallback retains its existing control semantics.

The default optimized admission budgets are 128 MiB host and 256 MiB device,
up to eight graphs, with 1 GiB free headroom. Admission counts pending
descriptors, full eligibility/support temporaries, source-cache capacity,
encoding chunks, tensor inputs, concatenation/offset buffers, target buffers
and conservative backward intermediates. Host autograd nodes and tensor
wrappers are charged separately, including on CUDA. CPU tensor workspace counts as host
RAM. The ledger also charges query-only teams, the quadratic league one-hot
temporary and pass-local detached loss scalars. The existing resident store,
model, optimizer and full-population observation cache are shared baseline
requirements outside the incremental workspace budget. A pack must also fit
the estimated native-singleton envelope. Dense or
oversized cases fall back explicitly to native. Empty cones over nonempty native
history also fall back: this preserves zero versus absent gradients and AdamW
weight decay. These are admission estimates,
not a process RSS or allocator hard cap; fallback retains native requirements.
Measured host and device peaks remain an acceptance gate. Free memory never
changes packing decisions: insufficient headroom raises before yielding a
pass, and OOM does not silently retry with another plan.

Mode and all budgets enter recipes, inference serialization, model identity,
experiment summaries and training-resume identity. Summaries retain bounded
last-pass counters, fallback reasons and plan hashes. Changing mode or budgets
rejects resume. Existing inference artifacts retain their default behavior and
fingerprint. Training checkpoints already bind package source: checkpoints
created before this source change must continue with their original source;
this change does not bypass that guard.

## Predeclared validation protocol

Bounded tests precede expensive fits. Test each ablation separately with the
same initialized state, then compare fixed-weight logits/probabilities,
all losses, gradients, parameters and AdamW moments. Report absolute maxima,
relative L2 differences, and near-zero/sign-changing gradients explicitly.
The historical 0.00162255 first-update difference is explained by AdamW's
unchanged first-step formula `-lr*g/(abs(g)+eps)`: gradients +3.3023966e-8 and
-5.8954406e-8, lr .001 and eps 1e-8 produce that difference. This explanation
applies to the saved prototype's `layers.1.u.weight`: changing reduction and
matrix shapes perturbed a nearly cancelled gradient across zero. Epsilon and
other model settings have not been changed to mask this sensitivity. This
does not establish numerical or predictive interchangeability.

User-approved full-training matrix (launch status recorded with the receipts):

| Dimension | Values |
|---|---|
| Model | plain GAT; Bayesian fixture features + GAT |
| Targets/head | 74 full-match totals, three quantiles, combined route, auxiliary weight 1 |
| Representative training populations | canonical fold 00 (9,516 fixtures); fold 02 (18,888 fixtures) |
| Inner validation | freeze latest 20% kickoff batches, purge training labels not known at the earliest validation history cutoff; same row manifests for both modes |
| Seeds | 17, 29, 43, paired |
| Execution | native; dependency_packed with fixed recorded budgets |
| Full fit | 100 updates, validation CE every update, same AdamW settings; select minimum validation CE |
| Total | 24 full fits |

Keep outer test populations untouched during this acceptance comparison.
Freeze and hash inner split manifests before launching. Record each pair's
selected checkpoint step, stopping reason and any patience differences
separately from predictive differences. Report full per-seed trajectories.

Predeclared predictive gates for the frozen inner validation: candidate CE
increase <=0.005 per pair and <=0.001 averaged across pairs; accuracy decrease
<=0.5 percentage points per pair; Brier increase <=0.003; auxiliary quantile
loss increase <=1% relative. Report all results, including violations; do not
retune thresholds after seeing trajectories. These are proposed acceptance
criteria, not evidence of equivalence.

Performance gate: >=1.5x median warm update-plus-validation improvement,
separate cold/preparation/evaluation/end-to-end reporting, and no increase in
measured peak resident RAM, private commit, CUDA allocated or reserved memory.
Use at least three timed repeats with variability and reverse-order checks,
one GPU process at a time. Measurements on one fold or one runtime do not
establish this gate elsewhere. Do not reuse the earlier scalar 3x claim.

## Reproduction and evidence

The regression tests are `test_graph_dependency.py`, `test_graph_packing.py`,
`test_graph_execution.py`, `test_graph_execution_fallback.py`,
`test_graph_execution_config.py` and the native fresh-process resume tests.
They include unequal graphs, repeated teams, empty neighborhoods, residuals,
missing targets, nonuniform weights, fallback, mode/config binding and UI
round-trips. `XDIYO_EXECUTION_DEVICE=cuda:0` runs the integration parity tests
on CUDA; otherwise they use CPU.

The synthetic float32 integration checks use logits `atol=2e-6, rtol=2e-5`,
probabilities `5e-7 / 2e-6`, loss components `3e-6 / 3e-5`, gradients
`2e-5 / 2e-3` and AdamW moments `2e-6 / 5e-3`. The first-update parameter
check deliberately uses a **diagnostic** absolute bound of 0.0021 (about two
learning-rate steps); this accommodates near-zero sign sensitivity and is
not a numerical-interchangeability acceptance threshold. Actual absolute and
relative errors are reported from saved states. Separate float64 cone tests
check logits at `1e-13 / 1e-12` and gradients at `1e-12 / 1e-11`. Same-mode
repeatability, native fallbacks and fresh-process resume require exact state
equality, including optimizer and RNG state.

`tools/benchmarks/gat_dependency_execution.py` reads hash-pinned canonical
inputs and writes only to a new external output directory. It measures cold
input preparation, warm update/validation, metric replay and prediction replay,
and can export fixed weights and optimizer states for numerical comparison.
Each repeat restores the same initial weights and a fresh AdamW. These are
bounded measurements, not full fits. Receipts include runtime, source/input
hashes, controls, phase timings, memory peaks and actual execution traces.

Bounded receipts and comparisons are preserved in
[`gat_dependency_packed_evidence/`](gat_dependency_packed_evidence/).
Large state captures and disposable experiments remain outside the repository.

## Scope of the bounded measurements

The real-data probes use the canonical selection population: 7,542 training
fixtures and 1,756 validation fixtures, with 592 and 153 issuance groups.
Both arms have 74 full-match-total quantile targets, 222 outputs and the
combined embeddings/statistics outcome route. The Bayesian arm adds eleven
fixture rating features. This is a different workload from the earlier scalar
prototype; its speed is measured independently.

Canonical recipes supply the feature and target schema. The new comparison
explicitly fixes the combined route and auxiliary weight 1 for both modes;
it is not a continuation or reproduction of the older auxiliary-weight-2
research runs. Existing saved recipes and running experiments are unchanged.

Each process performs three logical source-cache-cold preparation passes,
then three warm one-update repeats from identical initialized parameters and
fresh AdamW state. No repeat is discarded. Update timing includes validation;
training-metric replay and historical validation prediction replay are timed
separately. These historical replays use internal validation inputs, preserving
the public prediction API's timestamp restrictions. End-to-end time includes
imports, input verification, initialization, state capture and all repeats.
The cold preparation measurement does not claim a cold operating-system disk
cache. GPU runs are serial; ordinary desktop graphics remain present.

The measured policy uses 128 MiB host / 256 MiB device admission limits, eight
graphs maximum and an explicitly recorded 256 MiB free-memory headroom. The
product default headroom remains 1 GiB. Tests use a separately recorded 64 MiB
headroom for their small synthetic fixtures. Those values are never changed
automatically after allocation failures.

Whole-process peak RSS/private commit and CUDA allocated/reserved peaks are
the no-extra-memory acceptance measures. Packing can increase the isolated
preparation-only host peak because several graph inputs coexist. That phase
must therefore be reported separately and does not, by itself, qualify as a
no-extra-RAM improvement. Estimates do not constrain unrelated processes,
allocator fragmentation, driver allocations or future third-party kernels.

## Full-matrix operation

`tools/benchmarks/gat_dependency_matrix.py --preflight` writes a new external
manifest containing the source commit and content hashes, numerical runtime,
input hashes, frozen inner split rows, controls and all 24 fits. `--run
--manifest PATH` starts the serial coordinator. A separate `--preflight --smoke`
creates four one-update worker checks; it cannot satisfy the full-matrix
acceptance gate. Each fit has its own native checkpoints, per-update CE,
auxiliary/combined metrics, phase timings, selected model and memory receipt.
The coordinator preserves failed evidence and refuses implicit resume.

On completion, `summarize_gat_dependency_matrix.py` applies the frozen gates
to all twelve pairs and records checkpoint/stopping differences separately.
Missing fits, missing memory evidence or any failed gate prevent acceptance.
The source and runtime remain fixed throughout; the runner stops if they
change. Windows desktop graphics processes are identified in the GPU-gate
receipts, while other or unresolved GPU users block a new update. Existing
research processes and outputs are never stopped or modified.

## Bounded results, 2026-10-17

Windows 11, RTX 4070 SUPER, Python 3.14.0, PyTorch 2.9.0+cu130. Times below
are seconds, median [minimum, maximum], three repeats unless stated. No
sample was removed. The later plain native repeats were slower than the
earlier ones, so both are reported; the conservative speed ratio is 2.88x.

| Model / execution | Update including validation | Cold preparation | End-to-end probe |
|---|---:|---:|---:|
| Plain native, initial | 70.98 [70.42, 71.22] | 5.35 [4.47, 5.47] | 346.78 |
| Plain native, reverse check | 82.90 [78.64, 84.35] | 5.83 [5.43, 5.99] | 395.50 |
| Plain dependency snapshot | 73.14 [71.17, 73.29] | 7.59 [7.52, 7.92] | 369.52 |
| Plain dependency lean | 77.60 [68.98, 82.41] | 4.33 [3.69, 4.35] | 361.92 |
| Plain packed, final ledger | 24.62 [23.85, 28.15] | 3.96 [3.30, 4.10] | 143.27 |
| Bayesian native | 72.52 [70.41, 86.60] | 6.29 [5.90, 7.16] | 385.07 |
| Bayesian packed, final ledger | 25.07 [23.60, 29.11] | 4.46 [3.86, 4.54] | 149.39 |

Dependency selection and lean preparation alone do **not** meet the 1.5x
warm-performance gate. Packing is necessary for the observed substantial
gain. Plain warm ratios span 2.88x to 3.37x depending on the native run;
pooling all six native timings gives 3.04x. Bayesian's three-repeat ratio is
2.89x. Its native receipt predates final admission-ledger bookkeeping, which
is unused by native execution. A matching-source reverse check measured
78.18 seconds (one repeat), or 3.12x; it is corroboration, not a three-repeat
estimate. The matching-source plain comparison and final packed source
guards have no differing graph files. Full receipts retain source hashes.

For matching-source plain native versus packed, median validation time was
5.81 versus 2.71 seconds; training-metric replay 24.01 versus 7.70;
validation prediction replay 3.37 versus 1.37. Bayesian three-repeat values
were 5.59 versus 2.43, 20.56 versus 8.09, and 3.38 versus 1.58 respectively.
End-to-end probe ratios are 2.42x (initial plain native), 2.76x (reverse
plain native) and 2.58x (Bayesian). These probes include cold preparation,
initialization, state exports and repeated one-step fits, not 100-step fits.

Whole-process OS high-water values and CUDA allocator peaks, in decimal MB:

| Model / execution | RAM RSS | Private commit | CUDA allocated | CUDA reserved |
|---|---:|---:|---:|---:|
| Plain native, reverse | 2067.70 | 4243.09 | 221.36 | 253.76 |
| Plain packed | 2032.23 | 4174.66 | 197.40 | 211.81 |
| Bayesian native, three repeats | 2069.14 | 4251.14 | 221.39 | 253.76 |
| Bayesian packed | 2025.93 | 4176.97 | 197.42 | 211.81 |

This bounded whole-process comparison meets the no-extra-peak-memory gate.
Cold-preparation peaks do not: plain RSS/private rose from 1080.71/2795.24 MB
to 1102.05/2858.06 MB; Bayesian rose from 1086.43/2797.02 to
1102.05/2853.27 MB. Thus no claim of lower RAM at every phase or universal
hard memory bound is made. Cold preparation alone is not accepted as a
no-extra-RAM optimization. Larger folds must independently pass the full
matrix's memory gates.

Fixed initial parameters were identical. The table gives maximum absolute
difference / relative L2 difference for native versus final packed execution:

| Quantity | Plain | Bayesian |
|---|---:|---:|
| Fixed-weight logits | 1.669e-6 / 1.690e-7 | 7.153e-7 / 1.739e-7 |
| Fixed-weight probabilities | 3.576e-7 / 1.236e-7 | 2.980e-7 / 1.252e-7 |
| One-update gradients | 1.669e-6 / 4.086e-7 | 1.669e-6 / 4.038e-7 |
| One-update parameters | 3.427e-7 / 2.995e-8 | 1.317e-6 / 7.109e-8 |
| AdamW first moment | 1.639e-7 / 4.109e-7 | 1.639e-7 / 4.045e-7 |
| AdamW second moment | 4.307e-9 / 8.310e-7 | 5.821e-9 / 8.479e-7 |

Both arms' maximum quantile-output difference was 1.526e-5, with zero changed
predicted outcome classes. All 107,265 plain and 110,081 Bayesian trainable
parameters are covered. Near-zero means |native gradient| <=1e-7: 34,201
plain and 33,658 Bayesian entries, with no sign flips or changed exact-zero
entries in these captures. Maximum pointwise relative gradient errors were
0.00383 and 0.00525; reporting only absolute errors would hide this scale
sensitivity. AdamW step counters were identical. Each mode's three restored
one-update repeats had identical model, gradients, optimizer and loss state.
Per-tensor absolute/relative values, every initial and post-update loss
component and near-zero counts are retained in the comparison JSON files.

The separate lean-preparation comparison against snapshot dependency
preparation had exactly equal captured outputs, losses, gradients, parameters
and AdamW state. This is a result for these captures, not a general
interchangeability guarantee. Comparing packing against lean preparation
isolates a 4.095e-7 gradient relative L2 difference and 1.324e-6 maximum
parameter difference. Component tests and these ablations distinguish
selection arithmetic from packing's changed matrix shapes and reduction order.

The historical scalar prototype's 0.0016225483 parameter difference is
independently recovered from the original immutable states. AdamW's formula
predicts 0.0016225480, residual 2.89e-10, and reconstructs equal starting
parameters to the same tolerance. This establishes the amplification
mechanism, not the acceptability of diverging trained trajectories.

## Regression and launch checks

The final production-runtime CPU suite passed 144 cases; seven CUDA-only
checks were skipped in that CPU invocation. The dedicated CUDA integration
suite passed 37 cases. Three CUDA fresh-process resume cases and 56 CPU
configuration/resume cases passed, including the three execution ablations,
74-target Bayesian configuration, forced native fallback, saved mode/budget
rejection and UI recipe round-trips. Five pinned-source legacy compatibility
cases also passed. These counts overlap where suites intentionally share
configuration tests; they are not a unique-test total. Fifteen runner
contract tests and seventeen aggregation tests passed. JUnit receipts are
included beside the benchmark JSON.

The four one-update smoke fits completed on the frozen first fold for both
models and modes. Each wrote a native checkpoint, exported model, complete
metrics and a receipt; all recorded artifact hashes were independently
verified. The smoke manifest and receipts are included. Startup testing
caught and fixed the canonical persisted-index contract, Windows venv
launcher ownership and CUDA initialization order. A mismatched hidden-CUDA
runtime was rejected without weakening the runtime guard. These smoke fits
validate the coordinator and persistence pipeline, not predictive acceptance.

An earlier exploratory CPU sweep on PyTorch 2.14 encountered signed-log
derivative failures also reproduced on unchanged main, and archived-source
pandas 3 compatibility failures. Those are not evidence of optimized-mode
parity. The production PyTorch 2.9 runtime and pinned-source gate above are
the reported passing validation target. The stale UI help-string assertion
in the first final CPU run was corrected and the entire final CPU selection
rerun successfully.

Matrix preparation verified the physical canonical fixture index and froze
fold 00 to 6,425 fitting / 1,993 validation / 1,098 purged fixtures, and fold
02 to 14,430 / 3,844 / 614. The earliest validation history cutoffs are
2021-11-19 and 2023-08-18 UTC. Preflight failures are preserved externally;
they never authorize changing inputs, bypassing runtime guards or resuming
a partially started matrix. The final launch uses a fresh manifest after
the source commit is fixed. Its external `status.json` is authoritative for
progress; this committed report deliberately leaves full acceptance pending.

Remaining acceptance blockers: the 24 full fits, per-pair quality/memory
gates, selected-checkpoint differences and full end-to-end timings. Until
those pass, this is an experimental opt-in implementation with bounded
performance evidence. No numerical-interchangeability claim is made.
