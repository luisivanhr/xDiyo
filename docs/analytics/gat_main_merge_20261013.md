# GAT integration gate, 13 October 2026

## Candidate and scope

The starting remote tips were `c674a8832c476d2d9d02bc179d4e3f81aa0b3b5e`
(GAT) and `bfe249c53af8cee9ab801fde757542c5e55e552b` (main). Main was
integrated into GAT in `12565ed06e72c4eafffc1626ae2859a4b21a4423`.

All 483 main badge assets retain their recorded hashes. The catalogue selects
483 SVGs, with 199 alternative main asset records and their provenance retained.
The renderer accepted all selected assets; the real-browser regression passed.
No model implementation changed during this integration.

`codex/temporal-team-gat` has one commit absent from the starting main but already
contained in the published GAT branch. `codex/new-rating` is already contained in
both. Neither contains unique unpublished commits. Branches and worktrees are
retained. Main's existing data/scraping modifications are outside the integrated
file set and must remain untouched.

## Runtime and predeclared acceptance

- Executable: `C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe`
- Python 3.14.0, Torch 2.9.0+cu130, pandas 2.3.3, NumPy 2.4.6.
- RTX 4070 SUPER; one CPU thread, highest float32 matmul precision, matmul TF32
  disabled, deterministic CUDA algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Require exact deterministic comparisons. The user-prescribed fallback bounds
  are `atol=1e-6`, `rtol=1e-5`; they are recorded before execution and never relaxed.
- Original completed research files are read-only. Benchmarks use an independent,
  byte-verified copy of the saved expanded-totals fold 02.

## Regression and four-shape parity

**254 Python regression tests and 3 UI tests passed**, including asset inventory,
native orchestration, optional CNN, weighting, graph preparation lifetimes,
frozen-reference numerical comparisons, and real-browser reporter behavior.

`examples/verify_gat_merge.py` loads the exact four configurations and real
sidecars from the frozen study protocol: totals and split halves, each with and
without the 11 Bayesian fixture-head features. Each case uses three complete
training issuance groups, two validation groups and one later prediction group
from fold 02. Three updates are compared against frozen pre-optimization source.

All four cases passed **bit-exact deterministic CUDA** checks for prepared inputs,
causal eligibility, graph ordering, missing masks, fitted scaling, every update's
training/validation losses, gradients, parameters, full AdamW state, RNG,
predictions, checkpoint selection and native save/load behavior. Both training
and validation loss are evaluated after every update. Selection uses validation
loss and is checked against the actually retained parameter state. Bayesian
feature availability and fixture identities are checked by the frozen input
attachment helper. Input hashes are rechecked at the end.

These four-shape diagnostics use bounded real populations; they are not four
full-fold research fits. Cross-process training continuation was not tested in
this gate, and no additional resume-compatibility claim is made. Native
prediction-artifact save/load is tested separately from optimizer continuation.

## Benchmark receipt

The full-fold comparison passed **bit-exactly** for the saved update's gradients,
parameters and full AdamW state, both repetitions' training and post-update
losses, and sampled prepared tensors. Each repetition starts from the original
saved weights with fresh AdamW; this is not sequential training continuation.
The full benchmark covers 18,888 fixtures, 1,463 issuances and 27,188,308 directed
edge occurrences. Cold preparation is timed as a full
pass over every issuance before any warm update. Store loading/index construction
is measured separately. Every warm timing includes training and its full
post-update training-loss replay, with identical inputs and runtime for both
implementations. Peak process RSS and CUDA allocator memory are sampled/recorded.

| Measurement | Frozen reference | Integrated candidate | Speedup |
| --- | ---: | ---: | ---: |
| Cold full preparation | 154.107 s | 10.876 s | 14.17x |
| Store/index setup | 0.929 s | 0.975 s | 0.95x |
| Warm training update | 202.626 s | 22.940 s | 8.83x |
| Post-update loss replay | 180.134 s | 14.946 s | 12.05x |
| Training plus loss replay | 382.760 s | 37.886 s | 10.10x |
| Peak process RAM | 1.809 GiB | 1.761 GiB | |
| Peak CUDA allocated | 215.69 MiB | 216.41 MiB | |
| Peak CUDA reserved | 248 MiB | 248 MiB | |

Warm measurements are medians of two repetitions. RAM is sampled every 10 ms
from before store setup through the updates; CUDA peaks cover the same phases.
The candidate's slightly larger CUDA allocation is retained in the results.
There were no concurrent Python GPU training processes. Windows desktop GPU
clients remained active; these are local desktop measurements, not dedicated
headless-device timings. Cold means uncached native preparation in a fresh
process, not a flushed operating-system disk cache. Do not combine these ratios
with timings from a different Python/Torch environment.

The saved full-fold artifact has no validation partition; the four-shape gate
above supplies validation and checkpoint-selection coverage. Full-fold prepared
tensor hashes cover early/middle/late samples; the four-shape gate compares every
prepared group in its bounded populations. The full-fold benchmark also checks
that CPU/CUDA RNG does not advance during each update; its independently started
processes do not share an initial RNG seed. The seeded four-shape gate compares
RNG states exactly across implementations.

The first benchmark launch stopped before preparation because the new memory
measurement reset ran before Torch 2.9 had initialized CUDA. The harness now
initializes CUDA before resetting memory counters; the failed startup log is
retained. No model calculation, tolerance or input was changed to address it.
The standalone comparison script also needed its checkout's `src` path added
before importing test helpers; its initial import failure occurred before any
state comparison. Both corrected harnesses completed successfully.

Exact commands, source hashes, reference hashes, input hashes, configurations and
selected groups are included in the JSON receipts. All audit states and portable
test models are written outside the original completed experiment.

- [Four-shape parity](data/gat_main_merge_20261013/four_shape_parity.json)
- [Full-fold state comparison and speedups](data/gat_main_merge_20261013/full_fold_comparison.json)
- [Reference benchmark](data/gat_main_merge_20261013/reference.json)
- [Candidate benchmark](data/gat_main_merge_20261013/candidate.json)
- [Regression results](data/gat_main_merge_20261013/regression.txt)
- [Asset reconciliation](data/gat_main_merge_20261013/asset_merge.json)
- [Branch audit](data/gat_main_merge_20261013/branch_audit.json)
