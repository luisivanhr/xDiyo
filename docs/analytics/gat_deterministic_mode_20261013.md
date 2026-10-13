# Native GAT deterministic mode verification

## Change and compatibility

`TemporalGraphConfig(deterministic=True)` enables strict deterministic PyTorch
execution around native backend operations. The UI checkbox is under the temporal
graph model's Config section. Default `False` preserves the caller's existing
execution settings. The setting round-trips through native recipes, JSON/Python
exports and saved models. Default-off model fingerprints are backward compatible;
enabled mode is part of model identity. See the
[usage and CUDA session requirements](temporal_team_gat.md#optional-deterministic-execution).

This patch changes execution policy, not architecture or the numerical objective.
The network source remains Git blob `bcaef00f0b7eb99845abf6b536a86bdc8b0918cc`,
identical to frozen `11ffa446` and the integrated `044a39c` baseline. AdamW
parameters, graph eligibility/order, scaling/masks, initialization seed, and
per-update training/validation loss evaluation are retained. The optimized
snapshot, row-encoding, tensor-streaming and loss-transfer paths are unchanged.

The source base is `044a39c4643fa4a78f3c2edc59b551fa2aa0559f`; fresh receipts
identify the exact working-tree source SHA-256 values. They are authoritative
for the candidate implementation tested before committing this patch.

## Regression and numerical checks

- 208 Python tests passed across the temporal graph, native integration,
  deterministic execution, goal selection, weight normalization, preparation
  memory, streaming/snapshot/speed, CNN/KL, CNN repair and UI inventory suites.
- 4 JavaScript form tests passed, including toggling the real Deterministic
  checkbox and preserving its boolean value in JSON.
- CPU and RTX 4070 SUPER comparisons cover the scalar GAT and optional CNN with
  KL reconstruction: three updates, exact prepared inputs, gradients, weights,
  full AdamW state, losses, predictions, embeddings, attention and save/load.
- Flags restore after successful execution, nested calls and exceptions.
  Disabled mode inherits caller flags. Tests exercise automatic CUDA workspace
  setup in a fresh subprocess, rejection of unsafe late setup, and propagation
  of unsupported deterministic-kernel errors.
- All four frozen real recipe shapes pass bit-exact GPU comparison against
  frozen `11ffa446`: totals and split halves, each with and without the 11
  Bayesian head features. Each uses three real training issuances, two
  validation issuances, one prediction issuance and three consecutive updates.
  Checks include causal eligibility, graph order, masks/scaling, per-update
  losses, gradients, parameters, full AdamW state, RNG, checkpoint selection,
  predictions and saved-model reload. Both training and validation losses are
  evaluated after every update. The native candidate runs with caller
  deterministic flags disabled, exercising its own scoped option; the reference
  runs with externally enabled deterministic flags.

The parity gate does not relax the predeclared `atol=1e-6`, `rtol=1e-5` fallback:
these comparisons pass the stronger bit-exact requirement.

## Full saved-fold benchmark

The benchmark uses an isolated copy of expanded-totals fold 02 and verifies the
saved model commitment, supplied support hashes and unchanged inputs. No study
or runner files are edited. Each implementation runs in a separate process in
the same Python 3.14.0 / PyTorch 2.9.0+cu130 environment on the RTX 4070 SUPER,
with one CPU thread and unchanged precision settings. One complete cold tensor
preparation pass is timed separately from one disposable full training update
and its post-update training-loss evaluation. This is a bounded timing check,
not a statistical timing study or a resumed experiment.

| Operation | Frozen reference | Native deterministic | Speedup |
| --- | ---: | ---: | ---: |
| Cold preparation | 140.818 s | 10.034 s | 14.03x |
| Training update | 190.341 s | 23.135 s | 8.23x |
| Post-update loss | 158.645 s | 15.817 s | 10.03x |
| Training + post-update loss | 348.985 s | 38.953 s | 8.96x |

The pass covered 1,463 issuances, 18,888 fixtures and 27,188,308 directed-edge occurrences. Peak process RSS was 1.784 GiB versus 1.724 GiB; peak allocated CUDA memory was 215.16 MiB versus 215.87 MiB. The full-update losses, gradients, parameters and complete AdamW state match bit-exactly. Native caller deterministic flags are restored to off after the update. Cold preparation here is measured independently of the model update.

Fresh receipts include exact commands, executable, source/input hashes and scope:

- [Four-shape GPU parity](data/gat_deterministic_mode_20261013/parity.json)
- [Full-fold comparison](data/gat_deterministic_mode_20261013/comparison.json)
- [Reference timing](data/gat_deterministic_mode_20261013/reference.json)
- [Native timing](data/gat_deterministic_mode_20261013/native.json)
- [Python regressions](data/gat_deterministic_mode_20261013/regression.txt)
- [UI tests](data/gat_deterministic_mode_20261013/ui.txt)

### Optimized-path control

The already-optimized path with external deterministic flags took 38.825 seconds
for the same full update plus loss evaluation, compared with 38.953 seconds using
the native option (0.33% difference in this single timing). Prepared tensors,
losses, gradients, parameters and full AdamW state match exactly. This supports
retention of the speed improvements; it does not estimate timing variance.

- [Optimized-path timing](data/gat_deterministic_mode_20261013/native-external.json)
- [Exact control comparison](data/gat_deterministic_mode_20261013/policy-control-native.json)
- [Final focused tests, including coordinate/attention parity](data/gat_deterministic_mode_20261013/determinism-final.log)

## Limits

Exact matching is conditional on fixed inputs, seed, hardware, runtime and
precision settings. It does not establish bit-exact matching to previous
nondeterministic runs. Cross-process training resume remains unverified.
The CNN check is synthetic and verifies execution fidelity, not predictive
benefit. Full-fold timing covers the supplied scalar expanded-totals artifact;
it does not measure every CNN/grid/architecture configuration. PyTorch's flags
are process-global, so unrelated concurrent PyTorch work is not supported while
this scoped policy is active.
