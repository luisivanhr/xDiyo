# Temporal GAT preparation speed audit

## First-pass result

This records the first optimization pass and its source hashes. The subsequent
[snapshot and fixture preparation pass](gat_speed_second_pass.md) builds on it.

On the supplied completed expanded-totals fold 02, two disposable full updates
including the unchanged post-update training-loss replay took **101–112 seconds**
with the first-pass patch, versus **461–462 seconds** using frozen `11ffa446` source.
The ratio of median wall times is **4.35×** (77.0% less elapsed time).

These are bounded local measurements, not a guaranteed study speedup. Another
Python process was using the GPU and the desktop remained active. Both comparison
processes used the same RTX 4070 SUPER, one CPU thread, Python 3.12.14, Torch
2.14.0+cu130, pandas 3.0.6, NumPy 2.5.4, highest matmul precision, TF32 matmul off,
FP32 model parameters and the existing FP64 grouped softmax. Ayre's original
study telemetry used a different Python/Torch runtime; it is not the baseline
for the speed ratio above.

The benchmark loads the completed model's parameters and fitted preprocessing.
Each repeat starts from those same weights with a fresh disposable AdamW and
updates over all 1,463 training issuances / 18,888 fixtures. It is not a resumed
research update. The 55,976 source edges, configuration, model commitments, input
hashes and model/store identities are checked; original files are never written.

## What changed

- One bounded CPU cache encodes immutable source rows after fitting. It retains
  standardized numeric columns, missing masks, home flags and league channels,
  using float64 arithmetic before the original float32 representation. Only
  eligible requested rows are encoded; invalid future rows cannot fail an earlier
  forecast. Store/config/schema/preprocessing changes invalidate the cache.
- Issuance snapshots copy only required metadata and optional deterministic CNN
  grid columns. Native public snapshots retain their complete schema.
- Fixed timestamp arrays and season-entry counts replace repeated edge-by-season
  Python loops. Node populations, strict kickoff boundaries, inclusive release
  bounds, memberships, decay, support and rest remain issuance-specific.
- Detached scalar losses transfer to the host once per pass, then accumulate in
  the original Python-float order. No graph or learned representation is retained.
- Opt-in phase timing measures preparation, transfers, forward/backward, metric
  replay, optimizer, fingerprint and native export. It does not affect model
  configuration, recipe identity or serialization.

No metric pass is skipped. The number and order of backward calls, global weight
denominator, prior penalty, AdamW updates, validation and checkpoint selection
remain unchanged. Existing artifacts and the research full-state replay wrapper
continue to work.

## Measurements

Warm graph preparation includes metadata, encoding, tensor casting and CUDA
transfer. Each row reports the median of three repeats; cold preparation is in
the JSON evidence. The cold graph genuinely has zero edges.

| Issuance | Edges | Encoded width | Reference | Optimized |
| --- | ---: | ---: | ---: | ---: |
| Early | 0 | 163 | 8.01 ms | 6.59 ms |
| Middle | 18,406 | 163 | 94.36 ms | 24.48 ms |
| Late | 37,752 | 163 | 255.08 ms | 45.56 ms |

| Full update | Training pass | Post-update metric pass | Total |
| --- | ---: | ---: | ---: |
| Reference 1 | 232.29 s | 229.99 s | 462.28 s |
| Reference 2 | 259.77 s | 201.22 s | 460.99 s |
| Optimized 1 | 53.30 s | 47.55 s | 100.85 s |
| Optimized 2 | 55.54 s | 56.07 s | 111.62 s |

Sampled peak process RSS was **2.037 GB reference / 1.886 GB optimized**. These
include the harness and CUDA runtime. Peak live device allocation was
175.05 / 175.80 MB; both processes reserved 228.59 MB. The slight device increase
includes the retained detached scalars. The single source-row cache allocated
**34.86 MiB**, within its shared 128 MiB value/readiness budget. No overlapping
snapshot or prepared-graph population is cached; existing lifetime/cancellation
tests confirm only one transient graph survives at a time.

A third optimized update with synchronized phase timing took **149.22 seconds**,
40.5% above the unprofiled median. Profiling therefore stays opt-in. It reported
84.46 s snapshot/index work, 6.74 s row encoding/cache lookup, 5.67 s tensor
transfers, 16.54 s forward, 11.72 s backward, 0.0034 s scalar transfer, 0.0038 s
optimizer and 0.0010 s fingerprint. The 74.00 s metric-replay timing includes
nested phases: **do not add it to those totals**. External research checkpoint IO
was not part of these disposable updates; native exports have a separate timing
hook, and external observers can wrap their work in that phase.

## Numerical checks and limits

- **Byte-exact prepared inputs** on the actual early/middle/late graphs.
- **Bit-exact deterministic CUDA** on those three real graph sizes, comparing
  logits, losses, gradients, parameters, AdamW state and CPU/CUDA RNG. This is a
  three-graph diagnostic, not a full-fold deterministic training comparison.
- **209 passing Python tests**, including CPU and deterministic-CUDA comparisons
  with frozen code; optional MSE/KL CNN reconstruction; scaler/store cache
  invalidation and budget fallback; missing values, tiny values and signed zero;
  empty graphs, delayed results, postponed rounds and same-midnight distinct
  memberships; large integer identities; export/load; exception cleanup; and
  supplied-wrapper continuation plus completed-budget zero-update replay.
- **3 passing UI tests**. Model parameters and exported recipes are unchanged.

The full-fold default-CUDA update is **not bit-exact**. Its largest observed
gradient difference was 2.10e-9, AdamW-state difference 2.33e-10 and parameter
difference 1.68e-6. A tight initial `rtol=1e-6, atol=1e-7` parameter check failed;
we have not relabeled that result as exact or silently relaxed its tolerance.
Baseline repeats also differed slightly in loss. Deterministic large-graph checks
pass exactly, consistent with reduction nondeterminism and its amplification by
a fresh AdamW near small gradients. Default-CUDA full-fold identity is not claimed.

The original support `microbenchmark.py` was also run locally after pointing
`FOLD` to the supplied directory and `MODEL` to its model subfolder, with native
source-manifest and model-commitment checks retained. Its float64 encoding hashes
match Ayre's cloud evidence for all three graphs. No original experiment,
checkpoint, library dataset, or active run was changed.

## Reproduction and evidence

Run separate processes with the same completed fold and an output location
outside that fold:

```powershell
python examples/benchmark_gat_speed.py --fold $fold --mode reference --repeats 2 --evidence $supportZip --output reference.json
python examples/benchmark_gat_speed.py --fold $fold --mode native --repeats 2 --profile --evidence $supportZip --output native.json
python examples/verify_gat_speed.py --fold $fold --output deterministic.json
```

The scripts require development dependencies `torch`, `pytest` and `psutil` and
use frozen comparison sources under `tests/analytics/reference`. They produce
only disposable diagnostic states outside the study.

Evidence: [reference timings](data/gat_speed_11ffa446/reference.json),
[optimized timings and phases](data/gat_speed_11ffa446/native.json),
[observed default-CUDA differences](data/gat_speed_11ffa446/parity.json),
[deterministic real-graph parity](data/gat_speed_11ffa446/deterministic.json),
[original microbenchmark rerun locally](data/gat_speed_11ffa446/local_original_microbenchmark.json),
[verified support input hashes](data/gat_speed_11ffa446/verified_support_inputs.json),
[implementation file hashes](data/gat_speed_11ffa446/source_sha256.json).

Changed production files are `graphs/store.py`, `preprocessing.py`,
`streaming.py`, `adapter.py`, `persistence.py`, `__init__.py` and the new
`profiling.py`. The remaining additions are regression fixtures, tests,
reproduction scripts and this documentation/evidence.
