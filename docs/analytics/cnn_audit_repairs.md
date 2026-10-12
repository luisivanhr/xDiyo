# CNN audit repairs

Ayre's five synthetic findings against `5eded399` reproduced locally. The
repairs are confined to the GAT/CNN development branch; no research fits or
saved experiment artifacts were changed.

## Repairs

1. Grid identity is checked independently of table hashes at builder/direct
   adapter construction, fitting, store attachment and native loading. This
   rejects coincidentally identical mass/count grids with incompatible semantics.
   Failed attachment leaves the original store in place. Scalar GAT behavior is
   preserved, and compatible extensions and embedded/external stores still load.
2. When a training metric controls selection, the backend evaluates it at the
   post-update state before observers, scheduling, early stopping, restart
   comparison and checkpoint selection. Validation monitors retain their existing
   evaluation path. Existing saved curves are not rewritten.
3. Standalone CNN artifacts retain the declared model specification and a separate
   complete effective training policy. Legacy artifacts load with an explicitly
   unknown effective policy rather than a fabricated one.
4. Spatial lookups reject floating-point event/team identifiers before joining.
   Exact uint64 identities and intentional missing maps remain supported. General
   heatmap point-kind support is unchanged.
5. Standalone weights are normalized once in float64 with max scaling, separately
   for prediction and observed-map reconstruction. There is no absolute floor
   that changes the objective under rescaling. Zero observed weight gives zero
   auxiliary loss. Detached batch losses accumulate in order on the device,
   avoiding a host scalar read after every minibatch.

## Verification

- 604 distinct targeted Python tests passed across the CNN, GAT, streaming,
  training-control and spatial suites. Two imported SciPy array-API checks were
  skipped; the existing CUDA-specific test was deselected for this CPU audit.
- Two native CNN UI tests passed. JSON/Python recipe and native artifact paths
  remain covered by the Python integration tests.
- New assertions cover the original probes, both restoration settings, restart
  and scheduler policies, retained training objectives, validation controls,
  legacy controls, exact/rounded IDs, MSE/KL weight rescaling and batch gradients.
- The existing frozen GAT reference tests retain bit-exact validation training,
  optimizer state, preprocessing, group order, predictions and exports.
- Sequential device accumulation matches the host-scalar reference exactly on
  CPU for both losses and gradients. Only two scalar reads occur per loss pass.

This is implementation verification, not evidence of predictive improvement or
CNN CUDA parity. Point `observed_at` remains an unresolved source publication-time
contract; native match/result cutoffs do not establish publication availability.

## Bounded speed check

Run from the checkout root with the local source on `PYTHONPATH`:

```powershell
$env:PYTHONPATH='src'
python examples/benchmark_cnn_audit.py
```

Synthetic CPU test, one thread, PyTorch 2.14.0, pandas 3.0.6. Standalone input:
512 rows with two 10×10 maps and optional MSE reconstruction; 20 measured steps
after three warm-up steps. GAT: 20 fitting issuances and six validation issuances;
eight measured steps after two warm-up steps. Baseline backend source is loaded
read-only from Git. No research dataset is accessed.

| Path | Baseline median | Repaired median |
| --- | ---: | ---: |
| Standalone CNN, validation monitor | 15.95 ms/step | 15.40 ms/step |
| CNN-enabled GAT, validation monitor | 143.23 ms/step | 141.01 ms/step |

These small differences are timing noise, not a speedup claim. There is no
measured regression in these validation-monitor paths. Typed integer validation
for one million rows took about 0.041 ms because it checks the integer domain
without a Python row loop. Nullable/object IDs require their appropriate checks.

Training-based selection deliberately performs one additional forward evaluation
to align the selected metric and checkpoint. On the standalone benchmark with
validation also present, this took 19.56 ms/step (about 27% above repaired
validation-based selection). It adds no backward pass or retained activation
cache. This correctness cost should not be mistaken for zero overhead. Large
production folds and GPU throughput have not been benchmarked here.
