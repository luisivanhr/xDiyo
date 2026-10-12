# GAT observation-weight follow-up

Ayre's audit of `d22aae1` confirms the five preceding CNN repairs. The remaining
inherited graph weighting issue reproduced locally: scales `1e-50` and `1e100`
produced nonfinite losses/gradients; scale `1.2e38` produced zero data and MSE
reconstruction losses and zero data/reconstruction gradients. This does not mean
that prior penalties or AdamW decay would also contribute zero to an optimizer
update, nor does it establish corruption of previous research runs.

## Repair

`_targets` now scales fitting and validation weights independently by an exact
power of two in float64 before casting to float32. The maximum stored weight
is between 0.5 and 1. `frexp`/`ldexp` handle float64 subnormal inputs and maxima
near its upper limit without constructing an overflowing scale factor.

The existing weighted-average objective is preserved. No epsilon floor,
clipping or learned-model change is introduced. Individual zeros remain zero;
positive weights whose *relative* dynamic range cannot survive float32 are
rejected explicitly instead of silently disappearing. The tests include an
explicit rejection of `[1e-300, 1e300]`.

Reconstruction still receives the issuance's share of total fixture weight,
independent of historical edge count. That share is calculated before multiplying
by auxiliary loss. The full-population denominator is calculated once per replay
instead of repeatedly scanning weights for every issuance.

## Verification

279 distinct targeted Python tests passed locally:

- 17 new CPU checks cover fitting/validation, scalar GAT, CNN without a decoder,
  MSE and KL; scales include the original failure cases, `5e307` and the smallest
  positive float64. Values and gradients agree within float32 roundoff.
- Six new CUDA tests cover scalar/MSE/KL fitting and validation scale invariance.
- The existing CUDA CPU-equivalence/GPU-training test passes.
- 255 existing CNN, GAT, streaming and training-control tests pass on CPU.

The frozen eager reference files remain untouched. All existing bit-exact
checks of ordinary-weight gradients, optimizer state, histories, predictions,
embeddings and exports still pass. The one assertion comparing raw stored
weights now compares their normalized proportions because the native storage
uses binary rescaling. No model-value or optimizer tolerances were relaxed.

CUDA checks used the local RTX 4070 SUPER and PyTorch 2.14.0+cu130. New CUDA
scale checks compare losses and backward gradients; they do not establish
complete AdamW-state or parameter equality across devices. No research fit or
saved experiment artifact was modified.

## Speed check

The bounded synthetic CPU benchmark from the earlier audit was repeated against
`d22aae1`, one CPU thread, 20 fitting and six validation GAT issuances. Median
step times were 148.97 ms before and 149.13 ms after: effectively unchanged at
this measurement scale. A regression assertion verifies one full weight sum per
replay. This is not a large-fold or CUDA throughput benchmark.

```powershell
$env:PYTHONPATH='src'
python examples/benchmark_cnn_audit.py --baseline d22aae1
```

The separate point-publication-time limitation remains: native match/result
availability does not establish the meaning of point-level `observed_at`.
