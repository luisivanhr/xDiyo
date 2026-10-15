# Owned inputs and resumable training follow-up

## Scope

Branch: `codex/performance-owned-caches-resume`, based on `2e3b967`.
This branch is **not merge-ready until the remaining CUDA gates pass**.
The current completion pass is documented in [completion.md](completion.md);
that document supersedes the deferred-status matrix below. This file retains
the earlier implementation and validation history. Neither this branch nor
`codex/output-preserving-performance` has been merged. No frozen local/Hostinger
runtime, saved fit, dataset, or production experiment was changed. GPU tests are
masked explicitly; a separate Python GPU process (PID 83592) remained active.

The original EMA qualification stands: recurrence reuse still constructs and
compares full eligible-history prefixes. It is not fully linear-time EMA.

## Implementation matrix

| Handoff item | Delivered | Remaining acceptance boundary |
| --- | --- | --- |
| 1. Training checkpoint | Native GAT and standalone CNN `fit_resumable`, separate from inference export. Saves model, all AdamW state and gradients, preprocessing, Python/NumPy/torch CPU and applicable CUDA RNG, loop/restart/scheduler patience counters, best selection state, data/config/source identity and runtime/device profile. Payload and commit-marker hashes are verified before atomic directory/pointer publication. | Exact fresh-process continuation demonstrated on CPU only. CUDA continuation and CUDA inspection remain unverified. Fixture-head coverage uses a synthetic scalar fixture feature, not a freshly calibrated Bayesian feature. No scaler is used by these native backends. |
| 2. Finalization | `CheckpointPolicy(finalization="prediction_only")`, with `retain` default; exposed in native UI inventory and recipe serialization. Requires a completed, verified checkpoint matching this fitted object's parameters and preprocessing. Removes optimizer, gradients and prepared training/validation populations. Predictions, graph embeddings/attention, CNN encoding and inference export/load are tested after cleanup. Explicit capability required for custom adapters. | External/custom cleanup implementations remain their adapter's responsibility. Same-runtime training restoration requires a matching native checkpoint, not an inference export. |
| 3. Ownership | Standalone CNN numeric populations use immutable byte-backed arrays, a content generation, configuration/preprocessing/device binding, copy-isolated public reads and an explicit fit/finalize lifetime. Nested graph fixture metadata is isolated. Native event-store inputs and public frame access now isolate nested object cells/attrs. | No universal frozen execution snapshot spanning all custom stores, feature objects and GAT preparation caches is claimed. No derived CPU population is dropped on an unverified version-counter assumption. |
| 4. GPU residency | Not enabled. Existing single-graph streaming and CPU preparation remain. | Deferred: aggregate per-device accounting must include concurrent train/validation/inspection owners, active transfer leases and activation/optimizer headroom. A per-cache byte limit alone cannot demonstrate that guarantee. Active GPU work prevents the required plain/fixture-head GAT and CNN memory/eviction parity checks. |
| 5. Spatial/export/inspection | Owned standalone CNN KL target validation occurs once during preparation. Device normalization, loss reductions and gradients remain exact; arbitrary mutable populations still validate. Native graph export traverses internal tables synchronously, without allocating public full-frame copies; custom store subclasses retain public fallback. Byte-identical Parquet export test. | GAT shared for/against grid ownership and once-only KL validation are not extended to publicly mutable/custom graphs. CUDA select-before-transfer inspection tests remain pending. |
| 6. Search and run index | Existing full content/source validation remains intact. | Deferred: no frozen-search lease or manifest index was introduced. `execution_key` currently permits arbitrary mutable/callable inputs; caching a signature alone would not freeze the data actually consumed. `read_runs` remains authoritative, with visible corrupt-record failures. Need a verified freeze-to-fit boundary plus concurrent atomic-index publication/mutation tests before substituting cached decisions. |
| 7. Reports/UI retention | Existing eager completion validation, native report format and download links are retained and regression-tested. | Deferred: eager `result.to_html()` differs from the served `render_result_report`, which refreshes local badges and supplies spatial transport. Custom renderers are not serialized by `FootballExperiment.load`; unsaved preparation/prediction jobs can uniquely own data. Reusing eager HTML or evicting these payloads would change behavior without a complete verified reload/render capability. |
| 8. Acceptance gaps | Nested custom crosswalk cells tested through actual Parquet/hash validation. Frozen original recovery encoder checked byte-for-byte independently of candidate `pack`. Isolated pandas 2.2.3/3.0.0 environments tested, alongside current pandas 2.3.3 neural tests. CPU owned-input benchmark uses separate reference/candidate processes. | Interactive local-file preview was blocked by the browser URL policy; no workaround used. No repository GitHub workflow is tracked; remote integrated CI is not claimed. GPU memory/cold-preparation and production-scale multi-candidate/fold/report throughput remain pending. |

## Native checkpoint usage and boundaries

```python
from xdiyo_analytics.training import CheckpointPolicy

# Ordinary behavior remains the default. Native GAT/CNN now implement resume.
policy = CheckpointPolicy(every=10, finalization="retain")
# Opt-in cleanup is performed only after the completed checkpoint is published.
prediction_only = CheckpointPolicy(every=10, finalization="prediction_only")
```

Use the existing experiment/training checkpoint-policy slot. A policy namespace
allows only one active writer, as before. Checkpoint directories are trusted local
artifacts, loaded as data/tensors with `weights_only=True`, not arbitrary model
pickle. They contain `training.json` and `training.pt`; ordinary model exports are
not accepted as resumable checkpoints. Checkpoint offers rejected by `every` do
not clone optimizer/model buffers. Final publication is mandatory even if it falls
between periodic offers.

Resuming rebuilds deterministic prepared inputs from the matching fitting context,
verifies their preprocessing, then restores training state and RNG. Changed source,
data/config or execution profile is rejected. This strict source binding also means
an intentionally upgraded library must not silently resume an older native training
checkpoint. It can still use a compatible inference export through the existing
model-loading path. Cross-runtime/device bit-exact continuation is not promised.

CPU cold-process tests deliberately exit the checkpointing process immediately
after the second atomic pointer publication. A new process, with cold caches,
continues training. It compares every remaining update's gradients, parameters,
AdamW counters/buffers, loss/history, selected checkpoint, predictions and final
RNG against uninterrupted training. A further completed restart makes no updates.
Tests cover GAT, GAT with fixture inputs, CNN classification and CNN regression,
with optional KL reconstruction, validation and scheduler/restart state.

## CPU benchmark

`benchmarks/owned_cnn_inputs.py` loads the original CNN adapter and network from
`2e3b967` for its reference worker. The candidate worker uses this branch. Both use
Python 3.14, torch 2.9.0+cu130, pandas 2.3.3, one CPU thread and deterministic mode.
Each is a separate process. Three reference/candidate pairs alternate order,
with the same synthetic 2,000-fixture population, two
10x10 maps per row, whole-map missingness, batch size 128 and KL reconstruction.
No production experiment is opened. Five updates include post-update training-loss
evaluation. Timing summaries exclude the first warm-up update; the table reports medians
across the three process runs.

| Measurement | Frozen reference | Owned inputs |
| --- | ---: | ---: |
| Cold preparation | 1.08870 s | 1.09895 s |
| Median warm update + evaluation | 0.093157 s | 0.093379 s |
| Peak process working set | 723,124,224 bytes | 722,866,176 bytes |
| Working set after updates | 721,760,256 bytes | 721,129,472 bytes |
| GC-tracked objects at receipt | 346,736 | 346,926 |

All five loss records and hashes of full model/gradient/AdamW state match exactly
in all three pairs. Observed warm ratio is 0.998x (effectively unchanged); cold
preparation is 0.9% slower and RAM effectively unchanged. Earlier single-pair
probes varied; they do not establish a CPU speedup. This is a small CPU microbenchmark, not a demonstrated end-to-end GAT or
GPU speedup. Immutable ownership still makes transient batch copies so writable
Torch views cannot mutate retained input storage. Object counts include interpreter
and module state; they are not unique prepared-object counts. Windows process peak
working set includes imports/fixtures. OS file-cache usage is not separately
attributed; CUDA allocated/reserved peaks are null because CUDA was masked.

Final receipts: `owned-cnn-reference-{1,2,3}.json`,
`owned-cnn-candidate-{1,2,3}.json`, and `owned-cnn-summary.json`. Unnumbered
receipts preserve the earlier exploratory single pair. Each worker receipt includes exact
source/input hashes, package versions, every measured update and state hashes.

```powershell
$env:PYTHONPATH='src;tests/analytics;.'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUDA_VISIBLE_DEVICES='-1'
$python='C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe'
foreach ($round in 1..3) {
  $kinds = if ($round % 2) { @('reference','candidate') } else { @('candidate','reference') }
  foreach ($kind in $kinds) {
    & $python benchmarks/owned_cnn_inputs.py --kind $kind --output "docs/analytics/performance/owned-cnn-$kind-$round.json"
    if ($LASTEXITCODE -ne 0) { throw 'Benchmark failed' }
  }
}
```

## Validation receipts

- `follow-up-owned-final.xml`: **21 passed**, 89.28 s, on final implementation
  `6acca07`; owned-input checks, cold-process resume and cleanup/inspection.
- `follow-up-integrated.xml`: **2,166 passed, 8 skipped, 2 failed**, 277.14 s.
  One failure is the known baseline Bayesian serializer inventory expectation.
  The second is `WinError 5` during a Bayesian rating artifact-directory rename.
  Its failing Bayesian serializer/persistence files are unchanged from `2e3b967`;
  the isolated exact test passed on rerun without an implementation change
  (`follow-up-bayesian-save-recheck.xml`). Retain both receipts; the broad run is
  not represented as fully green. No claim is made that the Windows failure's
  underlying cause has been eliminated.

- `follow-up-pandas22-final.xml`: **398 passed**, 40.53 s, pandas 2.2.3.
- `follow-up-pandas30-final.xml`: **398 passed**, 43.16 s, pandas 3.0.0.
- Both isolated compatibility environments use Python 3.12 / NumPy 2.2.6,
  pyarrow 23.0.1, scipy 1.17.0 and scikit-learn 1.8.0. This covers tested supported
  versions, not every version combination in the declared dependency ranges.
- `follow-up-final-neural.xml`: **113 passed, 3 skipped**, 110.10 s. CUDA skipped
  explicitly via `CUDA_VISIBLE_DEVICES=-1`, not evidence of unavailable hardware.
- `follow-up-finalization-inspection.xml`: **4 passed**, graph inspection and CNN
  encoding/export after finalization. Counts overlap; do not sum as unique tests.
- Initial pandas 3.0 run had one `WinError 10053` local HTTP connection abort in
  `test_api_requires_token_and_matching_origin`; its focused recheck and complete
  final rerun passed without changing the implementation or test expectations.
- The independently frozen recovery source is stored at
  `tests/analytics/reference/recovery_01617da.txt`, with an exact SHA-256 assertion
  and Git `-text` attribute to prevent checkout newline changes.
- An interactive candidate report was generated at
  `%TEMP%/xdiyo-performance-followup.html`. Browser opening was rejected by its
  local-file URL policy. No interactive download/zoom success is claimed.
- The known baseline Bayesian UI serializer-inventory expectation remains separate;
  see `baseline-test-status.json` from the prior branch. Required integrated CI and
  interactive/CUDA checks remain pre-merge gates.

Compatibility selection (run with each isolated interpreter):

```powershell
$tests = @(Get-ChildItem tests/analytics/test_*.py | Where-Object {
  $_.Name -match '^test_(performance_(io|history)|recovery_|table_attribute_recovery|execution_recovery|reporting_viewer|ui_server|report_refresh_recovery)'
} | ForEach-Object FullName)
& $python -m pytest @tests -q --tb=short --junitxml=receipt.xml
```

The isolated environments are under `%TEMP%/xdiyo-pandas22-followup` and
`%TEMP%/xdiyo-pandas30-followup`; active training environments were not modified.

## Changed-file guide

- `training/native_resume.py`, `training/resume_loop.py`, adapter hooks and
  `training/checkpoints.py`: resumable state, publication, finalization lifecycle.
- `neural/prepared.py`, `neural/adapter.py`, `neural/network.py`: owned populations,
  generation/binding, copy-isolated tensor batches and once-only KL validation.
- `_ownership.py`, `graphs/store.py`, `graphs/streaming.py`, `odds/crosswalk.py`:
  nested object/metadata isolation; `graphs/persistence.py`: native internal export.
- `ui/build_inventory.py`, `ui/inventory.json`: finalization choice and help.
- `test_native_training_resume.py`, `native_resume_worker.py`,
  `test_owned_neural_inputs.py`, `test_performance_io.py`, frozen reference and
  `.gitattributes`: independent parity, lifecycle, encoding and mutation checks.
- `benchmarks/owned_cnn_inputs.py` and the receipts beside this document: repeatable
  CPU tradeoffs. No GPU performance claim or production experiment rerun.
