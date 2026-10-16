# Authorized CUDA integration verification

## Result and boundaries

The user authorized this gate and confirmed the GPU was idle. A process-command
and GPU-telemetry check found no Python compute job; graphics applications remained
open. Verification ran sequentially on the RTX 4070 SUPER with Python 3.14.0,
PyTorch 2.9.0+cu130 and driver 610.88. No production process was stopped or changed.

- Runtime candidate: `c4e26d855142cd4546ed6ddda905bbfcc30e7284`.
- Default reference: `861b00461943be7ff589c9bd7deace909e96c9c1`.
- Feature reference: `5ee87b5791f142f0c80432f1e0647d01f137a599`.
- 23 new CUDA integration cases passed in 212.74 seconds.
- 33 existing CUDA regressions passed in 93.96 seconds; 157 CPU cases deselected.
- Shared CPU helpers: **33 passed**, 23 deliberately skipped opt-in CUDA cases,
  185.81 seconds; one existing PyTorch TF32 API deprecation warning.
- Zero CUDA failures, errors or skips. No tolerances were relaxed.
- No runtime/model fixes were needed. Changes in this verification commit are
  tests and evidence only. Main, the research checkout and frozen artifacts were
  not changed. This branch remains unmerged and unpushed.

## Fresh evidence

| Gate | Coverage |
| --- | --- |
| Pinned source parity (5) | Main scalar/CNN defaults; feature memory, generic numeric statistics, and memory+statistics+CNN/KL. Independent source imports and processes. Initial prepared inputs/scaling/weights, two updates, losses, gradients, full AdamW, predictions, embeddings, attention and model identity compare exactly. |
| Real CUDA residency (12) | Heatmaps on/off, memory on/off, auxiliary/combined/bottleneck routes. Cached versus uncached two-update trajectories, post-update metrics, full optimizer/gradient/parameter comparisons and predictions are exact. Actual CUDA cache hits and budget checked; cached segments contain no learned initial state. |
| Extension continuation (6) | Memory, generic statistics and combined modes; interruption at regular update and early stopping. Fresh processes, residency enabled, full gradient/AdamW/model/scaler/RNG/metrics/winner parity. Completed checkpoints perform zero extra updates. |
| Reload | All five pinned-source cases compare pre-save outputs with same-GPU reload exactly, including embeddings and attention. CPU reload uses predeclared atol=1e-6, rtol=1e-5. |
| Existing regression (33) | Default GAT/CNN fresh-process continuation and residency, selective frozen inspection, owned reconstruction, statistics families/quantiles, half-only/fixture feature persistence and native CUDA updates. |

`cuda-environment.json` records exact runtime and 465 source/test hashes captured
before continuation completed. `cuda-verification.json` checks those hashes after
all runs, records receipt hashes and confirms runtime bytes still match the
candidate commit. Runtime/test files were not edited during any resume test.
Logs and JUnit XML are retained beside this report. Original CPU and renderer
receipts remain intact; the shared CPU gate helpers were rerun separately after
the CUDA suite, including verification that CUDA tests skip unless opted in.

## Reproduction

Working directory: `C:/Users/luisi/.codex/worktrees/gat-feature-integration/xDiyo`.
Run only with explicit GPU authorization and no competing training job.

```powershell
$env:CUDA_VISIBLE_DEVICES='0'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:XDIYO_CUDA_CHECKS='1'
$env:XDIYO_RUN_GPU_AUDIT='1'
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe' -m pytest tests/analytics/test_graph_integration_cuda.py -q --tb=short --maxfail=1 --junitxml=docs/analytics/gat_feature_integration/cuda-integration.xml *> docs/analytics/gat_feature_integration/cuda-integration.log
& 'C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe' -u -m pytest tests/analytics/test_temporal_graph.py tests/analytics/test_graph_extensions.py tests/analytics/test_graph_statistics_repair.py tests/analytics/test_graph_speed.py tests/analytics/test_graph_goal_features.py tests/analytics/test_input_residency.py tests/analytics/test_performance_inspection.py tests/analytics/test_native_training_resume.py -k 'cuda or idle_gpu or scheduled_gpu' -q --tb=short --maxfail=1 --junitxml=docs/analytics/gat_feature_integration/cuda-regression.xml *> docs/analytics/gat_feature_integration/cuda-regression.log
$env:CUDA_VISIBLE_DEVICES='-1'
$env:XDIYO_CUDA_CHECKS='0'
$env:XDIYO_RUN_GPU_AUDIT='0'
& 'C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe' -u -m pytest tests/analytics/test_graph_integration_gates.py tests/analytics/test_graph_integration_cuda.py -q --tb=short --maxfail=1 --junitxml=docs/analytics/gat_feature_integration/cpu-gate-regression.xml *> docs/analytics/gat_feature_integration/cpu-gate-regression.log
```

## Limits

This establishes bounded deterministic integration/continuation compatibility in
the recorded environment, not bit-exact behavior under nondeterministic CUDA or
across different runtime/source profiles. It does not establish predictive benefit,
large-fold speedups, hosted CI or live-browser QA. No selection/refit study or
large-fold benchmark was rerun. Input residency remains disabled by default.
