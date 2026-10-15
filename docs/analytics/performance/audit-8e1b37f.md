# Review repairs after 8e1b37f

Branch: `codex/performance-owned-caches-resume`. Keep unmerged. Repairs are local;
this audit does not authorize publishing another revision. No frozen experiment,
production source binding, model artifact or dataset is changed.

## Repairs

1. **KL inference.** `cached_maps` and `native_tensors` take an explicit
   `reconstruction_targets` mode, defaulting to forward-input validation.
   Training/validation loss streams opt into target validation; prediction,
   embedding and attention inspection do not. KL validation mode is part of
   source-row cache identity. Only validated loss streams bypass repeated device
   target checks; ordinary/public groups retain validating reconstruction loss.
   A fitted KL GAT accepts later eligible observed zero maps for prediction and
   embeddings, with exact agreement against the uncached forward path. Both
   bounded-cache and no-cache cases reject those same maps as training targets,
   including after inference has populated the cache.
2. **Request/eviction race.** Object/retained-loader snapshots are taken under the
   retention lock. A request holds a strong reference before eviction can remove
   the job entry; verified disk loading can then proceed outside that snapshot
   lock. HTML selection/publication also uses the retention lock. Deterministic
   event barriers exercise real HTTP report and model-export requests while
   another thread attempts eviction, followed by successful report reopening.
3. **Optional spool failures.** Result reload verification and document write/read
   verification form one guarded transaction. Job replacements and payload removal
   occur only after both succeed. Failure leaves the completed result and HTML in
   memory and adds a `pinned:` warning. Injected write and read errors cover both
   older and newly completed jobs; both stay complete and can report/export.
   Report regeneration after eviction also keeps rendered HTML if optional
   spooling fails.
4. **NumPy controls/checkpoints.** Accepted integral controls (steps, seed,
   restarts, nested patience) become Python integers; numeric policy values are
   normalized. Checkpoint scalar metadata recursively becomes weights-only-safe
   Python numbers/booleans without cloning tensor buffers again. The manifest
   completion flag is a Python bool. NumPy controls pass publication, fresh-process
   load, continuation and completed no-op checks for all four native model forms.

## Continuation boundaries

Sixteen new independent-process cases cover four native model forms times NumPy
controls, restart boundaries, early stopping, and nonfinite monitor termination.
Controlled monitor sequences exercise stopping deterministically while retaining
real native optimizer updates. Each compares every update's parameters, gradients
and full AdamW state, final backend/RNG, selected history/summary and predictions
against an uninterrupted reference. No comparison tolerance is relaxed.

## Verification status

- Corrected KL and HTTP/fault regressions: 11 passed, 1 CUDA-gated skip (14.36 s).
- pandas 2.2.3: 411 passed (48.57 s).
- pandas 3.0.0: 411 passed (54.66 s).
- Initial focused run: 33 passed, 5 CUDA-gated skips, 6 failures in new test
  scaffolding (keyword-only UI API and full-column uncached reference). Both
  scaffolding issues were fixed; production code did not change for these fixes.
- Integrated final run: **5,739 passed, 34 skipped, zero failures**, 58 warnings,
  1,227.97 s. Skips cover unavailable/gated CUDA and SciPy array-API checks;
  warnings include deprecations and intentional float32 overflow probes.
- CUDA on repaired source: pending production GPU release. Prior CUDA receipts
  remain evidence for 8e1b37f only, not a claim that these repairs passed CUDA.

The initial integrated run was stopped after the corrected reference harness had
been collected in its old form; its partial log is retained. The final integrated
run starts afresh with corrected tests. Source/test hashes and GPU gate are in
`audit-8e1b37f-environment.json`; logs/XML retain actual results without replacing
historical receipts.

## Commands

Use the existing Python 3.14 environment and `PYTHONPATH=src;tests/analytics;.`.
CPU runs use `CUDA_VISIBLE_DEVICES=-1`, `XDIYO_CUDA_CHECKS=0` and
`PYTHONDONTWRITEBYTECODE=1`.

```powershell
& $python -m pytest tests/analytics -q --tb=short --junitxml=docs/analytics/performance/audit-8e1b37f-integrated-final.xml
$tests = @(Get-ChildItem tests/analytics/test_*.py | Where-Object {
  $_.Name -match '^test_(performance_(io|history|remaining)|recovery_|table_attribute_recovery|execution_recovery|reporting_viewer|ui_server|ui_retention_audit|report_refresh_recovery)'
} | ForEach-Object FullName)
# Run $tests with each existing isolated pandas22/pandas30 interpreter.
```

After full process-command and telemetry inspection proves the GPU free:

```powershell
$env:CUDA_VISIBLE_DEVICES='0'
$env:XDIYO_CUDA_CHECKS='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& $python -m pytest tests/analytics/test_input_residency.py tests/analytics/test_graph_owned_kl.py tests/analytics/test_performance_inspection.py tests/analytics/test_native_training_resume.py -q --tb=short --junitxml=docs/analytics/performance/audit-8e1b37f-cuda.xml
```

Do not edit runtime source during cold-process resume checks. Capture fresh
source/runtime hashes. Do not run this command while a production worker or a
successor supervisor is active. Current wait gate is the Bayesian recovery
worker PID 145340 (launcher 139544), `long200_aux2_bayes11_fold03.json`; recheck
all processes, not only these historical PIDs.

## Performance and limitations

Residency remains disabled by default. The saved large-fold receipts showed no
benefit, with zero hits; synthetic CNN improvement is a separate measurement.
Enabled residency owns a fixed plan/preprocessor per stream, duplicating bounded
CPU caches (up to 128 MiB each for edge/map encodings per stream, plus plan and
transients). This CPU storage is outside the GPU budget and included in measured
process RSS. There is no claim that all source caches are shared when enabled.
No new speedup is claimed from these correctness repairs. The existing browser
CSV-download limitation and lack of remote CI remain; local HTTP report/export
checks passed. Keep the branch unmerged pending final verification/review. The existing quiet
heartbeat has been reactivated for the repaired-source CUDA gate; it must wait
for all production workers and successors to release the GPU.
