# Current handoff: audit verification complete

The repairs are committed at `745570fc4de0cdf857355ce283f21c154a6f0f35`.
Read [audit-8e1b37f.md](audit-8e1b37f.md), its environment receipt and the
fresh `audit-8e1b37f-cuda-environment.json` for the tested source/runtime hashes.

- CPU integrated: 5,739 passed, 34 skipped.
- pandas 2.2.3 and 3.0.0: 411 passed each.
- Final repaired-source CUDA selection: 56 passed, zero skips/failures, 464.61 s.
- Strict deterministic CUDA and fresh-process continuation comparisons passed.
- No runtime source changed during verification; production fits were untouched.

The user's subsequent instruction authorizes pushing this completed branch;
it supersedes historical no-push instructions below. Keep the branch unmerged.
Commit the evidence, push `codex/performance-owned-caches-resume`, verify the
remote tip, and pause the waiting heartbeat. Do not rerun frozen fits or change
their artifacts. Saved-fold benchmarks are already completed; no new performance
claim is made for these correctness repairs.

---

# Completion handoff: verification complete

## Completion status (2026-10-15)

All remaining CUDA checks completed after the production GPU workers exited.
37 tests passed with no skips; saved-fold plain/Bayesian parity and all three CNN
process pairs matched exactly. Fresh-process deterministic resume passed for all
four native model forms. See completion.md and completion-cuda-summary.json for
measured tradeoffs: large-fold GAT residency had no benefit and remains disabled
by default; synthetic CNN warm updates improved 1.223x with slower preparation.
No runtime source changed during final GPU verification. Pause the heartbeat.
The commands below are retained for reproduction, not pending work. Keep the
branch unmerged and unpushed.

## Original scope and safety

- Worktree: `C:/Users/luisi/.codex/worktrees/performance-owned-caches-resume/xDiyo`
- Branch: `codex/performance-owned-caches-resume`; keep unmerged and unpushed.
- Do not modify primary xDiyo checkout, frozen local/Hostinger runtimes, datasets,
  saved experiment artifacts or any unrelated branches.
- User explicitly said **wait for the fit to finish**. PID 83592 is the frozen
  worker, with launcher 82964 and continuation/supervisor processes that may
  launch successors. Do not use abbreviated process output as proof of exit.
- Heartbeat `finish-performance-cuda-verification` checks every 30 minutes,
  stays quiet while occupied and resumes the authorized work once free.
- Active worktree is outside writable roots; use explicit workdir and scoped
  escalation for commands. Never let the primary editable installation select
  source: set `PYTHONPATH=src;tests/analytics;.`.
- Do not edit source during checkpoint continuation tests; source fingerprints
  intentionally reject a changed implementation.

## CPU evidence

See completion.md and receipts. Full suite before the final native GAT KL
validation refinement: 5,710 passed, 33 skipped. Focused refinement run: 45 passed,
5 CUDA skips. Pandas 2.2.3 and 3.0.0: 407 passed each. Full affected neural run is
recorded in completion-neural-final.xml (218 passed, 28 skipped); final selection/recovery tests: 397 passed. The final CPU benchmark identifies implementation 592a6ff and all source hashes. Three frozen CNN reference/candidate CPU
pairs have exact losses and model/gradient/full-AdamW state hashes.

## Gate and environment

Read-only full process command lines:

```powershell
Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -match 'gpu_worker\.py|long200_aux2|expanded-gat'
} | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Depth 3
nvidia-smi
```

Worker progress:
`C:/Users/luisi/Documents/Codex/2026-10-11/task/expanded-gat-11ffa446/run/gpu_runs_long200_aux2_refits/gat_totals_bayes11_aux_quantiles/refit-fold-02/LATEST.json`
was step 050 during CPU verification. Confirm no successor worker takes over.
If ambiguous, wait; do not compete. Automated test gates alone do not prove idle.

Only after the GPU is free:

```powershell
$env:PYTHONPATH='src;tests/analytics;.'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUDA_VISIBLE_DEVICES='0'
$env:XDIYO_CUDA_CHECKS='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$python='C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe'
& $python -m pytest tests/analytics/test_input_residency.py tests/analytics/test_graph_owned_kl.py tests/analytics/test_performance_inspection.py tests/analytics/test_native_training_resume.py -q --tb=short --junitxml=docs/analytics/performance/completion-cuda.xml
```

The four fresh-process tests independently launch uninterrupted, intentionally
interrupted (exit 73), resumed and completed workers, checking every update,
optimizer, gradients, RNG, checkpoint history and prediction artifacts. Do not
relax exact expectations or claim cross-runtime resume.

## Saved-fold CUDA benchmarks

Plain:
`C:/Users/luisi/Documents/Codex/2026-10-11/task/expanded-gat-11ffa446/run/gpu_runs/gat_expanded_totals/refit-fold-02`

Bayesian:
`C:/Users/luisi/Documents/Codex/2026-10-11/task/expanded-gat-11ffa446/run/gpu_runs/gat_expanded_totals_bayes11/refit-fold-02-fresh-001-source-de32880e`

Run each in separate processes with cache 0 and a suitable enabled budget
(initially 6442450944 bytes, 2147483648 bytes headroom, if telemetry permits):

```powershell
& $python benchmarks/gpu_residency_saved_fold.py --fold $fold --cache-bytes $budget --headroom-bytes 2147483648 --updates 3 --output $receipt
```

Script verifies every MODEL_COMMITTED hash, copies to an isolated temporary
folder, verifies copied hashes/model version, and uses train_metadata columns
for the Bayesian fixture inputs. Do not use fixture_features.parquet as training
input: it is the test population. Verify disabled/enabled prepared hashes, all
metrics, per-update full-state hashes, source hashes and inputs match. Compare
cold preparation and warm updates separately. Include cache hit/miss and memory
receipts. Current script freshly initializes matching candidate models to isolate
residency; do not describe this as continuing or retraining the frozen experiment,
or as a comparison against an old graph source version.

## CNN CUDA benchmark

Run separate reference and candidate processes using the same environment:

```powershell
& $python benchmarks/owned_cnn_inputs.py --kind reference --device cuda:0 --output docs/analytics/performance/completion-cnn-cuda-reference.json
& $python benchmarks/owned_cnn_inputs.py --kind candidate --device cuda:0 --cache-bytes 268435456 --output docs/analytics/performance/completion-cnn-cuda-candidate.json
```

The reference adapter/network source is frozen at 2e3b967. Compare exact metrics
and every model/gradient/full-AdamW hash; both include post-update loss evaluation.
Record CUDA allocated/reserved peaks and host memory separately. Repeat alternating
order if needed for stable timing; do not launch alongside production workers.

## Finish

- Repair failures without weakening parity, capture fresh source hashes/commands.
- Update completion.md with CUDA receipts and measured tradeoffs, not promises.
- Keep limitations explicit: no remote CI workflow, browser CSV download event was
  not confirmed (links/data were verified), synthetic lifecycle is not realistic
  production multi-fold throughput, EMA prefixes remain non-linear.
- Commit coherent verified changes as the handoff requests; no push or merge.
- Pause the heartbeat when remaining work is complete and notify the user.
