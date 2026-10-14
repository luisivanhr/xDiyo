# Verification — 2026-10-14

Base: 1619c32921a6f1c76af319d582b3241ba6e18fd4, isolated
codex/gat-boundary-memory-stat-forecast worktree. No merge or experiment launch.

Python: C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe.
CPU synthetic tests, PYTHONPATH=src and PYTHONDONTWRITEBYTECODE=1.

## Test commands/results

    python -m pytest tests/analytics/test_graph_numeric_targets.py tests/analytics/test_graph_statistics_repair.py tests/analytics/test_graph_extensions.py tests/analytics/test_graph_integration.py tests/analytics/test_temporal_graph.py -q --disable-warnings --maxfail=2

135 passed, 15 skipped. Skips are explicit opt-in idle-GPU audits.
Following the final canonical-goals period validation and four further acceptance
checks, the complete numeric-target file was rerun: **27 passed** (23 previously).

    python -m pytest tests/analytics/test_graph_numeric_targets.py -q --disable-warnings --maxfail=1
    node --test tests/analytics/ui_cnn.mjs

27 Python tests passed; 7 form-renderer tests passed. UI tests exercise native
renderer controls/events and JSON persistence, not a live browser screenshot.

Coverage includes full 74-target/222-output construction, home/away mapping,
fractional percentages, signed/absolute covariance, hand pinball/weight normalization,
constant and fitting-only scales, zero weights, release boundaries, two side gathers,
auxiliary gradient routing, fixed-weight future-label safety, original-unit reports,
exact CPU model save/reload, changed schema rejection, native recipe preparation,
JSON/Python exports, and validation-CE checkpoint/patience versus improving auxiliary
loss. Two-arm example construction is tested. Existing legacy/disabled tests passed.

## Source semantics and preparation timing

Read-only support validation passed for all target declarations against the existing
saved expanded-totals fold 02 history: 55,976 directed rows. This is source validation,
not training coverage; per-fold positive-weight fitting coverage is saved natively
only during a fit. No retained research artifacts were changed.

A bounded label-row lookup microbenchmark used the first 2,000 home fixtures from
that history, all 74 targets. Exact array equality (including NaNs) passed.
The old method rebuilt a MultiIndex and reindexed the full target frame for each
target. The new method builds two side lookups and gathers individual columns.

Three repeats, seconds:

- Previous row gather: 0.148637, 0.146454, 0.149031.
- New row gather: 0.004870, 0.004238, 0.004257.

Median ratio: approximately 34.9x for row gathering only. This excludes Parquet
loading, availability/support validation, scaling, tensors, gradients and training.
No end-to-end training speedup is claimed. Implementation hashes and timings are in
label_lookup_benchmark.json.

## Files and limitations

- graphs: new numeric target schema, labels, transforms/scalers, manifests,
  quantile parameterization/loss, unavailable auxiliary validation evidence.
- reporting: original-unit signed/bounded quantile scoring and unavailable scores.
- UI: registration, metadata, typed controls, output preview, JSON/Python export.
- tests/docs: synthetic acceptance coverage, source semantics, native full74 manifest,
  two-arm construction example, preparation timing.

No CUDA correctness/performance run, research selection/refit, real predictive
improvement, cross-process optimizer resume, or live browser QA was performed.
Frozen Bayesian feature construction and the runner's choice of new refit budgets
remain the research workflow's responsibility; neither was executed here.
