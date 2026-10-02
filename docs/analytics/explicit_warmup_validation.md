# Explicit warm-start validation

Validated 2026-10-03 against checkout baseline
`915f31e3c00f34a0dabcb87a697ba2be2a5e0942`, preserving existing unrelated changes.

## Implemented surfaces

| File/surface | Change |
|---|---|
| `features/warmup.py` | Append backward-compatible mode/cohort/variance parameters; dispatch explicit modes separately |
| `features/seeded.py` | Boundary-window own/cohort seeds, moment updates, exact handoff, audits and source fingerprints |
| `features/movement.py` | Nullable TeamMovement leaves and stable-ID native/override evidence resolution |
| `features/evaluation.py`, public exports | Dispatch, scoped evaluation, nested audit retention and source/context identity |
| `datasets/assembly.py` | Carry audit/evidence/fingerprint in definitions; retain existing row assembly |
| UI catalog, inventory builder, generated inventory | Native constructors, readable conditional controls and parameter help |
| `examples/explicit_warmup.py` | Caller-parameterized Python and native recipe factories; preparation-only helper |
| Documentation and two new test modules | Equations, coverage matrix, deterministic fixtures, UI/export and preparation checks |

No source tables, production recipes, fitted models or saved studies were edited.
No model fitting, tuning, betting comparison, outcome lookup, publishing,
commit or push was performed. Glicko numerical code was not changed.

## Commands and results

PowerShell environment:

```powershell
$env:PYTHONPATH='.;src;tests/analytics'
$py='C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe'
& $py -m xdiyo_analytics.ui.build_inventory
& $py -m pytest tests/analytics/test_explicit_warmup.py tests/analytics/test_ui_explicit_warmup.py tests/analytics/test_warmup.py tests/analytics/test_transition_context.py tests/analytics/test_rating_transitions.py tests/analytics/test_ratings.py tests/analytics/test_features.py tests/analytics/test_league.py tests/analytics/test_keyed_features.py tests/analytics/test_datasets.py tests/analytics/test_spatial_fixture_features.py tests/analytics/test_heatmaps.py tests/analytics/test_ui_inventory.py -k 'not ui_recipe_autoloads_points_prepares_grids_and_renders_reports and not iterative_inventory_factory_keeps_preprocessing_and_restart_seed and not local_badges_merge_into_source_names_and_offline_report' -q -p no:cacheprovider
```

**403 passed, 4 deselected, 18.03 seconds.** The deselected cases run estimators
(two are parameterizations of the same badge/report case). The included cases
exercise legacy warm-up, Glicko/rating transitions, ordinary histories, pooled
League/LOO behavior, keyed assembly, spatial presentation and generated inventory.
The full repository suite was not run.

After the final UI label refinement:

```powershell
& $py -m xdiyo_analytics.ui.build_inventory
& $py -m pytest tests/analytics/test_ui_explicit_warmup.py -q -p no:cacheprovider
```

**2 passed, 6.83 seconds.** Includes actual browser interactions and preparation
of the two modes from temporary synthetic exports. No production data is loaded.

## Independent numerical acceptance

The new numerical module has **64 passing parameterized cases**. It checks final
boundary values, delayed availability, recursive updates/missing skips, pure
replacement, top/bottom/all/tied cohorts, eligibility before ranking, unequal
donor observation counts, missing origins/cohorts, explicit strengths, both ddof
conventions, exact handoff missingness, postponements, stale destination visits,
custom/H2H scopes, scalar arithmetic/covariance, nested lags, arbitrary Stat
identities/fields/periods/perspectives, grid and Gaussian heatmaps under all three
normalizations, regional moments, rotation, movement conflicts/overrides, large
IDs, row shuffling, both layouts, native policy recovery and content identities.

Representative audit: own seed 20, destination donor means 4 and 8, all-donor
cohort -> mean 6; first new observation 10 with alpha .5 -> 8. Separate paired
dispersion fixture: donor means 2 and 8, variances 1 and 9 -> variance 5, not 14.
These expectations are hand-calculated; tests do not use the production seed
helper to calculate expected answers.

## Rendered UI QA

**Flow:** local builder → Features & ratings → switch Uniform to With league
prior → choose all donors and corrected variance → supply strength and matching
ddof → save/reopen → export Python/notebook.

**Environment:** ephemeral `http://127.0.0.1:<port>/` synthetic no-job server;
Playwright Chromium; 1440×1000 and 480×850 viewports. Browser plugin not available,
so the existing pytest Playwright fixture was used. No new dependencies installed.

| Check | Result |
|---|---|
| Page identity/title | Pass: fixture URL and “Football experiment builder” |
| Meaningful nonblank content | Pass: navigation, feature tree and forms visible |
| Framework error overlay | None in inspected screenshots |
| Console/page errors | None; fixture asserts clean error collection |
| Conditional controls | Cohorts hidden in Uniform, shown in league-prior mode; strength exposed for corrected variance |
| Real state changes | Saved recipe contains selected mode, bottom=-1, estimator, strength and matching ddof |
| Reopen/export | Exact native recipe round-trip; exported Python/notebook source compiles without execution |
| Responsive check | Mode/cohort controls remain visible and readable at both widths |
| Job isolation | No run/prepare/predict requests in browser fixture; separate synthetic preparation test only |

Screenshots are retained outside the repository in the user's Temp directory:
`xdiyo-explicit-warmup-desktop.png` and `xdiyo-explicit-warmup-mobile.png`.

## Limits

Direct warm wrapping of native EMA and Lag is intentionally rejected; their
ordinary and nested-child behavior remains unchanged. Direct League/LOO
population-to-team seed mapping is rejected with an explicit explanation.
New corrected variance supports ddof=1, population variance ddof=0, and rejects
other ddof values. Distribution-mixture fade and James–Stein are outside this
implementation. Production parameters remain caller choices.

Browser verification covered Chromium at two widths, not every browser or
arbitrary imported extension. No large production performance benchmark or
model-quality study was run. See [complete semantics and coverage matrix](explicit_warmup.md).

## Ayre review remediation (2026-10-03)

Review baseline: `4fba7684f68e5072b86f8904416d887e74a50c11`. All five
reported findings were reproduced and corrected in the explicit statistical
path; the shared legacy/Glicko implementation was left unchanged.

| Finding | Correction and regression |
|---|---|
| R1: entrant evidence without current fixture rows | Use the complete supplied movement lookup for donor exclusion, evidence retention and hashing |
| R2: newer origin seasons entering seeds | Restrict boundary windows to the identified predecessor and provably older seasons in that competition |
| R3: undated targets | Keep their outputs missing even with finite cutoffs; ignore them in dated-kickoff duplicate checks |
| R4: independent nullable flags | Preserve known negatives and explicit nulls independently; distinguish omitted flags from explicit missing evidence; retain validation of contradictions |
| R5: alpha=1 cancellation | Assign the latest finite value and its moment state directly, including very large previous means |

Permanent coverage: [20 review regressions](../../tests/analytics/test_warmup_review_regressions.py).
The 64 existing numerical cases also pass (**84 combined**).

The four unmodified external suites (`math-audit/test_independent_math.py`,
`api-audit/test_independent_api.py`, `api-audit/test_movement_counterexamples.py`,
`causal-audit/test_independent_boundaries.py`) now give **43 passed in 2.71 s**.
The initial run reproduced ten failures; an additional temp-directory permission
error was resolved by supplying a fresh `--basetemp`, without editing the tests.

The broad command above, with `test_warmup_review_regressions.py` added, now
gives **423 passed, 4 deselected in 20.03 s**. The same four estimator-running
cases remain excluded. This includes the rendered Chromium save/reopen/export
checks at both viewport sizes. No production model was fitted.

For these reruns, use the environment above and a fresh writable test directory:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$testTemp = Join-Path $env:TEMP ('xdiyo-warmstart-' + [guid]::NewGuid().ToString('N'))
# Add --basetemp $testTemp to the pytest command.
```

Ayre's unchanged `causal-audit/compare_legacy.py` was rerun with its output in
a fresh temporary file. Its serialized results match
`baseline-legacy-values.json` **exactly**: ordinary mean/SD/Z-score, lag, EMA,
three legacy warm-up configurations, league/LOO and both Glicko variants, for
retained and moving teams. The supplied review files were not modified.

Documentation now states predecessor ordering, undated-target handling,
full entrant evidence, nullable-flag semantics and exact alpha=1 replacement.
No source datasets, production recipes, saved models or studies were changed
by this remediation. Unrelated checkout changes were preserved.
