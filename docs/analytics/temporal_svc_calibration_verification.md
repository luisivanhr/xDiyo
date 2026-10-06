# Temporal SVC calibration verification

Reviewed base: `fcda79ac6ec28725549b6b75b93e788c94473768`.
Environment: Python 3.14.0, scikit-learn 1.9.0, SciPy 1.17.1, pandas 2.3.3.

## Patch map

| Files | Purpose |
| --- | --- |
| `training/calibration.py` | Backward-compatible response selection, binary guards, calibrated adapter and retained diagnostics. |
| `training/margins.py` | Explicit margin/class schema, single stable Platt-smoothed sigmoid and convergence checks. |
| `training/calibration_split.py` | Whole-match/group chronological tail, availability guards, issue boundary and purge audit. |
| `training/estimators.py` | Binary `decision_function` output with fitted class identities and finite-margin validation. |
| `training/runner.py` | Shared populations for selectors, weights, fitting and calibration; response-dependent fitting. |
| `selection/core.py` | Original-position remapping of nested purge audits and calibration rows. |
| `ui/recipe.py` | Margin response selection, incompatible-setting guards and native exports. Existing factory identity already includes calibration settings. |
| `ui/build_inventory.py`, `ui/inventory.json` | Response, availability, cutoff/grouping and minimum-count controls; clear SVC probability-route help. |
| `tests/analytics/test_temporal_svc_calibration.py`, `ui_svm.mjs` | Numerical, population, recovery, reporter and real form-event regression tests. |
| `docs/analytics/support_vector_models.md`, `temporal_svc_calibration.md` | Route comparison, native configuration, exact objective, output and persistence contracts. |

## Evidence

- Focused Python integration suite: **145 passed**. Includes 39 dedicated temporal
  tests and unchanged probability-calibration/SVC tests, UI workflow/inventory and
  class weighting regressions.
- Form renderer/event tests: **3 passed** using the actual `forms.js` and generated
  inventory (Node harness, not a visual browser inspection).
- Final full analytics suite: **4,349 passed, 5 skipped, 1 failed**, in 359.32 s.
  The sole failure is the unchanged packaging entry-point assertion described
  below. The five skips require `SCIPY_ARRAY_API`; they are not calibration skips.
  The final run held Python source fixed throughout.
- An earlier full run also encountered a Windows checkpoint rename denial and
  a recovery test invalidated by concurrent source edits. Both passed on isolated
  rerun and in the final full suite. The adapter's validation error message was
  adjusted to preserve the existing `prediction_methods` contract.

Commands (project Python, `PYTHONPATH=src;.` and `PYTHONDONTWRITEBYTECODE=1`):

```text
python -m pytest tests/analytics/test_temporal_svc_calibration.py tests/analytics/test_probability_calibration.py tests/analytics/test_ui_svm.py tests/analytics/test_ui_inventory.py tests/analytics/test_ui_workflow.py tests/analytics/test_class_weighting.py -q -p no:cacheprovider
node --test tests/analytics/ui_svm.mjs
python -m pytest tests/analytics -q -p no:cacheprovider --tb=short
```

Key checks:

- Spied on the libsvm fit call: probability estimation is disabled. No probability
  prediction path is used by the margin route.
- Shuffled input, duplicate kickoffs and paired team rows preserve identity-based
  fitting/calibration populations and fitted parameters.
- Calibration feature/label and outer-test feature/label mutations leave earlier
  imputation, scaling, support vectors, coefficients and gamma unchanged.
- Selector and weight computation spies see only earlier fitting rows. Imputer
  medians and scaler means are checked independently; a train-empty feature is
  preserved without learning its calibration values.
- Sigmoid coefficients/probabilities/loss agree with an independent score-equation
  numerical oracle, including nonconsecutive/string/reversed class identities.
- Extreme inference margins remain finite probabilities summing to one. Constant
  margins, unknown classes, insufficient/one-class tails and failed optimization
  are explicit errors.
- Delayed or missing availability purges whole prediction groups. Availability is
  checked at the calibration prediction cutoff, including the one-hour lead.
- Nested search restores original positions in both retained model and report
  audits. Explicit refit requires an issue time and repeats temporal separation.
- JSON, Python and notebook recipes, model save/load and experiment reuse preserve
  outputs. Changing policy configuration invalidates reuse. Adding calibration/bet
  reporters reuses predictions without refitting.
- Old calibrators without `response_method` resolve to the probability-input path.

## Full-suite caveat identified before this patch

The unchanged packaging smoke test asserts that the only console entry point is
`xdiyo-ui`. The base revision's `pyproject.toml` declares **both** `xdiyo-ui` and
`xdiyo-report`. Both facts were verified with `git show HEAD:<file>`; neither file
is changed in this patch. This unrelated assertion is not suppressed.

All fitting was confined to synthetic regression fixtures. No discarded research
comparison or frozen experiment was rerun or modified. No new dependencies were
installed, and no commit or push was performed for this patch.
