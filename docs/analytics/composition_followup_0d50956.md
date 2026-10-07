# Follow-up to independent verification of 0d50956

Date: 7 October 2026. Branch: `codex/model-composition-stake-policy`.
Reviewed parent: `0d50956847acfff114738b3b409fa65b818a512d`.

Ayre independently verified all eight original repairs and reported two adjacent
remaining gaps. Both are fixed here. The two previously known local test-suite
failures are also corrected without suppressing tests or weakening assertions.

## Repairs

### Nonempty exposure identities

`RiskLimits.project` now treats strings satisfying `not value.strip()` as
missing. This check already recurses through tuples and is applied to both
selected-ticket and outstanding exposure identities for each configured cap.
It fails before allocation can return amounts. Numeric zero and nonblank strings
remain valid and are not normalized.

Regression tests cover empty strings, ordinary whitespace, tabs/newlines,
nonbreaking spaces, nested tuples, selected/open exposures, both native source
league directions and league/round exposure caps. Nonblank identity controls
verify that existing exposure is counted without altering identifier values.

### Native retrospective status at prediction time

The shared composition prediction boundary now excludes exact native `status`
and `is_awarded` keys, including named nested metadata dictionaries. Native label
creation uses these fields when applying void/award rules; their retrospective
values must not be offered to a prediction adapter.

The end-to-end regression creates labels using the review's native
`BetOption(Outcome(...))` example, assembles the fixture dataset, and verifies the
boundary in chronological OOF, outer predictions and restored inference for
stacks, honest residuals and ensembles. Restore runs with fitting disabled.
Source metadata remains unchanged for reporting. Explicit caller features in X
remain untouched, and their timing/semantics still require separate auditing.
This is not a sandbox for arbitrary callbacks or concealed outcome fields.

## Existing test-suite failures

- The wheel assertion now expects both existing console commands, `xdiyo-ui`
  and `xdiyo-report`, validates their declared destinations and exercises both
  help paths from the isolated installed wheel. The second command was already
  part of the package; no entry point was added by this repair.
- The discovery test harness now provides the existing `configureOdds` mock and
  links the real `grid-fields.js` module separately. Unknown module imports fail
  explicitly. All eleven original assertions remain in place and now execute.
  No production browser JavaScript needed changing.

## Verification

- Full analytics suite: **4509 passed, 5 skipped, 55 warnings in 314.55s**.
  This includes the installed-wheel and Chromium UI checks. The five skips are
  optional sklearn Array API checks because `SCIPY_ARRAY_API` is unset. Warnings
  include dependency deprecations, the existing nonconvergence test and overflow
  checks; there are no test failures.
- Focused checks: **29 new regression cases and both packaging tests passed**
  (31 total, 11.43s).
- All **23 Node tests passed**, including the eleven formerly failing discovery
  tests.

Full Python command (with `PYTHONPATH=src;.` and
`PYTHONDONTWRITEBYTECODE=1`):

```powershell
python -m pytest tests/analytics -q -p no:cacheprovider --tb=short
```

Local log: `%TEMP%/xdiyo-composition-followup-final.log`. Environment:
Python 3.14.0, sklearn 1.9.0, pandas 2.3.3, NumPy 2.4.6 and joblib 1.5.2.
No optional dependency or runtime version was changed. These results describe
this local environment, not a cross-version compatibility matrix.

## Scope

Production edits are confined to `composition/training.py` and
`evaluation/stake_policy.py`. Other changes are the focused regression file,
two existing test fixtures and documentation. Work remains on the separate
branch. Main, unrelated working files, datasets and frozen experiments are
untouched. No merge, real-data experiment or live betting operation was performed.
