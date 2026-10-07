# Quote-contract review repairs

Date: 7 October 2026. Branch: `codex/research-quote-availability`.
Reviewed parent: `b2a99271b98f6471551055ab7fcb87c0b10091b2`.

## Changes

1. **Empty native results.** Initialize missing disclosure columns with an
   index-aligned object Series before masked assignment. No replacement tickets
   or study-side fallback. Native empty ticket/membership schemas, declaration
   attributes, metric labels, auxiliary tables and exports remain available for
   empty input, all rejected fixtures and undersized pools.
2. **Complete timing preflight.** The example checks all candidate decision times
   and both model streams before grouping. Missing or malformed evidence on a
   fixture that would fail OR raises instead of disappearing from the audit.
   Grouping uses parsed UTC times, and valid empty input yields an empty audit
   with the declared schema.
3. **Reserved probability sources.** Apply the existing reserved-outcome-field
   check to `probability_columns` names at construction and before use. Validate
   again at finalization so mutable dictionaries cannot bypass the check. Native
   names such as `won`, `profit`, `payout`, `actual` and `outcome` are rejected
   before their values could be renamed to probability. `quote_status` metadata
   remains allowed. This closes the demonstrated inherited configuration gap;
   it does not authenticate model sources or detect arbitrary concealed aliases.

The Parquet regression compares timestamps at nanosecond resolution and
normalizes pandas 3's string inference for comparison only. PyArrow stores
seconds-resolution timestamps as milliseconds; this is not a changed instant.
All other dtypes, values, missing masks and metadata remain checked. Reloaded
report data itself is untouched by this test normalization.

## Validation

- Normal environment (Python 3.14.0 / pandas 2.3.3): focused review suite
  **106 passed** (4.81s). Full analytics suite: **4,612 passed, 5 skipped,
  4 failed** (332.91s). The four failures were Windows `PermissionError`
  (`WinError 5`) during directory renames in Bayesian checkpoint persistence
  and recipe refresh. All four passed on immediate targeted rerun (3.01s),
  without code changes. The cause of those access errors was not established;
  the initial full run is not represented as a clean pass.
- Python 3.12.6 / **pandas 2.2.3: 106 passed** (5.16s; 115 NumPy/pandas
  timedelta deprecation warnings).
- Python 3.12.6 / **pandas 3.0.6: 106 passed** (6.52s).
- The compatibility runs cover all 72 original quote tests and 34 new review
  cases, including Parquet/report reuse and native catalogue serialization.

Compatibility environments are separate temporary virtual environments under
`%TEMP%/xdiyo-quote-review-matrix`, with NumPy 2.5.3, PyArrow 25.0.1 and sklearn
1.9.1. No project runtime, dependency declaration or permission setting changed.
The compatibility runs do not constitute full-library pandas 3 certification.

```powershell
python -m pytest tests/analytics/test_quote_contract_review.py tests/analytics/test_quote_availability.py -q -p no:cacheprovider --tb=short
python -m pytest tests/analytics -q -p no:cacheprovider --tb=short
```

Both commands use `PYTHONPATH=src;.` and `PYTHONDONTWRITEBYTECODE=1`. Full-suite
log: `%TEMP%/xdiyo-quote-repair-full.log`. The pandas 2.2.3 compatibility log is
`%TEMP%/xdiyo-quote-review-matrix/pandas22-tests.log`; the final pandas 3.0.6
result and targeted integration rerun were captured in the task tool output.

The four rerun cases were:

- `test_bayesian_fold_boundaries.py::test_prediction_uses_declared_time_even_when_fixture_kickoff_is_later`
- `test_bayesian_kickoff_replay.py::test_legacy_checkpoint_can_forecast_but_requires_rebuild_to_assimilate_results`
- `test_recipe_refresh.py::test_materialize_then_refresh_stale_recipe_without_old_dataset`
- `test_ui_bayesian.py::test_native_bayesian_recipe_runs_through_existing_training_and_save`

## Scope and evidence limits

No real-data consensus, cached-model refit, dataset change, merge or push was
performed. The source composition branch remains `f76edff`; main remains
`2999493`. Opening quote times remain unknown. Research assumptions do not prove
historical tradability. Retrospective model timestamps used as simulation-clock
declarations must be documented as such and cannot be presented as observed
historical artifact issuance. Independent review of the repair remains separate
from these implementation checks.
