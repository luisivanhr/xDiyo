# V59 integration validation

The final regression run passed **121 tests**, including all 25 new round-layer
tests, the native loader, awarded-fixture filtering, supplemental fixtures and
named-dataset compatibility (including installed-wheel imports).

Environment: Python 3.12.14, pandas 3.0.1, PyArrow 25.0.1, pytest 9.1.1.

```powershell
$env:PYTHONPATH = 'src'
# Keep pip's wheel-test scratch files inside the writable workspace.
$env:TEMP = '<workspace>/scratch/pip-temp'
$env:TMP = $env:TEMP
python -m pytest `
  tests/analytics/test_fixture_rounds.py `
  tests/analytics/test_loading.py `
  tests/analytics/test_awarded_loading.py `
  tests/analytics/test_supplemental_fixtures.py `
  tests/analytics/test_named_datasets.py `
  tests/analytics/test_named_dataset_repairs.py `
  tests/analytics/test_named_movements.py -q
```

The first round test run exposed an overly broad test regex that counted an
older identity warning alongside a V56 source-conflict gate. The assertion now
matches the exact V56 identity-hold token and verifies 26 records. No source
gates or selected data changed. The first compatibility run passed 120 tests;
the installed-wheel test encountered a sandbox temporary-directory permission
error. Redirecting pip's temporary directory into this workspace allowed the
full 121-test run to pass.

`integration.json` records the real 146,000-row local dataset materialization,
61,488 checked assignments and exact source/output/release hashes. All original
columns stayed unchanged; all seven selected columns, nullable types and attached
provenance survived Parquet serialization. `verification.json` binds these
receipts to the tested code (UTF-8/LF normalized hashes) and exact release manifest.
The test receipt omits the machine hostname; outcomes and timings are unchanged.
No production experiment
was prepared or run, and no forecast timing was changed.
