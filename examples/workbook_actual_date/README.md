# Workbook-only actual-date study

Read `EXPERIMENT_PROTOCOL.md` for fixed choices and limitations. Run preparation from this checkout, using a new output directory. It verifies and applies the two committed round layers and preserves input files. Required private inputs are the retained workbook Matches_Results and Statistics_FT extracts, the supplied first/second-division inclusion lists and the existing same-workbook odds snapshot.

```powershell
python examples/workbook_actual_date/prepare_data.py --matches '<Matches_Results.parquet>' --statistics '<Statistics_FT.parquet>' --first '<first_division_leagues_coverage.csv>' --second '<second_division_leagues_coverage.csv>' --output '<new-run>'
python examples/workbook_actual_date/prepare_odds.py '<new-run>' '<same-workbook-odds-snapshot>'
```

Copy the current canonical Ayre `REPORTING_STANDARDS.md` into the new run and record its hash. Preparation freezes copies of source and examples. Using the CUDA Python environment, run `run.py smoke` with selection indices 0, 3, 6, 9, 12, 15, 18, 21, 24 (one seed per stage/family). Then run `preflight.py`, followed by `run.py seal` and `run.py run`. Run paths refer to the copies inside the new run. A new seal requires a passing full-population preflight and nine completed real CUDA smoke receipts. Do not edit sealed fitting inputs.

`results/STATUS.json`, per-job `progress.json`, `history.csv`, atomic `continuation.pt` and logs expose actual state. `run.py run` resumes only identical saved job controls/source. Every selection and refit starts fresh; continuation is crash recovery for the same job. A failed child stops the queue. The completion marker requires 27 selections, 108 refits and 108 outer prediction exports. Final analysis runs automatically; native reports, diagnostics, browser QA and Ayre upload are distinct completion work. Smoke scores are not experiment results.

Private data, fitted models, reports and odds remain in the ignored run directory, not in Git. Commit and push reusable source/tests/protocol and verification receipts on a non-main branch. Preserve other worktrees.
