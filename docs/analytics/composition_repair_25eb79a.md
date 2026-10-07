# Independent branch review repairs

Subsequent findings and test-harness fixes are recorded in the
[verification follow-up for `0d50956`](composition_followup_0d50956.md).
The results below describe this earlier checkpoint.

Review baseline: `25eb79a34a657d813467d881674c7dcfbad29164`.
Branch: `codex/model-composition-stake-policy`.
Date: 7 October 2026.

The eight findings in the supplied branch review have been addressed on the
existing isolated branch. Main, datasets, frozen experiments and unrelated files
were not modified. This report is implementation evidence, not merge approval.

## Findings and repairs

| Finding | Repair | Regression evidence |
| --- | --- | --- |
| G1: retained learned wrappers contaminate OOF | Recursively clone target-transform/device wrappers and their sklearn children. Recreate iterative adapters. Reject retained calibrators for refitting; require an unfitted base with `ModelNode.calibration` or a provenance-checked frozen artifact. Reject other retained native learned state without an unfitted specification. | Later-label perturbations leave warm-start forest OOF and outer predictions invariant; serial and process folds; save/load with fitting disabled; retained calibrators rejected for both perturbations, including nested device wrapper |
| G2: outer/restored feature timing | A label-free prediction validator checks nonmissing kickoff/issue, issue no later than kickoff, and declared feature availability no later than issue on composition and residual inference. | Stack and honest residual: future/missing feature time, absent timing column, issue after kickoff; original and restored inference; valid label-free prediction |
| S1: leg provenance lost in maxima | Validate each model/leg before multiplying probabilities or aggregating timestamps. Supplied probabilities require nonmissing provenance, strict training-before-issue and vintage no later than issue/decision. | Both models, both legs, reversed rows, missing cutoff/issue, future vintage and incompatible cutoff/issue |
| S2: exposure caps bypassed | Preserve source league/season fallbacks. Reject missing/empty configured exposure identities, including open exposures. | Zero league cap, league/round cap with outstanding positions, scalar and nested missing identities |
| S3: OR abstention ignored | Retain incomplete model valuations as missing, let the declared gate decide; supplied invalid probabilities still raise. | AND/OR with absent column, one missing leg, all missing legs, error/reject policy and invalid finite/infinite values |
| S4: future quote | Compare quote time to each candidate's decision, in addition to context availability. | Quote after candidate decision but before context is rejected |
| G3: residual mismatch | Reject target/kind/link/unit incompatibilities before fitting, and validate reconstructed output against the base schema. | Wrong target, nonidentity link, probability units on regression, nonregression kind, reconstruction overflow; existing regression/binary restore tests |
| G4: invalid distribution edge | Validate categorical PMF family, parameter meaning, units, link, support/columns and finite normalized mass at every boundary, including restored schemas. Unsupported families error explicitly. | Negative, nonfinite, unnormalized, reordered/wrong support; invalid family/parameters/units/link; valid mass; legacy restored schema revalidation |

## Additional contract repairs

- Composition prediction contexts exclude library-named outcomes and training-label
  fields from metadata and named metadata dictionaries, including `settlement::*`,
  `y`, `training_label`, `gross_return` and `net_return_per_unit`. Meta feature
  construction receives a prediction context; labels stay in its fitting context.
  Utility adapter inference uses the same boundary. This is not a sandbox for
  arbitrary custom feature names, callbacks, external data or concealed state.
- Duplicate comparison excludes outcome fields on the opt-in ticket path.
  Conflicting retrospective evidence becomes missing after decision validation;
  it cannot select a leg or change the allocation. The legacy no-policy path is
  unchanged, including its conflicting-evidence error.
- Ledger settlement checks proposed finite accounting state before changing open
  tickets, realized profit or events. The review's overflow case leaves the entire
  ledger unchanged; a subsequent valid settlement still works.

## Validation

Environment: Python 3.14.0, scikit-learn 1.9.0, pandas 2.3.3, NumPy 2.4.6,
joblib 1.5.2, existing Chromium installation. `PYTHONPATH=src;.` and
`PYTHONDONTWRITEBYTECODE=1`. No dependency versions were changed.

| Check | Result |
| --- | --- |
| Full final analytics run | **4,479 passed, 5 skipped, 1 failed**, 55 warnings, 316.98 seconds |
| Final focused graph/staking/composition/browser checks | **111 passed**, 6 warnings, 12.74 seconds; included in the subsequent full run |
| New repair acceptance cases | **79 passed** across graph, staking and native browser test files; included in the full run |
| Node composition, bundles and SVM controls | **12 passed** |
| Wider Node discovery harness | **11 failures**, the previously documented missing `configureOdds` mock export; no repair changes to the harness or application JS |
| Installed wheel | `test_composition_wheel.py` passed in the full run, including execution of the examples from the isolated installed package |
| Bounded smoke after final full run | All eight synthetic model examples, independent ticket gate, allocation and delayed bankroll replay passed; ending illustrative wealth/cash 150 and reserve zero |
| Patch whitespace check | `git diff --check` passed |

Full command:

```powershell
python -m pytest tests/analytics -q -p no:cacheprovider --tb=short
```

The remaining Python failure is the unchanged
`test_packaging.py::test_publishable_wheel_runs_without_editable_install` assertion
that the wheel exposes only `xdiyo-ui`. The package already exposed `xdiyo-report`
on main and the reviewed branch. The dedicated isolated-wheel check verifies both
existing commands. Five skips are array-API checks requiring `SCIPY_ARRAY_API`.
The baseline failure was not suppressed or changed; this is not a clean all-pass
suite claim. The discovery harness failures were already reproduced on unchanged
main during initial implementation; the repair rerun confirms the same error.

Temporary local logs: `xdiyo-composition-repair-final.log` and
`xdiyo-composition-repair-discovery.log` under the user's temporary directory.
The earlier repair full run passed 4,476 tests before the final metadata-boundary
tightening and three additional tests; the final count above supersedes it.

Tests use synthetic fixtures and temporary artifacts. No research experiment was
run. Actual Chromium checks prohibit preparation, fitting and prediction jobs;
they edit and round-trip timing settings, OR abstention and exposure caps through
save, reopen and Python/notebook export. A dependency-version matrix was not run.

## Changed-file summary

- `composition/training.py`: wrapper reset and inference metadata/timing boundary.
- `composition/adapter.py`, `residual.py`, `targets.py`: inference boundary use,
  residual compatibility/reconstruction checks, label-free meta feature context.
- `composition/contracts.py`: complete PMF declaration and per-edge validation.
- `evaluation/ticket_allocation.py`: leg-level provenance, missing valuations,
  supported exposure aliases.
- `evaluation/decision_layer.py`: candidate quote timing and invalid probability
  rejection even when other model evidence is missing.
- `evaluation/stake_policy.py`: fail-closed exposure identities.
- `evaluation/all_combinations.py`, `tickets.py`: opt-in outcome-free duplicate
  validation with unchanged legacy behavior.
- `evaluation/bankroll.py`: atomic settlement preflight.
- `test_composition_review_graph.py`, `test_composition_review_staking.py`:
  independent review reproductions and expanded adversarial acceptance tests.
- `test_ui_composition_review.py`: actual Chromium import/edit/save/open and
  Python/notebook export checks for timing, model gates and exposure caps.
- Guide, changelog and this repair report.

The existing limitations in [the guide](composition.md) still apply: one learned
meta layer, CPU support, regression/binary residuals, explicit utility datasets,
finite PMFs, binary per-ticket Kelly approximation, trusted model artifacts and
Python-only chronological bankroll replay. No architecture or dataset migration
was introduced.
