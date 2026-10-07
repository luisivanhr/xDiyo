# Composition and staking implementation status

Follow-up: [independent review repairs for `25eb79a`](composition_repair_25eb79a.md).
The initial implementation evidence below is retained as a historical checkpoint;
the linked repair report records the subsequent fixes and validation.

Baseline: `29994933902d213bccd13376bc91ab5af70a481b` (7 October 2026).
Branch: `codex/model-composition-stake-policy`.
Worktree: `C:/Users/luisi/.codex/worktrees/composition-stake-policy/xDiyo`.
Primary checkout: `C:/Users/luisi/Documents/Programming/Python/xDiyo`, still on
`main` at the baseline. Its unrelated data/collector changes were not edited,
stashed or moved. Frozen experiments and datasets were not rerun/changed.
No push, PR or merge was performed.

## Local checkpoints

| Commit | Content and verification |
| --- | --- |
| `e8e607e6c44d4f5bb9a4d091f058a44af1648b12` | Typed contracts, decisions, allocation interfaces and legacy stake golden baseline; 5 focused tests |
| `20f0a4eca5fc0f1e2d440b64c9d15722e1309250` | Integrated composition, stacking, residuals, gates, staking, replay, UI, persistence and examples; 186 focused tests |
| `54c190f1190b67b4959cde9c640f475c1d64b3d2` | Corrected process-safe closure and typed class-label UI; 52 process/weighting tests and 12 Node tests |

The final checkpoint contains this report, explicit historical-rate input and
typed allocation controls, plus the additional calibrated-stack restoration,
parallel-composition and isolated-wheel tests.
Its exact hash is returned with the handoff; `git log -1` identifies it locally.

## Acceptance and support matrix

| Milestone / path | Status | Evidence or boundary |
| --- | --- | --- |
| Baseline and typed contracts | Implemented | Schemas, identities, immutable contexts, static graph validation, legacy stake golden tests |
| Fixed ensembles and independent gates | Implemented | Same-schema means, labels/vote fractions, finite PMF mixtures, AND/OR, missing policy and threshold equality |
| Native complete-ticket multi-stream gate | Implemented | AllCombinations mapping, identical terms, independent leg-product assumption, per-model audit |
| Optional staking on singles and all final templates | Implemented | Existing path when absent; replacements after nominal resolution; shared cash; caps and unfunded accounting |
| Closed ledger and chronological replay | Implemented | Asynchronous reserve/release, latest required leg availability, ties, fees, refunds, native tables |
| Chronological regression/classification stack | Implemented | One meta layer, original row keys, grouped fixtures, local transforms/calibration/weights, drop/error warm-up, outer cutoff |
| Frozen children | Implemented | Matching retained provenance and per-prediction cutoff/vintage validation; caller owns evidence authenticity |
| Residual correction | Implemented | Algorithmic training residuals and honest chronological OOF error-meta; regression and binary logistic pseudoresiduals |
| Learned gates/weights/allocation | Implemented interfaces | Explicit utility targets and held-out evaluation; learned prediction combinations are meta models; saved utility artifacts usable by native allocation controls |
| Recipes, exports, restore, UI | Implemented | Native model field, nested child controls, JSON/Python/notebook export, fitted graph, custom serializer forwarding |
| Diagnostics and examples | Implemented | CompositionReporter, gate/allocation audit, bankroll tables, executable synthetic examples |
| Deeper learned graphs / repeated OOF resolution | Unsupported with errors | No cross-fitting beyond one meta layer; repeated='error' only |
| GPU / multiclass residual corrections | Unsupported with errors | CPU; regression or binary logistic correction |
| Parametric parameter averaging / nonbinary Kelly | Unsupported with errors | Convert distributions to events; Kelly requires declared binary payoffs |
| Joint portfolio Kelly, automatic online updates | Not implemented | Independent capped ticket approximation and frozen replay policies are explicit |
| Automatic utility dataset / replay button in builder | Not implemented | Utility labels require a declared prepared training population; replay is a Python API. Static allocation is native in the UI |

## Validation

Python checks used the existing `.misc314` environment with `PYTHONPATH=src;.`
and `PYTHONDONTWRITEBYTECODE=1`.

### Baseline and full regression

```powershell
python -m pytest tests/analytics -q -p no:cacheprovider --tb=short
```

- Baseline `2999493`: **4,349 passed, 5 skipped, 1 failed**, 335.66 seconds.
- Full regression at checkpoint `54c190f`: **4,397 passed, 5 skipped, 1 failed**,
  313.81 seconds. This full run includes 48 new Python cases.
- The unchanged failure is
  `test_packaging.py::test_publishable_wheel_runs_without_editable_install`:
  it asserts only `xdiyo-ui` exists, while the baseline package already also
  declares `xdiyo-report`. The assertion was neither suppressed nor changed.
- Five skips are sklearn/scipy array-API checks requiring `SCIPY_ARRAY_API`.
- Local temporary logs: `xdiyo-composition-baseline.log` and
  `xdiyo-composition-final-verified.log`. An intermediate full run exposed an
  accidental parent-observer closure capture; all seven affected process tests
  passed after correction in the final full rerun.

### Additional verification

After the final historical-rate input and allocation-control changes, all **51
new Python cases plus 13 inventory cases passed** (64 total, 16.76 seconds), and
the selected **12 Node tests passed** again. The full analytics suite was not
repeated after these narrow changes.

```powershell
python -m pytest tests/analytics/test_composition_completion.py -q -p no:cacheprovider
python -m pytest tests/analytics/test_composition_wheel.py -q -p no:cacheprovider
python examples/composition_and_staking.py
node --test tests/analytics/ui_composition.mjs tests/analytics/ui_feature_bundles.mjs tests/analytics/ui_svm.mjs
```

- Completion suite: **12 passed**, including two additional cases added after
  full-suite collection: calibrated-stack save/load with fitting disabled, and
  parallel outer-composition equality.
- Isolated-wheel test: **1 passed**. Built/installed without network, checked both
  existing entry points and new modules, executed all synthetic examples from the
  installed wheel, and verified imports originated in the isolated install.
- Synthetic examples: passed. The hand-checked delayed replay allocates 75 then
  25 of an illustrative 100, finishes with wealth/cash 150 and zero reserve.
- Node composition, bundle and SVM forms: **12 passed**, including numeric versus
  text class identity. No visual browser QA was performed.
- Wider `ui_feature_discovery.mjs`: **11 pre-existing harness failures**,
  reproduced on unchanged primary `2999493` with `--experimental-vm-modules`:
  its mocked forms module lacks the already-used `configureOdds` export. This
  existing test file was not changed.
- No lint/static-type task is configured by `pyproject.toml`. No minimum/latest
  dependency-version matrix or real research experiment was executed.

### Compatibility evidence

No-policy golden tests preserve numeric/Series stakes, template replacement,
MultiBet total division, membership, payouts and legacy thresholds. Existing
recipe preparation/execution/reuse tests pass. Whole graphs restore with child
pipelines and calibrators; restored outputs and decisions match with fitting
disabled. Future-label mutation, delayed releases, fixture partners, frozen
cutoffs, malformed allocations, outcome-independent selection, explicit missing
gates and order-independent caps are covered. Schemas, row identities and timing
remain contracts, not an assertion that arbitrary caller metadata is truthful.

## Changed-file map

- `composition/`: schemas, graph, reducers/transforms, local training,
  ensembles/stacks, residuals, utility targets and graph serializer.
- `evaluation/`: safe contexts/gates, stake policies and sources, final ticket
  allocation, bankroll ledger; narrow opt-in ticket changes.
- `training/runner.py`, `training/persistence.py`: preparation guard and composite
  artifact support.
- `experiments/exports.py`: explicit serializer forwarding on save and verification.
- `reporting/`: composition diagnostics and stake/gate audits with actual-stake math.
- `ui/`: catalog, generated inventory, nested child configuration, typed labels
  and top-level-wrapper action. Packaging includes the new composition module.
- New tests, `examples/composition_and_staking.py`, this report,
  [guide](composition.md) and [changelog](composition_changelog.md).

Migration is optional: existing ordinary models and fixed stakes need no edits.
