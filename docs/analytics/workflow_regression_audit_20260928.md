# Current experiment workflow compatibility audit — 28 September 2026

## Conclusion

**No workflow-blocking regression was found in the tested current checkout.**
The classifier notebook completed from setup through saved-model reload on a
disposable five-season synthetic dataset, both with its current disabled feature
selector and with optional 80% Spearman selection. **249 focused tests passed.**
No production notebook, user configuration, retained notebook output, library
source, saved experiment, or running user process was changed by this audit.

This verifies executable workflow compatibility and the listed invariants. It
does **not** reproduce real-data accuracy, run the user's full grid, establish
GPU compatibility, or prove historical artifacts are reusable after source changes.

## Verified source and scope

- Git HEAD: `f4502ed8de44637d70bdd5cd0d9a7b85f32eed97`.
- The working checkout, including uncommitted files, was the tested source.
  Exact SHA256 fingerprints for notebooks 19/20, shared notebook helpers and
  all analytics Python modules are retained in
  [the evidence manifest](workflow_regression_audit_20260928_evidence.json).
  These source fingerprints matched before/after both notebook executions.
- Runtime: Python environment `.misc314`; NumPy 2.4.6, pandas 2.3.3,
  XGBoost 3.1.1, scikit-learn 1.9.0, nbformat 5.10.4, nbclient 0.10.2.
- Baseline: current notebook source, its shared helper contracts, existing
  independent numerical tests, and the 27 September composition migration
  contract in `IMPLEMENTATION_PROGRESS.md`. Prior saved prediction outputs
  were preserved, not treated as freshly reproduced results.

## Actual notebook configuration

| Component | Current configuration verified from source |
|---|---|
| Population | All available leagues; `20_21`, `21_22`, `22_23`, `23_24`, `24_25`; awarded matches excluded |
| Outer evaluation | Latest loaded season held out; earlier four seasons used for development |
| Inner evaluation | Three expanding chronological season folds, pooled across leagues by `source_season` |
| Statistics | 22 requested full-match statistics; six eligible half-period keys; only development-discovered fields |
| Team history | Mean windows 3/5/10/20; lags 1/2/3; EMA spans 5/10; standard deviation/Z-score; for/against |
| Population features | League aggregates; LOO/H2H mean/std/Z at 3/5/10; completed-round league windows |
| Warm starts | SeededEMA alpha 0.5, hard handoff after one round, league weight 0.5; separate Glicko transitions; ordinary features retained |
| Other features | Standings, ratings, rest, calendar, paired sums/differences, short-minus-long trends and league indicators |
| Recent composition additions | Existing calculations use public helpers; ratios and team indicators are available library building blocks but are **not automatically added** to this notebook |
| Selection | `FEATURE_SELECTION_PROPORTION=None`; correlation setting `spearman`; all assembled inputs by default |
| Target | Raw nonnegative integer total corners; class IDs encoded from fitting labels and predictions decoded to actual counts |
| Grid | Depth 2/3 × child weight 20/30/60 × trees 100/150/200 × balance power 0/.05/.1 = **54 candidates** |
| Decision | Minimum inner-fold MAE; winner fitted to outer training population |
| Weights | Frequency powers calculated separately within each optimization population; mean-one sample weights |
| Scaling | No input or target scaler configured in this classifier workflow; raw count labels retained |
| Execution | CPU, one fitting job, four estimator threads, reuse enabled; no native early stopping or separate deployment refit |
| Reports | Training correlations; held-out metrics/count distributions; match results with numeric **±2 corner tolerance**; leaderboard and gain plot |

## Executed checks

The disposable fixture extends the established UI publication fixture to the
same five seasons and supplies all requested statistic/period families. It
contains exact unsigned match IDs above 2^63, exact team IDs above 2^53, completed
matches and one upcoming fixture without a target. Only CPU threads and model
grid size were reduced: two candidates, balance powers 0/0.1, three shallow trees.
Feature definitions, windows, periods, LOO/H2H and warm policies remained active.

| Check | Current default selector | Optional 80% Spearman |
|---|---:|---:|
| Labelled matches | 39 | 39 |
| Historical expressions | 1,023 | 1,023 |
| Assembled inputs | 2,741 | 2,741 |
| Outer training / test rows | 32 / 7 | 32 / 7 |
| Inner chronological folds | 3 | 3 |
| Winner retained inputs | 2,741 | 2,193 = ceil(0.8 × 2,741) |
| Notebook Run All duration | 15.59 s | 22.40 s |
| Outcome | Passed | Passed |

Both executions checked exact row identity/alignment, float32 model inputs,
whole-fold chronological separation, decoded count probability columns and
unit probability sums, preserved warm/baseline families, report generation,
numeric tolerance arithmetic, and saved-model prediction equality. Independent
direct pandas calculations matched all generated sums, differences and trends
within float32 rounding tolerance. League-indicator parity and ratio missing/zero
behavior passed their dedicated tests.

Both cached reruns passed with `XGBClassifier.fit` replaced by an assertion that
would fail on any fit call. Retained probability tables matched exactly. These
are reuse checks for compatible results created under the current source.

The focused tests additionally verified training-only correlation selection and
weight frequencies, sparse original-count class encoding, unseen test classes,
early-cutoff and future-outcome isolation, LOO individual-observation spread,
warm handoff/rating transitions, key-preserving assembly, model persistence,
candidate recovery, UI catalog/export round trips, and training-only scaling
with inverse-transformed predictions in workflows that configure scalers.

## Commands and execution evidence

Run from the repository root with
`C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe`:

```powershell
python -m pytest tests/analytics/test_classifier_corners.py tests/analytics/test_classifier_feature_extensions.py tests/analytics/test_quantile_corners_notebook.py tests/analytics/test_feature_composition.py tests/analytics/test_keyed_features.py tests/analytics/test_warmup.py tests/analytics/test_rating_transitions.py tests/analytics/test_selection_candidate_recovery.py tests/analytics/test_ui_workflow.py tests/analytics/test_ui_recipe_cutoffs.py -q -p no:cacheprovider
# 160 passed in 30.12s

python -m pytest tests/analytics/test_target_scaling.py tests/analytics/test_target_transformers.py tests/analytics/test_model_persistence.py tests/analytics/test_ui_inventory.py tests/analytics/test_football_experiment_checkpoints.py tests/analytics/test_football_experiment_fixed.py -q -p no:cacheprovider
# 89 passed in 12.10s

python .pytest_tmp/workflow_regression_audit_20260928/verify.py
python .pytest_tmp/workflow_regression_audit_20260928/verify.py --selected
```

The harness is intentionally isolated to its own disposable directory; use a
fresh copy/directory for another fresh execution. Executed notebook copies and
synthetic run artifacts remain beneath
`.pytest_tmp/workflow_regression_audit_20260928/` and its `selected/` subdirectory.
Both fresh kernels issued the known nonfatal Windows/ZMQ selector-thread warning;
execution and shutdown completed successfully.

## Nonblocking documentation finding and limits

`docs/analytics/xgboost_classifier_total_corners.md:182` still says the notebook
launches a 24-candidate grid, while its current configuration and the same guide's
grid section specify **54**. This is stale verification prose, not a runtime
failure. No corrective edit was made during this review.

The full real-data 54-candidate × three-fold grid, production-data memory/runtime,
live Jupyter frontend appearance, CUDA, parallel worker processes, explicit
promotion/relegation context, and numerical parity against an older full-data
trained model were not exercised by these notebook executions. Default transition
context remains `None`; movement behavior has separate synthetic unit coverage.
Feature discovery and league-indicator vocabulary are derived from the outer
development population, as before; fitted correlation ranks and class weights
are separately recomputed inside every training fold. Historical eligibility
continues to use the existing retrospective finished-match assumption when no
explicit cutoff is supplied.
