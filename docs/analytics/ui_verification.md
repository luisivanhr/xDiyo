# Experiment builder verification

Targeted Python verification completed on 2026-09-19. This page records bounded synthetic evidence, not a claim that every visual interaction or optional model backend has been validated.

## Synthetic coverage

`tests/analytics/test_ui_workflow.py` uses three generated season publications with completed and upcoming fixtures, exact unsigned event IDs above the JavaScript safe-integer range, explicit team names, and corners statistics. It covers:

- Constructor parameter exposure and JSON/default decoding.
- Numeric rolling-window controls and recursive immutable AST parameters.
- Recipe file round trips and syntactically valid Python/notebook exports.
- Preparation, fixed fitting, report generation, completed-result reuse, explicit model saving, and selected-round prediction with missing future labels.
- Holdout and nested grid selection, retaining original outer evaluation row positions.
- Human-readable selection parameters with internal IDs/hashes kept out of the comparison display.
- Named rating streams, warm-start feature composition, and cutoff offsets.
- Fitted top-k selection inside parallel fold execution.
- Season-statistic inspection and asynchronous preparation, including exact-recipe prepared-object reuse and changed-recipe preparation invalidation.
- Fresh estimator state for deferred restart factories.
- Seeded iterative restarts, validation loss history, and explicit deployment refitting after nested selection.
- Tiny CPU LightGBM/XGBoost regression, binary classification, and team-match ranking with training-only preprocessing and transformed validation inputs.
- Native XGBoost group weights, original Outcome class identities, and saved/reloaded classifier predictions and probability columns.

`tests/analytics/test_ui_server.py` covers token/origin checks, local asset serving and package-data declarations, recipe save/open/export and path resolution, discovery with underscored league names, and exact presentation of large IDs and missing/nonfinite values.

## Recorded results

- After the final display-name repair: **11 passed, 18 deselected in 8.65 seconds**, covering targeted grid/nested/presentation cases plus the existing experiment reporting/search files. Candidate names remain distinct, and an added assertion verifies that the selected `alpha=` appears exactly once in the final holdout run name. The temporary duplicate-candidate regression is resolved.
- Earlier combined run: **74 passed in 19.35 seconds**. This consists of **27 UI recipe/API tests** and **47 existing affected-contract tests**; the final naming-only change was then checked by the targeted run above.
- Both optional native boosting libraries were installed and exercised on CPU. The affected-contract tests cover experiment reporting/search, training diagnostic storage, and model-selection criteria.
- No known failure remains in these targeted suites. Earlier discovered failures in catalog recovery identity, temporal metadata serialization, deferred restart isolation, native validation support, and XGBoost Outcome class mapping were corrected before the recorded successful runs.

Commands, using the project's `C:/Users/luisi/Documents/Programming/Python/.misc314/Scripts/python.exe`:

```text
python -m pytest tests/analytics/test_ui_workflow.py tests/analytics/test_ui_server.py -q --tb=short -p no:cacheprovider
python -m pytest tests/analytics/test_football_experiment_reporting.py tests/analytics/test_football_experiment_search.py tests/analytics/test_training_controls_store.py tests/analytics/test_model_selection_criteria.py -q --tb=short -p no:cacheprovider
python -m pytest tests/analytics/test_ui_workflow.py tests/analytics/test_football_experiment_reporting.py tests/analytics/test_football_experiment_search.py -q --tb=short -p no:cacheprovider -k "grid_search or selection_presentation or nested_search or not test_ui"
```

The browser verification fixture is `.pytest_tmp/ui_browser/recipe.json`, with synthetic publications and isolated experiment artifacts beside it. It is disposable verification data, not a production experiment or packaged example.

## Launcher notebook

`notebooks/17_experiment_builder.ipynb` was created without changing notebooks 01–16. `nbformat.validate` passed and both code cells compiled. Execution counts and outputs remain empty. The launch cell was deliberately **not executed**, as requested, because it starts a persistent local server and opens a browser. To use it, choose the project's Python (`misc314`) kernel, run the import/launch cell, and optionally uncomment `builder.show()` for the embedded view or `builder.close()` to stop it.

No production season was fitted. No package or CUDA installation was performed. Browser visual/interaction verification and optional model-specific GPU/checkpoint tests are outside these Python test results and require their own evidence.
