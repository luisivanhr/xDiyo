# Bayesian rating maintainer handoff

Prepared 2026-10-03; review revisions verified 2026-10-04. This is an implementation and synthetic-verification
handoff, not a production model evaluation.

## Checkout and review boundary

- Managed worktree: `C:\Users\luisi\.codex\worktrees\new-rating\xDiyo`.
- Exact branch: `codex/new-rating`.
- Fetched base: `origin/main` at
  `75d25578e21ed166ca396367522a931347681ae7`
  (`Add standalone ticket and portfolio sharing reports`).
- The implementation is committed on this branch; the accompanying handoff
  message identifies the commit. Nothing has been merged or pushed.
- Main-checkout collector changes and data deletions were not edited, staged
  or cleaned. Baseline comparison tests read the main code and wrote only to
  unique temporary directories with bytecode and pytest caching disabled.
- Review findings should be addressed on this same branch in new commits.

## Implemented behavior

The bivariate Gamma-mixed Poisson model maintains attack, defensive vulnerability
and league home-advantage Gamma factors. It provides converged mean-field VB,
the epsilon-first one-step approximation, an independent-Poisson control,
discounted dynamics, pooled or independent league calibration, mirrored
promotion/relegation priors and optional fixed-gap state bridges.

Output selection is independent of filtering and persistence. `BayesianRating`
defaults to attack and vulnerability means for both team perspectives; optional
uncertainty, expected-log and shape/rate fields preserve requested order.
`BayesianFixture` supplies optional expected goals and outcome probabilities in
canonical home/away orientation. Generic `Rating(..., fields=None)` still
exports all numeric snapshot fields.

Frozen parameter JSON is distinct from complete state checkpoints. Checkpoints
retain full shape/rate states, shared home advantage, snapshots, processed event
identities, observation frontier and transfer-closure metadata. Results available
at a transfer boundary are assimilated before transfers. Future scheduled
anchors do not advance the durable observation frontier. Resumed new-season
entries for known teams require explicit movement/predecessor declarations.

The normal model adapter fits only exact `FitContext` matches and supplied goal
targets. Predictions may project known season movements from a frozen checkpoint
without consuming evaluation outcomes or changing that checkpoint. Native
serialization retains movement declarations and season-start anchors. Inactive
entry parameters without movement evidence, and univariate dispersion, are held
fixed and identified in calibration reports.

## Compatibility decisions

The existing Glicko pairwise update receives a scalar result and opponent state;
it cannot express the joint goal likelihood and shared home-advantage factor.
The new score-aware producer returns a `BayesianRatingRun` subclass through the
existing rating contract. Glicko's numerical engine and pairwise protocol are
unchanged.

The only additional label is `MatchGoals`, an ordinary `LabelExpr` producing
the two match-layout goal targets through existing `LabelData` and assembly.
The native adapter uses existing `ModelAdapter` and serializer contracts.
The builder registers the new constructors/loaders, selects the native adapter,
and exposes the existing explicit serializer option. No second experiment
pipeline, store, split engine or orchestration layer was introduced.

The maintained inventory includes the relevant new/updated entries. Unrelated
sklearn/XGBoost description and metric discovery drift from regenerating under
the local dependency versions was excluded.

## Implementation map

| Area | Files |
| --- | --- |
| Gamma likelihood, inference, predictions | `src/xdiyo_analytics/ratings/bayesian_core.py` |
| Configuration, output contracts, portable model/run | `src/xdiyo_analytics/ratings/bayesian.py` |
| Chronology, transitions, snapshots, resumable replay | `src/xdiyo_analytics/ratings/bayesian_replay.py` |
| Calibration, normal model adapter, native serializer | `src/xdiyo_analytics/ratings/bayesian_training.py` |
| Feature integration | `src/xdiyo_analytics/features/{ratings,evaluation,__init__}.py` |
| Goal-pair target | `src/xdiyo_analytics/labels/{expressions,creation,__init__}.py` |
| Builder, recipes, controls | `src/xdiyo_analytics/ui/{catalog,recipe,schema,build_inventory}.py`, `inventory.json`, `static/app.js` |
| Optional dependency | `pyproject.toml`, extra `ratings` with SciPy |
| Explanation | [Model equations and limits](bayesian_rating_model.md), [usage and UI](bayesian_rating_usage.md), [design boundary](bayesian_rating_design.md) |

## Initial implementation verification

All checks used the main checkout's existing `.venv` Python with the worktree's
`src` on `PYTHONPATH`. No dependencies were installed and no production data
fit was run.

1. All `tests/analytics/test_bayesian*.py`: **86 passed in 6.69s**.
2. Existing rating, transition, feature, label, dataset, persistence and UI
   tests selected below, including new UI/goal-label tests:
   **448 passed, 9 skipped, 2 deselected in 64.97s**.
3. Both deselected tests were separately reproduced as failures in the
   unchanged main checkout. They fail in the existing
   `training/persistence.py:39` fallback importing
   `joblib.externals.cloudpickle` after serializing a local iterative backend:
   - `test_ui_inventory.py::test_iterative_inventory_factory_keeps_preprocessing_and_restart_seed`
   - `test_ui_workflow.py::test_iterative_recipe_validation_and_restart_history`
4. All nine skips require unavailable `playwright.sync_api`, including the new
   Bayesian browser test. Python-level UI schema, selected-field routing,
   recipe JSON/Python export replay, native training and normal run-store
   checkpoint persistence passed. Browser interaction is not claimed verified.
5. `git diff --check` passed. Existing dependency deprecation/overflow warnings
   in unrelated UI fixtures were observed during the broad run.

The numerical tests include independent ELBO optimization, shared-effect
quadrature, negative-multinomial normalization/moments, Skellam and large-kappa
controls. Temporal tests perturb future targets, test delayed availability,
simultaneous results, transfer boundaries, save/load time barriers and resumed
state equality. Pipeline tests verify exact fit membership, target-only
outcomes, frozen prediction, metadata-only transfers and native artifact reuse.

Reproduction commands, from this worktree:

```powershell
$env:PYTHONPATH = "$PWD\src"
$env:PYTHONDONTWRITEBYTECODE = '1'
$ratingPython = 'C:\Users\luisi\Documents\Programming\Python\xDiyo\.venv\Scripts\python.exe'
$bayesianTests = @(rg --files tests/analytics | Where-Object { $_ -match 'test_bayesian.*\.py$' })
& $ratingPython -m pytest @bayesianTests -q -p no:cacheprovider --basetemp "$env:TEMP\xdiyo-bayes-$([guid]::NewGuid())"

$regressionTests = @(rg --files tests/analytics | Where-Object {
    $_ -match 'test_(ratings|rating_transitions|feature_composition|features|keyed_features|labels|label_settlement|match_goals|datasets|model_persistence|ui.*)\.py$'
})
& $ratingPython -m pytest @regressionTests -q -p no:cacheprovider --basetemp "$env:TEMP\xdiyo-bayes-$([guid]::NewGuid())" --deselect tests/analytics/test_ui_inventory.py::test_iterative_inventory_factory_keeps_preprocessing_and_restart_seed --deselect tests/analytics/test_ui_workflow.py::test_iterative_recipe_validation_and_restart_history
```

## Maintainer review revisions, 2026-10-04

All three requested findings have been addressed in new commits on the same
isolated branch. Nothing has been merged or pushed.

1. **Declared fold timing** (`42caa50`): fitting honors the existing
   `training_boundary` and `fit_at` metadata. Every selected kickoff must be
   strictly before the data boundary, and every selected result must be available
   by it. Invalid populations fail instead of being silently filtered or assigned
   a later cutoff. Native persistence preserves the model activation time;
   prediction validates it and the recorded fit population against supplied
   context. Standalone/refit calls without declared timing retain their behavior.
2. **Saved-run UI**: the Features & ratings page exposes saved rating inputs
   through the existing `feature_options.ratings` mapping and loaders. Name
   discovery and field choices distinguish generated Glicko and saved Bayesian
   sources. Incompatible selected fields remain visible with a warning; duplicate
   saved/generated names fail validation. Recipe saving, Python/notebook export,
   reopening and preparation are exercised in Chromium with training and replay
   patched to fail if called. Saved artifact hashes remain unchanged.
3. **Browser locators**: the original Bayesian control test uses the actual
   lowercase checkbox labels and passes in Chromium.

Revision verification uses the reviewer's existing browser-capable runtime,
`C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe`.
No dependencies were installed and no production model was fitted.

- All Bayesian tests plus model persistence, training runner, training controls,
  temporal split/mixed-gap tests and the maintainer's original boundary
  reproducer: **272 passed in 19.97s**.
- All `tests/analytics/test_ui*.py`: **121 passed in 79.60s**, with no skips or
  deselections. The two initial iterative serializer failures also pass in this
  runtime. Two existing pandas float32 overflow warnings occur in preparation
  parity fixtures.
- JavaScript syntax checks and `git diff --check` passed.

Reproduction commands, from this worktree:

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\tests\analytics"
$env:PYTHONDONTWRITEBYTECODE = '1'
$reviewPython = 'C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe'
$bayesianTests = @(rg --files tests/analytics | Where-Object { $_ -match 'test_bayesian.*\.py$' })
& $reviewPython -m pytest @bayesianTests tests/analytics/test_model_persistence.py tests/analytics/test_training_runner.py tests/analytics/test_training_controls_refit.py tests/analytics/test_temporal_splits.py tests/analytics/test_temporal_mixed_gaps.py "$env:TEMP\test_bayesian_review_boundary.py" -q -p no:cacheprovider --basetemp "$env:TEMP\xdiyo-review-numerics-$([guid]::NewGuid())"
$uiTests = @(rg --files tests/analytics | Where-Object { $_ -match 'test_ui.*\.py$' })
& $reviewPython -m pytest @uiTests -q -p no:cacheprovider --basetemp "$env:TEMP\xdiyo-review-ui-$([guid]::NewGuid())"
```

The boundary reproducer in the first command is the reviewer's local temporary
file; the new committed `test_bayesian_fold_boundaries.py` retains equivalent
coverage and additional gap, malformed-time, persistence and split integration
cases. The existing model context exposes scalar fold timing, not the original
per-row cutoff vector. Exact per-row cutoffs remain available through the rating
feature APIs; the adapter does not infer them from arbitrary feature columns.

## Remaining limits and evaluation work

- The default team clock is per observed appearance, with an optional elapsed-day
  clock; the paper uses rounds. Proper Gamma priors replace its initial
  reference-team constraint. Simultaneous fixtures use a shared joint VB batch.
- Forecasts integrate the match effect conditional on posterior state means;
  full state and hyperparameter uncertainty are not integrated.
- Pooled mode shares the entire parameter bundle, without hierarchical partial
  pooling. It does not infer absolute league strength from disconnected scores.
  Bridge gaps are supplied, not learned; independent bridge calibration is
  currently unsupported.
- Native current scores are not universally regulation-time counts. Regulation
  mode requires explicit caller preparation. Awarded matches are excluded.
- Checkpoint updates need new releases strictly after their frontier. Result
  revisions and equal-time partial append batches need a full rebuild.
  Previous-season observations arriving after a transition/bridge are rejected;
  retrospective seasonal smoothing is not implemented.
- A loaded parameter bundle does not contain trained team states. A static
  saved-run feature query does not automatically ingest new results. Use the
  documented replay/update path and explicit entry metadata.
- The three/five season gates count observed labels only. Completeness,
  consecutiveness, transition sample size and deployment suitability require
  data review and temporal held-out evaluation. These are pilot heuristics,
  not proven minimum sample sizes.
- Production convergence, calibration speed, probability calibration and any
  improvement over Glicko2 remain unmeasured. No betting-profit claim is made.
- The initial main-checkout `.venv` limitations are recorded above for provenance.
  Browser tests now run in the existing reviewer runtime; no global serializer
  or environment repair was included in this model change.
