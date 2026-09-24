# Builder controls and persistent inventory

The builder configures the existing analytics pipeline. Python APIs, mathematical definitions, recipe files, and saved experiment artifacts remain the underlying interface. The graphical controls are described by a packaged, reviewable JSON inventory rather than reconstructed from built-in constructor signatures whenever the page opens.

For launching, preparing an experiment, running it, and reopening saved results, see [the builder guide](ui.md). [Scaling verification](ui_scaling_verification.md) explains what is fitted on training rows and how saved predictions retain their units.

## Configuring an experiment

1. **Data:** discover the prepared exports and select leagues and seasons. All leagues are selected initially. Statistic, team, and round choices come from the selected publications. The short season inspection contains only matches, statistics, pregame, and shots, with row counts and descriptions.
2. **Features and ratings:** add named expressions using their own controls. Statistic selection shows available statistics and periods; the internal statistic group is resolved from that selection. Rolling windows, lags, EMA, league populations, leave-one-out, and warm-start policies retain their existing library behavior.
3. **Named rating states:** choose match results or a selected statistic, configure the engine, then refer to that stream from a `Rating` feature. Glicko strength (`rating`), uncertainty (`rd`), and volatility (`sigma`) are separate outputs. Initial values and `tau` are visible engine settings. Optional transition policies preserve the existing rating/feature warm-up distinction.
4. **Labels and layout:** choose the observed target and match or team-match rows. Label construction uses observed outcomes; historical feature eligibility stays in feature construction.
5. **Splits:** choose temporal, whole-match, grouped, or CPCV splitting. Training and evaluation scopes remain explicit. A scoring-round range is an inclusive pair of round numbers; it is separate from the time gap between training and test.

   Temporal **Seasons (pooled leagues)** groups the same `source_season` across
   leagues even when their native season IDs and kickoff dates differ.
   **League-seasons (separate leagues)** preserves the former per-league season
   behavior. For four training seasons and the next test season starting at
   round 11, choose train size 4, test size 1, Seasons, gap 10, and Gap unit Rounds.
   This removes ten observed rounds separately per league without adding them to
   training or extending the season end. A postponed fixture in an excluded round
   remains excluded. Gap unit disabled inherits the window unit; finer Kickoffs
   gaps instead count unique times pooled within the current calendar.
6. **Studies and selection:** add only the requested reporters. Correlation methods are selectable. MCC-specific category and threshold controls appear only when MCC is selected. Metric controls expose options relevant to that metric. Fitted feature selection remains separate from exploratory studies.
7. **Model and execution:** configure the model, ordered preprocessing, optional search, execution policy, and refitting. Preparation exposes the assembled data and fold sizes before fitting. Running uses a snapshot of the submitted recipe.

Controls include parameter-specific help, enumerated choices, numeric inputs, checkboxes, lists, and named entries. Less common controls appear under **Additional settings**. An optional field with a `None` default is enabled explicitly; leaving it disabled omits its override. The meaning of `None` belongs to that parameter. For example, no prediction `as_of` value means no clock filter.

Feature checkboxes require the **assembled** feature column names. In **Pre-training analysis**, click **Discover feature columns** before enabling the reporter's **Features** setting. This runs preparation without training and stays on that page; ordinary **Prepare** and completed run jobs also provide choices. Enable **Features** to select columns, or leave it disabled to include all. The choices survive reporter edits and navigation. Changes to data, statistic selection, features/timing, ratings, labels/target, layout, history, or split inputs invalidate the prepared choices and require another discovery. Results from an older in-flight preparation are not applied to changed inputs.

The same workflow applies to an embedded notebook view. Refresh its page to receive frontend changes; no kernel restart is needed for the discovery button. Backend fixes require reloading the updated Python modules and relaunching the builder. Save the recipe before closing it; merely opening another builder in a kernel with stale imported modules does not update that code.

For **Evaluation → Choose folds**, first use **Discover folds**. It discovers the entire outer plan without fitting and keeps the Evaluation page open. Preparation submits a cloned recipe with `fold_ids=None`; the live empty or existing selection is preserved, and Run still uses that selection. This also works when holdout search is configured before one outer fold has been chosen. Full outer choices are separate from post-report fold choices, so a selected run's locally renumbered preview cannot overwrite the complete outer list. Input changes invalidate both lists.

Post-report choices are mapped prospectively into the selected run's local IDs. Selecting outer `[2, 0]` produces local **Fold 0 (outer 2)** and **Fold 1 (outer 0)**. An outer selection/order or preparation-population change clears an explicit report subset; a stale local ID must not silently refer to another outer fold. Repeating discovery without changing that scope preserves the subset. If a selected run is completed without prior full-plan discovery, its local report choices are usable, but its preview is not treated as the complete outer plan.

Distribution reporters let you choose a binning/smoothing rule or supply a numeric bin count/bandwidth. Learning curves can show every attempt, the retained attempt, or selected attempt numbers. Composite preprocessing can define named pipeline steps and column-specific transformations. For an incremental estimator such as SGD, the iterative adapter's backend factory receives the native estimator and preprocessing separately; the preprocessing pipeline itself does not need to implement `partial_fit`.

Some similarly named settings have different meanings. Glicko's tolerance is a strictly positive numerical solver threshold; match-result tolerance is an allowed prediction error, and coefficient tolerance selects terms whose absolute coefficients exceed the threshold. A leaderboard `run_group` identifies one experiment run and its trials, not a statistical comparison group. Season transition boundaries default to the first prediction cutoff, and promotion/relegation requires movement evidence rather than merely observing a new team.

The API retains advanced configurations beyond the ordinary controls. Imported recipes can contain registered constructors, references, and registered callback/factory specifications. Arbitrary Python source is not evaluated from a recipe. Custom extensions still require registration in the Python process hosting the builder.

## Feature bundles and proportional selection

Under **Named features**, choose **New feature setup → Bundle** to configure one computation and apply it to several checked statistics. For example, configure a rolling mean with a five-match window and select corner kicks and total shots. The computation, period, observation field, for/against setting, and other parameters are shared while building the bundle. The statistic checklist is filtered by the selected period; **All periods** retains separate available period outputs. Statistics with the same key in different groups are distinguished in the choices.

The computation must contain exactly one `Stat` source so that replacement is unambiguous. Operators can be nested around it. Inventory defaults such as a rolling window, EMA span, or lag are materialized in generated expressions and names. Preview names can be edited; generated name collisions receive numeric suffixes and existing features are retained. Press **Add selected features** to create independent ordinary recipe expressions. Later edits to one added feature do not update its siblings, and there is no bundle object required during Python evaluation. Selecting raw `Stat` as a source does not change the rule that current-match observations cannot be used directly as historical predictors.

Top-k selector fields accept either whole-number counts or proportions: enter `10` for ten features or `0.25` for 25% of inputs, rounded upward. `1` and `1.0` mean one feature. The denominator follows explicit input restrictions and earlier selectors, includes undefined inputs, and does not force undefined scores into the selection. In consensus mode `fold_k` controls each fold's nominations and `k` controls the final selection; both accept proportions. See [the selector equations and scope rules](reporting.md#counts-or-proportions).

## Reports and team display

Analysis scope, fold, partition, and layout describe the rows actually analyzed. Post-training reporters consume retained predictions; choosing a different view does not retrain. Training/validation loss histories belong to the model diagnostics view. Their availability does not imply that validation predictions have been retained as an additional prediction partition.

Match reports use names from the loaded matches. Local badge catalogs may supplement those names with optional images; current source names take precedence over old names in a badge catalog. Badge paths inside a catalog resolve relative to that catalog. Available images are embedded in the generated report, so viewing them does not require an external image server. Missing images retain the text fallback.

Older UI recipes that omit the match reporter's badge setting acquire `show_badges=True` and the prepared team catalog when loaded. An explicit `False` or custom catalog is preserved, and loading does not mutate the original recipe object. This is a UI recipe compatibility default, not a change to every direct Python reporter construction.

Reopening a retained report can refresh its badge presentation from the local catalog without fitting or rebuilding predictions. The refresh copies presentation options, preserves the original artifact data, and honors an explicit badge opt-out even when the saved artifact already contains images.

`PostTrainingAnalysis(..., fold_ids=[...])` limits the folds included in reports while retaining every trained fold. Both pooled and per-fold studies use only those selected prediction occurrences. Calling `run(..., fold_ids=[...])` overrides the configured report selection; `None` uses the configured selection, and no configured selection means all fitted folds. Fold IDs must exist and be distinct; prediction studies require at least one. To override a configured selection with all folds, pass their IDs explicitly. This report filter is separate from choosing which experiment folds to train.

Run names show readable model parameters. Internal IDs and hashes remain available in saved evidence for identity and recovery rather than becoming primary leaderboard columns. The same experiment store pools completed final runs across sessions; search trials remain in selection details.

## Updating the inventory

The relevant package files are:

| File | Purpose |
|---|---|
| `ui/catalog.py` | Register callable component implementations. |
| `ui/schema.py` | Describe the pipeline-stage argument sources used during inventory generation. |
| `ui/build_inventory.py` | Assign widget types, discovery sources, help, conditional fields, and domain-specific choices. |
| `ui/inventory.json` | Persist the reviewed presentation contract included in the package. |
| `ui/inventory.py` | Load and cache that contract for normal operation. |

After deliberately adding or changing a built-in component:

```powershell
python -m xdiyo_analytics.ui.build_inventory
```

Run this in the intended project environment. Optional model packages influence which model schemas can be generated. Review the JSON diff, especially argument types, defaults, nullable fields, nested components, valid enum choices, and help. A signature alone cannot determine whether an object argument should be a statistic selector, a table reference, a round range, or a collection of named bets; add the semantic descriptor in the builder.

Commit the implementation and generated inventory together. The server reloads the inventory when its file modification time changes; reopen the browser page to fetch the revised catalog. Restart the builder after changing Python implementation code. Normal known-component schema requests use the saved descriptors; the explicit generation command may inspect constructors and docstrings. Catalog construction still validates arguments and creates actual Python components when executing a recipe.

Run focused checks after an inventory change:

```powershell
python -m pytest tests/analytics/test_ui_inventory.py tests/analytics/test_ui_workflow.py tests/analytics/test_ui_server.py -q
```

These synthetic checks cover descriptor loading without runtime signature inspection, data discovery boundaries, short season inspection, conditional MCC settings, local badges, recipe execution, persistence, and the local server contract. Browser interaction and visual inspection are separate checks. CPU adapter tests do not verify physical CUDA execution, third-party checkpoint recovery, or real-data predictive quality.

### Recorded verification — 2026-09-19

- **163 passed in 16.18 seconds:** UI workflow/server, scaling, reporting viewer, training-control diagnostics, and match-result rendering/core/catalog/probability checks.
- **15 passed, 20 deselected in 3.78 seconds:** all 13 new inventory integration cases plus the existing iterative/factory cases after the native-estimator seed repair. This verifies selected large team IDs, inclusive round scoring, a real rating transition context, multi-column auxiliary indexes, relative/absolute local badges, and iterative SGD with retained preprocessing and validation loss history.
- The native-estimator factory initially failed to propagate restart seeds. The regression now passes with distinct requested seeds, isolated preprocessing instances, and a successful bounded iterative run. No known failure remains in these targeted Python checks.
- Final inventory-only pass after the help audit and regeneration: **13 passed in 3.45 seconds**. The audit corrected season-entry timing, movement/count qualifications, run-group identity, and the distinct meanings of tolerance.
- Badge compatibility and report-fold follow-up: **17 new tests passed in 3.19 seconds**, plus **44 affected existing tests passed in 4.49 seconds**. These cover legacy badge defaults/explicit opt-out, immutable retained-report refresh, selected pooled/per-fold predictions, default and call-level fold overrides, invalid IDs, preserved training folds, numerical metric scope, and completed-result reuse.
- Feature bundles and proportional selection: **86 Python tests passed in 2.70 seconds** across the new boundary/composition and actual Node-to-recipe checks plus affected selector/training suites. A follow-up covering the public helper export and new static module serving passed **30 tests in 4.01 seconds**. **5 Node pure-helper tests passed** for shared parameters, inventory defaults, periods, same-key groups, collisions, cloning, and invalid templates. The executable recipe check independently confirms that the third match's generated rolling features use only the first two completed results. These results do not substitute for browser interaction checks.
- Reporter feature discovery: **30 Python tests passed in 19.20 seconds** across the new preparation/run preview checks and UI workflow/server regression suites. The checks select actual returned columns in match and team-match layouts, prove preparation does not construct a model, and verify run/recovered-run previews. **4 Node lifecycle tests passed** using the actual `app.js` with a minimal DOM/form harness: discovery without a results host, once-per-job publication, navigation while a run completes, stale preparation rejection, and reporter-edit persistence. This harness checks lifecycle logic; browser visual behavior is verified separately.
- Outer-fold discovery follow-up: **6 Node lifecycle tests passed** through the approved Python-to-Node runtime, including empty/prior selections, same-page Evaluation discovery, unfiltered Prepare payloads, preserved Run selections, and protection of full outer choices after a selected run. **4 Python discovery tests passed in 4.57 seconds**, including full-plan discovery with holdout search configured and model construction forbidden.
- Prospective report-fold correction: **8 lifecycle tests passed**, including selection-order mapping, clearing an explicit subset when the meaning of its local IDs changes, same-scope preservation, and selected-run fallback choices. **5 Python discovery tests passed in 5.06 seconds**, including actual outer fold 1 → fitted/report fold 0 for the correct held-out season. Fresh synthetic browser checks also confirmed all three outer choices appear from an enabled empty selection, repeated discovery preserves outer fold 2, and selecting that fold exposes only report Fold 0 (outer 2). Changing to outer fold 1 clears the previous explicit report subset; no console errors were observed.

The follow-up command was:

```powershell
python -m pytest tests/analytics/test_ui_inventory.py tests/analytics/test_ui_workflow.py -q --tb=short -p no:cacheprovider -k "inventory or iterative or factory"
```

Bundle and proportion checks:

```powershell
python -m pytest tests/analytics/test_selector_proportions.py tests/analytics/test_ui_feature_bundles.py tests/analytics/test_reporting_selection.py tests/analytics/test_reporting_selector_contract.py tests/analytics/test_training_selection.py -q --tb=short -p no:cacheprovider
node --test tests/analytics/ui_feature_bundles.mjs
```

Feature-discovery checks:

```powershell
python -m pytest tests/analytics/test_ui_feature_discovery.py tests/analytics/test_ui_workflow.py tests/analytics/test_ui_server.py -q --tb=short -p no:cacheprovider
node --experimental-vm-modules --test tests/analytics/ui_feature_discovery.mjs
```

## Temporal units and mixed-gap verification

The focused temporal, split-core and inventory suite passed **125 tests in
6.16 seconds**. Synthetic checks use two leagues, six shared source seasons,
distinct native season IDs, staggered dates, shuffled rows and both row layouts.
They verify two pooled folds versus four separate-league folds, expanding/sliding
membership, explicit grouping overrides, round-11 starts, excluded postponed
early-round fixtures, independently numbered leagues, pooled kickoff gaps,
cutoff/availability filtering, scoring subsets, invalid gaps, default gap-unit
replay, and the packaged UI choices. No production model was fitted.

```text
python -m pytest tests/analytics/test_temporal_mixed_gaps.py tests/analytics/test_temporal_splits.py tests/analytics/test_split_core.py tests/analytics/test_ui_inventory.py -q --tb=short -p no:cacheprovider
```

Fold-preview compatibility follow-up: **33 Python tests passed in 20.04 seconds**
across feature discovery, workflow and server tests; **8 actual-app Node lifecycle
tests passed**. Assertions check readable train/test seasons, the retained
rounds after a gap, and propagation of descriptions into both outer and local
post-report choices. Browser verification also confirmed pooled season
train 2/test 1/gap 2 rounds produces 16 training and 5 test rows, displays Alpha,
training seasons 22/23 and 23/24, test season 24/25 and test rounds 3–7; switching
to league-seasons remained usable.

Relevant implementation files are `splits/temporal.py`, `ui/catalog.py`,
`ui/build_inventory.py`, `ui/inventory.json`, `ui/server.py` and
`ui/static/app.js` under `src/xdiyo_analytics`. Follow-up tests are
`test_ui_feature_discovery.py` and `ui_feature_discovery.mjs` under
`tests/analytics`. No notebook was changed or executed.

## Distribution selections and observed labels

Distribution **Labels** is an opt-in checklist using discovered label names.
Disabled omits label plots; enabled requires at least one selection. Feature
selection keeps its existing default: disabled uses all inputs. An enabled empty
reporter checklist now produces a named correction before Run and routes to its
analysis page. Prepare and discovery remain available, and the recipe is not
silently changed. Labels and features share analysis rows/settings but keep
separate plots and downloadable tables.

Verification: **84 Python tests passed in 6.63 seconds**, covering label/feature
scope, numerical histogram/KDE values, identical feature/label names, label-only
contexts, nonfinite/constant labels, ordinary feature behavior and early named
errors before preparation. **10 actual-app Node lifecycle tests passed**,
including blocked Run, correction routing, continued Prepare discovery and
disabled-checklist recovery. The saved user recipe was not modified and no real
experiment was fitted. Tests: `test_distribution_labels.py`,
`test_reporting_statistics.py`, `test_ui_feature_discovery.py`, and
`ui_feature_discovery.mjs` in `tests/analytics`.

The follow-up persisted Labels-picker assertion also passed: **16 focused
distribution tests in 2.28 seconds**. Isolated browser QA confirmed that an empty
Distribution Features selection redirects to Pre-training analysis before a job
starts, discovery still returns four actual feature columns, and selecting one
feature plus Labels → corners produces both plots in a completed 23-row synthetic
Ridge report. The Label · corners histogram and KDE rendered in the report iframe;
no console errors occurred. The saved user recipe remained unchanged.

## Fitted analysis report retention

Correlation studies and selectors configured under Fitted feature selection
are now retained in `training.fitted_report` and displayed as `fitted/` studies
in the combined result. Validation/test rows are excluded from those fitting
populations. Final search reports show evaluated winners; nested runs group them
by outer fold, keeping inner-trial panels in trial evidence. New bundles preserve
the reports on recovery; old bundles without that field load with `None` and
cannot recreate unsaved diagnostics automatically.

Verification: **74 targeted and affected tests passed in 18.39 seconds** across
fitted report retention, experiment reporting, selection scopes/nested selection,
training selection and UI workflows. After strengthening inner-trial original-row
assertions, **all 6 focused tests passed in 3.36 seconds**. Tests cover fixed,
holdout and nested candidate paths, validation exclusion, shuffled original
positions, separate retained/consumed selector objects, numerical correlation
agreement, outer-fold presentation, saved-result reuse and legacy missing-field
recovery. Relevant new test file: `tests/analytics/test_fitted_report_retention.py`.
No real experiment was trained or notebook modified by this verification.

## Report viewport follow-up

Embedded reports use a viewport-relative height and an **Expand report** button.
The expanded native dialog receives the same retained HTML in a separate frame,
then removes itself when closed. A focused actual-app DOM test verified that both
frames retain the same sandbox, expansion makes no API calls, close removes the
dialog, and the viewport/flex sizing declarations are present. **All 11 lifecycle
tests passed**. Pixel sizing and browser interaction are covered by separate
browser QA, not by this DOM harness. No training or saved artifacts were modified.

## Execution boundaries

- Fold jobs can run concurrently through the existing execution policy. This does not make every estimator GPU-capable.
- Scaling is fitted on fitting rows. Validation and held-out inputs use that fitted state. Input scaling does not call for inverse-transforming predictions; explicit target transformations do.
- Native LightGBM/XGBoost stopping uses native adapter support. The shared iterative training-control protocol requires an adapter that implements it.
- Completed preparation is reused in the same server only for the exact same resolved recipe. Press **Prepare** again after changing data files.
- Candidate search tunes supported candidate-stage settings. It does not silently rebuild arbitrary data or feature-preparation grids.
- Existing advanced recipes remain valid subject to their component contracts. A graphical choice does not bypass those contracts or supply missing custom callbacks.
