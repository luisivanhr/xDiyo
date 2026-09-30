# Reproducing notebook 20 in the experiment builder

**All items from the original UI gap audit are now implemented.** Use the
[importable current-notebook recipe](feature_inventory/current_notebook_ui_recipe.json)
and [concise settings guide](feature_inventory/current_notebook_ui_settings.md).
The public preset, derived/context/identity controls, class-weight parameter grid,
exact-count adapter, diagnostics and artifact export are ordinary builder options.
Importing the supplied recipe is a convenient starting point; its settings remain editable.

Implementation status: **28 September 2026**. The reference is the current saved
configuration in [`20_xgboost_classifier_total_corners.ipynb`](../../notebooks/20_xgboost_classifier_total_corners.ipynb),
its [`classifier_corners.py`](../../notebooks/helpers/classifier_corners.py) runner
and shared [`quantile_corners.py`](../../notebooks/helpers/quantile_corners.py)
preparation. The original audit identified the missing pieces; Section 10 records
where each is now implemented. No production experiment was retrained for this work.

The [verified hierarchical inventory](feature_inventory/README.md) separates the
latest completed run (**2,145 selected from 2,681**, Spearman 80%, run `ca713acf-d775-4d77-8a17-87afc61d8a39`)
from today's configuration (**all 2,681**, selection disabled). Their prepared
expression bank and ordered columns match; their selector and search grid differ.

## 1. Current configuration versus retained results

The notebook currently requests five seasons, expanded LOO/H2H features and additional warmed feature/rating columns. Its saved outputs still include an earlier **601-expression / 1,589-column** preparation. Those counts are not confirmation of the current expanded configuration. The current feature count is determined by preparation and its feature manifest; it should not be copied from the retained output.

The retained run `e5d7db17-146f-4152-937b-b846f161f187` selected:

- `max_depth=2`, `min_child_weight=60`, `n_estimators=100`, `balance_power=0.0`.
- It belongs to `total-corners-xgboost-classifier-all-features--dc32754e` under `experiments/total_corners_xgboost_classifier`.
- This is the winner of that saved run, **not a replacement for the current 54-candidate grid** and not evidence that the newly added warm feature bank has been evaluated.

The notebook has league indicators. **Neither its current configuration nor classifier helper adds team identity dummy columns.** Team/opponent history grouping, H2H and display names are different from team dummy predictors.

## 2. Data

In **Data → Source data**, use:

| Control | Value |
|---|---|
| Data root | `C:\Users\luisi\Documents\Programming\Python\xDiyo\data\xDiyo_data` |
| Seasons | `20_21`, `21_22`, `22_23`, `23_24`, `24_25` |
| Leagues | Disabled / `None`: all available leagues |
| Tables | `matches`, `statistics`, `pregame` |
| Include awarded | `False` |
| Team-history options | Existing defaults |

The helper discovers eligible statistic identities from **20_21 through 23_24 only**, and reserves 24_25 for evaluation. It then loads the selected statistics plus the `standings` bundle. In Features, enable Feature bank preset and select FeatureBankPreset. Set Preset discovery seasons explicitly to 20_21 through 23_24. Choosing `all_stats` alone does not create temporal expressions; the preset does.

## 3. Target and row layout

In **Target → Named labels**, add `total_corners`:

- Component: `MatchTotal`.
- Source: `Stat(period="ALL", group="Match overview", key="cornerKicks", field="value")`.
- Select `total_corners` as the training target.
- Dataset layout: `match` (one row per fixture).
- Drop missing targets: `True`.

The classifier treats each observed integer total as its own class. It predicts the most probable total, without rounding a continuous regression prediction. Its class encoder is fitted separately on each training population; held-out totals absent from training remain actual observations and have no supported class probability. The shared XGBoost adapter already supplies this fit-local encoding and decodes predictions/probability columns back to original totals.

## 4. Features

### Broad statistic bank

The following is the exact shortlist, filtered to identities available in development data:

| Group | Keys |
|---|---|
| Match overview | `cornerKicks`, `ballPossession`, `totalShotsOnGoal`, `expectedGoals`, `bigChanceCreated`, `goalkeeperSaves`, `totalTackle` |
| Shots | `shotsOnGoal`, `shotsOffGoal`, `blockedScoringAttempt`, `totalShotsInsideBox`, `totalShotsOutsideBox` |
| Attack | `touchesInOppBox`, `bigChanceMissed`, `offsides` |
| Passes | `accurateCross`, `finalThirdEntries`, `finalThirdPhaseStatistic` |
| Defending | `ballRecovery`, `interceptionWon`, `totalClearance`, `errorsLeadToShot` |

For every available `ALL` statistic, add both `ForAgainst(side="for")` and `ForAgainst(side="against")`, then each operator below as a separate named feature:

| Operator | Parameters |
|---|---|
| Lag | `periods=1`, `2`, `3` |
| RollingMean | `window=3`, `5`, `10`, `20`; `min_periods=1` |
| RollingStd | `window=5`, `10`; `min_periods=1`, `ddof=1` |
| RollingZScore | `window=5`; `min_periods=1`, `ddof=1`; default historical reference |
| EMA | `span=5`, `10`; `min_periods=1` |

For periods `1ST` and `2ND`, use only `cornerKicks`, `ballPossession`, `totalShotsOnGoal`, `shotsOnGoal`, `shotsOffGoal`, `touchesInOppBox` when available. For each side, add **Lag 1 and RollingMean 5/10 only**. This is not the full ALL operator set repeated for each half.

Enable **Features → Feature bank preset** to generate this bank with the matching availability filter, names and column order. Leave Named features empty when reproducing just the bank; those entries are additional. Individual operators remain available in Named features. Names follow `{period}_{group}_{key}_{side}_{operator}{window}`. Preserve the manifest and column order for exact reproduction: changing order can change seeded XGBoost column subsampling even when the unordered set of features matches.

### League, LOO and H2H

All the following use `Stat("ALL", "Match overview", "cornerKicks")`:

- League population: `League(unit="team", schedule="completed_rounds", window_unit="rounds")`.
- Ordinary league mean and std: windows **3, 5**; std `ddof=1`.
- LOO is **enabled**: wrap that population in `LeaveOneOut(exclude="team_contributions")`; mean/std/z windows **3, 5, 10**. Z-score reference is explicitly `Lag(ForAgainst(corners, side="for"), periods=1)`.
- H2H is **enabled**: `H2H(ForAgainst(corners, side="for"/"against"))`; mean/std/z windows **3, 5, 10**, std/z `ddof=1`.
- Keep the default partial-window behavior (`min_periods=1`). LOO selects the history window before excluding contributions; it does not replenish the window with older observations.

These expression families are in the UI catalog. The preset expands its configured window/reducer lists into this bank; custom named expressions can supplement it.

### Standings, ratings and warm start

- `NormalizedStanding(side="for", missing_value=0)`.
- Baseline `MatchResultGlicko(side="for", fields=["rating", "rd"])`.
- Baseline `StatGlicko(corners, side="for", fields=["rating", "rd"])`.
- Both rating streams retain default `Glicko2` settings and `scope=["competition_id"]`.

**Keep every unwarmed baseline.** The notebook additionally prefixes warmed copies with `warm::`:

1. For every generated RollingMean/Std/Z feature, use `WarmStart` with `SeededEMA(alpha=0.5, handoff=Hard(rounds=1), league_weight=0.5)`. `WARM_STAT_KEYS=None` means all eligible statistic keys, including the half-period, league, LOO and H2H rolling expressions. Lag, ordinary EMA and standings do not get warmed copies. For H2H the constructed form is `H2H(WarmStart(Rolling...(ForAgainst(...)), policy))`.
2. Add independently warmed result/corners rating copies using `GlickoTransition(phi_scale=1.1, movement_phi_scale=1.0, shrinkage=0.5, top=3, bottom=3, rank_by="standings", aggregate="mean")`.

`TEAM_SEASONS=None` and `SEASON_STARTS=None`: no explicit promotion/relegation membership evidence or custom season boundary table is supplied. The policy supports movements when evidence is supplied; these settings do not establish such movements merely from a newly appearing team. Warm start here is historical feature/rating initialization, not XGBoost continuation.

**Warm-rating editor:** WarmStart now accepts MatchResultGlicko/StatGlicko
sources and switches its policy editor to GlickoTransition. Rolling sources use
SeededEMA. The preset's rating_warm_policy configures the additional warmed rating
copies directly, retaining both baseline and warmed columns. Separate Named rating
states plus Rating expressions remain an alternative; they are not required for
this recipe.

### Timing

In **Historical timing and grouping**, leave Hours before kickoff disabled (`None`), result-availability inputs disabled, and standard history grouping unchanged. Features use earlier finished matches under the documented kickoff-time retrospective assumption. No current-match raw statistic is a predictor.

### Features added after assembly

The preset and UI preparation stages also reproduce the following:

- `rest_days`: days since the last eligible finished fixture, separately for each team perspective; missing without eligible history.
- For feature names containing `_mean5`, `_mean10`, `standing`, `glicko` or `rest_days`, add home+away and home−away columns, preserving the original home/away columns.
- For every available ALL statistic/for-against side, add each home's/away's mean3−mean20 trend.
- Calendar `month_sin`, `month_cos` using angle `2π × month / 12`, weekday (Monday=0), and numeric round.
- League indicators fitted on alphabetically ordered **outer-training** league identities, then transformed over all rows. Unknown identities give all-zero indicators. No team indicators.
- Cast assembled numeric predictors to `float32`, replacing infinities with missing values. No all-missing or constant feature deletion is applied.

Keep preset include_rest/include_calendar/include_combinations enabled. In
**Features → Identity indicators**, select source_league with prefix league and
alphabetical ordering. In **Numeric feature conversion**, choose float32 and
infinities-to-missing. For custom expressions, **Derived match features** exposes
prepared Column/Sum/Difference/Ratio/Constant after assembly and **Fixture context**
exposes CalendarFeature. Discover assembled columns before choosing operands.
The preset already supplies the notebook's arithmetic/calendar columns, so their
manual maps remain empty in the supplied recipe.

## 5. Outer evaluation and inner tuning

In **Evaluation → Outer evaluation folds**, choose `TemporalSplit`:

| Parameter | Outer | Inner, under Model selection |
|---|---:|---:|
| train_size | 4 | 1 |
| test_size | 1 | 1 |
| unit | `seasons` | `seasons` |
| window | `expanding` | `expanding` |
| calendar_by | Explicit empty list `[]` | Explicit empty list `[]` |
| block_by | `["source_season"]` | `["source_season"]` |
| gap | 0 | 0 |
| step | Default / None | Default / None |
| gap_time | None | None |
| score_start | 0 | 0 |
| score_rounds | None | None |
| allow_partial_test | False | False |

The explicit empty calendar groups all leagues into one chronology. Leaving it at the season splitter's inferred default would separate competition calendars. `source_season` groups calendar seasons consistently across provider season IDs.

This yields one outer evaluation: train20_21–23_24, test24_25. Inner tuning uses expanding development folds: train20_21/test21_22; train20_21–21_22/test22_23; train20_21–22_23/test23_24. Disable **Repeat selection within each outer fold** (`nested=False`). Keep fold inputs and fold selection unset; only one outer fold exists. Do not use random K-fold for this reproduction.

## 6. Pre-analysis, feature selection and weights

### Exploratory report

Add `CorrelationAnalysis`: `type="overall"`, `partition="train"`, `methods=["spearman"]`, `percentile_ranks=True`, all features/targets. It is descriptive and does not select features.

### Feature selector

**No selector is enabled in the current notebook** (`FEATURE_SELECTION_PROPORTION=None`). Leave `Features from` unset and do not add TopKCorrelationSelector. The model receives **100% of the assembled features**. `FEATURE_SELECTION_CORRELATION="spearman"` only takes effect if selection is enabled. `colsample_bytree=0.65` is tree-level column sampling, not a retained feature subset; displaying top 30 gain features is also not feature selection.

If later matching a notebook run with selection enabled, add a fitted `TopKCorrelationSelector(type="per_fold", partition="train", k=<proportion>, method=<configured correlation>)` and select it in Features from. The existing selector rounds proportion×column count upward, then ranks eligible finite correlations. Non-rankable columns can make the retained count smaller; setting proportion 1 does not necessarily reproduce selector-disabled behavior. Use the fitted-preparation section so each inner fit and final outer-training fit recomputes its own selection.

### Class balancing

Under **Fitted preparation**, add a named `ClassWeightReporter`, e.g. `count_weights`:

- `type="per_fold"`, `partition="train"`, `mode="power"`, target `total_corners`.
- Set power to the candidate's `0.0`, `0.05` or `0.1`.
- Under **Model → Candidate**, set `Weights from=count_weights`.
- Keep native `class_weight` unset, `scale_pos_weight=1` and other native imbalance weighting disabled.

The new public reporter/policy is mathematically equivalent to the notebook helper's fit-local weighting:

\[
\widetilde w_i=\left(\frac{N}{K n_{y_i}}\right)^p,\qquad
w_i=\frac{\widetilde w_i}{N^{-1}\sum_{j=1}^{N}\widetilde w_j}.
\]

Counts and normalization use only actual fitting rows. Power 0 is unweighted. The shared implementation uses stabilized logarithms, so equivalent floating-point results may differ at rounding precision. No external class-weight dictionary or global prevalence table is needed.

**Class-weight power grid:** the parameter picker now includes fitted
ClassWeightReporter parameters. Add the weights reporter's Power values
`0.0,0.05,0.1` beside the estimator parameters. Its serialized path is
`fitted_reporters.weights.params.power` in the supplied recipe; the ordinary grid
control creates that path. The complete search has 54 candidates.

## 7. Model and grid

Select **XGBClassifier**. Remove the builder's default imputer and StandardScaler: the notebook uses **no preprocessing**, retaining native XGBoost missing-value handling. Leave target scaling, native validation, training controls, calibration disabled. Under Custom training adapter choose BoostingAdapter, enable exact_counts, retain the estimator reference and both predict/predict_proba. This checks the notebook's finite nonnegative integer target constraint.

| Fixed model setting | Value |
|---|---:|
| objective | `multi:softprob` |
| eval_metric | `mlogloss` |
| tree_method | `hist` |
| learning_rate | 0.04 |
| subsample | 0.8 |
| colsample_bytree | 0.65 |
| reg_alpha | 0.2 |
| reg_lambda | 10.0 |
| max_bin | 128 |
| random_state | 42 |
| early_stopping_rounds | None |

Class count comes from each training fit; do not supply a class universe discovered from the holdout. Notebook base options are `n_estimators=400`, `max_depth=3`, `min_child_weight=10`, but **every candidate overrides all three** with the grid:

| Tuned parameter | Values |
|---|---|
| max_depth | 2, 3 |
| min_child_weight | 20, 30, 60 |
| n_estimators | 100, 150, 200 |
| class-weight power | 0.0, 0.05, 0.1 (fitted reporter Power control) |

Total **54 candidates**, each assessed on three inner folds. Choose search metrics `mae`, `mse`, `accuracy` and **MetricSelection(metric="mae")**, minimizing MAE. Do not leave the UI's default MSE-only evidence while choosing an MAE decision. Selection uses count-distance MAE; exact-class accuracy is an additional metric. There is no early-stopping subset, calibration subset, calibration policy or optimizer checkpoint in this study.

## 8. Execution, reuse and model saving

- ExecutionPolicy: `n_jobs=1`, `device="cpu"`, `inner_threads=4`; keep other defaults.
- Reuse completed results: `True`.
- Intermediate training checkpoints: disabled.
- Final deployment refit: disabled. The winner is still fitted on the complete outer-training population for evaluation.
- Experiment name: `Total corners XGBoost classifier all features`.
- Artifacts folder: `C:\Users\luisi\Documents\Programming\Python\xDiyo\experiments\total_corners_xgboost_classifier`.

The UI can retain completed search candidates and save a fitted evaluated model through **Results → Save a fitted model**: choose outer fold0, leave Final refit unchecked, and supply the destination folder. The notebook automatically saves `<run>/fitted_model` and tests a load_model
prediction round trip. The supplied UI recipe enables **Execution → Export tables and fitted models** with
report_tables, predictions, save_models and verify_reload all True, so saving
and reloaded-prediction verification are automatic after a completed run. Each
export creates `<run>/exports/<id>` with a manifest mapping all CSV tables and
fold model folders. It intentionally uses generic names rather than the legacy
classifier_summary filenames; the manifest records what each file contains.

Using the same named experiment directory can pool runs in its leaderboard. It does not make a separately constructed UI recipe cache-identical to the notebook's helper/config identity. Do not assume the UI will reuse the notebook's existing model or candidate artifacts simply because visible hyperparameters match.

## 9. Post-training report controls

Keep all evaluated test folds (the one outer holdout), using the score partition:

| Reporter | Settings |
|---|---|
| PerformanceReporter | overall/score; `mae`, `mse`, `accuracy`, plus `Metric("f1_score", parameters={"average":"macro","zero_division":0}, key="macro_f1")` |
| MatchResultReporter | overall/score; **comparison=`numeric`**, tolerance=`2`; league_column=`source_league`, season_column=`source_season`; team catalog from loaded data; Show badges=`False` to match notebook |
| PredictionDistributionReporter | overall/score; mode=`frequency` |
| ExperimentLeaderboardReporter | overall; weights `{"mae":1.0}` |
| CountClassificationReporter | overall/score; pooling=last; tolerance=2; epsilon=1e-12; group_by=[source_league] |
| FeatureImportanceReporter | overall/model; top_k=30; importance_type=gain; include_zeros=True |
| PredictionTimelineReporter | timeline/score; pooling=last; max_points=300 |

Numeric comparison is required for ±2 coloring even though the estimator is a classifier. Categorical comparison tests exact equality and does not use this tolerance. Leave season/league/team/round filters unset initially. Do not add fabricated odds or a bet-performance report.

The notebook originally computed the following outside the common report pipeline; they are now covered by the added public reporters and artifact export:

- Overall/per-league within-two rate, unseen-count fraction, actual-class probabilities, probability-weighted count, clipped log loss (floor\(10^{-12}\)), supported-only log loss, and whether exact log loss is infinite.
- Outer-training majority-count baseline MAE/accuracy.
- Native gain importance with a top 30 plot.
- An observed/predicted-count timeline capped at 300 matches.
- `classifier_summary` CSVs: predictions, probabilities, metrics, per_league, class_balance, feature_importance, feature_manifest, candidate_comparison, selected_features; its JSON summary and notebook plot HTML.

MatchResultReporter colors ±2 errors, ClassWeightReporter exposes fit-local balance, and CountClassificationReporter retains tolerance/support/probability diagnostics plus a baseline derived from each prediction's fitting population. FeatureImportanceReporter and PredictionTimelineReporter provide the gain and capped timeline plots. ArtifactExport retains complete tables and prediction outputs using its manifest; plotting caps do not truncate those tables. Raw weighted probabilities are **not calibrated**; do not relabel their clipped log loss as ordinary finite log loss for unseen classes. Report differences do not by themselves change the trained predictor.

## 10. Original audit checklist: implemented

The original audit identified the following gaps. Every item now has a public
implementation and ordinary UI configuration. These entries record the final
locations, rather than leaving obsolete proposed remedies in the guide.

| Original audit item | Current implementation | Builder location |
|---|---|---|
| Ordered broad feature bank and development-only discovery | `features/presets.py:FeatureBankPreset`; `ui/recipe.py:prepare_recipe` | Features → Feature bank preset / Preset discovery |
| Home/away sums, differences and trends | Preset.postassembly_definitions; public Column/combine_features | Preset combinations or Derived match features |
| Rest days and calendar | `features/contextual.py:RestDays`, CalendarFeature, evaluate_context_features | Preset switches; Named features / Fixture context |
| Training-fitted league identities and frozen future vocabulary | `features/preparation.py:IdentityFeatureSpec`; saved definitions consumed on prediction | Identity indicators; source_league / prefix league |
| Numeric dtype and infinity handling | `features/preparation.py:NumericFeatures` | Numeric feature conversion |
| Joint estimator/class-weight search | `ui/static/grid-fields.js`; dotted fitted-reporter paths in recipe grid | Model selection → Parameter grid → weights Power |
| Direct rating WarmStart | WarmStart source/policy inventory and dependent editor in forms.js | WarmStart with rating source, or preset rating_warm_policy |
| Exact-count target guard | `ui/adapters.py:BoostingAdapter(exact_counts=True)` | Custom training adapter → Exact counts |
| Unsupported counts, clipped/supported log loss, ±2 and training-majority baseline | `reporting/classifier_diagnostics.py:CountClassificationReporter` | Post-training reporters |
| Per-league diagnostic grouping | CountClassificationReporter.group_by | Group by source_league |
| Native gain/top 30 and capped 300 timeline | FeatureImportanceReporter; PredictionTimelineReporter.max_points | Post-training reporters |
| Automatic model saving, reload verification and tabular exports | `experiments/exports.py:ArtifactExport` | Execution → Export tables and fitted models |

Generic exports use `exports/<id>/manifest.json` rather than duplicating legacy
classifier_summary filenames. The manifest maps full report tables, observations,
metadata, prediction/probability outputs, feature manifests, selected columns and
saved/reload-verified evaluated fold models. This is an intentional file-layout
change, not missing information. Native importance includes zero-importance inputs
when requested; top_k limits the display while retaining the complete table.

Verification: current recipe components and all 54 candidates construct. Bounded
synthetic preparation matched notebook helper X, y, metadata and outer positions
exactly for 23 matches × 293 float32 features. The real configured schema's ordered
names and 1,003 expression definitions match the latest completed prepared bank.
No new full 54-candidate production search was performed; this guide does not claim
identical winning models across separate recipe/cache identities or environments.

