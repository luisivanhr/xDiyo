# Import notebook20's current experiment into the builder

Use **Import recipe** in the experiment builder and select
[current_notebook_ui_recipe.json](current_notebook_ui_recipe.json). This is an
importable configuration whose fields remain editable through the builder.
It reproduces the **current saved notebook settings**, not the latest completed
winner's hyperparameters or its 80% selector.

The local data/artifact paths point to this checkout. Change those paths if
moving the recipe to another machine. Inspect preparation before starting:
the configured search performs **54 candidates × 3 inner folds plus the winner's
outer-training fit**. Importing or preparing does not start training.

## Settings to expect

| Stage | Configuration |
|---|---|
| Data | All leagues; 20_21, 21_22, 22_23, 23_24, 24_25; matches/statistics/pregame; awarded excluded |
| Feature discovery | Explicitly frozen to 20_21–23_24; no holdout coverage discovery |
| Public feature bank | Full notebook bank; ALL/first/second periods; for/against; lags 1/2/3; means 3/5/10/20; ALL std 5/10, Z 5, EMA 5/10; half-period limits unchanged |
| Population features | LOO and H2H enabled; mean/std/Z over 3/5/10; ordinary league mean/std 3/5 |
| Warm features | Additional SeededEMA variants, alpha .5, Hard 1 round, league weight .5; baseline columns retained |
| Ratings | Baseline and transitioned result/corners rating+RD; phi×1.1,shrinkage .5, top/bottom 3, standings rank, mean prior |
| Other predictors | Rest days; original home/away sums/differences/trends; calendar; outer-training league indicators prefixed `league::`; no team dummies |
| Numeric preparation | float32; infinities→missing; no imputer/scaler |
| Target | MatchTotal of ALL/Match overview/cornerKicks; one match row; missing targets dropped |
| Outer split | Four development seasons, 24_25 holdout; pooled leagues; source_season blocks |
| Inner splits | Expanding seasons, train 1/test 1; three folds inside outer training; nested selection disabled |
| Feature selector | Disabled; all 2,681 configured inputs retained |
| Class weighting | Fitted ClassWeightReporter,mode power,mean-one weights; grid powers 0, .05, .1 |
| XGBoost fixed | hist, .04 learning rate, .8 subsample, .65 column subsample, .2 L1, 10 L2, max_bin 128, seed 42, multi:softprob |
| Grid | depth 2/3 × min_child_weight 20/30/60 × trees 100/150/200 × weight power 0/.05/.1 |
| Decision | Lowest inner MAE; retain MAE,MSE,accuracy evidence |
| Controls | No early stopping/native validation/calibration/checkpoint/target scaling/final deployment refit |
| Execution | CPU; one fit job; four estimator threads; reuse enabled |
| Reports | Training Spearman correlations; evaluation errors/classification metrics; numeric match tolerance ±2; label/prediction frequencies; MAE leaderboard; grouped count support/baseline diagnostics; top 30 native gain; prediction timeline capped at 300 |
| Export | Report tables,predictions,evaluated models,and reload verification enabled |

The public exact-count BoostingAdapter checks finite nonnegative integer labels,
fits the class encoder on each training population, and returns original count
labels/probability columns. ClassWeightReporter supplies training-only sample
weights. Its grid path is `fitted_reporters.weights.params.power`.

Additional named feature, derived-feature and context-feature maps are empty:
the preset creates those columns. Adding duplicate manual features would alter
the study or conflict with preset names. Team-season movement and custom boundary
inputs remain unset, matching the notebook.

## Scope of reproduction

The current prepared bank has 2,681 columns and 1,003 historical expressions. Its
ordered schema matches the latest completed run's prepared bank, but that run
selected 2,145 columns using Spearman 80% and used a different 36-candidate grid.
See the [verified inventory](README.md) for that distinction.

The common UI weighting policy implements the same normalized power weights as
the notebook's adapter; stabilization can change floating-point rounding. UI
recipes have their own configuration/cache identities, so existing notebook runs
are not automatically cache-equivalent even if their visible settings match.

Execution → Export tables and fitted models enables ArtifactExport. It creates an `exports/<id>` folder with a manifest mapping
CSV tables and saved models. It does not promise the notebook helper's exact
`classifier_summary` filenames. The recipe also includes the three public diagnostics reporters:

- CountClassificationReporter: overall/score, pooling last, tolerance 2, epsilon 1e-12,
  grouped by source_league. It includes per-fitting-population majority baselines
  when that provenance is available, rather than estimating a baseline from test labels.
- FeatureImportanceReporter: overall/model, gain, top_k 30, include_zeros=True.
- PredictionTimelineReporter: timeline/score, pooling last, max_points 300.

Plot limits apply to presentation; complete underlying tables remain exportable.
Raw weighted probabilities remain uncalibrated. Unsupported-class diagnostics
retain their explicit clipping/support definitions.

## Verified without training

- Every configured component and all 54 candidates construct successfully.
- A bounded three-season fixture produced **23 labelled matches × 293 features**.
  UI and notebook helper preparation matched exactly for predictor values/order/
  dtype, labels, metadata and outer train/score positions, including warm, LOO,
  H2H, identities and calendar features. Every predictor was float32.
- This check did not execute the 54-candidate grid or refit any production model.

After importing, use the builder's preparation inspection to confirm the real
data's schema and folds, then start the run when ready.


