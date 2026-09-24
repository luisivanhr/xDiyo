# Total-corners prediction spread study

## Finding

The saved run reproducing the reporter screenshot is the Lasso run `bd20f431-2828-4e43-b0b4-026eeda650aa`, created 2026-09-19 10:29 UTC, under `experiments/corners_lasso_first_grid_search--fcd9789c`. Its report contains the exact `post/Residuals` view with `overall`, `partition: score`, `layout: match`, `9,204 rows`, `9,204 matches`, and `prediction pooling: last`. The stored fold artifacts contain two folds with 4,604 and 4,600 eligible match rows.

Predictions are narrowly distributed: observed corners have mean 9.872, SD 3.395, and 5th–95th percentile 5–16; predictions have mean 9.988, SD 0.426, and 5th–95th percentile 9.353–10.723 (range 8.749–11.830). Narrow SD alone does not establish harmful conditional-mean compression. The descriptive regression of observed on predicted has slope 0.980 and intercept 0.089 pooled: the observed change across the prediction range is close to the identity slope. Prediction deciles rise from mean prediction 9.321 / observed 9.373 to 10.793 / 10.622. The evidence supports weak signal and limited ranking, with no pooled evidence that predictions need amplitude stretching. It is not evidence that marginal prediction distribution matching is required.

Ranking is weak but nonzero: Pearson 0.123 and Spearman 0.120. MSE is 11.362, MAE 2.695, RMSE 3.371, against an observed-mean baseline MSE of 11.522 (R² versus that baseline 0.014). The observed-mean baseline is an oracle descriptive calculation on the scored outcomes, not a deployable training baseline or calibration target.

## Fold summaries

| scope | n | observed mean / SD | predicted mean / SD | MSE | MAE | RMSE | R² vs mean | Pearson | Spearman | descriptive slope / intercept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pooled | 9,204 | 9.872 / 3.395 | 9.988 / 0.426 | 11.362 | 2.695 | 3.371 | 0.014 | 0.123 | 0.120 | 0.980 / 0.089 |
| fold 0 | 4,604 | 9.970 / 3.428 | 9.854 / 0.373 | 11.514 | 2.703 | 3.393 | 0.020 | 0.152 | 0.147 | 1.400 / -3.820 |
| fold 1 | 4,600 | 9.773 / 3.358 | 10.123 / 0.433 | 11.209 | 2.686 | 3.348 | 0.006 | 0.129 | 0.127 | 0.999 / -0.345 |

## Prediction deciles

The decile table in [`prediction_deciles.csv`](prediction_deciles.csv) forms bins by sorted prediction and moves a tied prediction value as a whole; no equal predictions were split in this run. Mean observed corners increase from 9.373 in the lowest bin to 10.622 in the highest, but the fitted relationship is shallow relative to observed variability. The displayed IID standard errors are only descriptive; matches can be dependent through teams, leagues, seasons, and repeated temporal histories.

## Configuration and evidence limits

The selected winning trial in both folds is Lasso alpha 0.01 (the recipe-level alpha 0.1 is not the stored winning trial), with `fit_intercept=true`, median imputation followed by `StandardScaler`, a standardized target transform, and 116 retained feature columns per fold. Training summaries report estimator-managed termination with `n_iter=41` and `n_iter=345`; convergence history is unavailable. Train predictions are not retained, so train/test spread comparison cannot be made without refitting. The saved output is in original target units after inverse target transformation. The five identity columns are unique across all 9,204 rows, so the analyzed population has no duplicate match identifiers.

League metadata is present. The same compression appears within leagues: prediction SD ranges roughly 0.22–0.43 while observed means range 9.28–10.98; league-level Pearson correlations are mostly near zero. These are descriptive subgroup checks, not independent validation because all leagues share the same fitted run and temporal structures.

The run’s top-level metric records label pooling as `occurrences`, while the screenshot-matching residual view explicitly records `prediction pooling: last`. This study follows the screenshot-matching residual population (all 9,204 rows) and reports the discrepancy rather than silently conflating the two labels. Fold-specific decile tables and identity plots are included because fold 0 has slope 1.400/intercept -3.820 and fold 1 has slope 0.999/intercept -0.339; these fold differences require inspection in a future experiment.

## Recommended next experiment

Keep the target and temporal split fixed and compare a less heavily shrunk model or a controlled lower-penalty Lasso against this run, recording train and held-out prediction spread, calibration-by-prediction-bin, ranking, and the same mean baseline. Treat any fitted test slope as descriptive only; do not stretch predictions using test labels. Preserve an untouched holdout for the final comparison.

Artifacts: [`metrics.csv`](metrics.csv), [`prediction_deciles.csv`](prediction_deciles.csv), [`prediction_deciles_by_fold.csv`](prediction_deciles_by_fold.csv), pooled and fold-specific [`observed_vs_predicted_identity.png`](observed_vs_predicted_identity.png) plots, [`source_manifest.json`](source_manifest.json), and the reproducible analysis script [`total_corners_calibration_study_20260919.py`](total_corners_calibration_study_20260919.py).
