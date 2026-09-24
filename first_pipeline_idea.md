# First pipeline idea

**Purpose:** use one concrete experiment as the first test of the reusable football analytics library. Build the system and initial feature catalog now; the real-data adapter and experiment run can follow when the data is ready. The pipeline is an example configuration, not the library's fixed architecture.

**Confirmed pilot:** predict total match corners with **Lasso regression**, using historical corners/shots and match-result Glicko. **No seasonal warm-start/blending in this experiment.** This replaces the earlier open LightGBM/XGBoost pilot choice; those models remain later alternatives. This is a planning document, not an implemented or executed pipeline.

## Reusable preprocessing

Derive promotion/relegation flags once per input/evidence version and save a reusable team-season artifact. Experiment runs load those flags instead of recomputing them. They describe entry into the current season; unknown evidence remains unknown. Rebuild when source membership, evidence or derivation rules change.

The ID-based derivation/join core exists in [team_seasons.py](C:/Users/luisi/Documents/Programming/Python/xDiyo/utils/team_seasons.py). Its current builder reads legacy CSVs, so a new-export adapter and availability policy are still needed. Inspect a representative prepared export before implementing the loader.

## Experiment sequence

| Stage | Intended work/output |
| --- | --- |
| 1. Load | Historical matches, required statistics and saved season-entry context; retain IDs, season/round, source versions and timing. Resolve and verify versions internally from user-selected seasons/tables; save/reuse season-selection records automatically when a record path is supplied. Keep shots separate from other model inputs. |
| 2. Build histories and ratings | One chronological history per team; update both teams' win/draw/loss Glicko from the same preceding states when a completed result becomes available. |
| 3. Construct features | Build the lag/rolling features below and attach each team's eligible rating state to its upcoming home/away role. |
| 4. Prepare targets/options and folds | Keep total-corner targets, configurable bet-option definitions and realized option outcomes separate from predictors. Select temporal folds and the prediction cutoffs. |
| 5. Pre-training report | Fold periods/counts, feature definitions, missingness, sample feature plots and preparation/selection settings. Lasso's fitted selection is not yet available. |
| 6. Fit Lasso | Within each training fold, fit the missing-input treatment and scaling, then Lasso. Fix regularization in configuration or tune within training-only temporal inner folds. These details remain proposals to specify. |
| 7. Post-training report | Per-fold nonzero coefficients, signs/magnitudes and selection stability; held-out predictions versus labels, distributions, paired errors and baseline comparisons. |
| 8. Evaluate a strategy | Apply a configured decision rule to held-out predictions/options; report bets placed, wins, losses, pushes/voids, abstentions and win rate with its denominator. Profitability requires valid odds, stakes and costs. |

## Feature recipe

Prepare reusable definitions for **both upcoming teams**: corners earned, corners conceded, shots for and shots against. Each experiment selects its own subset; all lags, means and deviations need not appear together. This supports one or two initial comparisons without rebuilding the feature library.

| Family | Requested contents |
| --- | --- |
| Individual lags | A separate column for each of the previous five eligible matches: lags 1, 2, 3, 4 and 5. Four series × five lags × two teams = **40 lag columns**. |
| Rolling summaries | Rolling mean, standard deviation and Z-score for all four series. Using the same five-match horizon is the working interpretation; exact endpoints remain to be specified. |
| Glicko | All match-result rating state: provisionally `mu`, `phi`, `sigma` for each team, based on the existing rating record. Additional derived outputs are not implicitly selected. |
| Earlier season context | Preserve the previously discussed normalized standings and promotion/relegation context; normalization, predictor inclusion and any rating-adjustment role need explicit definitions. |

**Model-independent reporting:** model adapters supply predictions and declared optional capabilities. Separate importance providers return named scores with method, scale, fold and population metadata. Coefficients, native tree importance, permutation, leave-one-covariate-out and Shapley explanations are distinct methods, added as models require them. Gain is a tree-specific measure; leave-one-covariate-out requires explicit refits. Renderers consume explanation results without fitting models or inventing an unsupported importance score.

Use available eligible samples for partial rolling windows. Missing lag positions stay missing; do not pad with invented games. No prior/EMA/seasonal blend is applied. Glicko still requires an initial state. Actual earlier-season observations remain usable under the selected history scope.

## Bet options and remaining definitions

Example configurable options: **under 2.5, under 7.5, over 7.5, over 12.5 and over 16** total corners. Define the option before prediction and its observed outcome after the match. An integer line requires an explicit equality/push rule. Lasso continues to predict the count; option outcomes must not become inputs to that prediction.

- Specify the shots statistic, historical venue/competition scope, extra-time coverage, and whether the separately mentioned plain "mean" differs from the rolling mean.
- Define the Z-score's known historical input and common mean/std baseline, missing/zero-spread handling, and Glicko initialization/time accounting.
- Select competition/seasons, one initial CV configuration, baseline, evaluation metrics and regularization. Both previously agreed prediction-timing modes remain available for comparison.
- Clarify the optional "quantile calibration curve": quantile-binned point-prediction diagnostics or an empirical quantile comparison are possible; predictive-quantile coverage requires quantile outputs. Ordinary [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) supplies point predictions, not a probability distribution or [conditional quantile forecasts](https://scikit-learn.org/stable/auto_examples/linear_model/plot_quantile_regression.html).
- Define the strategy, including eligibility, ties and abstention. "Riskiest" is an example, not yet a rule; risk cannot be ranked from a predicted corner count alone. Five-match means repeat information in the five lag columns, so interpret sparse coefficients and selection stability with that redundancy in mind.

The [working notes](C:/Users/luisi/Documents/Programming/Python/xDiyo/football_analytics_working_notes.md) retain the broader library design and later extensions, including CPCV.
