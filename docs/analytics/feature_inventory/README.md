# Classifier feature inventory

Verified 28 September 2026; no model was trained to create this inventory.

Reproduce the current configuration with the [importable UI recipe](current_notebook_ui_recipe.json)
and [settings guide](current_notebook_ui_settings.md). The recipe uses the current
54-candidate grid with selection disabled, not the completed winner's configuration.

## Latest completed run: what the fitted winner actually used

Run `ca713acf-d775-4d77-8a17-87afc61d8a39`, created `2026-09-23T15:18:46.538219+00:00`. This is newer than the old
1,589-column result still displayed in notebook20. Source: `experiments\total_corners_xgboost_classifier\total-corners-xgboost-classifier-all-features--dc32754e\runs\ca713acf-d775-4d77-8a17-87afc61d8a39`.
The fitted fold's feature_columns, selected_features.csv and full feature manifest
agree: **2,145 selected from 2,681 prepared columns**, with
TopK **Spearman**, proportion **0.8**
(ceil(0.8 × 2,681)=2,145). The completed search had 36 candidates.
Winner: `XGB count 28: max_depth=5, min_child_weight=20, n_estimators=500, balance_power=1.0 · model=XGBClassifier · balance_power=1.0 · target=exact total corners · validation=None`.

| Family of columns | Actually selected | Prepared |
|---|---:|---:|
| direct | 1,603 | 2,016 |
| sum | 265 | 280 |
| difference | 232 | 280 |
| trend | 31 | 88 |
| calendar | 1 | 4 |
| league | 13 | 13 |

This tree counts the actual selected columns. A group can include different
subsets for home/away or for/against; listed dimensions are observed members,
not a claim that every Cartesian combination survived selection. The grouped
JSON retains exact counts by computation/window/period. Ordered-name hashes
allow comparison with the underlying saved schema without dumping thousands of rows.

<details>
<summary>Expand the selected feature hierarchy: computation → window → period → statistics</summary>

- **direct: 1,603 / 2,016 available columns**
  - baseline / statistic: 941
    - z, window 5, ALL: 32 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`
    - std, window 5, ALL: 75 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - lag, window 3, ALL: 76 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - std, window 10, ALL: 75 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 20, ALL: 76 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 10, ALL: 77 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - ema, window 10, ALL: 76 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 5, ALL: 71 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - lag, window 1, ALL: 64 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - ema, window 5, ALL: 76 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 3, ALL: 69 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/touchesInOppBox`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - lag, window 2, ALL: 68 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 10, 2ND: 18 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 10, 1ST: 18 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 5, 2ND: 18 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 5, 1ST: 18 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - lag, window 1, 1ST: 18 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - lag, window 1, 2ND: 16 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
  - warmed / statistic: 545
    - z, window 5, ALL: 29 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`
    - std, window 5, ALL: 77 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 20, ALL: 74 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 10, ALL: 76 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - std, window 10, ALL: 77 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 5, ALL: 70 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 10, 2ND: 18 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 10, 1ST: 18 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 3, ALL: 70 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 5, 1ST: 18 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 5, 2ND: 18 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
  - warmed / loo: 18
    - mean, window 10, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 5, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 3, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 10, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 5, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 3, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - z, window 5, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - z, window 10, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - z, window 3, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
  - baseline / loo: 18
    - mean, window 10, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 5, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 3, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 10, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 5, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 3, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - z, window 5, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - z, window 10, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - z, window 3, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
  - warmed / league: 8
    - mean, window 5, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 3, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 5, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 3, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
  - baseline / league: 8
    - mean, window 5, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 3, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 5, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 3, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
  - baseline / H2H: 26
    - mean, window 10, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 5, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 3, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 10, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 5, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 3, ALL: 2 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - z, window 10, ALL: 2 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - z, window 5, ALL: 2 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
  - warmed / H2H: 22
    - mean, window 10, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 5, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 3, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 10, ALL: 2 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 5, ALL: 2 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 3, ALL: 2 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - z, window 10, ALL: 2 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - z, window 5, ALL: 2 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
  - baseline / rating: 7
    - corners_glicko: 4 columns; venues away/home; fields rating/rd
    - result_glicko: 3 columns; venues away/home; fields rating/rd
  - warmed / rating: 7
    - corners_glicko: 4 columns; venues away/home; fields rating/rd
    - result_glicko: 3 columns; venues away/home; fields rating/rd
  - baseline / standings: 1
    - normalized_position: 1 columns; venues home
  - baseline / context: 2
    - rest_days: 2 columns; venues away/home
- **difference: 232 / 280 available columns**
  - baseline / statistic: 106
    - mean, window 5, ALL: 36 columns; sides against/for
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 10, ALL: 35 columns; sides against/for
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 10, 2ND: 9 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 5, 1ST: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 10, 1ST: 9 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 5, 2ND: 7 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
  - warmed / statistic: 107
    - mean, window 10, ALL: 35 columns; sides against/for
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 5, ALL: 36 columns; sides against/for
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 10, 2ND: 9 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 5, 1ST: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 10, 1ST: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 5, 2ND: 7 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
  - baseline / rating: 3
    - result_glicko: 1 columns; fields rating
    - corners_glicko: 2 columns; fields rating/rd
  - warmed / rating: 3
    - result_glicko: 1 columns; fields rating
    - corners_glicko: 2 columns; fields rating/rd
  - baseline / standings: 1
    - normalized_position: 1 columns
  - warmed / loo: 2
    - mean, window 10, ALL: 1 columns
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 5, ALL: 1 columns
      - Stats/identities: `Match overview/cornerKicks`
  - baseline / loo: 2
    - mean, window 10, ALL: 1 columns
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 5, ALL: 1 columns
      - Stats/identities: `Match overview/cornerKicks`
  - baseline / H2H: 4
    - mean, window 10, ALL: 2 columns; sides against/for
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 5, ALL: 2 columns; sides against/for
      - Stats/identities: `Match overview/cornerKicks`
  - warmed / H2H: 4
    - mean, window 10, ALL: 2 columns; sides against/for
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 5, ALL: 2 columns; sides against/for
      - Stats/identities: `Match overview/cornerKicks`
- **sum: 265 / 280 available columns**
  - baseline / statistic: 121
    - mean, window 5, ALL: 40 columns; sides against/for
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 10, ALL: 41 columns; sides against/for
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 10, 2ND: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 10, 1ST: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 5, 1ST: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 5, 2ND: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
  - warmed / loo: 2
    - mean, window 10, ALL: 1 columns
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 5, ALL: 1 columns
      - Stats/identities: `Match overview/cornerKicks`
  - baseline / loo: 2
    - mean, window 10, ALL: 1 columns
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 5, ALL: 1 columns
      - Stats/identities: `Match overview/cornerKicks`
  - warmed / league: 1
    - mean, window 5, ALL: 1 columns
      - Stats/identities: `Match overview/cornerKicks`
  - baseline / league: 1
    - mean, window 5, ALL: 1 columns
      - Stats/identities: `Match overview/cornerKicks`
  - warmed / statistic: 121
    - mean, window 10, ALL: 41 columns; sides against/for
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 5, ALL: 40 columns; sides against/for
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 10, 2ND: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 10, 1ST: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 5, 1ST: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 5, 2ND: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
  - warmed / H2H: 4
    - mean, window 10, ALL: 2 columns; sides against/for
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 5, ALL: 2 columns; sides against/for
      - Stats/identities: `Match overview/cornerKicks`
  - baseline / H2H: 4
    - mean, window 5, ALL: 2 columns; sides against/for
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 10, ALL: 2 columns; sides against/for
      - Stats/identities: `Match overview/cornerKicks`
  - baseline / rating: 4
    - result_glicko: 2 columns; fields rating/rd
    - corners_glicko: 2 columns; fields rating/rd
  - warmed / rating: 4
    - result_glicko: 2 columns; fields rating/rd
    - corners_glicko: 2 columns; fields rating/rd
  - baseline / context: 1
    - rest_days: 1 columns
- **trend: 31 / 88 available columns**
  - baseline / statistic: 31
    - mean_difference, window 3-20, ALL: 31 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
- **league: 13 / 13 available columns**
  - baseline / identity: 13
    - one_hot_league: 13 columns
      - Stats/identities: `Bundesliga`, `Bundesliga_2`, `Championship`, `Eredivisie`, `La_Liga`, `La_Liga_2`, `Ligue_1`, `Ligue_2`, `Premier_League`, `Premiership`, `Pro_League`, `Serie_A`, `Serie_B`
- **calendar: 1 / 4 available columns**
  - baseline / context: 1
    - weekday: 1 columns

</details>

## Current saved notebook configuration: available inputs if run now

The current source requests **no feature selector** (`None`); all **2,681**
assembled columns would be offered to the model. Its grid contains 54 candidates.
This is a configured schema, not a newly evaluated result. Export availability
was read only from 20_21–23_24; 24_25 remains held out. There are
**1,003 historical expressions**, plus rest/calendar/identity and
post-assembly arithmetic columns. Baseline and warmed variants coexist.

Current expression definitions equal the latest completed run's prepared bank:
**True**. Ordered column names also match:
**True**. The differing
selector/grid therefore matters even when the feature bank matches.

<details>
<summary>Expand the complete current configured feature hierarchy</summary>

- **direct: 2,016 columns**
  - baseline / statistic: 1,176
    - lag, window 1, ALL: 88 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - lag, window 2, ALL: 88 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - lag, window 3, ALL: 88 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 3, ALL: 88 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 5, ALL: 88 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 10, ALL: 88 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 20, ALL: 88 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - std, window 5, ALL: 88 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - std, window 10, ALL: 88 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - z, window 5, ALL: 88 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - ema, window 5, ALL: 88 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - ema, window 10, ALL: 88 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - lag, window 1, 1ST: 20 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 5, 1ST: 20 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 10, 1ST: 20 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - lag, window 1, 2ND: 20 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 5, 2ND: 20 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 10, 2ND: 20 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
  - baseline / H2H: 36
    - mean, window 3, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 3, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - z, window 3, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 5, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 5, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - z, window 5, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 10, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 10, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - z, window 10, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
  - baseline / league: 8
    - mean, window 3, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 3, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 5, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 5, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
  - baseline / loo: 18
    - mean, window 3, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 3, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - z, window 3, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 5, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 5, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - z, window 5, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 10, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 10, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - z, window 10, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
  - baseline / standings: 2
    - normalized_position: 2 columns; venues away/home
  - baseline / rating: 8
    - result_glicko: 4 columns; venues away/home; fields rating/rd
    - corners_glicko: 4 columns; venues away/home; fields rating/rd
  - warmed / statistic: 696
    - mean, window 3, ALL: 88 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 5, ALL: 88 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 10, ALL: 88 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 20, ALL: 88 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - std, window 5, ALL: 88 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - std, window 10, ALL: 88 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - z, window 5, ALL: 88 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 5, 1ST: 20 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 10, 1ST: 20 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 5, 2ND: 20 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 10, 2ND: 20 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
  - warmed / H2H: 36
    - mean, window 3, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 3, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - z, window 3, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 5, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 5, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - z, window 5, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 10, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 10, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - z, window 10, ALL: 4 columns; sides against/for; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
  - warmed / league: 8
    - mean, window 3, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 3, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 5, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 5, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
  - warmed / loo: 18
    - mean, window 3, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 3, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - z, window 3, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 5, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 5, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - z, window 5, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 10, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - std, window 10, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
    - z, window 10, ALL: 2 columns; venues away/home
      - Stats/identities: `Match overview/cornerKicks`
  - warmed / rating: 8
    - result_glicko: 4 columns; venues away/home; fields rating/rd
    - corners_glicko: 4 columns; venues away/home; fields rating/rd
  - baseline / context: 2
    - rest_days: 2 columns; venues away/home
- **sum: 280 columns**
  - baseline / statistic: 128
    - mean, window 5, ALL: 44 columns; sides against/for
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 10, ALL: 44 columns; sides against/for
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 5, 1ST: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 10, 1ST: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 5, 2ND: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 10, 2ND: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
  - baseline / H2H: 4
    - mean, window 5, ALL: 2 columns; sides against/for
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 10, ALL: 2 columns; sides against/for
      - Stats/identities: `Match overview/cornerKicks`
  - baseline / league: 1
    - mean, window 5, ALL: 1 columns
      - Stats/identities: `Match overview/cornerKicks`
  - baseline / loo: 2
    - mean, window 5, ALL: 1 columns
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 10, ALL: 1 columns
      - Stats/identities: `Match overview/cornerKicks`
  - baseline / standings: 1
    - normalized_position: 1 columns
  - baseline / rating: 4
    - result_glicko: 2 columns; fields rating/rd
    - corners_glicko: 2 columns; fields rating/rd
  - warmed / statistic: 128
    - mean, window 5, ALL: 44 columns; sides against/for
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 10, ALL: 44 columns; sides against/for
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 5, 1ST: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 10, 1ST: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 5, 2ND: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 10, 2ND: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
  - warmed / H2H: 4
    - mean, window 5, ALL: 2 columns; sides against/for
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 10, ALL: 2 columns; sides against/for
      - Stats/identities: `Match overview/cornerKicks`
  - warmed / league: 1
    - mean, window 5, ALL: 1 columns
      - Stats/identities: `Match overview/cornerKicks`
  - warmed / loo: 2
    - mean, window 5, ALL: 1 columns
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 10, ALL: 1 columns
      - Stats/identities: `Match overview/cornerKicks`
  - warmed / rating: 4
    - result_glicko: 2 columns; fields rating/rd
    - corners_glicko: 2 columns; fields rating/rd
  - baseline / context: 1
    - rest_days: 1 columns
- **difference: 280 columns**
  - baseline / statistic: 128
    - mean, window 5, ALL: 44 columns; sides against/for
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 10, ALL: 44 columns; sides against/for
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 5, 1ST: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 10, 1ST: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 5, 2ND: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 10, 2ND: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
  - baseline / H2H: 4
    - mean, window 5, ALL: 2 columns; sides against/for
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 10, ALL: 2 columns; sides against/for
      - Stats/identities: `Match overview/cornerKicks`
  - baseline / league: 1
    - mean, window 5, ALL: 1 columns
      - Stats/identities: `Match overview/cornerKicks`
  - baseline / loo: 2
    - mean, window 5, ALL: 1 columns
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 10, ALL: 1 columns
      - Stats/identities: `Match overview/cornerKicks`
  - baseline / standings: 1
    - normalized_position: 1 columns
  - baseline / rating: 4
    - result_glicko: 2 columns; fields rating/rd
    - corners_glicko: 2 columns; fields rating/rd
  - warmed / statistic: 128
    - mean, window 5, ALL: 44 columns; sides against/for
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 10, ALL: 44 columns; sides against/for
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
    - mean, window 5, 1ST: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 10, 1ST: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 5, 2ND: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
    - mean, window 10, 2ND: 10 columns; sides against/for
      - Stats/identities: `Match overview/ballPossession`, `Match overview/cornerKicks`, `Match overview/totalShotsOnGoal`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`
  - warmed / H2H: 4
    - mean, window 5, ALL: 2 columns; sides against/for
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 10, ALL: 2 columns; sides against/for
      - Stats/identities: `Match overview/cornerKicks`
  - warmed / league: 1
    - mean, window 5, ALL: 1 columns
      - Stats/identities: `Match overview/cornerKicks`
  - warmed / loo: 2
    - mean, window 5, ALL: 1 columns
      - Stats/identities: `Match overview/cornerKicks`
    - mean, window 10, ALL: 1 columns
      - Stats/identities: `Match overview/cornerKicks`
  - warmed / rating: 4
    - result_glicko: 2 columns; fields rating/rd
    - corners_glicko: 2 columns; fields rating/rd
  - baseline / context: 1
    - rest_days: 1 columns
- **trend: 88 columns**
  - baseline / statistic: 88
    - mean_difference, window 3-20, ALL: 88 columns; sides against/for; venues away/home
      - Stats/identities: `Attack/bigChanceMissed`, `Attack/offsides`, `Attack/touchesInOppBox`, `Defending/ballRecovery`, `Defending/errorsLeadToShot`, `Defending/interceptionWon`, `Defending/totalClearance`, `Match overview/ballPossession`, `Match overview/bigChanceCreated`, `Match overview/cornerKicks`, `Match overview/expectedGoals`, `Match overview/goalkeeperSaves`, `Match overview/totalShotsOnGoal`, `Match overview/totalTackle`, `Passes/accurateCross`, `Passes/finalThirdEntries`, `Passes/finalThirdPhaseStatistic`, `Shots/blockedScoringAttempt`, `Shots/shotsOffGoal`, `Shots/shotsOnGoal`, `Shots/totalShotsInsideBox`, `Shots/totalShotsOutsideBox`
- **calendar: 4 columns**
  - baseline / context: 4
    - month_sin: 1 columns
    - month_cos: 1 columns
    - weekday: 1 columns
    - round: 1 columns
- **league: 13 columns**
  - baseline / identity: 13
    - one_hot_league: 13 columns
      - Stats/identities: `Bundesliga`, `Bundesliga_2`, `Championship`, `Eredivisie`, `La_Liga`, `La_Liga_2`, `Ligue_1`, `Ligue_2`, `Premier_League`, `Premiership`, `Pro_League`, `Serie_A`, `Serie_B`

</details>

## Reading the hierarchy

- `direct` contains home/away versions of team-history expressions. `for` means
  the focal team's statistic, `against` its opponent's; venue is independent.
- `sum` and `difference` combine same-match home/away historical values.
  `trend` is each perspective's mean3 minus mean20.
- League/LOO windows count completed rounds. H2H windows count prior encounters.
- Warmed columns use the configured prior-seeded EMA or rating transition;
  they do not replace their baseline counterparts. No movement evidence table
  is supplied, so promotion/relegation is not inferred from new appearance.
- Calendar has month sine/cosine, weekday and round. Identity columns represent
  leagues only; there are no team dummy variables.
- Feature selection is fitted inside each training fold. This actual-winner
  inventory describes the final outer-training fit; inner selected sets can differ.
- [Completed grouped JSON](completed_run.json), [current grouped JSON](current_configuration.json),
  and [read-only reproducer](build_inventory.py).
