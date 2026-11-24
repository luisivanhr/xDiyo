# Explicit fitting-row missingness policy

Missing predictor values are kept by default. Existing recipes omit this policy
and keep their current populations and imputation behavior. HistoryInspector's
coverage/diagnostic features do not implicitly enable this policy.

Opt in via Candidate `missing_fit_rows=MissingFitRows(required_features=(...))`,
TrainingRunner, `fit_predict`, or `refit_model`. In the builder, use Model →
Candidate, feature selection and training controls → Missing fit rows and choose
explicit required **constructed model input columns**. Recipe JSON:

```json
{"candidate":{"missing_fit_rows":{"component":"training.MissingFitRows","params":{"required_features":["home::corners_mean","away::corners_mean"]}}}}
```

## Ordering and populations

1. Construct history and features on the full eligible historical input. Never
   remove raw history observations using this fitting policy.
2. Split the development population; reserve calibration and monitoring validation
   using the original population, exactly as without this policy.
3. Inspect the required columns' missingness (`pandas.isna`) on remaining fitting
   rows. Exclude a whole match when either team-row has a missing required value.
   Missing unrelated columns do not cause removal. Infinity is not missing.
4. Fit selectors/weights and estimator preprocessing (including imputers) only
   on eligible fitting rows. Fit the estimator on those rows. Explicit required
   columns must survive feature selection or the fit fails rather than silently
   changing eligibility.
5. Transform/predict **all** original validation, calibration and test rows.
   Test/score membership, identities and target values remain unchanged, so paired
   candidate comparisons share the evaluation universe. Any adapter unable to
   process remaining evaluation missingness must use appropriate preprocessing.

An empty required list, unknown/unselected required column, or zero remaining
fitting rows is an error. This is not a generic `dropna` option.

## Audit and reuse

`FoldResult.fit_positions` records actual fitting rows, while `train_positions`
retains the development universe. `training_summary['missing_fit_rows']` records
input/retained counts, excluded original row positions, missing counts by required
feature and the paired-match exclusion reason. These counts are not disjoint
per-feature exclusion counts. Nested selection remaps positions to original rows.
Refit and candidate/parallel execution use the same central policy. Native resume
receives the same filtered context; its existing context/recipe fingerprints
prevent resuming a different fit population. Recipe roundtrips preserve explicit
configuration and disabled defaults.

## Explicit limitations

Estimator-owned internal validation/early stopping is rejected when enabled
(including sklearn `early_stopping='auto'` and `n_iter_no_change`), because splitting
inside the estimator after filtering would change its monitoring universe. Disable
it and use an adapter with externally reserved validation. Custom adapters must
honor the FitContext contract: validation is monitoring only, never optimization.
Chronological composition with internal preparation is rejected; a top-level
filter cannot guarantee its inner monitoring populations. Do not claim that
hidden splits in arbitrary custom adapters can be inspected automatically.

Keeping missing rows and dropping them estimate models on different training
populations. Fixed evaluation membership makes comparisons meaningful but does
not remove selection bias or guarantee better predictions.
