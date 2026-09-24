# Recovered Negative Binomial search — September 19, 2026

All **90 completed candidates** from run group
`933d5702-69f3-478c-9278-7a8ed7cdb281` were recovered from stored prediction bundles.
No model was fitted or asked to predict during recovery. Original artifacts were
left unchanged. Recomputed MSE agreed with every original overall MSE within
\(10^{-10}\); MAE was calculated on the same retained rows.

## Selected configuration

Both MSE and MAE select trial `3af7fd75-ff68-416c-a198-83167f300b52`:

| Setting | Value |
|---|---|
| Model | Negative Binomial regression |
| Prediction | mode |
| Dispersion | 0.001 |
| Penalty | elasticnet |
| Alpha | 0.01 |
| L1 ratio | 1 (pure L1) |
| Maximum iterations | 500 |
| Input preprocessing | Median imputation, then StandardScaler |
| Target scaling | None |
| Inner CV MSE | 11.472832936563506 |
| Inner CV MAE | 2.6498389581291137 |

These are **inner cross-validation scores for outer fold 0**, on 14,282 scored
observations. They are not final holdout scores. The winning configuration still
needs its outer training fit and test evaluation if those were not completed.
Later outer folds need their own searches; this winner must not be substituted
for independent nested searches across all folds.

The old trial cache used the decision as part of its identity. New compatible
searches can reuse candidates after changing standard metrics or the decision;
these historical artifacts were recovered explicitly and were not rewritten to
pretend they carry the new cache identity. Importing the JSON below does not resume
a pipeline: it is a record of the recovered selection and its model settings.

## Files

- [All 90 candidates](nb_90_candidates.csv), sorted by MSE.
- [Winner settings and source identifiers](nb_recovered_selection.json).
- [Reproducible recovery script](recover_nb_comparison.py). Run from the repository
  environment with `python docs/analytics/search_recovery/recover_nb_comparison.py`.

Existing predictions suffice to repeat the comparison. A data-only prediction
bundle does not restore a fitted estimator or resume a partially completed
optimizer. See [experiment recovery](../football_experiment.md) for those separate
operations.
