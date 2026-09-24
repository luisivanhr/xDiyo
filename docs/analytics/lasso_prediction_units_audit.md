# Saved Lasso prediction-unit audit

The frozen run `corners_lasso_first_grid_search--fcd9789c/runs/82828cfd-0bb1-44ad-8306-aedd9c26f58d` used an input median imputer and StandardScaler, with **no target transform**. Its Lasso recipe set `fit_intercept=False`; nested selection changed alpha only. Therefore its retained predictions were already in original corner-count units. Applying an inverse input transformation to them would be incorrect.

| Outer fold | Selected alpha | Test rows | Actual mean | Prediction mean |
|---|---:|---:|---:|---:|
| 0 | 0.01 | 4,604 | 9.970026 | 0.274946 |
| 1 | 0.5 | 4,600 | 9.773478 | 0 |

Fold 1 had 112 zero coefficients and a zero intercept, so all 4,600 predictions were exactly zero. Input standardization centers each fitting feature. With the intercept disabled, the fitted prediction has zero mean on those centered fitting inputs. Stronger regularization can set every slope to zero. This explains the observed predictions without a missing inverse target transformation.

The direct synthetic check uses training labels centered at 1,000 and X centered at 100. With alpha 0.1, two test predictions were approximately 20.787868 and 27.717157 when the intercept was disabled, versus 1,020.787868 and 1,027.717157 when enabled. Both matched independently constructed sklearn pipelines and closed-form coefficients. The test truth was 1,021 and 1,028. This deliberately large offset separates baseline omission from scaling mistakes.

`tests/analytics/test_lasso_prediction_units.py` also checks explicit target inversion and data-only recovery. `tests/analytics/test_target_scaling.py` checks the new optional scaler through actual UI factories, nested selection, final refitting, native boosting validation, coefficient inspection and saved-model prediction. Together with `test_ui_scaling.py`, the focused suite passed **16 tests** on 19 September 2026. The historical run, its recipe and its predictions were not modified or retrained.

For new runs, keep the intercept enabled ordinarily, or deliberately configure centered targets using the separate **Scale the target** option. Target scaling defaults to disabled and changes the interpretation of alpha. See [scaling and prediction units](ui_scaling_verification.md).
