# CatBoost models

Install `pip install -e ".[training,catboost]"` in the environment running xDiyo.
CatBoost is optional: other models remain usable without it, and unavailable
CatBoost models are omitted from the live UI catalogue.

The Model & training selector includes **CatBoostRegressor**,
**CatBoostClassifier**, and **CatBoostRanker**. All constructor parameters are
exposed, with typed controls for iterations, depth, learning rate, regularization,
seed, early stopping, categorical feature indices/names, class weights and native
device settings. Disabled optional fields retain native defaults. Native parameter
aliases remain available; do not supply both an alias and its canonical parameter.
Loss strings can include parameters, such as `Quantile:alpha=0.9`.

The UI automatically selects the existing `training.BoostingAdapter`. JSON,
Python and notebook recipes preserve model configuration. Standard model saving
and loading retain the fitted model, preprocessing and original class identities.
Prediction frames retain input row order and target/class column identities.
The shared adapter currently supports one target and one prediction per row;
multi-target or multi-quantile output requires a separately defined output contract.

## Python and recipes

```python
from xdiyo_analytics.training.catboost import CatBoostRegressor
from xdiyo_analytics.ui.adapters import BoostingAdapter

def model_factory():
    return BoostingAdapter(CatBoostRegressor(
        iterations=300, depth=6, learning_rate=0.05,
        random_seed=42, verbose=False,
    ))
```

Use this model block in an existing native recipe:

```json
{
  "component": "catboost.CatBoostRegressor",
  "params": {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.05,
    "random_seed": 42,
    "verbose": false
  }
}
```

For classification, select `catboost.CatBoostClassifier`; the recipe factory
includes class probabilities automatically. Class labels need not be consecutive
integers. `exact_counts` on the adapter validates count labels when desired.
Common fitting-only class weights are supported; combining them with native
`class_weights`, `auto_class_weights` or `scale_pos_weight` is rejected.

The constructors registered under `catboost.*` use small subclasses from
`xdiyo_analytics.training.catboost`. Their only override is sklearn's public
`__sklearn_clone__` hook: CatBoost's copied mutable parameters otherwise fail
sklearn's constructor-identity check. Clones contain configuration only, never
fitted trees. Native training and prediction methods are unchanged. Use these
constructors for sklearn pipelines, search and library compositions.
The native `composition.Estimator` also selects the boosting adapter for these
models, retaining grouped validation for rankers. This does not claim universal
sklearn common-check compliance: native CatBoost raises its own error rather than
`NotFittedError` for prediction before fitting; the shim preserves that behavior.

## Validation, ranking and categorical data

Preprocessing is fitted on fitting rows only. A native `ValidationTail` or an
explicit runner validation split is transformed with that same fitted state and
passed to CatBoost. Configure native `early_stopping_rounds` / `use_best_model`
and `eval_metric` as appropriate. Common iterative `TrainingControl` is not
translated into native boosting semantics. Native training and validation curves
are retained for learning-curve reporting.

The ranker uses `team_match` layout. Each exact composite match identity is one
query group; rows and observation weights are reordered together into contiguous
groups for fitting and validation. Predictions retain the original input order.
Other query groupings and explicit pair/graph arrays need a custom adapter.
Runner-owned `group_id` and `eval_set` cannot be replaced through fit kwargs.
This follows CatBoost's [native ranking input contract](https://catboost.ai/docs/en/concepts/python-reference_catboostranker_fit).

Set `cat_features` to indices or string names **in the transformed input**.
String DataFrame column names are preserved. Structured/non-string columns use
`f0`, `f1`, etc.; indices avoid ambiguity. Disable generic numerical scaling or
one-hot encoding for categorical strings if native categorical handling is wanted.
The adapter does not invent categorical values or impute missing categories;
provide an explicit train-only transformation when needed. Numeric NaNs follow
native `nan_mode`. `has_time` preserves native input order but does not substitute
for temporal splits, causal historical features or availability checks.

## Execution and outputs

An explicit `ExecutionPolicy` owns resources: CPU and auto select CPU, while
`cuda` / `cuda:N` configure native `task_type="GPU"` and the selected device.
`inner_threads` maps to `thread_count`, including when it was omitted from the
constructor. Without an execution policy, explicit native settings are retained.
Unsupported GPU settings fail natively; no silent fallback is introduced.
GPU numerical determinism is not claimed.

The adapter defaults `allow_writing_files=False` to prevent concurrent folds
sharing a `catboost_info` directory. Explicit file-output settings are respected;
use isolated output directories per fit. Saved library artifacts use the existing
trusted-artifact Joblib serializer, not those incidental native log files.

Integration tests use small synthetic CPU fits; they demonstrate compatibility,
not predictive improvement on football data. No production experiment is run.
