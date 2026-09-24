# AI for sports betting

## Football analytics

Install the analytics package and its test, reporting, and training dependencies with `pip install -e ".[test,reporting,training]"`. XGBoost is included for the classifier notebook tests; LightGBM remains optional. The GitHub dataset contains the top-level season Parquet files and their relational `_tables/` exports. The `_collection/` directory stays local.

Use the [experiment builder](docs/analytics/ui.md) to configure the analytics pipeline through a local UI with the same visual style as its reports. [Notebook 17](notebooks/17_experiment_builder.ipynb) launches it or embeds it in a notebook.

```python
from xdiyo_analytics.ui import launch_ui

builder = launch_ui(workspace=".")
# builder.show()  # Optional notebook view.
```

The [implementation progress](IMPLEMENTATION_PROGRESS.md) records available layers and verification boundaries.

For a code-first experiment, use [notebook 18: Lasso for total match corners](notebooks/18_lasso_total_corners_pipeline.ipynb). It configures the loading, features, target, temporal split, model, reports and persistence stages explicitly. Its [companion guide](docs/analytics/lasso_total_corners.md) explains the defaults and optional stages.

For a broad feature experiment with prediction intervals, use [notebook 19: XGBoost quantiles for total corners](notebooks/19_xgboost_quantile_total_corners.ipynb). It trains separate 10th, 50th and 90th percentile models and inspects held-out pinball loss, interval coverage and quantile crossings. The [guide](docs/analytics/xgboost_quantile_total_corners.md) explains its features, GPU configuration and saved results.

For exact-count classification using the same full feature bank, use [notebook 20: XGBoost classifier for total corners](notebooks/20_xgboost_classifier_total_corners.ipynb). It adds a chronological hyperparameter grid, fitting-only class weights and match predictions with a two-corner tolerance. The [guide](docs/analytics/xgboost_classifier_total_corners.md) explains class probabilities, saved results and execution scope.
