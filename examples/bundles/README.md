# Reusable experiment inputs

This directory contains the small inputs needed by the published examples. Its
Git allowlist contains only this README and `corners_lasso.json`. Generated runs,
predictions, checkpoints, fitted models, logs and caches remain under the ignored
root `experiments/` and `experiment/` directories. No previous model or run is
needed to run this example. Private `_collection/` data is not an input.

## Corners Lasso recipe

Run from the repository root after installing the analytics dependencies:

```python
from xdiyo_analytics.ui import read_recipe, prepare_recipe, run_recipe

recipe = read_recipe("examples/bundles/corners_lasso.json")
prepared = prepare_recipe(recipe)  # Load inputs and construct features/splits.
result = run_recipe(recipe, prepared=prepared)  # Fit a new experiment, or reuse it.
```

The recipe reads the published `data/xDiyo_data` manifests and their `matches`,
`statistics` and `pregame` Parquet tables. It uses all available leagues for
2020/21 through 2024/25, trains on the first four seasons and evaluates 2024/25.
Input hashes are checked by the loader. The shared `source_season` calendar pools
the leagues. Earlier held-out results can enter later historical features; the
fitted model remains fixed for the final season. Preparation does not fit a model.
Results are written to `experiments/example_bundles`, which stays local.

The configuration is derived from the local recipe previously referenced by the
UI guide, `experiments/_recipes/Corners_experiment.json`, whose source SHA-256 is
`f39afea22727d81fa21d9e1647bd66a04e51a93cc0a86eb01993a4c6b7db2dc6`.
The 53 feature definitions, target, preprocessing, Lasso settings and reporter
types are retained. The Distribution reporter uses `features: null` to include
all available feature columns; an explicit empty selection is invalid. The fitted
Correlation reporter runs per fold, matching the Selector that consumes its
training-row results. The example has a relative data path and its own output/name,
enables input hash verification, orders the same five seasons chronologically,
and uses one
explicit four-season/one-season holdout with zero gap, disables the optional grid,
and uses one CPU worker. These execution choices make a fresh run explicit. The
original local recipe and all historical results remain unchanged. This bundle
contains configuration only; it makes no forecasting-quality claim.

## Other documented inputs

| Examples | Required inputs and generated outputs |
|---|---|
| Notebooks 01–08 | Published season tables. Selection records and rating-state bundles are created on first execution under `experiment/`. In notebooks with an absolute `root` or `project_root`, set that path to the local checkout. |
| Notebooks 09–16 | Inline synthetic fixtures and the tracked `notebooks/helpers` modules; the examples create their own saved results and fitted models. Notebook 12 also uses the tracked team catalog. |
| Notebook 17 and the UI guide | The installed UI plus this recipe, or a newly configured recipe. |
| Notebooks 18–20 | Published season tables and tracked helper modules. Each notebook prepares data and trains its configured experiment; saved models and prediction summaries are outputs. Their season settings are unchanged. |

The older [NB search recovery](../../docs/analytics/search_recovery/README.md) and
[Lasso calibration study](../../docs/analytics/calibration_study/total_corners_calibration_study_20260919.md)
describe particular historical runs. Reproducing those old numerical results
requires their original local prediction stores, which are outside this bundle.
The quantile guide's historical run links likewise refer to local outputs. None
of those historical artifacts is required to execute the current notebooks from
their source inputs.
