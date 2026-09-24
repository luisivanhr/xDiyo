# Control fitting, inspect models and refit explicitly

Optional training controls split a fold's development rows into fitting and
validation rows. A compatible adapter can then stop early, reduce its learning
rate, restore its best in-memory state, and try fresh seeded restarts. Ordinary
`EstimatorAdapter` fitting still uses the estimator's own training procedure.

This guide uses sixteen synthetic rows. See the [reference](training_controls_reference.md)
for all arguments and helper contracts, and the [equations](training_controls_equations.md)
for exact decisions and denominators. The short [notebook 11](../../notebooks/11_training_controls_quickstart.ipynb)
is a runnable companion using the existing `misc314_py314` kernel.

## 1. Declare data and outer folds

Applications obtain these objects from [dataset assembly](datasets.md) and
[splits](splits.md). The small example declares them directly. Histories, ratings,
availability cutoffs and fold-aware feature construction remain upstream work.

```python
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xdiyo_analytics.datasets import ModelDataset
from xdiyo_analytics.splits import Fold, SplitPlan
from xdiyo_analytics.training import (
    TrainingRunner, EstimatorAdapter, IterativeAdapter, PartialFitBackend,
    TrainingControl, EarlyStopping, ReduceOnPlateau, ValidationTail, refit_model,
)

positions = np.arange(16)
X = pd.DataFrame({'signal': np.linspace(-1, 1, 16), 'cycle': positions % 4})
y = pd.DataFrame({'count': 2 + 1.5 * X.signal})
metadata = pd.DataFrame({
    'event_id': pd.array([2**63 + 101 + int(i) for i in positions], dtype='uint64[pyarrow]'),
    'kickoff_at': pd.date_range('2025-01-01', periods=16, tz='UTC'),
})
dataset = ModelDataset(X, y, metadata, 'match', ('event_id',), ('event_id',),
                       'total', {'kind': 'synthetic training controls demonstration'})
plan = SplitPlan([
    Fold(np.arange(8), np.arange(8, 12), np.array([10, 11])),
    Fold(np.arange(12), np.arange(12, 16), np.array([14, 15])),
], 16, positions)
```

## 2. Choose a compatible model and controls

The backend factory creates a new estimator and preprocessor for each attempt.
`PartialFitBackend` fits preprocessing on fitting rows only, then performs one
`partial_fit` call per step. Validation rows are transformed and monitored without
being passed to that update. Both fitting and validation must contain whole
matches, including both rows in `team_match` layout.

```python
def model_factory():
    return IterativeAdapter(lambda seed: PartialFitBackend(
        SGDRegressor(random_state=seed, shuffle=False, penalty=None,
                     learning_rate='constant', eta0=.02),
        loss='mse', preprocessor=StandardScaler(),
    ))

control = TrainingControl(
    max_steps=12, early_stopping=EarlyStopping(patience=3, min_delta=1e-5),
    scheduler=ReduceOnPlateau(patience=2, factor=.5, min_lr=1e-5),
    restarts=1, seed=7, restore_best=True,
)
runner = TrainingRunner(model_factory, validation=ValidationTail(.25), control=control)
fitted = runner.run(dataset, plan)
fitted.folds[0].training_summary
```

`restarts=1` means two fresh attempts, with seeds 7 and 8. The default monitor is
`validation_loss` when validation exists, otherwise `train_loss`. An attempt's
best exact value and its patience counter are separate: `min_delta` controls what
resets patience; every strict improvement can become the restored state.
Arbitrary backend exceptions propagate. A nonfinite monitored value ends that
attempt and is retained in its history.

When using supervised feature selection, run the selector on
`runner.selection_plan(dataset, plan)`. This derived plan excludes
validation rows from the selector's fitting population. Supply the report through
`analysis_report=selection_report, features_from='selector study name'` in `run`.
The original outer plan
still defines evaluation. See the [existing selection example](training.md).

## 3. Read curves and coefficients without another fit

Model diagnostics use `partition='model'`. Their `overall` view combines separate
fold/attempt traces; it does not average losses or require prediction rows.
Prediction metrics continue to require an explicit `score` or `test` population.

```python
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.reporting import (
    LearningCurveReporter, CoefficientReporter, PerformanceReporter,
    ExperimentLeaderboardReporter,
)

report = PostTrainingAnalysis({
    'Loss by fold': LearningCurveReporter(type='per_fold'),
    'Selected attempts': LearningCurveReporter(type='overall', attempts='selected'),
    'Coefficients': CoefficientReporter(type='per_fold', tolerance=1e-8),
    'Feature survival': CoefficientReporter(type='overall', tolerance=1e-8),
    'Held-out error': PerformanceReporter(type='overall', partition='score', metrics=['mse']),
}).run(fitted)
output = Path('experiment/training_controls_demo/guide')
output.mkdir(parents=True, exist_ok=True)
report.to_html(output / 'controls.html')
```

Coefficients retain their signs and fitted feature names. After standardization,
their units refer to standardized inputs. Binary classifier coefficients describe
`classes_[1]`; multi-output and multiclass rows retain their target/class labels.
Intercepts have a separate table. Survival means absolute coefficient exceeds
the configured tolerance. Its denominator includes every inspected fold exposing
that target/class, including an intercept-only fold; an absent feature counts zero.

Lasso and ElasticNet expose coefficients but generally no usable iteration loss
curve through this adapter. The report explicitly records unavailable histories.

```python
lasso = TrainingRunner(lambda: EstimatorAdapter(make_pipeline(
    StandardScaler(), Lasso(alpha=.1, max_iter=2000)))).run(dataset, plan)
elastic = TrainingRunner(lambda: EstimatorAdapter(make_pipeline(
    StandardScaler(), ElasticNet(alpha=.1, l1_ratio=.5, max_iter=2000)))).run(dataset, plan)
sparse_report = PostTrainingAnalysis({
    'Lasso coefficients': CoefficientReporter(type='overall', include_zeros=True),
    'Available history': LearningCurveReporter(type='overall'),
}).run(lasso)
sparse_report.to_html(output / 'lasso.html')
elastic.folds[0].model.coefficient_table()
```

## 4. Add an optional lightweight display

`LiveLossPlot` observes actual update events. It maintains one IPython display
handle, refreshes at most once per interval except for final events, and draws
SVG without a widget or server. Display downsampling leaves saved histories intact.
Use a fresh observer per desired panel. A final event refreshes the last observed
step; consult `training_summary` for the restored step and selected attempt.

```python
from xdiyo_analytics.training import LiveLossPlot

live = LiveLossPlot(interval=.5, max_points=100)
observed = TrainingRunner(model_factory, control=TrainingControl(max_steps=3),
                         validation=ValidationTail(.25), observer=live).run(dataset, plan)
(output / 'loss_snapshot.html').write_text(live.to_html(), encoding='utf-8')
```

Fresh-kernel execution, saved SVG and display-handle behavior are checked
separately. Live Jupyter frontend trust/update behavior has not been verified in
this environment. Ordinary runs need no observer or IPython display.

## 5. Fit the selected configuration on declared development rows

`refit_model` creates a fresh model on explicitly supplied rows. It does not choose
hyperparameters or schedule future retraining. This example fixes a twelve-step
budget and uses all twelve declared development rows, with no validation tail.
The later four rows are used only for evaluation.

```python
final = refit_model(
    dataset, model_factory, train_positions=np.arange(12),
    control=TrainingControl(max_steps=12, early_stopping=None, scheduler=None,
                            restarts=0, seed=7, restore_best=False),
)
evaluated = final.evaluate(dataset, test_positions=np.arange(12, 16))
final_report = PostTrainingAnalysis({
    'Final evaluation': PerformanceReporter(type='overall', partition='test', metrics=['mse']),
    'Final history': LearningCurveReporter(type='overall'),
}).run(evaluated)
```

On the same dataset, evaluation rejects any development-row overlap, including
validation rows. On a genuinely different dataset, explicitly set
`same_dataset=False`; original development positions remain provenance, while
local train/fit/validation arrays are empty. This flag is a caller assertion, not
an identity-based proof of independence. `final.predict(new_dataset)` can return
predictions directly; it does not require observed target values.

## 6. Distinguish intermediate trials and final results

A group records identity only. The following saves one already computed trial
and a separately evaluated final result. It performs no grid search or Optuna run.
Linking a trial does not copy its metrics or establish independence. In this
teaching example the earlier outer folds also used the later rows for scoring;
the final output therefore illustrates the API, not an untouched research holdout.

```python
from xdiyo_analytics.experiments import ExperimentStore

store = ExperimentStore(output / 'experiments', 'Synthetic controls')
group = store.start_run('Manually configured example', config={'seed': 7})
trial = store.save_run(
    fitted, report, name='Development trial', config={'control': control},
    role='trial', run_group=group, save_predictions=False,
)
saved = store.save_run(
    evaluated, final_report, name='Final evaluation',
    config={'steps': 12, 'seed': 7, 'features': list(X)},
    run_group=group, selected_trial_id=trial['run_id'], save_html=True,
)
leaderboard = PostTrainingAnalysis({
    'Final results': ExperimentLeaderboardReporter(weights={'mse': 1.}),
}).run(experiment=store)
leaderboard.to_html(output / 'leaderboard.html')
```

`save_run` defaults to `role='final'`. The default leaderboard shows final results;
use `include_trials=True` and optionally `run_group=group` for detailed records.
`read_runs()` returns all roles. Each explicit group permits one final result,
including a failed final; serialize writes to that group. `training.json` retains
memberships, histories and summaries even with `save_predictions=False`.
It stores no model pickle or checkpoint for resuming training.

## Verification boundary

The [checklist](training_controls_documentation_checklist.md) and
[evidence](training_controls_check.json) record synthetic checks against the
preserved controls source revision. A later MatchResult reporter batch may change
the shared checkout and has a separate verification boundary. Earlier notebooks,
exports and completed evidence remain unchanged.
