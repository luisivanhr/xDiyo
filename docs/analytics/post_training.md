# Inspect predictions, account for bets and compare saved runs

Post-training analysis consumes an existing `TrainingResult`. It computes reusable
tables and figures without fitting models or recomputing predictions. The examples
below use twelve synthetic observations and two explicitly configured regressors.
They demonstrate interfaces; they do not run a football experiment or establish
predictive or betting performance.

Use the existing `misc314_py314` environment. The `reporting` optional dependency
group includes plotting, scientific calculation and scikit-learn metrics. The
[API reference](post_training_reference.md) covers every option and helper;
[equations](post_training_equations.md) define the calculations and denominators.

## Start with a fitted result

Ordinary applications obtain `ModelDataset` and `SplitPlan` from the existing
[assembly](datasets.md) and [split](splits.md) APIs. This small setup writes their
identities and memberships explicitly. Information cutoffs, ratings and fold-aware
state construction remain upstream responsibilities.

```python
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from xdiyo_analytics.datasets import ModelDataset
from xdiyo_analytics.splits import Fold, SplitPlan
from xdiyo_analytics.training import EstimatorAdapter, TrainingRunner

positions = np.arange(12)
X = pd.DataFrame({'trend': positions / 10, 'cycle': positions % 3})
y = pd.DataFrame({'count': 5. + positions + positions % 3})
metadata = pd.DataFrame({
    'competition_id': 17, 'season_id': 2025,
    'event_id': pd.array([2**63 + 101 + int(i) for i in positions],
                         dtype='uint64[pyarrow]'),
    'kickoff_at': pd.date_range('2025-01-01', periods=12, freq='2D', tz='UTC'),
})
keys = ('competition_id', 'season_id', 'event_id')
dataset = ModelDataset(X, y, metadata, 'match', keys, keys, 'total',
                       {'kind': 'synthetic demonstration'})
plan = SplitPlan([
    Fold(np.arange(6), np.arange(6, 9), np.array([7, 8])),
    Fold(np.arange(9), np.arange(9, 12), np.arange(9, 12)),
], 12, positions)

def make_adapter(alpha):
    return EstimatorAdapter(make_pipeline(
        SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=alpha)))

fitted = TrainingRunner(lambda: make_adapter(1.)).run(dataset, plan)
```

## Choose studies and their populations

Every prediction reporter requires both `type` and `partition`. `per_fold` produces
one view per selected fold; `overall` computes on pooled prediction rows;
`timeline` also sorts the population by UTC kickoff. `score` uses the retained
scoring subset and `test` uses all held-out predictions. Overall metrics are
calculated once from their eligible rows, never implicitly averaged from folds.

```python
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.reporting import (
    AnalysisReport, PerformanceReporter, ResidualAnalysisReporter,
    PredictionDistributionReporter, PredictionTimelineReporter,
    CalibrationReporter, BetPerformanceReporter, ExperimentLeaderboardReporter,
)

analysis = PostTrainingAnalysis({
    'Fold performance': PerformanceReporter(
        type='per_fold', partition='score', metrics=['mse', 'mae', 'r2']),
    'Overall performance': PerformanceReporter(
        type='overall', partition='score', metrics=['mse', 'mae', 'r2']),
    'Residuals': ResidualAnalysisReporter(type='overall', partition='score'),
    'Density overlay': PredictionDistributionReporter(
        type='overall', partition='score', mode='kde', grid_size=128),
    'Prediction timeline': PredictionTimelineReporter(
        type='timeline', partition='test', group_by='competition_id'),
})
prediction_report = analysis.run(fitted)
prediction_report.studies[2].result.tables['metrics']
```

The KDE overlay contains separate observed and predicted curves, each normalized
as a density, on a common grid. It has no histogram bars. A constant sample uses a
vertical marker. `mode='ecdf'` shows cumulative frequencies; `mode='frequency'`
compares categorical relative frequencies. Similar marginal distributions do not
establish accurate predictions for individual matches. Residuals are observed
minus predicted values. Timeline gaps remain visible; no time interpolation occurs.

```python
output_dir = Path('experiment/post_training_demo/guide')
output_dir.mkdir(parents=True, exist_ok=True)
prediction_report.to_html(output_dir / 'prediction_report.html')
# In a trusted notebook, use prediction_report.to_notebook() or its rich display.
```

The shared offline viewer provides navigation, per-fold selectors, plots and full
CSV downloads. Scope CSVs retain both `fold_id` and `row_position`, so repeated
predictions remain identifiable. Rendering a saved report does not rerun studies.
The notebook companion uses an isolated iframe; live frontend trust behavior is
distinct from fresh-kernel execution and reviewing the saved HTML export.

## Use numerical metrics without rendering

```python
from xdiyo_analytics.evaluation import Metric, evaluate_metrics, list_metrics, register_metric

fold = fitted.folds[0]
metric_table = evaluate_metrics(
    fold.y_true, fold.predictions,
    [Metric('mse', target='count'), Metric('mae', target='count')],
    metadata=fold.metadata,
)
metric_table[['metric', 'value', 'n', 'n_missing', 'status']]
```

Each result records its actual complete-case population, status, calculation,
parameters, output, target, direction and sample fingerprint. Numeric metrics
exclude missing/nonfinite pairs. Probability metrics require complete class
vectors; labels are matched to explicit class columns. Undefined results remain
missing with a reason. Losses are positive values and logarithms use nats.
Precision, recall and F1 default to macro averaging; binary averaging and its
positive label must be explicit.

```python
def mean_bias(observed, predicted, scale=1.):
    return scale * (predicted - observed).mean()

register_metric('mean_bias_demo', mean_bias, kind='numeric', replace=True)
custom = evaluate_metrics(fold.y_true, fold.predictions, [
    Metric('mean_bias_demo', target='count', parameters={'scale': 1.}, key='bias'),
], metadata=fold.metadata)
list_metrics().query("name == 'mean_bias_demo'")
```

Custom functions return one scalar. `kind` selects numeric values, class labels,
class-probability frames or uncertainty inputs. A descriptive metric such as signed
bias or entropy has no universal ranking direction. Supply a direction only when
the intended preference is explicit. Replacing a registry entry is deliberate;
use distinct keys for differently configured versions of a calculation.

## Declare how repeated held-out rows are pooled

```python
repeated = replace(fitted, folds=[*fitted.folds, replace(fitted.folds[0], fold_id=20)])
pooled = PostTrainingAnalysis({
    'Mean-pooled predictions': PerformanceReporter(
        type='overall', partition='test', pooling='mean', metrics=['mse']),
}).run(repeated)
pooled.studies[0].scope
```

This explicitly duplicates a recorded fold to illustrate the contract; it creates
no new independent evidence. With repeated rows, omitting `pooling` raises.
`occurrences` retains every prediction; `first` and `last` choose by selected fold
order. `mean` averages numeric outputs and propagates missing cells, including
class columns absent from one contributing fold. Mean-pooled scope uses
`fold_id=-1`. It does not turn numeric class decisions into meaningful probabilities;
use first/last for class decisions or supply probability-only outputs. Repeated
rows with inconsistent targets or metadata always raise.

## Inspect classification probabilities

```python
from sklearn.dummy import DummyClassifier

classified_dataset = replace(dataset, y=pd.DataFrame({'result': positions % 2}))
classified = TrainingRunner(lambda: EstimatorAdapter(
    DummyClassifier(strategy='prior'), prediction_methods=('predict', 'predict_proba'),
)).run(classified_dataset, plan)
classification_report = PostTrainingAnalysis({
    'Classification metrics': PerformanceReporter(
        type='overall', partition='test', metrics=[
            'accuracy', 'f1', 'binary_cross_entropy', 'brier_score', 'roc_auc']),
    'Calibration': CalibrationReporter(type='per_fold', partition='test', n_bins=3),
    'Class frequencies': PredictionDistributionReporter(
        type='overall', partition='test', mode='frequency'),
    'Predictive entropy': PerformanceReporter(
        type='overall', partition='test', metrics=['binary_entropy']),
}).run(classified)
```

Probability columns have `(target, class)` identities. Cross-entropy selects the
probability assigned to the observed label even when class order is reversed.
Binary entropy measures uncertainty in two predicted probabilities and needs no
observed label value. Calibration bins show mean predicted probability against
observed one-vs-rest frequency; they do not fit a recalibration model. Incomplete
probability vectors are excluded and counted. See the full metric list, aliases,
probability checks and default parameters in the reference.

## Supply explicit bet options, quotes and decisions

Bet accounting uses existing `BetOption` labels. It supports `match` and
`team_match`; the settlement label's unit must equal the prediction layout, and
team-level settlement joins include team identity. Each option must yield one
concrete settlement column. The following paired synthetic history has the same
match identities and total counts as the regression setup.

```python
from xdiyo_analytics.features import Stat
from xdiyo_analytics.labels import BetOption, MatchTotal, create_labels
from xdiyo_analytics.evaluation import BetSpec

own = 'team::ALL::Match overview::cornerKicks::value'
other = 'opponent::ALL::Match overview::cornerKicks::value'
rows = []
for i in positions:
    home_value = 2 + int(i % 4)
    away_value = float(y.loc[i, 'count']) - home_value
    for side, team, opponent, value, against in [
        ('home', 101, 202, home_value, away_value),
        ('away', 202, 101, away_value, home_value),
    ]:
        rows.append(dict(competition_id=17, season_id=2025,
                         event_id=int(metadata.event_id.iloc[i]),
                         kickoff_at=metadata.kickoff_at.iloc[i],
                         team_id=team, opponent_id=opponent, side=side,
                         status='finished', **{own: value, other: against}))
history = pd.DataFrame(rows)
history['event_id'] = pd.array(history.event_id, dtype='uint64[pyarrow]')
history.attrs['stat_columns'] = {
    name: dict(role=role, period='ALL', group_name='Match overview',
               key='cornerKicks', field='value')
    for name, role in [(own, 'team'), (other, 'opponent')]
}
option = BetOption(MatchTotal(Stat('ALL', 'Match overview', 'cornerKicks')),
                   selection='over', line=15., on_equal='push')
settlements = create_labels(history, {'over15': option})
bet = BetSpec(option, odds=2.1, take=True, stake=1.,
              policy='synthetic_all_offers_one_unit')
```

```python
bet_reporter = BetPerformanceReporter(
    type='overall', partition='test', bets={'over15': bet}, labels=settlements,
)
combined_analysis = PostTrainingAnalysis({**analysis.reporters, 'Bet accounting': bet_reporter})
report = combined_analysis.run(fitted)
report.studies[-1].result.tables['ledger']
```

`take=True` deliberately takes every offered option in this demonstration. The
library does not infer a bet from predictions. Supply exactly one of `labels` or
`history`. Quotes/stakes can be scalars or exact identity-indexed Series;
decisions can additionally be a callable returning an occurrence-indexed Series.
Its information eligibility and avoiding realized outcomes are caller obligations.

Wins earn stake times decimal odds minus stake; losses lose stake; push/void
return stake. Missing quotes, stakes or settlement remain unresolved. Nonselected
rows have zero stake and profit. ROI divides known profit by settled stakes,
including push/void stakes; partial results are labeled and cannot qualify as
complete leaderboard metrics. The cumulative line sums known profit in kickoff
order by default. It is not a cash-flow or bankroll simulation. Fees, bankroll
limits, dynamic strategy execution, split lines and parlays are outside this layer.

## Save explicit configurations and compare runs

```python
from xdiyo_analytics.experiments import ExperimentStore, configuration_hash, rank_runs

store = ExperimentStore(output_dir / 'experiments', 'Synthetic guide comparison')
first_record = store.save_run(
    fitted, report, name='Ridge alpha 1',
    config={'model': 'Ridge', 'alpha': 1., 'features': list(X),
            'population': 'twelve synthetic rows', 'bet': bet}, save_html=True,
)
second_fitted = TrainingRunner(lambda: make_adapter(4.)).run(dataset, plan)
second_report = combined_analysis.run(second_fitted)
second_record = store.save_run(
    second_fitted, second_report, name='Ridge alpha 4',
    config={'model': 'Ridge', 'alpha': 4., 'features': list(X),
            'population': 'twelve synthetic rows', 'bet': bet},
)
failure_record = store.save_failure(
    name='Explicit failed-candidate example', config={'model': 'demonstration'},
    error='Illustrative recorded failure; no exception was suppressed.',
)
```

These are two manually specified configurations, not a search loop. The store
creates experiment/run UUIDs and configuration hashes. It saves numerical metric
records, JSON tables and optional indexed Parquet predictions/targets/metadata.
It saves no model pickle and performs no automatic model introspection. Configs
should describe the relevant model, features, data, seed and decision rule;
hash equality alone does not establish reproducibility. Rerunning these examples
adds new run IDs to the named experiment.

Publication uses a directory rename after all artifacts are written. On Windows,
only access/sharing error codes 5, 32 and 33 receive five short delays and at most
six attempts. Persistent failures raise and retain an invisible `.pending-*`
directory; the store neither copies a partial run into publication nor deletes
the diagnostic artifacts. `read_runs()` returns published records only.

```python
selectors = {
    'error': {'metric': 'mse', 'target': 'count', 'study': 'Overall performance'},
    'return': {'metric': 'roi', 'target': 'over15', 'study': 'Bet accounting'},
}
weights = {'error': 3., 'return': 1.}  # Illustrative preferences, not a strategy.
ranking = rank_runs(store.read_runs(), weights, selectors=selectors)
leaderboard = PostTrainingAnalysis({
    'Experiment leaderboard': ExperimentLeaderboardReporter(
        weights=weights, selectors=selectors),
}).run(experiment=store)
ranking[['name', 'status', 'comparison_group', 'raw::error', 'raw::return', 'score', 'rank']]
```

Positive weights are normalized once. Higher utility is better; minimizing losses
are reversed. Percentile scaling ranks only comparable runs and assigns 0.5 to a
singleton or an all-tied metric. Adding candidates can change percentile scores.
For fixed scales, pass `scaling='fixed'` and `reference_scales` with finite
increasing `(low, high)` bounds for every positively weighted metric; utilities
are clipped to the unit interval. The [equations](post_training_equations.md#weighted-experiment-comparison)
give the exact transformations.

Metric selectors disambiguate study, target, output, execution type and fold.
Definitions, evaluated-sample fingerprints and scope determine comparison groups;
ranks from different groups are not comparable. Missing/partial/undefined required
metrics leave a run visible but unranked. No per-run weight renormalization occurs.
Bet comparison includes quotes, settlement opportunities, option and declared
policy/stake settings, while allowing candidate decision masks to differ.

```python
preview = AnalysisReport(
    studies=[*report.studies, *classification_report.studies, *leaderboard.studies],
    title='Synthetic post-training reporting preview',
)
preview.to_html(output_dir / 'post_training_report.html')
```

## What this increment establishes

The [coverage checklist](post_training_documentation_checklist.md) and
[verification evidence](post_training_check.json) distinguish numerical, browser,
notebook, packaging and preservation checks. The
[tenth notebook](../../notebooks/10_post_training_quickstart.ipynb) is a short
companion; advanced contracts remain in the reference.

Training/search decisions, final holdout design, scheduled refitting, feature
importance, fold-aware state reconstruction and real pilot performance remain
outside these post-training reporters. A leaderboard is a transparent comparison
of recorded evidence; it does not correct selection bias or create new held-out
observations.
