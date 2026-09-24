"""All-feature exact-count classification companion for notebook 20.

Feature preparation is imported unchanged from notebook 19's helper. Class
encoding and balancing are fitted independently in each optimization population.
"""
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from notebooks.helpers.quantile_corners import prepare, STAT_CHOICES, HALF_KEYS
from xdiyo_analytics.analysis import PreTrainingAnalysis, PostTrainingAnalysis
from xdiyo_analytics.evaluation import Metric
from xdiyo_analytics.experiments import FootballExperiment
from xdiyo_analytics.reporting import (CorrelationAnalysis, PerformanceReporter,
    MatchResultReporter, PredictionDistributionReporter, ExperimentLeaderboardReporter,
    TopKCorrelationSelector)
from xdiyo_analytics.selection import Candidate, ModelSelection, MetricSelection
from xdiyo_analytics.splits import TemporalSplit, SplitPlan, create_split_plan
from xdiyo_analytics.training import ExecutionPolicy, save_model, load_model
from xdiyo_analytics.ui.adapters import BoostingAdapter


DEFAULT_MODEL_OPTIONS = dict(n_estimators=400, learning_rate=.04, max_depth=3,
    min_child_weight=10, subsample=.8, colsample_bytree=.65, reg_alpha=.2,
    reg_lambda=10., max_bin=128, random_state=42, early_stopping_rounds=None)
DEFAULT_GRID = dict(max_depth=[3, 5], min_child_weight=[10, 30],
                    n_estimators=[200, 400], balance_power=[0., .5, 1.])


def count_labels(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError('Exact-count classification needs finite nonnegative integer corner labels.')
    return values.astype(np.int64)


def class_weights(labels, power):
    """Return per-observation weights with fitting-population mean exactly one."""
    if not np.isfinite(power) or power < 0:
        raise ValueError('balance_power must be finite and nonnegative.')
    y = count_labels(labels)
    classes, counts = np.unique(y, return_counts=True)
    inverse = (len(y) / (len(classes) * counts)) ** power
    weights = inverse[np.searchsorted(classes, y)]
    normalizer = weights.mean()
    return weights / normalizer, pd.DataFrame({'count_class': classes, 'fit_count': counts,
        'sample_weight': inverse / normalizer})


@dataclass
class BalancedCountClassifierAdapter(BoostingAdapter):
    """Fit-local multiclass weights plus existing original-count encoding.

This is an analytics adapter, not a new sklearn estimator. Validation supplied
by a caller must contain supported fitting classes, as required by native XGB.
The notebook avoids native validation/early stopping and tunes through inner CV.
"""
    balance_power: float = 0.

    def fit(self, context):
        y = count_labels(context.y.iloc[:, 0])
        if len(np.unique(y)) < 2:
            raise ValueError('A classifier fitting fold needs at least two observed count classes.')
        weights, table = class_weights(y, self.balance_power)
        self.class_balance_ = table
        self._native().set_params(objective='multi:softprob', num_class=len(table))
        self.fit_kwargs = {**self.fit_kwargs, 'sample_weight': weights}
        super().fit(context)
        self.training_summary_.update(balance_power=float(self.balance_power),
            weight_mean=float(weights.mean()), class_balance=table.to_dict('records'),
            class_universe='fitting labels only', prediction_statistic='most probable count')


def model_factory(options, balance_power):
    from xgboost import XGBClassifier
    return BalancedCountClassifierAdapter(XGBClassifier(tree_method='hist', eval_metric='mlogloss', **options),
        prediction_methods=('predict', 'predict_proba'), fit_kwargs={'verbose': False}, balance_power=balance_power)


def build_candidates(*, model_options=None, grid=None, feature_selection_proportion=None,
                     feature_selection_correlation='spearman', feature_count=None):
    options = {**DEFAULT_MODEL_OPTIONS, **(model_options or {})}
    if options.get('early_stopping_rounds') is not None:
        raise ValueError('This notebook uses chronological inner CV, with early_stopping_rounds=None.')
    if feature_selection_correlation not in {'pearson', 'spearman', 'kendall'}:
        raise ValueError('Feature-selection correlation must be pearson, spearman or kendall.')
    proportion = feature_selection_proportion
    if proportion is not None:
        if isinstance(proportion, (bool, np.bool_)) or not np.isfinite(proportion) or not 0 < proportion <= 1:
            raise ValueError('Feature-selection proportion must be None or in (0, 1].')
        if proportion == 1 and (feature_count is None or feature_count < 1):
            raise ValueError('Proportion 1 requires the assembled feature_count.')
    selector_name = 'Fold training correlation selection'
    candidates = []
    for number, values in enumerate(ParameterGrid(DEFAULT_GRID if grid is None else grid), 1):
        values = dict(values)
        power = values.pop('balance_power', 0.)
        current = {**options, **values}
        label = ', '.join(f'{key}={value}' for key, value in {**values, 'balance_power': power}.items())
        config = {'model': 'XGBClassifier', 'options': current, 'balance_power': power,
                  'target': 'exact total corners', 'validation': None}
        selection = None
        if proportion is not None:
            # The common selector interprets literal 1 as a count, so translate
            # this notebook's proportion=1 into the whole input-column count.
            k = int(feature_count) if proportion == 1 else float(proportion)
            selection = PreTrainingAnalysis({selector_name: TopKCorrelationSelector(
                type='per_fold', partition='train', k=k, method=feature_selection_correlation)})
            config['feature_selection'] = dict(proportion=float(proportion),
                correlation=feature_selection_correlation, k=k, scope='per_fold/train')
        candidates.append(Candidate(f'XGB count {number}: {label}', partial(model_factory, current, power),
            config=config, validation=None, pre_analysis=selection,
            features_from=selector_name if selection is not None else None))
    return candidates


def inner_plan(prepared, splitter=None):
    """Keep only chronological candidate folds wholly inside outer training."""
    data, outer = prepared.dataset, prepared.split_plan.folds[0]
    splitter = splitter or TemporalSplit(train_size=1, test_size=1, unit='seasons',
                                         calendar_by=(), block_by=('source_season',))
    cutoffs = None
    hours = prepared.config.get('cutoff_hours')
    if hours is not None:
        cutoffs = data.metadata.kickoff_at - pd.Timedelta(hours=hours)
    proposed = create_split_plan(data, splitter, cutoffs=cutoffs)
    folds = [fold for fold in proposed.folds if np.isin(fold.train, outer.train).all()
             and np.isin(fold.test, outer.train).all()]
    if not folds:
        raise ValueError('No inner chronological fold fits entirely within the outer training seasons.')
    return SplitPlan(folds, len(data.X), proposed.row_order)


def classification_diagnostics(observed, predicted, probabilities, *, epsilon=1e-12):
    """Include every held-out label, explicitly marking unsupported count classes."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
    y, prediction = count_labels(observed), count_labels(predicted)
    if len(y) != len(prediction) or len(y) != len(probabilities):
        raise ValueError('Labels, predictions and probability rows must align.')
    if isinstance(observed, pd.Series) and not observed.index.equals(probabilities.index):
        raise ValueError('Observed and probability row identities/order differ.')
    if isinstance(predicted, pd.Series) and not predicted.index.equals(probabilities.index):
        raise ValueError('Prediction and probability row identities/order differ.')
    p = probabilities.to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1., atol=1e-6):
        raise ValueError('Class probabilities must be finite, nonnegative and sum to one.')
    classes = count_labels(probabilities.columns)
    if len(np.unique(classes)) != len(classes):
        raise ValueError('Probability class columns must be distinct original counts.')
    lookup = {value: position for position, value in enumerate(classes)}
    supported = np.isin(y, classes)
    actual_p = np.array([p[row, lookup[value]] if value in lookup else 0. for row, value in enumerate(y)])
    rows = pd.DataFrame({'observed_total_corners': y, 'predicted_total_corners': prediction,
        'absolute_error': np.abs(y - prediction), 'within_two': np.abs(y - prediction) <= 2,
        'target_in_fitted_classes': supported, 'observed_class_probability': actual_p,
        'probability_weighted_count': p @ classes}, index=probabilities.index)
    scores = dict(n=len(y), mae=mean_absolute_error(y, prediction), mse=mean_squared_error(y, prediction),
        accuracy=accuracy_score(y, prediction), macro_f1=f1_score(y, prediction, average='macro', zero_division=0),
        within_two=float(rows.within_two.mean()), unseen_count_rows=int((~supported).sum()),
        unseen_count_fraction=float((~supported).mean()),
        log_loss_clipped_all=float(-np.log(np.maximum(actual_p, epsilon)).mean()), log_loss_floor=epsilon,
        log_loss_supported_only=float(-np.log(np.maximum(actual_p[supported], epsilon)).mean()) if supported.any() else np.nan,
        exact_log_loss_infinite=bool((actual_p == 0).any()))
    return rows, pd.DataFrame([scores])


@dataclass
class ClassifierRun:
    result: object
    comparison: pd.DataFrame
    predictions: pd.DataFrame
    probabilities: pd.DataFrame
    metrics: pd.DataFrame
    by_league: pd.DataFrame
    class_balance: pd.DataFrame
    importance: pd.DataFrame
    output_path: Path


def run_classifier(bundle, output_dir, *, model_options=None, grid=None, device='cpu', threads=4,
                   reuse=True, splitter=None, pre_analysis=True, feature_selection_proportion=None,
                   feature_selection_correlation='spearman'):
    """Select within all development seasons, then evaluate the held-out season."""
    original = bundle.prepared
    labels = original.dataset.y.copy()
    for column in labels:
        labels[column] = count_labels(labels[column])
    prepared = replace(original, dataset=replace(original.dataset, y=labels),
        config={**original.config, 'task': 'exact-count multiclass XGBoost; all notebook19 features'})
    data = prepared.dataset
    choices = build_candidates(model_options=model_options, grid=grid,
        feature_selection_proportion=feature_selection_proportion,
        feature_selection_correlation=feature_selection_correlation, feature_count=len(data.X.columns))
    selection_plan = inner_plan(prepared, splitter)
    print(f'{len(choices)} candidates × {len(selection_plan.folds)} inner fold(s), then one outer fit; {data.X.shape[1]:,} features.', flush=True)
    reports = {
        'Errors and classification': PerformanceReporter(type='overall', partition='score', metrics=[
            'mae', 'mse', 'accuracy', Metric('f1_score', parameters={'average': 'macro', 'zero_division': 0}, key='macro_f1')]),
        'Match predictions': MatchResultReporter(type='overall', partition='score', comparison='numeric', tolerance=2.,
            league_column='source_league', season_column='source_season', catalog=bundle.team_catalog),
        'Label and prediction counts': PredictionDistributionReporter(type='overall', partition='score', mode='frequency'),
        'Experiment leaderboard': ExperimentLeaderboardReporter(type='overall', weights={'mae': 1.}),
    }
    before = PreTrainingAnalysis({'Training rank correlations': CorrelationAnalysis(type='overall', partition='train',
        methods=('spearman',), percentile_ranks=True)}) if pre_analysis else None
    experiment = FootballExperiment('Total corners XGBoost classifier all features', output_dir=output_dir)
    result = experiment.run(prepared, model_selection=ModelSelection(choices, metrics=['mae', 'mse', 'accuracy'],
        decision=MetricSelection('mae')), selection_plan=selection_plan, pre_analysis=before,
        post_analysis=PostTrainingAnalysis(reports), execution=ExecutionPolicy(n_jobs=1, device=device, inner_threads=threads),
        reuse=reuse)
    fold = result.training.folds[0]
    target = fold.target_columns[0]
    prediction = fold.predictions['predict'].loc[fold.score_positions, target]
    probability = fold.predictions['predict_proba'].loc[fold.score_positions].xs(target, level=0, axis=1)
    observed = fold.y_true.loc[fold.score_positions, target]
    values, metrics = classification_diagnostics(observed, prediction, probability)
    values = data.metadata.loc[values.index].join(values)
    baseline = int(labels.iloc[fold.fit_positions, 0].mode().iloc[0])
    metrics['baseline_majority_count'] = baseline
    metrics['baseline_mae'] = np.abs(observed.to_numpy() - baseline).mean()
    metrics['baseline_accuracy'] = np.mean(observed.to_numpy() == baseline)
    by_league = []
    for league, part in values.groupby('source_league', sort=True):
        _, scores = classification_diagnostics(part.observed_total_corners, part.predicted_total_corners,
                                               probability.loc[part.index])
        scores.insert(0, 'league', league)
        by_league.append(scores)
    by_league = pd.concat(by_league, ignore_index=True)
    path = Path(result.path) / 'fitted_model'
    if fold.model is not None and not path.exists():
        save_model(result.training, path)
    model = fold.model if fold.model is not None else load_model(path).model if path.exists() else None
    balance = model.class_balance_ if model is not None else pd.DataFrame(fold.training_summary.get('class_balance', []))
    importance = []
    if model is not None:
        for key, gain in model._native().get_booster().get_score(importance_type='gain').items():
            importance.append({'feature': fold.feature_columns[int(key[1:])], 'gain': gain})
    importance = pd.DataFrame(importance, columns=['feature', 'gain']).sort_values('gain', ascending=False)
    output = Path(result.path) / 'classifier_summary'
    output.mkdir(exist_ok=True)
    comparison = result.selection.comparison
    selected_features = pd.DataFrame({'feature': fold.feature_columns})
    for name, frame in {'predictions': values, 'probabilities': probability, 'metrics': metrics,
                        'per_league': by_league, 'class_balance': balance, 'feature_importance': importance,
                        'feature_manifest': bundle.feature_manifest, 'candidate_comparison': comparison,
                        'selected_features': selected_features}.items():
        frame.to_csv(output / f'{name}.csv', index=True, index_label='row_position')
    summary = dict(features=data.X.shape[1], historical_expressions=len(bundle.definitions), rows=len(data.X),
        fit_rows=len(fold.fit_positions), test_rows=len(fold.test_positions), candidates=len(choices),
        inner_folds=len(selection_plan.folds), device=device, reused=result.reused, model_path=str(path),
        feature_selection_proportion=feature_selection_proportion,
        feature_selection_correlation=feature_selection_correlation,
        selected_feature_count=len(fold.feature_columns),
        selected_feature_fraction=len(fold.feature_columns) / len(data.X.columns),
        class_universe='optimization rows only', validation=None, tolerance=2,
        unseen_test_classes=sorted(set(observed) - set(probability.columns)),
        probability_note='Weighted classifier probabilities are not automatically calibrated. Unknown fitting classes have zero support; clipped log loss uses an explicit floor.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return ClassifierRun(result, comparison, values, probability, metrics, by_league, balance, importance, output)
