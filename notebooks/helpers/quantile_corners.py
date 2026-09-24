"""Reproducible broad-feature quantile experiment used by notebook 19.

Uses public analytics contracts. Each quantile is a separate scalar model, so
its predictions retain the ordinary target/row identities throughout storage.
"""
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from xdiyo_analytics.data import load_seasons, select_stats
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.features import (
    Stat, ForAgainst, Lag, RollingMean, RollingStd, RollingZScore, EMA, H2H,
    League, LeaveOneOut, NormalizedStanding, MatchResultGlicko, StatGlicko,
    WarmStart, evaluate_features, eligible_history_rows,
)
from xdiyo_analytics.labels import MatchTotal, create_labels
from xdiyo_analytics.datasets import assemble_dataset
from xdiyo_analytics.splits import TemporalSplit, create_split_plan
from xdiyo_analytics.experiments import FootballExperiment, PreparedExperiment
from xdiyo_analytics.selection import Candidate
from xdiyo_analytics.analysis import PreTrainingAnalysis, PostTrainingAnalysis
from xdiyo_analytics.reporting import (
    CorrelationAnalysis, PerformanceReporter, MatchResultReporter,
    PredictionDistributionReporter, LearningCurveReporter, TeamCatalog,
)
from xdiyo_analytics.evaluation import Metric, register_metric
from xdiyo_analytics.training import ExecutionPolicy, ValidationTail, save_model, load_model
from xdiyo_analytics.ui.adapters import BoostingAdapter


STAT_CHOICES = (
    ('Match overview', 'cornerKicks'), ('Match overview', 'ballPossession'),
    ('Match overview', 'totalShotsOnGoal'), ('Match overview', 'expectedGoals'),
    ('Match overview', 'bigChanceCreated'), ('Match overview', 'goalkeeperSaves'),
    ('Match overview', 'totalTackle'), ('Shots', 'shotsOnGoal'),
    ('Shots', 'shotsOffGoal'), ('Shots', 'blockedScoringAttempt'),
    ('Shots', 'totalShotsInsideBox'), ('Shots', 'totalShotsOutsideBox'),
    ('Attack', 'touchesInOppBox'), ('Attack', 'bigChanceMissed'),
    ('Attack', 'offsides'), ('Passes', 'accurateCross'),
    ('Passes', 'finalThirdEntries'), ('Passes', 'finalThirdPhaseStatistic'),
    ('Defending', 'ballRecovery'), ('Defending', 'interceptionWon'),
    ('Defending', 'totalClearance'), ('Defending', 'errorsLeadToShot'),
)
HALF_KEYS = ('cornerKicks', 'ballPossession', 'totalShotsOnGoal',
             'shotsOnGoal', 'shotsOffGoal', 'touchesInOppBox')
DEFAULT_MODEL_OPTIONS = dict(
    n_estimators=800, learning_rate=0.035, max_depth=3, min_child_weight=20,
    subsample=0.8, colsample_bytree=0.65, reg_alpha=0.2, reg_lambda=10.,
    max_bin=128, early_stopping_rounds=60, random_state=42,
)


def build_feature_definitions(stat_identities, *, windows=(3, 5, 10, 20),
                              lags=(1, 2, 3), spans=(5, 10),
                              periods=('ALL', '1ST', '2ND'), include_ratings=True,
                              include_loo=True, loo_windows=(3, 5), loo_reducers=('mean',),
                              include_h2h=True, h2h_windows=(3, 5), h2h_reducers=('mean',),
                              warm_policy=None, warm_stat_keys=None, rating_warm_policy=None):
    """Broad, explicit AST feature bank, restricted to available identities.

    Full-match statistics receive every rolling mean, two std windows, two EMA
    spans, three lags and one z-score. Half statistics receive lag1 and means5/10.
    Missing individual match observations remain missing; no raw target features.
    """
    available = set(map(tuple, stat_identities))
    definitions = {}
    for group, key in STAT_CHOICES:
        for period in periods:
            if (period, group, key) not in available or (period != 'ALL' and key not in HALF_KEYS):
                continue
            for side in ('for', 'against'):
                prefix = f'{period}_{group}_{key}_{side}'
                source = ForAgainst(Stat(period, group, key), side=side)
                for lag in lags if period == 'ALL' else (1,):
                    definitions[f'{prefix}_lag{lag}'] = Lag(source, periods=lag)
                for window in windows if period == 'ALL' else (5, 10):
                    definitions[f'{prefix}_mean{window}'] = RollingMean(source, window=window)
                if period == 'ALL':
                    for window in (5, 10):
                        definitions[f'{prefix}_std{window}'] = RollingStd(source, window=window, ddof=1)
                    definitions[f'{prefix}_z5'] = RollingZScore(source, window=5, ddof=1)
                    for span in spans:
                        definitions[f'{prefix}_ema{span}'] = EMA(source, span=span)
    corners = Stat('ALL', 'Match overview', 'cornerKicks')
    if include_h2h:
        for side in ('for', 'against'):
            source = ForAgainst(corners, side=side)
            for window in h2h_windows:
                for reducer in h2h_reducers:
                    definitions[f'h2h_corners_{side}_{reducer}{window}'] = _reducer(
                        reducer, H2H(source), window)
    population = League(corners, unit='team', schedule='completed_rounds', window_unit='rounds')
    for window in (3, 5):
        definitions[f'league_corners_mean{window}'] = RollingMean(population, window=window)
        definitions[f'league_corners_std{window}'] = RollingStd(population, window=window, ddof=1)
    if include_loo:
        for window in loo_windows:
            for reducer in loo_reducers:
                definitions[f'loo_corners_{reducer}{window}'] = _reducer(
                    reducer, LeaveOneOut(population), window,
                    reference=Lag(ForAgainst(corners, side='for')))
    definitions['standing'] = NormalizedStanding(side='for', missing_value=0.)
    if include_ratings:
        definitions['result_glicko'] = MatchResultGlicko(side='for', fields=('rating', 'rd'))
        definitions['corners_glicko'] = StatGlicko(corners, side='for', fields=('rating', 'rd'))
    # Additional columns retain all unwarmed definitions for direct comparison.
    if warm_policy is not None:
        for name, operator in list(definitions.items()):
            if not isinstance(operator, (RollingMean, RollingStd, RollingZScore)):
                continue
            source = operator.source
            h2h = isinstance(source, H2H)
            if h2h:
                source = source.source
            raw = source
            while isinstance(raw, (ForAgainst, League, LeaveOneOut)):
                raw = raw.source
            if warm_stat_keys is not None and raw.key not in warm_stat_keys:
                continue
            wrapped = WarmStart(replace(operator, source=source), warm_policy)
            definitions[f'warm::{name}'] = H2H(wrapped) if h2h else wrapped
    if include_ratings and rating_warm_policy is not None:
        for name in ('result_glicko', 'corners_glicko'):
            definitions[f'warm::{name}'] = WarmStart(definitions[name], rating_warm_policy)
    return definitions


def _reducer(name, source, window, reference=None):
    if name == 'mean':
        return RollingMean(source, window=window)
    if name == 'std':
        return RollingStd(source, window=window, ddof=1)
    if name == 'z':
        return RollingZScore(source, window=window, ddof=1, reference=reference)
    raise ValueError('Population reducers must be mean, std or z.')


@dataclass
class FeatureBundle:
    prepared: PreparedExperiment
    history: pd.DataFrame
    definitions: dict
    feature_manifest: pd.DataFrame
    data_summary: pd.DataFrame
    team_catalog: object
    elapsed_seconds: float


def season_populations(matches):
    """Order loaded seasons by first kickoff; reserve the latest for evaluation."""
    first = matches.groupby('source_season')['kickoff_utc'].min().sort_values(kind='stable')
    if len(first) < 2 or first.isna().any():
        raise ValueError('Preparation needs at least two loaded seasons with known kickoffs.')
    return tuple(first.index[:-1]), first.index[-1]


def training_stat_identities(statistics, training_seasons):
    """Discover predictors across all development seasons, never the holdout."""
    return statistics.loc[statistics.source_season.isin(training_seasons),
                          ['period', 'group_name', 'key']].drop_duplicates()


def prepare(data_root, *, seasons=('22_23', '23_24', '24_25'), leagues=None,
            windows=(3, 5, 10, 20), lags=(1, 2, 3), spans=(5, 10),
            periods=('ALL', '1ST', '2ND'), include_ratings=True, cutoff_hours=None,
            include_loo=True, loo_windows=(3, 5), loo_reducers=('mean',),
            include_h2h=True, h2h_windows=(3, 5), h2h_reducers=('mean',),
            warm_policy=None, warm_stat_keys=None, rating_warm_policy=None,
            team_seasons=None, season_starts=None):
    """Use all earlier loaded seasons for development and the latest as holdout."""
    started = time.perf_counter()
    print('Loading season tables...', flush=True)
    data = load_seasons(data_root, list(seasons), leagues=leagues,
                        tables=('matches', 'statistics', 'pregame'), include_awarded=False)
    summary = data.matches.groupby(['source_league', 'source_season']).size().rename('matches').reset_index()
    catalog = TeamCatalog.from_matches(data.matches)
    training_seasons, holdout_season = season_populations(data.matches)
    print(f'Development seasons: {training_seasons}; held out: {holdout_season}', flush=True)
    # Discover identities only in development seasons; holdout coverage
    # cannot decide which new fields enter this experiment.
    available = training_stat_identities(data.statistics, training_seasons)
    definitions = build_feature_definitions(available.itertuples(index=False, name=None),
        windows=windows, lags=lags, spans=spans, periods=periods, include_ratings=include_ratings,
        include_loo=include_loo, loo_windows=loo_windows, loo_reducers=loo_reducers,
        include_h2h=include_h2h, h2h_windows=h2h_windows, h2h_reducers=h2h_reducers,
        warm_policy=warm_policy, warm_stat_keys=warm_stat_keys, rating_warm_policy=rating_warm_policy)
    chosen = [tuple(row) for row in available.itertuples(index=False, name=None)
              if (row[1], row[2]) in STAT_CHOICES and row[0] in periods]
    history = build_team_history(select_stats(data, stats=chosen, bundles=['standings']))
    del data
    cutoff = None if cutoff_hours is None else history.kickoff_at - pd.Timedelta(hours=cutoff_hours)
    print(f'Evaluating {len(definitions):,} historical expressions on {len(history):,} team rows...', flush=True)
    # Bounded groups provide useful notebook progress without changing any
    # eligibility or numerical calculation in the shared evaluator.
    items, blocks = list(definitions.items()), []
    for start in range(0, len(items), 64):
        block = dict(items[start:start + 64])
        blocks.append(evaluate_features(history, block, cutoffs=cutoff, keyed=True,
                                       team_seasons=team_seasons, season_starts=season_starts))
        print(f'  Historical expressions: {min(start+64, len(items))}/{len(items)}', flush=True)
    attributes = dict(blocks[0].attrs)
    attributes['features'] = {key: repr(value) for key, value in definitions.items()}
    for block in blocks:
        block.attrs = {}
    values = pd.concat(blocks, axis=1)
    values.attrs = attributes
    # Rest is based on the same eligible finished-match history as all operators.
    eligible = eligible_history_rows(history, cutoffs=cutoff)
    kicks = history.kickoff_at
    rest = [np.nan if len(rows) == 0 else (kicks.iloc[i] - kicks.iloc[rows[-1]]).total_seconds() / 86400
            for i, rows in enumerate(eligible)]
    values['rest_days'] = rest
    labels = create_labels(history, {'total_corners': MatchTotal(Stat('ALL', 'Match overview', 'cornerKicks'))})
    dataset = assemble_dataset(values, labels['total_corners'], layout='match', drop_missing_targets=True)
    extras = {}
    # Same-match combinations of historical values are known at prediction time.
    for name in values.columns:
        if any(token in name for token in ('_mean5', '_mean10', 'standing', 'glicko', 'rest_days')):
            home, away = dataset.X[f'home::{name}'], dataset.X[f'away::{name}']
            extras[f'sum::{name}'] = home + away
            extras[f'difference::{name}'] = home - away
    for side in ('home', 'away'):
        for group, key in STAT_CHOICES:
            for role in ('for', 'against'):
                stem = f'{side}::ALL_{group}_{key}_{role}_mean'
                if stem + '3' in dataset.X and stem + '20' in dataset.X:
                    extras[f'trend::{stem}3_vs_20'] = dataset.X[stem + '3'] - dataset.X[stem + '20']
    kickoff = dataset.metadata.kickoff_at
    extras['calendar::month_sin'] = np.sin(2 * np.pi * kickoff.dt.month / 12)
    extras['calendar::month_cos'] = np.cos(2 * np.pi * kickoff.dt.month / 12)
    extras['calendar::weekday'] = kickoff.dt.dayofweek.astype(float)
    extras['calendar::round'] = pd.to_numeric(dataset.metadata['round'], errors='coerce')
    dataset.X = pd.concat([dataset.X, pd.DataFrame(extras, index=dataset.X.index)], axis=1).astype('float32')
    dataset.X = dataset.X.replace([np.inf, -np.inf], np.nan)
    split_cutoff = None if cutoff_hours is None else kickoff - pd.Timedelta(hours=cutoff_hours)
    plan = create_split_plan(dataset, TemporalSplit(train_size=len(training_seasons), test_size=1, unit='seasons',
        calendar_by=(), block_by=('source_season',)), cutoffs=split_cutoff)
    if len(plan.folds) != 1:
        raise ValueError('This experiment needs one pooled holdout with labelled matches in every loaded season.')
    if set(dataset.metadata.iloc[plan.folds[0].test].source_season) != {holdout_season}:
        raise ValueError('The chronological split must evaluate the latest loaded season.')
    # League indicators are fitted from training identities only.
    train = plan.folds[0].train
    for league in sorted(dataset.metadata.iloc[train].source_league.unique()):
        dataset.X[f'league::{league}'] = dataset.metadata.source_league.eq(league).astype('float32')
    manifest = pd.DataFrame({'feature': dataset.X.columns,
        'training_nonmissing_fraction': dataset.X.iloc[train].notna().mean().to_numpy(),
        'training_unique_values': dataset.X.iloc[train].nunique().to_numpy()})
    manifest['variant'] = np.where(manifest.feature.str.contains('warm::', regex=False), 'warm', 'baseline')
    manifest['family'] = manifest.feature.map(lambda name: next(
        (family for token, family in (('h2h_', 'h2h'), ('loo_', 'loo'), ('league_', 'league'),
                                     ('glicko', 'rating')) if token in name), 'team_or_context'))
    prepared = PreparedExperiment(dataset, plan, config={
        'seasons': tuple(seasons), 'leagues': leagues, 'windows': windows, 'lags': lags,
        'training_seasons': training_seasons, 'holdout_season': holdout_season,
        'spans': spans, 'periods': periods, 'ratings': include_ratings, 'cutoff_hours': cutoff_hours,
        'population_features': dict(include_loo=include_loo, loo_windows=tuple(loo_windows), loo_reducers=tuple(loo_reducers),
            include_h2h=include_h2h, h2h_windows=tuple(h2h_windows), h2h_reducers=tuple(h2h_reducers)),
        'warm_start': dict(feature_policy=repr(warm_policy), stat_keys=warm_stat_keys,
            rating_policy=repr(rating_warm_policy),
            team_seasons=team_seasons.to_dict('records') if isinstance(team_seasons, pd.DataFrame) else team_seasons,
            season_starts=season_starts.to_dict('records') if isinstance(season_starts, pd.DataFrame) else season_starts),
        'feature_definitions': {key: repr(value) for key, value in definitions.items()},
        'derived': 'home-away sums/differences, short-long trends, rest, calendar, train-league indicators',
    })
    print(f'Prepared {len(dataset.X):,} matches and {dataset.X.shape[1]:,} features in {time.perf_counter()-started:.1f}s.', flush=True)
    return FeatureBundle(prepared, history, definitions, manifest, summary, catalog, time.perf_counter()-started)


def pinball_loss(observed, predicted, alpha=0.5):
    error = np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.maximum(alpha * error, (alpha - 1) * error).mean())


def quantile_diagnostics(observed, predictions, quantiles, baseline_values):
    """Raw quantile calibration, pinball baseline and interval/crossing summaries."""
    y = np.asarray(observed, dtype=float)
    matrix = np.asarray(predictions, dtype=float)
    alphas = np.asarray(quantiles, dtype=float)
    if matrix.shape != (len(y), len(alphas)) or not np.all(np.diff(alphas) > 0):
        raise ValueError('Quantile columns must align with observations and ascending alphas.')
    rows = []
    for i, alpha in enumerate(alphas):
        loss = pinball_loss(y, matrix[:, i], alpha)
        baseline = pinball_loss(y, np.full(len(y), baseline_values[i]), alpha)
        rows.append(dict(quantile=alpha, pinball_loss=loss, baseline_pinball=baseline,
                         skill_vs_baseline=1-loss/baseline if baseline else np.nan,
                         observed_at_or_below=float(np.mean(y <= matrix[:, i]))))
    lower, upper = matrix[:, 0], matrix[:, -1]
    interval = pd.DataFrame([dict(n=len(y), nominal_coverage=float(alphas[-1]-alphas[0]),
        coverage=float(np.mean((y >= lower) & (y <= upper))), mean_width=float(np.mean(upper-lower)),
        quantile_crossing_fraction=float(np.mean(np.any(np.diff(matrix, axis=1) < 0, axis=1))),
        negative_quantile_fraction=float(np.mean(matrix < 0)))])
    return pd.DataFrame(rows), interval


def _model_factory(alpha, options):
    from xgboost import XGBRegressor
    return BoostingAdapter(XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha,
        tree_method='hist', **options), fit_kwargs={'verbose': False})


@dataclass
class QuantileRun:
    results: dict
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    intervals: pd.DataFrame
    by_league: pd.DataFrame
    importance: pd.DataFrame
    output_path: Path


def run_quantiles(bundle, output_dir, *, quantiles=(0.1, 0.5, 0.9), model_options=None,
                  device='cuda:0', threads=4, validation_fraction=0.15, reuse=True):
    """Fit once per quantile using training-tail early stopping; save reports/models.

    No hyperparameter search or test-driven choice. Separate scalar quantiles
    retain the standard pipeline output contract. Predictions are never sorted,
    clipped or rounded: crossing is measured and reported.
    """
    quantiles = tuple(quantiles)
    if not quantiles or any(not 0 < q < 1 for q in quantiles) or any(a >= b for a, b in zip(quantiles, quantiles[1:])):
        raise ValueError('Supply strictly increasing quantiles between zero and one.')
    options = {**DEFAULT_MODEL_OPTIONS, **(model_options or {})}
    register_metric('corners_pinball', pinball_loss, kind='numeric', direction='minimize', replace=True)
    output_dir = Path(output_dir)
    experiment = FootballExperiment('Total corners XGBoost quantiles broad features', output_dir=output_dir)
    results, predictions, importance, fitted_details = {}, None, [], {}
    dataset, fold = bundle.prepared.dataset, bundle.prepared.split_plan.folds[0]
    for alpha in quantiles:
        print(f'Fitting quantile {alpha:g} on {device} (validation from training only)...', flush=True)
        candidate = Candidate(f'XGBoost q={alpha:g}', partial(_model_factory, alpha, options),
            config={'quantile': alpha, 'model_options': options, 'feature_count': dataset.X.shape[1]},
            validation=ValidationTail(fraction=validation_fraction))
        reports = {
            'Errors and pinball': PerformanceReporter(type='overall', partition='score',
                metrics=['mae', 'mse', Metric('corners_pinball', parameters={'alpha': alpha}, key='pinball')]),
            'Learning curves': LearningCurveReporter(type='per_fold'),
            'Match predictions': MatchResultReporter(type='overall', partition='score', comparison='numeric',
                tolerance=2., league_column='source_league', season_column='source_season', catalog=bundle.team_catalog),
        }
        if alpha == 0.5:
            reports['Median distribution'] = PredictionDistributionReporter(type='overall', partition='score')
        before = PreTrainingAnalysis({'Training rank correlations': CorrelationAnalysis(
            type='overall', partition='train', methods=('spearman',), percentile_ranks=True)}) if alpha == 0.5 else None
        result = experiment.run(bundle.prepared, model=candidate, pre_analysis=before,
            post_analysis=PostTrainingAnalysis(reports),
            execution=ExecutionPolicy(n_jobs=1, device=device, inner_threads=threads), reuse=reuse)
        results[alpha] = result
        fitted = result.training.folds[0]
        current = fitted.predictions['predict'].loc[fitted.score_positions].iloc[:, 0]
        if predictions is None:
            predictions = dataset.metadata.loc[current.index].copy()
            predictions['observed_total_corners'] = dataset.y.loc[current.index].iloc[:, 0]
        elif not current.index.equals(predictions.index):
            raise ValueError('Quantile predictions have different row identities/order.')
        predictions[f'q{alpha:g}'] = current
        model_path = Path(result.path) / 'fitted_model'
        if fitted.model is not None and not model_path.exists():
            save_model(result.training, model_path)
        model = fitted.model
        if model is None and model_path.exists():
            model = load_model(model_path).model
        if model is not None:
            native = model._native()
            booster = native.get_booster()
            best_iteration = getattr(native, 'best_iteration', None)
            best_score = getattr(native, 'best_score', None)
            # XGBoost predicts through best_iteration after early stopping;
            # patience trees must not receive credit in the prediction report.
            prediction_rounds = (booster.num_boosted_rounds() if best_iteration is None
                                 else int(best_iteration) + 1)
            prediction_booster = booster if best_iteration is None else booster[:prediction_rounds]
            scores = prediction_booster.get_score(importance_type='gain')
            importance.extend({'quantile': alpha, 'feature': fitted.feature_columns[int(key[1:])], 'gain': value}
                              for key, value in scores.items())
            native_config = json.loads(booster.save_config())
            fitted_details[str(alpha)] = {
                'device': native_config['learner']['generic_param']['device'],
                'best_iteration': None if best_iteration is None else int(best_iteration),
                'best_validation_pinball': None if best_score is None else float(best_score),
                'prediction_rounds': prediction_rounds,
                'gain_scope': 'prediction_trees',
                'fit_rows': len(fitted.fit_positions),
                'validation_rows': len(fitted.validation_positions),
            }
        print(f'Quantile {alpha:g}: {"reused" if result.reused else "complete"}; artifacts {result.path}', flush=True)
    # Same fitting rows as model optimization, excluding the validation tail.
    fit_rows = next(iter(results.values())).training.folds[0].fit_positions
    baseline_values = np.quantile(dataset.y.iloc[fit_rows].iloc[:, 0].astype(float), quantiles)
    columns = [f'q{q:g}' for q in quantiles]
    metrics, intervals = quantile_diagnostics(predictions.observed_total_corners,
        predictions[columns], quantiles, baseline_values)
    league_frames = []
    for league, part in predictions.groupby('source_league'):
        scores, _ = quantile_diagnostics(part.observed_total_corners, part[columns], quantiles, baseline_values)
        scores.insert(0, 'league', league)
        league_frames.append(scores)
    by_league = pd.concat(league_frames, ignore_index=True)
    importance = pd.DataFrame(importance, columns=['quantile', 'feature', 'gain'])
    # Use a run-specific directory; independent trials never overwrite a previous summary.
    output = Path(next(iter(results.values())).path).parent.parent / 'quantile_summaries' / next(iter(results.values())).record['run_id']
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in {'predictions': predictions, 'pinball_metrics': metrics, 'interval_diagnostics': intervals,
                        'per_league_metrics': by_league, 'feature_importance': importance,
                        'feature_manifest': bundle.feature_manifest, 'data_summary': bundle.data_summary}.items():
        frame.to_csv(output / f'{name}.csv', index=False)
    summary = {'quantiles': quantiles, 'device_requested': device,
        'rows': len(dataset.X), 'features': dataset.X.shape[1], 'train_rows': len(fold.train),
        'fit_rows': len(fit_rows), 'test_rows': len(fold.test),
        'model_options': options, 'validation_fraction': validation_fraction,
        'run_paths': {str(q): str(r.path) for q, r in results.items()},
        'baseline_quantiles': baseline_values.tolist(), 'preparation_seconds': bundle.elapsed_seconds,
        'fitted_models': fitted_details}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return QuantileRun(results, predictions, metrics, intervals, by_league, importance, output)
