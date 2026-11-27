"""Portable experiment recipes compiled into the existing public pipeline."""

from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import default_catalog


def node(component, **params):
    return {'component': component, 'params': params}


def default_recipe(data_root='data/xDiyo_data', *, league_timezones=None):
    stat = node('features.Stat', period='ALL', group='Match overview', key='cornerKicks')
    return {
        'version': 1, 'name': 'Corners experiment', 'output_dir': 'experiments', 'config': {}, 'team_badges': None,
        'data': {'data_root': str(data_root), 'seasons': ['22_23', '23_24', '24_25'],
                 'leagues': None, 'tables': ['matches', 'statistics', 'pregame'], 'include_awarded': False},
        **({'league_timezones': league_timezones} if league_timezones is not None else {}),
        'statistics_overlay': None, 'data_selection': None, 'dataset_migration': None, 'native_preprocessing': None, 'stat_selection': None, 'history': {}, 'feature_options': {}, 'cutoff_hours': None,
        'feature_preset': None, 'preset_seasons': None, 'derived_features': {}, 'context_features': {},
        'identity_features': None, 'numeric_features': None,
        'ratings': {}, 'features': {'corners_mean': node('features.RollingMean', source=stat, window=5),
                                    'is_home': node('features.IsHome')},
        'labels': {'corners': node('labels.MatchTotal', source=deepcopy(stat))}, 'target': 'corners',
        'assembly': {'layout': 'match', 'drop_missing_targets': True},
        'split': node('splits.TemporalSplit', train_size=2, test_size=1, unit='seasons'),
        'split_options': {}, 'fold_ids': None,
        'pre_reporters': {}, 'fitted_reporters': {}, 'analysis_options': {},
        'model': node('sklearn.linear_model.Lasso', alpha=0.1, max_iter=5000),
        'preprocessors': [node('sklearn.impute.SimpleImputer', strategy='median'), node('sklearn.preprocessing.StandardScaler')],
        'adapter': None, 'target_transformer': None, 'prediction_methods': ['predict'], 'candidate': {},
        'search': None, 'execution': node('training.ExecutionPolicy'), 'refit': None, 'checkpoint': None,
        'post_reporters': {'performance': node('reporting.PerformanceReporter', type='overall', partition='score', metrics=['mse', 'mae']),
                           'matches': node('reporting.MatchResultReporter', type='overall', partition='score', catalog={'ref': 'team_catalog'}, show_badges=True)},
        'run': {'reuse': True}, 'artifact_export': None,
        'prediction': {'model_path': '', 'rounds': None, 'statuses': ['notstarted'], 'as_of': None},
    }


def read_recipe(path):
    return validate_recipe(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def validate_recipe(recipe):
    if not isinstance(recipe, dict) or recipe.get('version') != 1:
        raise ValueError('Choose a version 1 experiment recipe.')
    if not isinstance(recipe.get('name'), str) or not recipe['name'].strip():
        raise ValueError('Give the experiment a name.')
    unknown = set(recipe) - set(default_recipe()) - {'league_timezones'}
    if recipe.get('dataset_migration'):
        from ..data.catalogue.migration import validate_migration_envelope
        unknown -= validate_migration_envelope(recipe)
    if unknown:
        raise ValueError(f'Unknown recipe sections: {sorted(unknown)}')
    saved_ratings = recipe.get('feature_options', {}).get('ratings') or {}
    duplicate_ratings = set(saved_ratings) & set(recipe.get('ratings', {}))
    if duplicate_ratings:
        raise ValueError(f'Saved and generated rating names must be distinct: {sorted(duplicate_ratings)}')
    from .countdown_migration import reject_obsolete_countdown
    reject_obsolete_countdown(recipe)
    recipe = deepcopy(recipe)
    if recipe.get('league_timezones') is not None:
        from ..data.league_day import league_zones
        recipe['league_timezones'] = league_zones(recipe['league_timezones'])
    if recipe.get('statistics_overlay') is not None:
        from ..data.statistics_overlay import normalize_overlay_selection
        recipe['statistics_overlay'] = normalize_overlay_selection(recipe['statistics_overlay'])
    # Older UI recipes predate the badge toggle. Preserve an explicit opt-out.
    for reporter in [*recipe.get('post_reporters', {}).values(), *recipe.get('pre_reporters', {}).values()]:
        if isinstance(reporter, dict) and reporter.get('component') in ('reporting.MatchResultReporter', 'reporting.BetOutcomeReporter', 'reporting.HeatmapReporter', 'reporting.RatingReporter', 'reporting.EmbeddingReporter', 'reporting.GraphAttentionWeightReporter'):
            params = reporter.setdefault('params', {})
            params.setdefault('show_badges', True)
            params.setdefault('catalog', {'ref': 'team_catalog'})
            if reporter['component'] == 'reporting.BetOutcomeReporter' and params.get('labels') is None:
                params.setdefault('history', {'ref':'history'})
    return recipe


def _context_file(path, column=None, index=None):
    """Explicit CSV/Parquet auxiliary data; no pickles or executable imports."""
    path = Path(path)
    if path.suffix.lower() == '.csv':
        frame = pd.read_csv(path)
    elif path.suffix.lower() == '.parquet':
        frame = pd.read_parquet(path)
    else:
        raise ValueError('Auxiliary tables use CSV or Parquet.')
    if index is not None:
        frame = frame.set_index(index)
    return frame if column is None else frame[column]


def catalog_for_ui():
    catalog = default_catalog()
    from ..reporting import TeamCatalog
    from ..features import TransitionContext, build_team_seasons
    from ..ratings import (
        BayesianModel, BayesianRatingRun, BayesianScoreAdapter,
        BayesianScoreSerializer, RatingRun,
    )
    from ..training import JoblibSerializer
    from .adapters import BoostingAdapter
    from ..features.presets import FeatureBankPreset
    from ..features.contextual import RestDays, CalendarFeature
    from ..features.preparation import IdentityFeatureSpec, NumericFeatures
    from ..features.composition import Abs, Column, Constant, Sum, Product, Difference, Ratio
    from ..experiments.exports import ArtifactExport
    catalog.register('experiments.ArtifactExport', ArtifactExport, category='export')
    for name, constructor in [('Column', Column), ('Constant', Constant), ('Abs', Abs), ('Sum', Sum), ('Product', Product), ('Difference', Difference), ('Ratio', Ratio)]:
        catalog.register('prepared.' + name, constructor, category='derived_feature')
    catalog.register('features.RestDays', RestDays, category='feature')
    catalog.register('context.CalendarFeature', CalendarFeature, category='context_feature')
    catalog.register('preparation.FeatureBankPreset', FeatureBankPreset, category='preparation')
    catalog.register('preparation.IdentityFeatureSpec', IdentityFeatureSpec, category='preparation')
    catalog.register('preparation.NumericFeatures', NumericFeatures, category='preparation')
    catalog.register('input.Table', _context_file, category='input')
    from ..odds import OddsSeries
    catalog.register('input.OddsSeries', OddsSeries, category='odds_input')
    catalog.register('input.TeamCatalog', TeamCatalog.from_json, category='input')
    catalog.register('input.RatingRun', RatingRun.load, category='input')
    from ..ratings import StandingsRun
    catalog.register('input.StandingsRun', StandingsRun.load, category='input')
    catalog.register('input.BayesianModel', BayesianModel.load, category='input',
                     description='Load fixed Bayesian parameter JSON trained before the prediction period. Loading parameters does not fit them on this recipe history.')
    catalog.register('input.BayesianRatingRun', BayesianRatingRun.load, category='input',
                     description='Load a saved Bayesian run, including its full state checkpoint, model and opponent/venue-dependent predictions.')
    catalog.register('ratings.BayesianScoreAdapter', BayesianScoreAdapter, category='model',
                     title='Bayesian goal-score rating',
                     description='Fit fixed Bayesian rating parameters using the fitting partition only, with pooled or separate league parameters. Requires match-layout metadata and the observed home and away goal targets. No feature or target preprocessing is supported.')
    catalog.register('training.BayesianScoreSerializer', BayesianScoreSerializer, category='training')
    from ..graphs import TemporalGraphModel, TemporalGraphConfig, TemporalGraphSerializer
    catalog.register('graphs.TemporalGraphModel', TemporalGraphModel, category='model', title='Temporal team GAT')
    catalog.register('graphs.TemporalGraphConfig', TemporalGraphConfig, category='graph_config')
    from ..graphs import GATResearchBNV1, GATResearchCE1V1, GATResearchBCDDWV1, RetainedGATTraining, RetainedGATExecution
    catalog.register("graphs.RetainedGATTraining", RetainedGATTraining, category="graph_config")
    catalog.register("graphs.RetainedGATExecution", RetainedGATExecution, category="graph_config")
    for factory in (GATResearchBNV1, GATResearchCE1V1, GATResearchBCDDWV1):
        catalog.register('graphs.' + factory.__name__, factory, category='graph_config')
    catalog.register('graphs.TemporalGraphSerializer', TemporalGraphSerializer, category='training')
    from ..graphs import GraphHeatmaps
    from ..neural import HeatmapCNNModel, HeatmapCNNConfig, Reconstruction, HeatmapCNNSerializer
    catalog.register('graphs.GraphHeatmaps', GraphHeatmaps, category='graph_config')
    from ..graphs import BoundaryMemory, StatisticTarget, NumericStatisticTarget, StatisticsForecast, GoalAuxiliary, PhaseRankFeatures
    for cls in (BoundaryMemory, StatisticTarget, NumericStatisticTarget, StatisticsForecast, GoalAuxiliary, PhaseRankFeatures):
        catalog.register('graphs.' + cls.__name__, cls, category='graph_config')
    catalog.register('neural.HeatmapCNNModel', HeatmapCNNModel, category='model', title='Heatmap CNN')
    catalog.register('neural.HeatmapCNNConfig', HeatmapCNNConfig, category='neural_config')
    catalog.register('neural.Reconstruction', Reconstruction, category='neural_config')
    catalog.register('neural.HeatmapCNNSerializer', HeatmapCNNSerializer, category='training')
    catalog.register('features.TransitionContext', TransitionContext, category='input')
    catalog.register('features.TeamSeasons', build_team_seasons, category='input')
    catalog.register('training.JoblibSerializer', JoblibSerializer, category='training')
    catalog.register('training.BoostingAdapter', BoostingAdapter, category='training')
    from .. import composition as comp
    from ..composition.persistence import CompositeSerializer
    for name in ('Estimator', 'OutputSchema', 'OutputRef', 'TargetSpec', 'ModelNode', 'TransformNode', 'ReducerNode',
                 'PredictionGraphSpec', 'TrainingPlan', 'Mean', 'HardVote', 'DistributionMixture', 'OutputFeatures', 'UtilityTarget', 'SavedUtilityModel'):
        catalog.register('composition.' + name, getattr(comp, name), category='composition')
    for name in ('CompositeModelAdapter', 'ModelEnsemble', 'ModelStack', 'ResidualModel', 'LearnedTargetAdapter'):
        catalog.register('composition.' + name, getattr(comp, name), category='model',
                         description='Native composition. Put preprocessing and calibration inside each child; disable top-level preprocessors and adapter overrides.')
    catalog.register('composition.CompositeSerializer', CompositeSerializer, category='training')
    from ..evaluation import stake_policy as stakes
    from ..evaluation.decision_layer import DecisionLayer, LearnedGate
    for name in ('FixedStake', 'FixedFraction', 'FractionalKelly', 'ModelProbabilitySource', 'HistoricalRateSource', 'RiskLimits', 'StakeContext', 'LearnedAllocation'):
        catalog.register('evaluation.' + name, getattr(stakes,name), category='allocation')
    catalog.register('evaluation.DecisionLayer', DecisionLayer, category='decision')
    from ..evaluation.quote_availability import QuoteAvailability
    catalog.register('evaluation.QuoteAvailability', QuoteAvailability, category='decision')
    catalog.register('evaluation.LearnedGate', LearnedGate, category='decision')
    from ..reporting.composition import CompositionReporter
    catalog.register('reporting.CompositionReporter', CompositionReporter, category='post_reporter')
    from ..weighting import ClassWeightPolicy
    catalog.register('weighting.ClassWeightPolicy', ClassWeightPolicy, category='weighting')
    return catalog


def _recipe_split_plan(dataset, splitter, options, cutoff_hours, league_timezones=None):
    from ..splits import create_split_plan, TemporalSplit
    options = dict(options)
    if isinstance(splitter, TemporalSplit) and cutoff_hours is not None and 'cutoffs' not in options:
        # Assembly and search subsets can pivot or drop history rows. Use their
        # aligned timestamps; explicit cutoffs (including None) take precedence.
        options['cutoffs'] = dataset.metadata.kickoff_at - pd.Timedelta(hours=float(cutoff_hours))
    if isinstance(splitter, TemporalSplit) and league_timezones is not None:
        from ..splits.core import _times
        options['league_timezones'] = league_timezones
        current = dataset.metadata['league_day_issue_at']
        if options.get('cutoffs') is not None:
            requested = _times(dataset, options['cutoffs'], 'cutoffs')
            if requested.isna().any() or (requested > dataset.metadata.kickoff_at).any():
                raise ValueError('Invalid daily fold issuance cutoff')
            current = current.mask(requested.lt(current), requested)
        options['cutoffs'] = current
    return create_split_plan(dataset, splitter, **options)


def _verify_overlay_selection(recipe, definitions):
    """Reuse prepared data and fitted artifacts only with matching overlay semantics."""
    selected = recipe.get('statistics_overlay')
    saved = definitions.get('statistics_overlay')
    if selected is None and saved is None:
        return
    if (selected is None or saved is None
            or saved.get('mode', 'retrospective') != 'retrospective' or saved.get('as_of') is not None
            or selected['manifest_sha256'] != saved.get('manifest_sha256')
            or sorted(selected['keys']) != saved.get('applied_keys')):
        raise ValueError('Statistics overlay selection differs from the prepared dataset or fitted model')


def prepare_recipe(recipe, *, catalog=None, prediction=False, preparation_state=None):
    from ..data import load_seasons, select_stats
    from ..histories import build_team_history
    from ..features import evaluate_features
    from ..ratings import build_ratings
    from ..labels import create_labels
    from ..datasets import assemble_dataset
    from ..splits import SplitPlan
    from ..features.composition import combine_features
    from ..features.preparation import common_training_rows, add_identity_features
    from ..experiments.football import PreparedExperiment
    from ..reporting import TeamCatalog
    recipe = validate_recipe(recipe)
    catalog = catalog or catalog_for_ui()
    if prediction and preparation_state is not None:
        _verify_overlay_selection(recipe, preparation_state)
    zones = recipe.get('league_timezones')
    if prediction and preparation_state is not None and zones != preparation_state.get('league_timezones'):
        raise ValueError('League calendar differs from fitted model')
    named_data = None
    if recipe.get('data_selection'):
        if prediction: raise ValueError('Named historical selections are not live future-fixture inputs')
        from ..data.catalogue.integration import named_recipe_data
        named_data, named_history = named_recipe_data(recipe)
    data_options = catalog.build(recipe['data'])
    native_query = data_options.get('query') is not None or data_options.get('native_selection') is not None
    overlay = recipe.get('statistics_overlay')
    if overlay is not None:
        if named_data is not None:
            raise ValueError('Statistics backfill requires native SofaScore publications')
        tables = data_options.get('tables', ['matches'])
        tables = [tables] if isinstance(tables, str) else tables
        data_options['tables'] = list(dict.fromkeys([*tables, 'matches', 'statistics']))
        data_options['allow_missing_statistics'] = True
    native_report = None
    if recipe.get('native_preprocessing') is not None and not native_query:
        raise ValueError('Native preprocessing requires an explicit SofaScore query/selection; legacy recipes remain unchanged')
    if native_query:
        if named_data is not None:
            raise ValueError('Choose native query or named dataset selection, not both')
        tables = data_options.get('tables', ['matches'])
        data_options['tables'] = list(dict.fromkeys([*([tables] if isinstance(tables,str) else tables),'matches','team_seasons']))
    native_library = None
    if native_query:
        from ..data import NativeLibrary
        native_library = NativeLibrary(data_options['data_root'])
    def load_recipe_data(**selected_options):
        if native_library is None:
            return load_seasons(**selected_options)
        selected_options=dict(selected_options)
        selected_options.pop('data_root')
        if selected_options.pop('seasons',None) is not None or selected_options.pop('leagues',None) is not None:
            raise ValueError('Choose native query or legacy seasons/leagues, not both')
        selected_options['selection']=selected_options.pop('native_selection',None)
        return native_library.load(**selected_options)
    def needs_heatmaps(value):
        if isinstance(value, dict):
            return value.get('component') in ('features.Heatmap', 'features.SpatialPointSummary', 'features.SpatialHistoricalDeviation', 'features.SpatialFixtureDistance') or any(needs_heatmaps(v) for v in value.values())
        return isinstance(value, list) and any(needs_heatmaps(v) for v in value)
    if needs_heatmaps(recipe.get('features', {})) and data_options.get('tables') is not None:
        tables = data_options['tables']
        tables = [tables] if isinstance(tables, str) else tables
        data_options['tables'] = list(dict.fromkeys([*tables, 'heatmap_points']))
    graph_spec = None
    if recipe['model'].get('component') == 'graphs.TemporalGraphModel':
        graph_spec = catalog.build(recipe['model'])
        if graph_spec.config.heatmaps is not None and data_options.get('tables') is not None:
            tables = data_options['tables']
            tables = [tables] if isinstance(tables, str) else list(tables)
            data_options['tables'] = list(dict.fromkeys([*tables, 'heatmap_points']))
        if graph_spec.memberships_path is None and data_options.get('tables') is not None:
            tables = data_options['tables']
            tables = [tables] if isinstance(tables, str) else list(tables)
            data_options['tables'] = list(dict.fromkeys([*tables, 'team_seasons']))
        if recipe.get('cutoff_hours') is not None or recipe.get('feature_options', {}).get('cutoffs') is not None:
            raise ValueError('Temporal GAT supplies its own previous-day feature cutoffs; disable cutoff overrides.')
        if recipe['assembly'].get('layout', 'match') != 'match':
            raise ValueError('Temporal GAT requires match layout.')
        label = recipe['labels'].get(recipe['target'], {})
        if label.get('component') != 'labels.Outcome' or label.get('params', {}).get('perspective') != 'home' or label.get('params', {}).get('source') is not None:
            raise ValueError('Temporal GAT requires labels.Outcome(perspective="home") from match goals.')
    def needs_standings(value):
        if isinstance(value, dict):
            return value.get('component') == 'features.DerivedStandings' or any(needs_standings(v) for v in value.values())
        return isinstance(value, list) and any(needs_standings(v) for v in value)
    derived_standings = needs_standings(recipe.get('features', {}))
    standings_source = None
    saved_standings_options = recipe.get('feature_options', {})
    needs_standings_source = (derived_standings and not saved_standings_options.get('standings_run')
                              and saved_standings_options.get('standings_history') is None)
    if needs_standings_source:
        # Standings must see the full result population even when the modelling
        # population excludes awarded games or lacks statistics/odds.
        standings_options = {**data_options, 'tables': ['matches', 'team_seasons'], 'include_awarded': True}
        standings_data = load_recipe_data(**standings_options)
        standings_source = build_team_history(standings_data)
    def countdown_nodes(value):
        if isinstance(value, dict):
            if value.get('component') == 'features.SeasonCountdown':
                yield value.get('params', {})
            for child in value.values():
                yield from countdown_nodes(child)
        elif isinstance(value, list):
            for child in value:
                yield from countdown_nodes(child)
    countdown_data = None
    countdown_specs = list(countdown_nodes([recipe.get('features', {}), recipe.get('context_features', {})]))
    if countdown_specs:
        tables = ['matches']
        if any(p.get('total_rounds') is None and p.get('team_count') is None
               and p.get('format') not in ('scotland', 'japan_transition') for p in countdown_specs):
            tables.append('team_seasons')
        countdown_data = load_recipe_data(**{**data_options, 'tables': tables, 'include_awarded': True})
    data = named_data if named_data is not None else load_recipe_data(**data_options)
    if overlay is not None:
        from ..data.statistics_overlay import backfill_statistics
        overlay_options = dict(overlay)
        overlay_keys = overlay_options.pop('keys')
        data, overlay_receipt = backfill_statistics(data, overlay_options, keys=overlay_keys)
    else:
        overlay_receipt = None
    if native_query:
        from ..data.native_preflight import preflight_loaded, validate_preprocessing
        native_settings = validate_preprocessing(recipe.get('native_preprocessing'))
        def uses_movement(value):
            if isinstance(value,dict):
                return value.get('component') in ('features.WasPromoted','features.WasRelegated','features.TeamMovement') or any(uses_movement(v) for v in value.values())
            return isinstance(value,list) and any(uses_movement(v) for v in value)
        required = (['countdown'] if countdown_specs else []) + (['standings'] if derived_standings else []) + (['movements'] if uses_movement(recipe.get('features',{})) else [])
        native_report = preflight_loaded(data, required_features=required, options=native_settings)
        failures = [(r['publication'],r['phase_id'],r['blockers']) for r in native_report['cohorts'] if r['blockers']]
        if failures:
            raise ValueError(f'Native preprocessing preflight blocked: {failures}. Use NativeLibrary.preflight for the full report.')
    heatmaps = data.tables.get('heatmap_points')
    if heatmaps is not None:
        heatmaps = heatmaps.copy(deep=False)
        heatmaps.attrs['source'] = data.provenance
    if recipe.get('stat_selection') is not None:
        data = select_stats(data, **catalog.build(recipe['stat_selection']))
    history = named_history if named_data is not None else build_team_history(data, **catalog.build(recipe.get('history', {})))
    if native_query:
        from ..data.native_preflight import prepare_native_history
        history = prepare_native_history(data, history, native_settings)
        if standings_source is not None:
            standings_source = prepare_native_history(standings_data, standings_source, native_settings)
    names = data.matches.copy(deep=False)
    for column in ('home_name', 'away_name'):
        if column not in names:
            names = names.assign(**{column: pd.NA})
        display = column.removesuffix('_name') + '_display_name'
        if display in names:
            names = names.assign(**{column: names[display]})
    team_catalog = TeamCatalog.from_matches(names)
    if recipe.get('team_badges'):
        badge_path = Path(recipe['team_badges'])
        saved_catalog = TeamCatalog.from_json(badge_path) if badge_path.is_file() else TeamCatalog({})
    else:
        saved_catalog = TeamCatalog.builtin()
    team_catalog = TeamCatalog({key: {**saved_catalog.entries.get(key, {}), **entry}
                                for key, entry in team_catalog.entries.items()})
    context = {'history': history, 'matches': data.matches, 'team_catalog': team_catalog}
    options = catalog.build(recipe.get('feature_options', {}), context)
    if native_query and native_settings['availability']=='explicit':
        if options.get('available_at') not in (None,'result_available_at'):
            raise ValueError('Explicit native availability must use result_available_at evidence')
        options['available_at']='result_available_at'
    elif native_query:
        # Explicitly selected retrospective convention, still bounded by every
        # reviewed completion/publication time in availability_times.
        options.setdefault('available_at','kickoff_at')
    if named_data is not None:
        options.update(cutoffs=history.kickoff_at, available_at=history.result_available_at)
    if countdown_data is not None:
        options.setdefault('schedule_history', countdown_data.matches)
        if 'team_seasons' in countdown_data.tables:
            options.setdefault('team_seasons', countdown_data['team_seasons'])
    if needs_standings_source:
        options.setdefault('standings_history', standings_source)
        options.setdefault('team_seasons', standings_data['team_seasons'])
    if derived_standings and options.get('standings_history') is not None:
        if options.get('available_at') is not None and options.get('standings_available_at') is None:
            if not isinstance(options['available_at'], str):
                raise ValueError('With separate standings history, supply standings_available_at for full-population release times.')
            options['standings_available_at'] = options['available_at']
    if heatmaps is not None:
        options.setdefault('heatmaps', heatmaps)
    if 'team_seasons' in data.tables:
        options.setdefault('team_seasons', data['team_seasons'])
    hours = recipe.get('cutoff_hours')
    if hours is not None:
        if options.get('cutoffs') is not None:
            raise ValueError('Choose either cutoff hours or explicit cutoffs.')
        options['cutoffs'] = history.kickoff_at - pd.Timedelta(hours=float(hours))
    ratings = {}
    for name, spec in recipe.get('ratings', {}).items():
        rating_options = catalog.build(spec, context)
        if native_query and native_settings['availability']=='explicit':
            if rating_options.get('available_at') not in (None,'result_available_at'):
                raise ValueError('Native ratings must respect explicit result availability')
            rating_options['available_at']='result_available_at'
        if 'team_seasons' in data.tables:
            rating_options.setdefault('team_seasons', data['team_seasons'])
        ratings[name] = build_ratings(history, **rating_options)
    if ratings:
        options['ratings'] = {**options.get('ratings', {}), **ratings}
    if options.get('keyed') is False:
        raise ValueError('Assembly needs keyed features; leave keyed=True.')
    options['keyed'] = True
    graph_store = None
    if graph_spec is not None:
        from ..graphs.integration import history_cutoffs
        history = graph_spec.prepare_history(history)
        if native_query and graph_spec.config.forecast_policy=='by_round':
            from ..data.native_preflight import validate_round_groups
            validate_round_groups(history,graph_spec.config.round_keys,'GAT issuance')
        graph_store = graph_spec.event_store(history, data.tables.get('team_seasons'), heatmaps=heatmaps)
        options['cutoffs'] = history_cutoffs(graph_store, graph_spec.config, history)
        context['graph_store'] = graph_store
    labels = create_labels(history, catalog.build(recipe['labels']))
    if zones is not None:
        from ..features.history import aligned_times
        options['league_timezones'] = zones
        issued = aligned_times(history, options.get('cutoffs'), default='kickoff_at')
        if issued.isna().any() or (issued > history.kickoff_at).any():
            raise ValueError('Invalid daily recipe issuance cutoff')
        ids = ['competition_id', 'season_id', 'event_id']
        if issued.groupby([history[k] for k in ids],sort=False).nunique().gt(1).any():
            raise ValueError('Daily recipe needs one issuance per fixture')
        for label in labels.values():
            label.metadata['league_day_issue_at'] = issued.loc[label.metadata.index]

    if native_query and native_settings['entry_information']=='explicit':
        from ..data.native_preflight import validate_entry_cutoffs
        validate_entry_cutoffs(history,data['team_seasons'],options.get('cutoffs'))
    assembly = catalog.build(recipe['assembly'])
    if prediction:
        assembly['drop_missing_targets'] = False
    def make_plan(frame):
        if named_data is not None:
            from ..data.catalogue.integration import frozen_split
            frozen = frozen_split(frame, data.provenance['data_selection'])
            if frozen is not None:return frozen
        splitter = catalog.build(recipe['split'])
        if native_query:
            from ..splits import TemporalSplit
            from ..data.native_preflight import validate_round_groups
            from dataclasses import replace
            if isinstance(splitter,TemporalSplit) and splitter.unit=='rounds':
                if splitter.block_by is None:
                    splitter=replace(splitter,block_by=('season_id','countdown_calendar_id','round'))
                validate_round_groups(history,splitter.block_by,'TemporalSplit')
            if isinstance(splitter,TemporalSplit) and splitter.gap_unit=='rounds' and splitter.unit!='rounds':
                validate_round_groups(history,('season_id','round'),'TemporalSplit round gap')
        split_options = catalog.build(recipe.get('split_options', {}), {**context, 'dataset': frame})
        if native_query and native_settings['availability']=='explicit':
            from ..splits import TemporalSplit
            if not isinstance(splitter,TemporalSplit):
                raise ValueError('Explicit native availability requires TemporalSplit label-availability validation')
            if split_options.get('available_at') not in (None,'result_available_at'):
                raise ValueError('Native training labels must use result_available_at evidence')
            split_options['available_at']='result_available_at'
        plan = (splitter if isinstance(splitter, SplitPlan) else
                _recipe_split_plan(frame, splitter, split_options, hours, zones))
        if recipe.get('fold_ids') is not None:
            plan = SplitPlan([plan.folds[int(i)] for i in recipe['fold_ids']], plan.n_rows, plan.row_order, paths=None)
        if named_data is not None:
            from ..data.catalogue.integration import validate_selected_plan
            validate_selected_plan(frame,plan)
        return plan
    definitions = catalog.build(recipe['features'])
    preset = catalog.build(recipe.get('feature_preset'))
    identities = None
    if preset is not None:
        from ..features import IsHome
        from ..features.contextual import RestDays
        saved = (preparation_state or {}).get('feature_preset')
        if prediction and saved is not None:
            identities = saved['stat_identities']
        else:
            stats = data.statistics
            seasons = recipe.get('preset_seasons')
            if seasons is not None:
                if not seasons:
                    raise ValueError('Choose at least one preset discovery season, or disable the override.')
                stats = stats.loc[stats.source_season.isin(seasons)]
            else:
                probe = assemble_dataset(evaluate_features(history, {'probe': IsHome()}, **options), labels[recipe['target']], **assembly)
                rows = common_training_rows(make_plan(probe))
                match_ids = probe.metadata.iloc[rows].event_id
                stats = stats.loc[stats.event_id.isin(match_ids)]
            group = 'group_name' if 'group_name' in stats else 'group'
            identities = list(stats[['period', group, 'key']].drop_duplicates().itertuples(index=False, name=None))
        bank = preset.build(identities)
        if preset.include_rest:
            bank['rest_days'] = RestDays()
        overlap = set(bank).intersection(definitions)
        if overlap:
            raise ValueError(f'Explicit feature names duplicate the preset: {sorted(overlap)}')
        definitions = {**bank, **definitions}
    retained_ratings = {}
    features = evaluate_features(history, definitions, rating_runs=retained_ratings, **options)
    dataset = assemble_dataset(features, labels[recipe['target']], **assembly)
    if native_query:
        from ..data.native_preflight import sporting_mask
        keys=['competition_id','season_id','event_id']
        eligible=data.matches.native_target.copy()
        if not prediction:
            eligible &= sporting_mask(data.matches)
        targets=set(data.matches.loc[eligible,keys].itertuples(index=False,name=None))
        selected=np.array([k in targets for k in dataset.metadata[keys].itertuples(index=False,name=None)])
        dataset.X=dataset.X.loc[selected].reset_index(drop=True)
        dataset.y=dataset.y.loc[selected].reset_index(drop=True)
        dataset.metadata=dataset.metadata.loc[selected].reset_index(drop=True)
        dataset.definitions['native_preprocessing']=native_report
    if zones is not None:
        dataset.definitions['league_timezones'] = zones
    if overlay_receipt is not None:
        dataset.definitions['statistics_overlay'] = overlay_receipt
    if named_data is not None:
        from ..data.catalogue.integration import select_targets
        dataset = select_targets(dataset, data.provenance['data_selection'])
    if preset is not None:
        dataset.definitions['feature_preset'] = {'stat_identities': identities, 'configuration': recipe['feature_preset']}
        if preset.include_combinations:
            dataset.X = combine_features(dataset.X, preset.postassembly_definitions(
                features.columns, dataset.X.columns,
                feature_scopes=features.attrs.get('feature_scopes', {})))
    if recipe.get('derived_features'):
        dataset.X = combine_features(dataset.X, catalog.build(recipe['derived_features']))
    calendar = preset.calendar_definitions() if preset is not None and preset.include_calendar else {}
    calendar.update(catalog.build(recipe.get('context_features', {})))
    if calendar:
        from ..features.contextual import evaluate_context_features
        context_cutoffs = None
        if countdown_specs:
            from ..features.history import aligned_times
            keys = ['competition_id', 'season_id', 'event_id']
            if 'team_id' in dataset.metadata:
                keys.append('team_id')
            timing = history[keys].copy()
            timing['_context_cutoff'] = aligned_times(history, options.get('cutoffs'), default='kickoff_at').to_numpy()
            if timing.groupby(keys, dropna=False)._context_cutoff.nunique(dropna=False).gt(1).any():
                raise ValueError('Context countdown requires matching fixture prediction cutoffs.')
            joined = dataset.metadata[keys].merge(timing.drop_duplicates(keys), on=keys, how='left', validate='many_to_one', sort=False)
            context_cutoffs = pd.Series(joined._context_cutoff.to_numpy(), index=dataset.metadata.index)
        extra = evaluate_context_features(dataset.metadata, calendar, schedule_history=options.get('schedule_history'), team_seasons=options.get('team_seasons'), cutoffs=context_cutoffs)
        if set(extra).intersection(dataset.X):
            raise ValueError('Context features must have distinct names from existing inputs.')
        dataset.X = pd.concat([dataset.X, extra], axis=1)
    numeric = catalog.build(recipe.get('numeric_features'))
    if numeric is not None:
        dataset.X = numeric.transform(dataset.X)
        dataset.definitions['numeric_features'] = recipe['numeric_features']
    identity_spec = catalog.build(recipe.get('identity_features'))
    if identity_spec is not None:
        state = (preparation_state or {}).get('identity_features') if prediction else None
        if state is None:
            if prediction:
                raise ValueError('Future identity features need the fitted model vocabulary; use predict_recipe or pass preparation_state.')
            rows = common_training_rows(make_plan(dataset))
            state = identity_spec.fit(dataset.metadata.iloc[rows])
        dataset = add_identity_features(dataset, state)
    if graph_spec is not None:
        from ..graphs.integration import fixture_cutoffs
        if graph_spec.round_stage_column is not None:
            from ..graphs.config import MATCH_KEYS
            schedule = graph_store.schedule.set_index(list(MATCH_KEYS))
            keys = pd.MultiIndex.from_frame(dataset.metadata[list(MATCH_KEYS)])
            dataset.metadata['round_stage'] = schedule.loc[keys, 'round_stage'].to_numpy()
        dataset.metadata['feature_cutoff_at'] = fixture_cutoffs(graph_store, graph_spec.config, dataset.metadata)
    context.update(features=features, labels=labels, dataset=dataset)
    if prediction:
        return data, dataset, context
    plan = make_plan(dataset)
    return PreparedExperiment(dataset, plan, {'history': history, 'matches': data.matches, 'features': features,
                                            'labels': labels, 'ratings': retained_ratings, 'team_catalog': team_catalog,
                                            **({'native_preflight': native_report} if native_report is not None else {}),
                                            'rating_cutoffs': options.get('cutoffs'), 'graph_inputs': None if graph_store is None else dict(history=graph_store.history, team_seasons=graph_store.team_seasons, schedule=graph_store.schedule, provenance=graph_store.provenance,
                                                **({"phase_context_state": graph_store.phase_context.state()} if graph_store.phase_context is not None else {}))},
                              config={'recipe': recipe})


@dataclass
class ModelFactory:
    """Fresh estimator, preprocessing and adapter per fit; process serializable."""
    recipe: dict
    catalog: object
    graph_store: object = None

    def cache_key(self):
        """Only configuration consumed when constructing a fitted model."""
        return {"catalog": self.catalog, "model": self.recipe["model"],
                "graph_store": getattr(self.graph_store, "version", None),
                "preprocessors": self.recipe.get("preprocessors", []),
                "adapter": self.recipe.get("adapter"),
                "target_transformer": self.recipe.get("target_transformer"),
                "calibration": self.recipe.get("candidate", {}).get("calibration"),
                "prediction_methods": self.recipe.get("prediction_methods", ["predict"])}

    def __call__(self):
        from sklearn.pipeline import Pipeline
        from ..ratings import BayesianScoreAdapter
        from ..training import EstimatorAdapter, TargetTransformAdapter
        estimator = self.catalog.build(self.recipe['model'])
        from ..neural import HeatmapCNNModel
        if isinstance(estimator, HeatmapCNNModel):
            if self.recipe.get('preprocessors') or self.recipe.get('adapter') is not None or self.recipe.get('target_transformer') is not None:
                raise ValueError('Heatmap CNN owns whole-map missingness and numeric scaling; disable top-level preprocessing, adapter and target scaling.')
            return estimator.build()
        from ..graphs import TemporalGraphModel
        if isinstance(estimator, TemporalGraphModel):
            if self.graph_store is None:
                raise ValueError('Prepare the graph recipe before constructing its model.')
            if self.recipe.get('preprocessors') or self.recipe.get('adapter') is not None or self.recipe.get('target_transformer') is not None:
                raise ValueError('Temporal GAT owns masked input scaling; disable top-level preprocessing, adapter and target scaling.')
            return estimator.build(self.graph_store)
        from ..composition import CompositeModelAdapter, ModelEnsemble, ModelStack, ResidualModel, LearnedTargetAdapter
        if isinstance(estimator, (CompositeModelAdapter, ModelEnsemble, ModelStack, ResidualModel, LearnedTargetAdapter)):
            if self.recipe.get('preprocessors') or self.recipe.get('adapter') is not None or self.recipe.get('target_transformer') is not None:
                raise ValueError('Composition owns child preprocessing/adapters/target transforms; disable top-level wrappers.')
            if self.recipe.get('candidate', {}).get('calibration') is not None:
                raise ValueError('Configure calibration inside composition children.')
            built = estimator.build() if hasattr(estimator, 'build') else estimator
            if hasattr(built, 'graph'):
                built.graph.order()
            return built
        if isinstance(estimator, BayesianScoreAdapter):
            if self.recipe.get('preprocessors') or self.recipe.get('target_transformer') is not None:
                raise ValueError('Bayesian goal-score training requires untransformed scores and match metadata; disable preprocessors and target_transformer.')
            if self.recipe.get('adapter') is not None:
                raise ValueError('BayesianScoreAdapter is already a training adapter; disable the adapter override.')
            return estimator
        native_estimator = estimator
        steps = [(f'prepare_{i}', self.catalog.build(spec)) for i, spec in enumerate(self.recipe.get('preprocessors', []))]
        if steps:
            estimator = Pipeline([*steps, ('model', estimator)])
        adapter = self.recipe.get('adapter')
        methods = list(self.recipe.get('prediction_methods', ['predict']))
        calibration = self.recipe.get('candidate', {}).get('calibration')
        response = self.catalog.build(calibration).response_method if calibration is not None else None
        if response == 'decision_function':
            if self.recipe.get('target_transformer') is not None:
                raise ValueError('Decision-margin calibration needs original discrete labels; disable label scaling.')
            from sklearn.svm import SVC
            if not isinstance(native_estimator, SVC) or native_estimator.probability not in (False, 'deprecated'):
                raise ValueError('Decision-margin calibration requires SVC(probability=False).')
            if 'predict_proba' in methods:
                raise ValueError('Do not request base predict_proba in decision-margin calibration; calibrated probabilities are returned automatically.')
            if response not in methods:
                methods.append(response)
        elif (calibration is not None or callable(getattr(estimator, 'predict_proba', None))) and 'predict_proba' not in methods:
            methods.append('predict_proba')
        if adapter is not None:
            built = self.catalog.build(adapter, {'estimator': estimator, 'native_estimator': native_estimator,
                                                 'preprocessor': Pipeline(steps) if steps else None})
        elif self.recipe['model']['component'].startswith(('lightgbm.', 'xgboost.', 'catboost.')):
            from .adapters import BoostingAdapter
            built = BoostingAdapter(estimator, tuple(methods))
        else:
            built = EstimatorAdapter(estimator, tuple(methods))
        if 'predict_proba' in methods and adapter is not None and hasattr(built, 'prediction_methods'):
            built.prediction_methods = tuple(dict.fromkeys((*built.prediction_methods, 'predict_proba')))
        if response == 'decision_function' and adapter is not None and hasattr(built, 'prediction_methods'):
            built.prediction_methods = tuple(dict.fromkeys((*built.prediction_methods, response)))
        target_transformer = self.recipe.get('target_transformer')
        if target_transformer is not None:
            built = TargetTransformAdapter(built, self.catalog.build(target_transformer))
        return built


def make_candidate(recipe, catalog, context=None):
    from ..selection import Candidate
    from ..analysis import PreTrainingAnalysis
    config = catalog.build(recipe.get('candidate', {}), context)
    config.setdefault('name', recipe['model']['component'].split('.')[-1])
    parameters = recipe['model'].get('params', {})
    display = {k: v for k, v in parameters.items() if v is None or isinstance(v, (str, bool, int, float))}
    config.setdefault('config', {'model': recipe['model'], 'preprocessors': recipe.get('preprocessors', []),
                                 'target_transformer': recipe.get('target_transformer'), **display})
    if recipe.get('fitted_reporters'):
        config['pre_analysis'] = PreTrainingAnalysis(catalog.build(recipe['fitted_reporters'], context),
                                                    **catalog.build(recipe.get('analysis_options', {}).get('fitted', {})))
    graph_store = (context or {}).get('graph_store')
    if graph_store is None and (context or {}).get('graph_inputs') is not None:
        from ..graphs import TemporalEventStore
        inputs = dict(context['graph_inputs'])
        if 'phase_context_state' in inputs:
            from ..graphs import PhaseRankLookup
            inputs['phase_context'] = PhaseRankLookup.from_state(inputs.pop('phase_context_state'))
        graph_store = TemporalEventStore(**inputs)
    return Candidate(model_factory=ModelFactory(recipe, catalog, graph_store), **config)


@dataclass
class InnerPlan:
    spec: dict
    options: dict
    catalog: object
    cutoff_hours: object = None
    league_timezones: object = None

    def __call__(self, dataset):
        return _recipe_split_plan(dataset, self.catalog.build(self.spec), self.catalog.build(self.options), self.cutoff_hours, self.league_timezones)


def _grid_candidates(recipe, search, catalog, context):
    from sklearn.model_selection import ParameterGrid
    candidates = []
    base = search.get('candidates') or [recipe]
    for candidate_recipe in base:
        allowed = {'model', 'preprocessors', 'adapter', 'target_transformer', 'prediction_methods', 'candidate', 'fitted_reporters'}
        if candidate_recipe is not recipe and set(candidate_recipe) - allowed:
            raise ValueError('Candidate overrides configure model, preprocessors, adapter, target_transformer, prediction_methods, candidate or fitted_reporters.')
        candidate_recipe = {**deepcopy(recipe), **deepcopy(candidate_recipe)}
        for combination in ParameterGrid(search.get('grid') or {}):
            current = deepcopy(candidate_recipe)
            for path, value in combination.items():
                # Friendly alpha aliases model.params.alpha; full paths tune candidate settings.
                parts = path.split('.') if '.' in path else ['model', 'params', path]
                if parts[0] not in allowed:
                    raise ValueError(f'Grid path {path!r} is not a candidate setting; preparation is shared across candidates.')
                cursor = current
                for part in parts[:-1]:
                    if isinstance(cursor, list):
                        cursor = cursor[int(part)]
                    else:
                        cursor = cursor.setdefault(part, {})
                cursor[int(parts[-1]) if isinstance(cursor, list) else parts[-1]] = value
            item = make_candidate(current, catalog, context)
            item.config['grid_parameters'] = combination
            if combination:
                item.name += ' · ' + ' · '.join(f'{k}={v}' for k, v in combination.items())
            candidates.append(item)
    return candidates


def _preparation_settings(recipe, catalog):
    fields = ('data', 'data_selection', 'dataset_migration', 'native_preprocessing', 'league_timezones',
              'stat_selection', 'history', 'feature_options', 'cutoff_hours', 'feature_preset',
              'preset_seasons', 'derived_features', 'context_features', 'identity_features',
              'numeric_features', 'ratings', 'features', 'labels', 'target', 'assembly',
              'split', 'split_options', 'fold_ids', 'team_badges')
    settings = {name: recipe.get(name) for name in fields}
    # Mapping order determines the public feature matrix column order.
    for name in ('features', 'derived_features', 'context_features'):
        settings[name] = tuple((recipe.get(name) or {}).items())
    if recipe['model'].get('component') == 'graphs.TemporalGraphModel':
        from ..graphs.spatial import grid_identity
        graph = catalog.build(recipe['model'])
        settings['graph inputs'] = dict(membership=graph.membership_policy, members=graph.memberships_path,
            phase=graph.phase_context_path, stage=graph.round_stage_column,
            phase_features=graph.config.phase_rank_features, timezone=graph.config.timezone,
            policy=graph.config.forecast_policy, round_keys=graph.config.round_keys,
            grid=grid_identity(graph.config.heatmaps) if graph.config.heatmaps is not None else None)
    return settings


def run_recipe(recipe, *, prepared=None, catalog=None):
    from ..experiments.football import FootballExperiment, RefitPolicy
    from ..analysis import PreTrainingAnalysis, PostTrainingAnalysis
    from ..selection import ModelSelection
    from ..selection.core import _development
    from ..splits import Fold, SplitPlan
    recipe = validate_recipe(recipe)
    for section, title in [('pre_reporters', 'Pre-training analysis'),
                           ('fitted_reporters', 'Pre-training analysis'),
                           ('post_reporters', 'Post-training analysis')]:
        for name, reporter in recipe.get(section, {}).items():
            if not isinstance(reporter, dict) or not reporter.get('component', '').startswith('reporting.'):
                continue
            for field in ('features', 'targets'):
                if reporter.get('params', {}).get(field) == []:
                    label_plot = field == 'targets' and reporter['component'] == 'reporting.FeatureDistributionReporter'
                    title_field = 'Labels' if label_plot else field.title()
                    disabled = 'omit label plots' if label_plot else 'use all available columns'
                    raise ValueError(
                        f'{title} → {name} → {title_field} is empty. '
                        f'Select at least one column, or disable {title_field} to {disabled}. '
                        'Discover feature columns first if the choices are not loaded.')
    catalog = catalog or catalog_for_ui()
    if prepared is not None:
        _verify_overlay_selection(recipe, prepared.dataset.definitions)
        original = prepared.config.get('recipe')
        if original is None:
            raise ValueError('Cannot verify prepared recipe inputs. Use prepare_recipe to rebuild them.')
        from ..experiments.recovery import signature
        old = _preparation_settings(original, catalog)
        new = _preparation_settings(recipe, catalog)
        changed = [k for k in sorted(old.keys() | new.keys()) if signature(old.get(k)) != signature(new.get(k))]
        if changed:
            raise ValueError('Prepared recipe inputs changed: ' + ', '.join(changed) + '. Run prepare_recipe again.')
    prepared = prepared or prepare_recipe(recipe, catalog=catalog)
    context = {**prepared.outputs, 'dataset': prepared.dataset}
    experiment = FootballExperiment(recipe['name'], output_dir=recipe.get('output_dir', 'experiments'), config=recipe.get('config'))
    options = catalog.build(recipe.get('run', {}), context)
    analysis = catalog.build(recipe.get('analysis_options', {}))
    options.update(pre_analysis=PreTrainingAnalysis(catalog.build(recipe.get('pre_reporters', {}), context), **analysis.get('pre', {})),
                   post_analysis=PostTrainingAnalysis(catalog.build(recipe.get('post_reporters', {}), context), **analysis.get('post', {})),
                   execution=catalog.build(recipe.get('execution')),
                   checkpoint_policy=catalog.build(recipe.get('checkpoint')))
    if recipe.get('refit') is not None:
        refit_spec = deepcopy(recipe['refit'])
        candidate_spec = refit_spec.pop('candidate', None)
        refit = catalog.build(refit_spec, context)
        if candidate_spec is not None:
            refit['candidate'] = (catalog.build(candidate_spec, context) if 'component' in candidate_spec else
                                  make_candidate({**deepcopy(recipe), **candidate_spec}, catalog, context))
        positions = refit.get('train_positions')
        if isinstance(positions, str) and positions == 'all':
            refit['train_positions'] = np.arange(len(prepared.dataset.X))
        elif isinstance(positions, str) and positions == 'training':
            refit['train_positions'] = np.unique(np.concatenate([f.train for f in prepared.split_plan.folds]))
        options['refit_policy'] = RefitPolicy(**refit)
    search = recipe.get('search')
    if search is None:
        options['model'] = make_candidate(recipe, catalog, context)
    else:
        candidates = (catalog.build(search['candidate_source'], context) if search.get('candidate_source') is not None
                      else _grid_candidates(recipe, search, catalog, context))
        options['model_selection'] = ModelSelection(candidates, **catalog.build(search.get('options', {}), context))
        inner = InnerPlan(search['split'], search.get('split_options', {}), catalog, recipe.get('cutoff_hours'), recipe.get('league_timezones'))
        if search.get('nested', False):
            options['inner_plan_factory'] = inner
        else:
            if len(prepared.split_plan.folds) != 1:
                raise ValueError('Holdout search needs one outer fold. Select one fold or enable nested search.')
            rows = np.asarray(search.get('development_positions', prepared.split_plan.folds[0].train))
            subset, rows = _development(prepared.dataset, rows)
            local = inner(subset)
            options['development_positions'] = rows
            options['selection_plan'] = SplitPlan([Fold(rows[f.train], rows[f.test], rows[f.score], f.metadata)
                                                  for f in local.folds], len(prepared.dataset.X), np.arange(len(prepared.dataset.X)))
    result = experiment.run(prepared, **options)
    if recipe.get('artifact_export') is not None:
        result.export_path = catalog.build(recipe['artifact_export']).run(result)
    return result


def predict_recipe(recipe, *, catalog=None):
    from ..data import select_prediction_fixtures
    from ..training import load_model
    catalog = catalog or catalog_for_ui()
    options = catalog.build(recipe.get('prediction', {}))
    path = options.pop('model_path')
    serializer = options.pop('serializer', None)
    fitted = load_model(path, serializer=serializer)
    data, dataset, context = prepare_recipe(recipe, catalog=catalog, prediction=True, preparation_state=fitted.definitions)
    from ..graphs import TemporalGraphAdapter
    from ..training.calibration import CalibratedAdapter
    graph_model = fitted.model.estimator if isinstance(fitted.model, CalibratedAdapter) else fitted.model
    if isinstance(graph_model, TemporalGraphAdapter):
        store = context.get('graph_store')
        if store is None:
            raise ValueError('Load graph artifacts with a Temporal team GAT recipe and full retained history.')
        if graph_model.config != catalog.build(recipe['model']).config:
            raise ValueError('Prediction graph configuration must match the saved model.')
        graph_model.attach_event_store(store)
    fixtures = select_prediction_fixtures(data, **options)
    aligned = fixtures.align(dataset)
    return aligned, fitted.predict(aligned)


def export_python(recipe):
    """Executable, inspectable Python; no string interpolation into Python code."""
    payload = json.dumps(validate_recipe(recipe), indent=2, ensure_ascii=False)
    return ('import json\nfrom xdiyo_analytics.ui import prepare_recipe, run_recipe\n\n'
            f'recipe = json.loads({payload!r})\n\n'
            'prepared = prepare_recipe(recipe)\n'
            '# Inspect prepared.dataset and prepared.split_plan before fitting.\n'
            'result = run_recipe(recipe, prepared=prepared)\nresult.show()\n')


def export_notebook(recipe):
    source = export_python(recipe)
    marker = 'result = run_recipe'
    before, after = source.split(marker, 1)
    return {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
            'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': ['# ' + recipe['name']]},
                      *[{'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text.splitlines(True)}
                        for text in (before, 'prepared.dataset.X.head()\n', marker + after)]]}
