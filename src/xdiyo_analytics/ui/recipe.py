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


def default_recipe(data_root='data/xDiyo_data'):
    stat = node('features.Stat', period='ALL', group='Match overview', key='cornerKicks')
    return {
        'version': 1, 'name': 'Corners experiment', 'output_dir': 'experiments', 'config': {}, 'team_badges': None,
        'data': {'data_root': str(data_root), 'seasons': ['22_23', '23_24', '24_25'],
                 'leagues': None, 'tables': ['matches', 'statistics', 'pregame'], 'include_awarded': False},
        'stat_selection': None, 'history': {}, 'feature_options': {}, 'cutoff_hours': None,
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
        'run': {'reuse': True},
        'prediction': {'model_path': '', 'rounds': None, 'statuses': ['notstarted'], 'as_of': None},
    }


def read_recipe(path):
    return validate_recipe(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def validate_recipe(recipe):
    if not isinstance(recipe, dict) or recipe.get('version') != 1:
        raise ValueError('Choose a version 1 experiment recipe.')
    if not isinstance(recipe.get('name'), str) or not recipe['name'].strip():
        raise ValueError('Give the experiment a name.')
    unknown = set(recipe) - set(default_recipe())
    if unknown:
        raise ValueError(f'Unknown recipe sections: {sorted(unknown)}')
    recipe = deepcopy(recipe)
    # Older UI recipes predate the badge toggle. Preserve an explicit opt-out.
    for reporter in recipe.get('post_reporters', {}).values():
        if isinstance(reporter, dict) and reporter.get('component') == 'reporting.MatchResultReporter':
            params = reporter.setdefault('params', {})
            params.setdefault('show_badges', True)
            params.setdefault('catalog', {'ref': 'team_catalog'})
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
    from ..ratings import RatingRun
    from ..training import JoblibSerializer
    from .adapters import BoostingAdapter
    catalog.register('input.Table', _context_file, category='input')
    catalog.register('input.TeamCatalog', TeamCatalog.from_json, category='input')
    catalog.register('input.RatingRun', RatingRun.load, category='input')
    catalog.register('features.TransitionContext', TransitionContext, category='input')
    catalog.register('features.TeamSeasons', build_team_seasons, category='input')
    catalog.register('training.JoblibSerializer', JoblibSerializer, category='training')
    catalog.register('training.BoostingAdapter', BoostingAdapter, category='training')
    return catalog


def prepare_recipe(recipe, *, catalog=None, prediction=False):
    from ..data import load_seasons, select_stats
    from ..histories import build_team_history
    from ..features import evaluate_features
    from ..ratings import build_ratings
    from ..labels import create_labels
    from ..datasets import assemble_dataset
    from ..splits import create_split_plan, SplitPlan
    from ..experiments.football import PreparedExperiment
    from ..reporting import TeamCatalog
    recipe = validate_recipe(recipe)
    catalog = catalog or catalog_for_ui()
    data = load_seasons(**catalog.build(recipe['data']))
    if recipe.get('stat_selection') is not None:
        data = select_stats(data, **catalog.build(recipe['stat_selection']))
    history = build_team_history(data, **catalog.build(recipe.get('history', {})))
    names = data.matches.copy(deep=False)
    for column in ('home_name', 'away_name'):
        if column not in names:
            names = names.assign(**{column: pd.NA})
    team_catalog = TeamCatalog.from_matches(names)
    badge_path = Path(recipe.get('team_badges') or Path(__file__).resolve().parents[3] / 'docs/analytics/team_assets/catalog.json')
    if badge_path.is_file():
        saved_catalog = TeamCatalog.from_json(badge_path)
        team_catalog = TeamCatalog({key: {**saved_catalog.entries.get(key, {}), **entry}
                                    for key, entry in team_catalog.entries.items()})
    context = {'history': history, 'matches': data.matches, 'team_catalog': team_catalog}
    options = catalog.build(recipe.get('feature_options', {}), context)
    hours = recipe.get('cutoff_hours')
    if hours is not None:
        if options.get('cutoffs') is not None:
            raise ValueError('Choose either cutoff hours or explicit cutoffs.')
        options['cutoffs'] = history.kickoff_at - pd.Timedelta(hours=float(hours))
    ratings = {name: build_ratings(history, **catalog.build(spec, context))
               for name, spec in recipe.get('ratings', {}).items()}
    if ratings:
        options['ratings'] = {**options.get('ratings', {}), **ratings}
    if options.get('keyed') is False:
        raise ValueError('Assembly needs keyed features; leave keyed=True.')
    options['keyed'] = True
    features = evaluate_features(history, catalog.build(recipe['features']), **options)
    labels = create_labels(history, catalog.build(recipe['labels']))
    assembly = catalog.build(recipe['assembly'])
    if prediction:
        assembly['drop_missing_targets'] = False
    dataset = assemble_dataset(features, labels[recipe['target']], **assembly)
    context.update(features=features, labels=labels, dataset=dataset)
    if prediction:
        return data, dataset, context
    splitter = catalog.build(recipe['split'])
    plan = splitter if isinstance(splitter, SplitPlan) else create_split_plan(dataset, splitter, **catalog.build(recipe.get('split_options', {}), context))
    if recipe.get('fold_ids') is not None:
        ids = recipe['fold_ids']
        plan = SplitPlan([plan.folds[int(i)] for i in ids], plan.n_rows, plan.row_order,
                         paths=None)  # a subset is not a complete CPCV path assembly
    return PreparedExperiment(dataset, plan, {'history': history, 'matches': data.matches, 'features': features,
                                            'labels': labels, 'ratings': ratings, 'team_catalog': team_catalog},
                              config={'recipe': recipe})


@dataclass
class ModelFactory:
    """Fresh estimator, preprocessing and adapter per fit; process serializable."""
    recipe: dict
    catalog: object

    def cache_key(self):
        """Only configuration consumed when constructing a fitted model."""
        return {"catalog": self.catalog, "model": self.recipe["model"],
                "preprocessors": self.recipe.get("preprocessors", []),
                "adapter": self.recipe.get("adapter"),
                "target_transformer": self.recipe.get("target_transformer"),
                "prediction_methods": self.recipe.get("prediction_methods", ["predict"])}

    def __call__(self):
        from sklearn.pipeline import Pipeline
        from ..training import EstimatorAdapter, TargetTransformAdapter
        estimator = self.catalog.build(self.recipe['model'])
        native_estimator = estimator
        steps = [(f'prepare_{i}', self.catalog.build(spec)) for i, spec in enumerate(self.recipe.get('preprocessors', []))]
        if steps:
            estimator = Pipeline([*steps, ('model', estimator)])
        adapter = self.recipe.get('adapter')
        if adapter is not None:
            built = self.catalog.build(adapter, {'estimator': estimator, 'native_estimator': native_estimator,
                                                 'preprocessor': Pipeline(steps) if steps else None})
        elif self.recipe['model']['component'].startswith(('lightgbm.', 'xgboost.')):
            from .adapters import BoostingAdapter
            built = BoostingAdapter(estimator, tuple(self.recipe.get('prediction_methods', ['predict'])))
        else:
            built = EstimatorAdapter(estimator, tuple(self.recipe.get('prediction_methods', ['predict'])))
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
    return Candidate(model_factory=ModelFactory(recipe, catalog), **config)


@dataclass
class InnerPlan:
    spec: dict
    options: dict
    catalog: object

    def __call__(self, dataset):
        from ..splits import create_split_plan
        return create_split_plan(dataset, self.catalog.build(self.spec), **self.catalog.build(self.options))


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
        inner = InnerPlan(search['split'], search.get('split_options', {}), catalog)
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
    return experiment.run(prepared, **options)


def predict_recipe(recipe, *, catalog=None):
    from ..data import select_prediction_fixtures
    from ..training import load_model
    catalog = catalog or catalog_for_ui()
    data, dataset, _ = prepare_recipe(recipe, catalog=catalog, prediction=True)
    options = catalog.build(recipe.get('prediction', {}))
    path = options.pop('model_path')
    serializer = options.pop('serializer', None)
    fixtures = select_prediction_fixtures(data, **options)
    aligned = fixtures.align(dataset)
    fitted = load_model(path, serializer=serializer)
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
