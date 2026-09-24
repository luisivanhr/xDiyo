"""Constructor-backed form catalog and explicit, local extension registry.

Recipes contain data, never Python expressions or import paths. A host may
register additional adapters/callables before starting its local UI server.
"""

from dataclasses import MISSING, fields, is_dataclass
from importlib import import_module
from copy import deepcopy
import inspect
import json
import math


class ComponentFactory:
    """Serializable deferred construction, for native iterative adapter factories."""
    def __init__(self, catalog, spec, context):
        self.catalog, self.spec, self.context = catalog, spec, context

    def __call__(self, seed=None):
        context = deepcopy(self.context)
        for key in ('estimator', 'native_estimator'):
            estimator = context.get(key)
            if estimator is not None and hasattr(estimator, 'get_params'):
                from sklearn.base import clone
                estimator = clone(estimator)
                if seed is not None:
                    names = [name for name in estimator.get_params(deep=True) if name.split('__')[-1] == 'random_state']
                    if names:
                        estimator.set_params(**{name: seed for name in names})
                context[key] = estimator
        return self.catalog.build(self.spec, {**context, 'seed': seed})

    def cache_key(self):
        return {'catalog': self.catalog, 'spec': self.spec, 'context': self.context}


class Catalog:
    def __init__(self):
        self.entries = {}

    def register(self, key, constructor, *, category, title=None, fields=None, description=None):
        if key in self.entries:
            raise ValueError(f"Component already registered: {key}")
        self.entries[key] = dict(constructor=constructor, category=category,
                                 title=title or key.split('.')[-1], overrides=fields or {},
                                 description=description or inspect.getdoc(constructor) or '')
        return self

    def cache_key(self):
        # Recovery's signature walker hashes callable code and package versions.
        # Display text is irrelevant to numerical identity, constructors are not.
        return {key: item['constructor'] for key, item in sorted(self.entries.items())}

    def encode(self, value):
        import numpy as np
        if isinstance(value, float) and not math.isfinite(value):
            return {'constant': str(value)}
        if isinstance(value, type) and value in (float, int, np.float64, np.float32, np.int64, np.int32):
            return {'constant': value.__name__}
        if is_dataclass(value) and not isinstance(value, type):
            key = next((k for k, v in self.entries.items() if v['constructor'] is type(value)), None)
            if key:
                return {'component': key, 'params': {f.name: self.encode(getattr(value, f.name))
                                                    for f in fields(value) if f.init}}
        if isinstance(value, (tuple, list)):
            return [self.encode(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self.encode(v) for k, v in value.items()}
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise TypeError(f"Register a serializable default for {type(value).__name__}")

    def schema(self, *, refresh=False):
        result = []
        for key, item in self.entries.items():
            if not refresh:
                from .inventory import component_schema
                saved = component_schema(key)
                if saved is not None:
                    saved = deepcopy(saved)
                    for field in saved['fields']:
                        field.update(item['overrides'].get(field['name'], {}))
                    result.append(saved)
                    continue
            constructor = item['constructor']
            dc = {f.name: f for f in fields(constructor)} if is_dataclass(constructor) else {}
            params = []
            for name, parameter in inspect.signature(constructor).parameters.items():
                if parameter.kind in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL):
                    continue
                default = parameter.default
                required = default is parameter.empty
                if name in dc and dc[name].default_factory is not MISSING:
                    default = dc[name].default_factory()
                annotation = str(parameter.annotation).replace('<class ', '').replace('>', '')
                kind = 'number' if isinstance(default, (int, float)) and not isinstance(default, bool) else 'text'
                if isinstance(default, bool) or annotation in ("'bool'", 'bool'):
                    kind = 'boolean'
                elif isinstance(default, (tuple, list)):
                    kind = 'list'
                elif isinstance(default, dict):
                    kind = 'map'
                elif annotation in ("'int'", "'float'", 'int', 'float'):
                    kind = 'number'
                elif default is None or (not required and is_dataclass(default)):
                    kind = 'value'
                spec = dict(name=name, title=name.replace('_', ' ').capitalize(),
                            required=required, kind=kind, annotation=annotation)
                if not required:
                    spec['default'] = self.encode(default)
                spec.update(item['overrides'].get(name, {}))
                params.append(spec)
            # Boosting estimators deliberately declare most settings through
            # **kwargs. Their public get_params() supplies those defaults.
            if any(p.kind == p.VAR_KEYWORD for p in inspect.signature(constructor).parameters.values()):
                try:
                    defaults = constructor().get_params(deep=False)
                except (TypeError, AttributeError):
                    defaults = {}
                known = {p['name'] for p in params}
                for name, value in defaults.items():
                    if name not in known:
                        params.append(dict(name=name, title=name.replace('_', ' ').capitalize(), required=False,
                                           kind='value', default=self.encode(value), annotation=''))
            result.append(dict(id=key, title=item['title'], category=item['category'],
                               description=item['description'], fields=params,
                               accepts_extra=any(p.kind == p.VAR_KEYWORD for p in inspect.signature(constructor).parameters.values())))
        return result

    def build(self, value, context=None):
        context = context or {}
        if isinstance(value, list):
            return [self.build(v, context) for v in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {'factory'}:
            return ComponentFactory(self, value['factory'], context)
        if set(value) == {'callable'}:
            return self.entries[value['callable']]['constructor']
        if set(value) == {'constant'}:
            import numpy as np
            constants = {'nan': float('nan'), 'inf': float('inf'), '-inf': -float('inf'),
                         'float': float, 'int': int, 'float64': np.float64, 'float32': np.float32,
                         'int64': np.int64, 'int32': np.int32}
            return constants[value['constant']]
        if set(value) == {'ref'}:
            if value['ref'] not in context:
                raise ValueError(f"Reference {value['ref']!r} is unavailable at this stage")
            return context[value['ref']]
        if 'component' not in value:
            return {k: self.build(v, context) for k, v in value.items()}
        if set(value) - {'component', 'params'}:
            raise ValueError('A component contains only component and params')
        key = value['component']
        if key not in self.entries:
            raise ValueError(f"Unknown registered component: {key}")
        constructor = self.entries[key]['constructor']
        params = {k: self.build(v, context) for k, v in value.get('params', {}).items()}
        signature = inspect.signature(constructor)
        signature.bind(**params)
        # Frozen AST nodes use tuple parameters as cache keys.
        for name, p in signature.parameters.items():
            if name in params and (isinstance(p.default, tuple) or p.annotation is tuple
                                   or str(p.annotation).startswith('tuple')):
                if isinstance(params[name], list):
                    params[name] = tuple(params[name])
        return constructor(**params)


def default_catalog():
    catalog = Catalog()
    groups = {
        'features': ('feature', 'Stat ForAgainst H2H IsHome NormalizedStanding Lag RollingMean RollingStd RollingZScore EMA Rating MatchResultGlicko StatGlicko League LeaveOneOut WarmStart'),
        'labels': ('label', 'TeamValue MatchTotal Outcome Above BetOption'),
        'ratings': ('rating', 'Glicko2 GlickoTransition'),
        'splits': ('split', 'TemporalSplit MatchKFold GroupKFold CPCV Fold SplitPlan'),
        'training': ('training', 'TrainingControl EarlyStopping ReduceOnPlateau ValidationTail CheckpointPolicy ExecutionPolicy EstimatorAdapter PartialFitBackend IterativeAdapter DeviceAdapter'),
        'selection': ('selection', 'MetricSelection WeightedSelection ParsimonySelection FitStatistics'),
        'evaluation': ('evaluation', 'Metric BetSpec'),
        'reporting': ('reporter', 'FeatureDistributionReporter CorrelationAnalysis FeatureTimeline TopKCorrelationSelector PredictionReporter PerformanceReporter ResidualAnalysisReporter CalibrationReporter PredictionTimelineReporter PredictionDistributionReporter BetPerformanceReporter ExperimentLeaderboardReporter LearningCurveReporter CoefficientReporter MatchResultReporter TeamCatalog'),
    }
    for module, (category, names) in groups.items():
        namespace = import_module('xdiyo_analytics.' + module)
        for name in names.split():
            cat = category
            if module == 'reporting':
                cat = 'pre_reporter' if name in ('FeatureDistributionReporter', 'CorrelationAnalysis', 'FeatureTimeline', 'TopKCorrelationSelector') else 'post_reporter'
                if name == 'TeamCatalog':
                    cat = 'display'
            catalog.register(f'{module}.{name}', getattr(namespace, name), category=cat)
    namespace = import_module('xdiyo_analytics.features')
    for name in ('SeededEMA', 'Hard', 'LinearFade', 'ObservationCount'):
        catalog.register('features.' + name, getattr(namespace, name), category='warmup')
    for module, names, category in (
        ('sklearn.linear_model', 'Lasso ElasticNet Ridge LinearRegression PoissonRegressor LogisticRegression SGDRegressor SGDClassifier', 'model'),
        ('sklearn.ensemble', 'RandomForestRegressor RandomForestClassifier HistGradientBoostingRegressor HistGradientBoostingClassifier', 'model'),
        ('sklearn.preprocessing', 'StandardScaler MinMaxScaler MaxAbsScaler RobustScaler QuantileTransformer PowerTransformer OneHotEncoder PolynomialFeatures', 'preprocessor'),
        ('sklearn.impute', 'SimpleImputer', 'preprocessor'),
        ('sklearn.compose', 'ColumnTransformer', 'preprocessor'),
        ('sklearn.pipeline', 'Pipeline', 'preprocessor'),
        ('lightgbm', 'LGBMRegressor LGBMClassifier LGBMRanker', 'model'),
        ('xgboost', 'XGBRegressor XGBClassifier XGBRanker', 'model'),
    ):
        try:
            namespace = import_module(module)
        except ImportError:
            continue
        for name in names.split():
            catalog.register(module + '.' + name, getattr(namespace, name), category=category)
    try:
        import_module('statsmodels.genmod.generalized_linear_model')
        counts = import_module('xdiyo_analytics.training.counts')
    except ImportError:
        pass
    else:
        catalog.register('training.NegativeBinomialRegressor', counts.NegativeBinomialRegressor,
                         category='model', title='Negative Binomial regression',
                         description='Count regression with a log link and optional Ridge, Lasso or Elastic Net penalties. Choose mean or mode predictions; variance is mean + dispersion × mean². Keep target labels unscaled; standardize inputs when penalizing coefficients.')
    preprocessing = import_module('sklearn.preprocessing')
    for name, title, description in (
        ('StandardScaler', 'Standard target scaler', 'Center labels by their mean and scale by their standard deviation.'),
        ('MinMaxScaler', 'Min-max target scaler', 'Map the fitting label range to a chosen interval, usually 0 to 1.'),
        ('MaxAbsScaler', 'Max-absolute target scaler', 'Divide labels by their largest absolute fitting value without centering.'),
        ('RobustScaler', 'Robust target scaler', 'Center labels by their median and scale by a chosen percentile range.'),
        ('QuantileTransformer', 'Quantile target transformer', 'Map label ranks to a uniform or normal distribution. Inverse predictions stay within the fitted quantile range.'),
        ('PowerTransformer', 'Power target transformer', 'Reduce skew with Yeo-Johnson or Box-Cox. Box-Cox requires strictly positive labels.'),
    ):
        catalog.register('targets.' + name, getattr(preprocessing, name), category='target_transformer',
                         title=title, description=description + ' Learns from fitting rows only; predictions are automatically restored to original units.')
    return catalog


def _decorate(catalog):
    """UI hints supplement signatures; no constructor parameter is removed."""
    enums = {
        'side': ['for', 'against', 'both'], 'type': ['per_fold', 'overall', 'timeline'],
        'pooling': [None, 'occurrences', 'first', 'last', 'mean'],
        'direction': [None, 'minimize', 'maximize'], 'rank_by': ['standings', 'rating'],
        'aggregate': ['mean', 'median'], 'schedule': ['completed_rounds', 'kickoff'],
        'window_unit': ['rounds', 'matches', 'days'],
        'perspective': ['team', 'home', 'away'], 'exclude': ['team_contributions', 'fixtures'],
        'on_equal': ['push', 'loss'], 'draw': ['loss', 'push'],
        'scaling': ['percentile', 'fixed'], 'unsupported': ['skip', 'raise'],
    }
    for key, item in catalog.entries.items():
        names = inspect.signature(item['constructor']).parameters
        for name, choices in enums.items():
            if name in names:
                item['overrides'][name] = {'choices': choices}
        if 'partition' in names:
            item['overrides']['partition'] = {'choices': ['train', 'test', 'score', 'all'] if item['category'] == 'pre_reporter' else ['test', 'score', 'model', 'experiment']}
        if 'type' in names and hasattr(item['constructor'], 'supported_types'):
            item['overrides']['type'] = {'choices': list(item['constructor'].supported_types)}
        for name in ('source', 'reference', 'engine', 'handoff', 'policy', 'early_stopping', 'scheduler', 'estimator', 'preprocessor', 'option'):
            if name in names and key != 'reporting.TopKCorrelationSelector':
                item['overrides'][name] = {'kind': 'component'}
                categories = {'source': ['feature'], 'reference': ['feature'], 'engine': ['rating'],
                              'handoff': ['warmup'], 'policy': ['warmup', 'rating'],
                              'early_stopping': ['training'], 'scheduler': ['training'],
                              'preprocessor': ['preprocessor'], 'option': ['label']}.get(name)
                if name == 'source' and key in ('labels.Above', 'labels.BetOption'):
                    categories = ['label']
                if categories:
                    item['overrides'][name]['categories'] = categories
        if key == 'features.Stat':
            item['overrides'].update(period={'suggestions': ['ALL', '1ST', '2ND']}, group={'discovery': 'groups'}, key={'discovery': 'stats'}, field={'choices': ['value', 'total', 'display']})
        if key == 'splits.TemporalSplit':
            item['overrides']['unit'] = {'choices': ['rounds', 'seasons', 'league_seasons', 'kickoffs']}
            item['overrides']['gap_unit'] = {'choices': ['rounds', 'seasons', 'kickoffs']}
            item['overrides']['window'] = {'choices': ['expanding', 'sliding']}
        if key == 'features.League':
            item['overrides']['unit'] = {'choices': ['team', 'match']}
        if key == 'training.ExecutionPolicy':
            item['overrides']['device'] = {'suggestions': ['cpu', 'auto', 'cuda', 'cuda:0']}
        if key == 'evaluation.Metric':
            item['overrides']['name'] = {'discovery': 'metrics'}
