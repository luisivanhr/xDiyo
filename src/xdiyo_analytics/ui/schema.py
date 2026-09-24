"""Stage form definitions. Function signatures are the parameter source of truth."""

from .catalog import Catalog


def stage_schema(*, refresh=False):
    if not refresh:
        from copy import deepcopy
        from .inventory import inventory
        return deepcopy(inventory()['stages'])
    from ..data import load_seasons, select_stats, select_prediction_fixtures
    from ..histories import build_team_history
    from ..features import evaluate_features
    from ..ratings import build_ratings
    from ..datasets import assemble_dataset
    from ..selection import Candidate, ModelSelection
    from ..experiments.football import FootballExperiment, RefitPolicy
    from ..training import load_model
    functions = [
        ('data', load_seasons, set()), ('stat_selection', select_stats, {'data'}),
        ('history', build_team_history, {'data'}),
        ('feature_options', evaluate_features, {'history', 'features', 'keyed'}),
        ('rating_options', build_ratings, {'history'}),
        ('assembly', assemble_dataset, {'features', 'label'}),
        ('candidate', Candidate, {'model_factory', 'pre_analysis'}),
        ('search_options', ModelSelection, {'candidates'}),
        ('refit', RefitPolicy, set()),
        ('run', FootballExperiment.run, {'self', 'prepared', 'model', 'model_selection', 'selection_plan',
                                       'development_positions', 'inner_plan_factory', 'pre_analysis', 'post_analysis',
                                       'refit_policy', 'checkpoint_policy', 'execution'}),
        ('prediction', select_prediction_fixtures, {'data'}),
        ('model_loading', load_model, set()),
    ]
    catalog = Catalog()
    for key, function, _ in functions:
        catalog.register(key, function, category='stage')
    schema = {entry['id']: entry for entry in catalog.schema(refresh=True)}
    for key, _, excluded in functions:
        schema[key]['fields'] = [f for f in schema[key]['fields'] if f['name'] not in excluded]
    for item in schema['assembly']['fields']:
        if item['name'] == 'layout':
            item['choices'] = ['match', 'team_match']
    for item in schema['candidate']['fields']:
        if item['name'] in ('control', 'validation', 'observer', 'complexity', 'fit_statistics'):
            item['kind'] = 'component'
    schema['split_options'] = {'fields': [dict(name=name, title=name.replace('_', ' ').capitalize(),
                                              kind='value', default=None, required=False)
                                        for name in ('cutoffs', 'available_at', 'groups', 'information_start')]}
    return schema
