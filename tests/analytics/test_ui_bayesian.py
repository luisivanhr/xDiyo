"""Bayesian catalog, recipe preparation, native adapter dispatch and exports."""

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from test_ui_workflow import ui_recipe
from xdiyo_analytics.features import BayesianFixture, BayesianRating
from xdiyo_analytics.ui.inventory import inventory
from xdiyo_analytics.ui.recipe import (
    ModelFactory, catalog_for_ui, export_notebook, export_python, node, prepare_recipe,
    run_recipe,
)


def test_bayesian_inventory_distinguishes_team_fixture_and_training_fields():
    from xdiyo_analytics.ratings.bayesian import FIXTURE_FIELDS, TEAM_FIELDS

    components = inventory()['components']
    for key, choices in [('features.BayesianRating', TEAM_FIELDS),
                         ('features.BayesianFixture', FIXTURE_FIELDS)]:
        fields = {field['name']: field for field in components[key]['fields']}
        assert fields['fields']['kind'] == 'multiselect'
        assert fields['fields']['choices'] == list(choices)
        assert fields['model']['components'] == ['input.BayesianModel', 'ratings.BayesianModel']
    config = {field['name']: field for field in components['ratings.BayesianConfig']['fields']}
    assert config['transition']['choices'] == ['mirrored', 'bridge']
    assert config['transition']['kind'] == 'select'
    assert 'components' not in config['transition']
    model = {field['name']: field for field in components['ratings.BayesianScoreAdapter']['fields']}
    assert model['mode']['choices'] == ['pooled', 'per_league']
    assert model['home_goal_target']['discovery'] == 'targets'
    assert components['ratings.BayesianScoreAdapter']['category'] == 'model'
    run = {field['name']: field for field in inventory()['stages']['run']['fields']}
    assert run['model_serializer']['components'] == [
        'training.JoblibSerializer', 'training.BayesianScoreSerializer']


def test_bayesian_ast_model_json_roundtrip_and_parameter_loader(tmp_path):
    from xdiyo_analytics.ratings import BayesianConfig, BayesianModel, BayesianParameters

    catalog = catalog_for_ui()
    model = BayesianModel(per_league=((17, BayesianParameters(kappa=8.)),),
                          config=BayesianConfig(transition='bridge', bridge_gaps=((17, 18, .2),)))
    definitions = {
        'state': BayesianRating(model=model, fields=('attack_sd', 'log_attack')),
        'fixture': BayesianFixture(model=model, fields=('p_draw', 'expected_home_goals')),
    }
    encoded = catalog.encode(definitions)
    rebuilt = catalog.build(json.loads(json.dumps(encoded)))
    assert rebuilt == definitions
    assert hash(rebuilt['state']) == hash(definitions['state'])
    path = tmp_path / 'parameters.json'
    model.save(path)
    assert catalog.build(node('input.BayesianModel', path=str(path))) == model


def test_preparation_native_goal_label_and_exported_feature_selection(ui_recipe):
    from test_ui_odds_roundtrip import export_payload

    ui_recipe['labels'] = {'score': node('labels.MatchGoals')}
    ui_recipe['target'] = 'score'
    ui_recipe['features'] = {
        'state': node('features.BayesianRating', fields=['defence_vulnerability_mean', 'attack_mean']),
        'fixture': node('features.BayesianFixture', fields=['p_draw', 'expected_home_goals']),
    }
    before = deepcopy(ui_recipe)
    prepared = prepare_recipe(ui_recipe)
    assert list(prepared.dataset.y) == ['score::home_goals', 'score::away_goals']
    assert prepared.dataset.X.shape[1] == 10  # Four team values per side + two fixture values once.
    assert any('defence_vulnerability_mean' in name for name in prepared.dataset.X)
    assert not any('attack_sd' in name for name in prepared.dataset.X)
    assert prepared.dataset.X.notna().all().all()
    catalog = catalog_for_ui()
    for source in [export_python(ui_recipe), ''.join(export_notebook(ui_recipe)['cells'][1]['source'])]:
        restored = export_payload(source)
        assert catalog.build(restored['features']) == catalog.build(ui_recipe['features'])
        pd.testing.assert_frame_equal(prepare_recipe(restored).dataset.X, prepared.dataset.X)
    assert ui_recipe == before


def test_native_model_factory_uses_adapter_without_sklearn_wrapper(ui_recipe):
    from xdiyo_analytics.ratings import BayesianScoreAdapter

    ui_recipe['model'] = node('ratings.BayesianScoreAdapter', home_goal_target='score::home_goals',
                              away_goal_target='score::away_goals', mode='per_league')
    ui_recipe['preprocessors'] = []
    factory = ModelFactory(ui_recipe, catalog_for_ui())
    first, second = factory(), factory()
    assert isinstance(first, BayesianScoreAdapter) and first is not second
    assert first.mode == 'per_league'
    ui_recipe['target_transformer'] = node('targets.StandardScaler')
    with pytest.raises(ValueError, match='untransformed scores'):
        factory()
    ui_recipe['target_transformer'] = None
    ui_recipe['adapter'] = node('training.EstimatorAdapter', estimator={'ref': 'estimator'})
    with pytest.raises(ValueError, match='already a training adapter'):
        factory()


def test_native_bayesian_recipe_runs_through_existing_training_and_save(ui_recipe, tmp_path):
    from xdiyo_analytics.ratings import BayesianScoreAdapter

    ui_recipe['labels'] = {'score': node('labels.MatchGoals')}
    ui_recipe['target'] = 'score'
    ui_recipe['features'] = {'venue': node('features.IsHome')}
    movements = pd.DataFrame([
        dict(competition_id=17, season_id=202425, team_id=team, movement='retained',
             previous_competition_id=17, previous_season_id=202324)
        for team in (2**53 + 3, 2**53 + 5)
    ])
    movement_path = tmp_path / 'known-season-entries.parquet'
    movements.to_parquet(movement_path, index=False)
    ui_recipe['model'] = node('ratings.BayesianScoreAdapter', home_goal_target='score::home_goals',
                              away_goal_target='score::away_goals', fit_fields=[], min_seasons=1,
                              team_seasons=node('input.Table', path=str(movement_path)))
    ui_recipe['preprocessors'] = []
    ui_recipe['post_reporters'] = {}
    ui_recipe['run']['model_serializer'] = node('training.BayesianScoreSerializer')
    result = run_recipe(ui_recipe)
    fitted = result.training.folds[0].model
    assert isinstance(fitted, BayesianScoreAdapter)
    assert fitted.model_.training_cutoff is not None
    assert fitted.run_.checkpoint
    # The ordinary run store accepts the explicit native serializer unchanged.
    assert list(Path(ui_recipe['output_dir']).rglob('bayesian.json'))


def test_bayesian_controls_save_and_export_selected_fields(browser_ui, tmp_path):
    from test_ui_odds_roundtrip import import_recipe, save_recipe, export_payload
    from xdiyo_analytics.ui import default_recipe

    page, handle, calls = browser_ui
    recipe = default_recipe('/synthetic')
    recipe['features'] = {'state': node('features.BayesianRating'),
                           'fixture': node('features.BayesianFixture')}
    import_recipe(page, recipe)
    page.get_by_role('link', name='Features & ratings', exact=True).click()
    assert page.get_by_role('checkbox', name='attack mean', exact=True).is_checked()
    page.get_by_role('checkbox', name='attack sd', exact=True).check()
    page.get_by_role('checkbox', name='p draw', exact=True).check()
    saved = save_recipe(page, handle)
    assert saved['features']['state']['params']['fields'] == [
        'attack_mean', 'defence_vulnerability_mean', 'attack_sd']
    assert saved['features']['fixture']['params']['fields'] == [
        'expected_home_goals', 'expected_away_goals', 'p_draw']
    for kind in ('Python', 'notebook'):
        with page.expect_download() as download:
            page.get_by_role('button', name=f'Export {kind}', exact=True).click()
        payload = Path(download.value.path()).read_text()
        source = payload if kind == 'Python' else ''.join(json.loads(payload)['cells'][1]['source'])
        assert export_payload(source)['features'] == saved['features']
    assert not any(route in ('run', 'prepare', 'predict') for route, _ in calls)


# Reuse the established local builder browser fixture without starting any jobs.
from test_ui_odds_roundtrip import browser_ui
