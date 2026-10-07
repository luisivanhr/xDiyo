"""Native browser recipe edits and exports; no preparation or experiment jobs."""
from copy import deepcopy
import json
from pathlib import Path
from test_ui_odds_roundtrip import browser_ui, import_recipe, save_recipe, export_payload, recipe_fixture
from xdiyo_analytics.ui.recipe import node


def test_composition_timing_gates_and_risk_controls_roundtrip(browser_ui):
    page,handle,calls=browser_ui
    recipe=recipe_fixture()
    schema=node('composition.OutputSchema',target='corners')
    child=node('composition.Estimator',estimator=node('sklearn.linear_model.Ridge'))
    recipe['model']=node('composition.ModelStack',
        base_models={'base':node('composition.ModelNode',model=deepcopy(child),schemas={'predict':deepcopy(schema)})},
        meta_model=child,meta_features=node('composition.OutputFeatures',names=['base']),schema=schema,
        training=node('composition.TrainingPlan',mode='chronological',availability_delay='3h',feature_availability_column='feature_at'))
    recipe['preprocessors']=[]
    params=recipe['post_reporters']['decisions']['params']
    params.update(stake_policy=node('evaluation.FixedStake',amount=10,currency='u'),
        stake_context=node('evaluation.StakeContext',bankroll=100,available_cash=100,currency='u',time='2025-01-01'),
        risk_limits=node('evaluation.RiskLimits',exposure_caps=[['league_keys',25.]]),
        composition=node('evaluation.AllCombinations',legs=2,stage_column=None,probability_mode='independent',payoff='binary',
            probability_columns={'a':'a::probability','b':'b::probability'},
            ticket_gate=node('evaluation.DecisionLayer',models=['a','b'],gate='or',missing='reject')))
    import_recipe(page,recipe)
    assert save_recipe(page,handle)==recipe
    page.locator('details').evaluate_all('elements => elements.forEach(e => e.open = true)')
    cap=page.get_by_role('spinbutton',name='Maximum outstanding amount',exact=True)
    cap.fill('20'); cap.blur()
    params['risk_limits']['params']['exposure_caps']=[['league_keys',20.]]
    assert save_recipe(page,handle)==recipe
    page.get_by_role('link',name='Model & training',exact=True).click()
    page.locator('details').evaluate_all('elements => elements.forEach(e => e.open = true)')
    timing=page.get_by_role('textbox',name='Feature availability column',exact=True)
    timing.fill('features_ready_at'); timing.blur()
    recipe['model']['params']['training']['params']['feature_availability_column']='features_ready_at'
    assert save_recipe(page,handle)==recipe
    for fmt in ('Python','notebook'):
        with page.expect_download() as downloading:
            page.get_by_role('button',name=f'Export {fmt}',exact=True).click()
        text=Path(downloading.value.path()).read_text()
        source=text if fmt=='Python' else ''.join(json.loads(text)['cells'][1]['source'])
        assert export_payload(source)==handle.state.recipe_paths(recipe)
    page.get_by_role('button',name='Open recipe',exact=True).click()
    page.get_by_role('button',name='Open',exact=True).click()
    page.get_by_text('Recipe opened.',exact=True).wait_for()
    assert save_recipe(page,handle)==recipe
    assert not any(route in ('run','prepare','predict') for route,_ in calls)


def test_research_quote_contract_browser_and_exports(browser_ui):
    page,handle,calls=browser_ui
    recipe=recipe_fixture()
    params=recipe['post_reporters']['decisions']['params']
    contract=node('evaluation.QuoteAvailability',mode='research_assumed',
        assumption_id='opening-round-v1',rationale='Unverified opening quotes before the round.',reference='protocol/v1')
    params['composition']=node('evaluation.AllCombinations',legs=2,stage_column='stage',
        probability_mode='independent',payoff='binary',probability_columns={'a':'a::probability','b':'b::probability'},
        ticket_gate=node('evaluation.DecisionLayer',models=['a','b'],missing='error'),quote_availability=contract)
    import_recipe(page,recipe)
    page.locator('details').evaluate_all('elements => elements.forEach(e => e.open = true)')
    field=page.get_by_role('textbox',name='Reference',exact=True)
    field.fill('protocol/v2'); field.blur()
    contract['params']['reference']='protocol/v2'
    page.get_by_role('combobox',name='Audit level',exact=True).select_option('summary')
    params['composition']['params']['audit_level']='summary'
    assert save_recipe(page,handle)==recipe
    for fmt in ('Python','notebook'):
        with page.expect_download() as downloading:
            page.get_by_role('button',name=f'Export {fmt}',exact=True).click()
        text=Path(downloading.value.path()).read_text()
        source=text if fmt=='Python' else ''.join(json.loads(text)['cells'][1]['source'])
        assert export_payload(source)==handle.state.recipe_paths(recipe)
    assert not any(route in ('run','prepare','predict') for route,_ in calls)
