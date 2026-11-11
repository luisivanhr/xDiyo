"""Real browser edits, save/open and downloads, without experiment jobs."""
import json
from pathlib import Path
import pytest

pytest.importorskip('catboost')
from test_ui_odds_roundtrip import browser_ui, import_recipe, save_recipe, export_payload
from xdiyo_analytics.ui import default_recipe
from xdiyo_analytics.ui.recipe import node


@pytest.mark.parametrize('suffix', ['Regressor', 'Classifier', 'Ranker'])
def test_catboost_browser_roundtrip(browser_ui, suffix):
    page, handle, calls = browser_ui
    recipe = default_recipe('/synthetic/catboost')
    recipe['model'] = node('catboost.CatBoost' + suffix, iterations=3, depth=2, verbose=False)
    if suffix == 'Ranker':
        recipe['assembly']['layout'] = 'team_match'
        source = recipe['labels']['corners']['params']['source']
        recipe['labels']['corners'] = node('labels.TeamValue', source=source)
    import_recipe(page, recipe)
    page.get_by_role('link', name='Model & training', exact=True).click()
    field = page.get_by_role('spinbutton', name='Iterations', exact=True)
    field.fill('12'); field.blur()
    recipe['model']['params']['iterations'] = 12
    page.locator('details').evaluate_all('(nodes) => nodes.forEach(node => node.open = true)')
    page.get_by_role('checkbox', name='Enable Penalties coefficient', exact=True).check()
    penalty = page.get_by_role('spinbutton', name='Penalties coefficient', exact=True)
    penalty.fill('0.75'); penalty.blur()
    recipe['model']['params']['penalties_coefficient'] = 0.75
    assert save_recipe(page, handle) == recipe
    for fmt in ('Python', 'notebook'):
        with page.expect_download() as downloading:
            page.get_by_role('button', name=f'Export {fmt}', exact=True).click()
        text = Path(downloading.value.path()).read_text()
        source = text if fmt == 'Python' else ''.join(json.loads(text)['cells'][1]['source'])
        assert export_payload(source) == handle.state.recipe_paths(recipe)
    page.get_by_role('button', name='Open recipe', exact=True).click()
    page.get_by_role('button', name='Open', exact=True).click()
    page.get_by_text('Recipe opened.', exact=True).wait_for()
    assert save_recipe(page, handle) == recipe
    assert not any(route in ('run', 'prepare', 'predict') for route, _ in calls)
