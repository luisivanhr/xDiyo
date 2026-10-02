"""Actual browser recipe controls; uses the synthetic no-job server fixture."""
import json
from pathlib import Path
import runpy

from test_ui_odds_roundtrip import browser_ui, import_recipe, save_recipe, export_payload


def test_combinations_and_binary_policy_render_save_export(browser_ui):
    page,handle,calls=browser_ui
    factory=runpy.run_path(str(Path(__file__).resolve().parents[2]/'examples/all_combinations_draw.py'))['draw_recipe']
    recipe=factory(snapshot='/synthetic/pinned',crosswalk='/synthetic/mapping',
                   snapshot_hash='a'*64,crosswalk_hash='b'*64)
    import_recipe(page,recipe)
    assert save_recipe(page,handle)==recipe
    assert page.get_by_role('spinbutton',name='Legs per ticket',exact=True).count()==6
    assert page.get_by_role('checkbox',name='Enable Maximum non-draw probability',exact=True).count()==2
    assert not page.get_by_role('checkbox',name='Enable Maximum non-draw probability',exact=True).first.is_checked()
    assert page.get_by_role('checkbox',name='Enable Maximum non-draw probability',exact=True).nth(1).is_checked()
    page.get_by_role('spinbutton',name='Legs per ticket',exact=True).first.fill('3')
    page.get_by_role('spinbutton',name='Legs per ticket',exact=True).first.blur()
    recipe['post_reporters']['Baseline']['params']['composition']['params']['tickets']['Pairs']['params']['legs']=3
    assert save_recipe(page,handle)==recipe
    page.get_by_role('combobox',name='Stage column',exact=True).first.select_option(label='Single-stage seasons (explicit)')
    recipe['post_reporters']['Baseline']['params']['composition']['params']['tickets']['Pairs']['params']['stage_column']=None
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
