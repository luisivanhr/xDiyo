"""Actual browser recipe controls; uses the synthetic no-job server fixture."""
import json
from pathlib import Path
import runpy
import tempfile
import pytest

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


@pytest.mark.parametrize('legs',[2,3,4])
def test_ticket_ev_controls_nested_roundtrip_and_no_jobs(browser_ui,legs,tmp_path):
    page,handle,calls=browser_ui
    factory=runpy.run_path(str(Path(__file__).resolve().parents[2]/'examples/all_combinations_draw.py'))['draw_recipe']
    recipe=factory(snapshot='/synthetic/pinned',crosswalk='/synthetic/mapping',
                   snapshot_hash='a'*64,crosswalk_hash='b'*64)
    import_recipe(page,recipe)
    enabled=page.get_by_role('checkbox',name='Enable Minimum ticket EV (per unit stake)',exact=True).first
    assert not enabled.is_checked()
    enabled.check()
    value=page.get_by_role('spinbutton',name='Minimum ticket EV (per unit stake)',exact=True).first
    assert value.input_value()=='0'
    assert page.get_by_role('alert').filter(has_text='Ticket EV filtering requires Independent').is_visible()
    params=recipe['post_reporters']['Baseline']['params']['composition']['params']['tickets']['Pairs']['params']
    params['min_ev']=0.
    assert save_recipe(page,handle)==recipe  # Invalid intermediate state is kept, not silently rewritten.
    page.get_by_role('combobox',name='Probability mode',exact=True).first.select_option(label='independent')
    params['probability_mode']='independent'
    assert page.get_by_role('alert').filter(has_text='Ticket EV filtering requires Independent').count()==0
    page.get_by_role('spinbutton',name='Legs per ticket',exact=True).first.fill(str(legs))
    page.get_by_role('spinbutton',name='Legs per ticket',exact=True).first.blur()
    params['legs']=legs
    value.fill('0.5'); value.blur(); params['min_ev']=.5
    assert save_recipe(page,handle)==recipe
    page.get_by_role('button',name='Open recipe',exact=True).click()
    page.get_by_role('button',name='Open',exact=True).click()
    page.get_by_text('Recipe opened.',exact=True).wait_for()
    assert value.input_value()=='0.5'
    value.evaluate("element => element.scrollIntoView({block: 'center'})")
    page.screenshot(path=str(Path(tempfile.gettempdir())/f'xdiyo-ticket-ev-{legs}.png'))
    assert save_recipe(page,handle)==recipe
    for fmt in ('Python','notebook'):
        with page.expect_download() as downloading:
            page.get_by_role('button',name=f'Export {fmt}',exact=True).click()
        text=Path(downloading.value.path()).read_text()
        source=text if fmt=='Python' else ''.join(json.loads(text)['cells'][1]['source'])
        compile(source,'export','exec')  # Never execute generated experiment commands.
        assert export_payload(source)==handle.state.recipe_paths(recipe)
    enabled.uncheck()
    del params['min_ev']  # Native optional controls omit disabled fields.
    assert save_recipe(page,handle)==recipe
    enabled.check()
    params['min_ev']=0.
    assert value.input_value()=='0'
    assert save_recipe(page,handle)==recipe
    assert not any(route in ('run','prepare','predict') for route,_ in calls)
