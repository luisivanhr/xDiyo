"""Actual builder controls and exports, without preparation or training jobs."""
from pathlib import Path
import json

from test_ui_odds_roundtrip import browser_ui, import_recipe, save_recipe, export_payload
from test_match_score import score_recipe
from xdiyo_analytics.ui.recipe import catalog_for_ui


def test_score_source_controls_and_export_roundtrip(browser_ui,tmp_path):
    page,handle,calls=browser_ui
    recipe=score_recipe()
    import_recipe(page,recipe)
    page.get_by_role('link',name='Features & ratings',exact=True).click()
    basis=page.get_by_role('combobox',name='Score basis',exact=True)
    assert basis.count()==2
    assert basis.first.locator('option:checked').inner_text().lower()=='current'
    assert 'regulation time' in page.locator('body').inner_text()
    # This is a normal nested source choice, not a statistics bundle entry.
    sources=page.get_by_role('combobox',name='Component',exact=True)
    assert sources.locator('option[value="features.MatchScore"]').count()>=2
    sides=page.get_by_role('combobox',name='Side',exact=True)
    assert sides.count()==2
    sides.first.select_option(label='against')
    saved=save_recipe(page,handle)
    expected=catalog_for_ui().build(recipe['features'])
    from dataclasses import replace
    expected['goals_for_r20']=replace(expected['goals_for_r20'],source=replace(expected['goals_for_r20'].source,side='against'))
    assert catalog_for_ui().build(saved['features'])==expected
    for kind in ('Python','notebook'):
        with page.expect_download() as download:
            page.get_by_role('button',name=f'Export {kind}',exact=True).click()
        payload=Path(download.value.path()).read_text()
        source=payload if kind=='Python' else ''.join(json.loads(payload)['cells'][1]['source'])
        assert catalog_for_ui().build(export_payload(source)['features'])==expected
    page.screenshot(path=str(tmp_path/'match-score-controls.png'),full_page=True)
    assert not any(route in ('run','prepare','predict') for route,_ in calls)
