"""Saved Bayesian runs can be configured and prepared entirely through the UI."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from test_ui_workflow import ui_recipe
from test_ui_odds_roundtrip import import_recipe, save_recipe, export_payload
from xdiyo_analytics.ui import launch_ui
from xdiyo_analytics.ui.inventory import inventory
from xdiyo_analytics.ui.recipe import node, prepare_recipe, validate_recipe


def test_saved_run_inventory_exposes_mapping_and_typed_name_discovery():
    saved = inventory()
    mapping = next(f for f in saved['stages']['feature_options']['fields'] if f['name'] == 'ratings')
    assert not mapping.get('hidden', False)
    assert mapping['kind'] == 'map'
    assert mapping['item']['components'] == ['input.BayesianRatingRun', 'input.RatingRun']
    fields = {f['name']: f for f in saved['components']['features.BayesianFixture']['fields']}
    assert fields['name']['discovery'] == 'bayesian_ratings'
    rating = next(f for f in saved['components']['features.Rating']['fields'] if f['name'] == 'fields')
    assert rating['source_choices']['generated'] == ['rating', 'rd', 'sigma']
    assert 'attack_mean' in rating['source_choices']['input.BayesianRatingRun']
    assert 'rating' not in rating['source_choices']['input.BayesianRatingRun']


def test_duplicate_saved_and_generated_rating_names_fail_before_loading(ui_recipe, monkeypatch):
    import xdiyo_analytics.data as data

    ui_recipe['ratings'] = {'goals': {}}
    ui_recipe['feature_options']['ratings'] = {'goals': node('input.BayesianRatingRun', path='does-not-exist')}

    def forbidden(*args, **kwargs):
        pytest.fail('Duplicate names must fail before data or saved artifacts are loaded')

    monkeypatch.setattr(data, 'load_seasons', forbidden)
    for action in (validate_recipe, prepare_recipe):
        with pytest.raises(ValueError, match=r"Saved and generated rating names must be distinct: \['goals'\]"):
            action(ui_recipe)


def test_saved_run_ui_create_save_export_reopen_prepare_without_retraining(ui_recipe, tmp_path, monkeypatch):
    api = pytest.importorskip('playwright.sync_api')
    from playwright.sync_api import expect
    import xdiyo_analytics.ratings as ratings
    import xdiyo_analytics.ratings.bayesian as bayesian
    import xdiyo_analytics.ratings.bayesian_training as training

    history = prepare_recipe(ui_recipe).outputs['history']
    run = ratings.build_bayesian_ratings(history)
    run_path = run.save(tmp_path / 'saved-goals')
    original_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run_path.iterdir()}

    def forbidden(*args, **kwargs):
        pytest.fail('Preparing saved rating features must not train parameters or replay a new Bayesian run')

    monkeypatch.setattr(ratings, 'train_bayesian', forbidden)
    monkeypatch.setattr(training, 'train_bayesian', forbidden)
    monkeypatch.setattr(ratings, 'build_bayesian_ratings', forbidden)
    monkeypatch.setattr(bayesian, 'build_bayesian_ratings', forbidden)
    ui_recipe['features'] = {}
    ui_recipe['ratings'] = {'glicko': {}}
    handle = launch_ui(workspace=tmp_path, open_browser=False)
    original_dispatch = handle.state.dispatch
    calls, errors = [], []

    def dispatch(route, payload):
        calls.append((route, deepcopy(payload)))
        assert route not in ('run', 'predict'), 'This test performs preparation only'
        if route == 'odds-discover':
            return {'odds_crosswalks': []}
        if route in ('discover', 'choices'):
            return {'publications': [], 'stats': [], 'teams': [], 'rounds': []}
        return original_dispatch(route, payload)

    handle.state.dispatch = dispatch
    try:
        with api.sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('console', lambda message: errors.append(message.text) if message.type == 'error' else None)
            page.goto(handle.url)
            page.get_by_text('Data choices are ready.', exact=False).wait_for()
            import_recipe(page, ui_recipe)
            page.get_by_role('link', name='Features & ratings', exact=True).click()
            sources = page.locator('section.section').filter(has=page.get_by_role('heading', name='Saved rating runs', exact=True))
            sources.get_by_role('button', name='+ Add', exact=True).click()
            sources.get_by_role('textbox', name='Entry name', exact=True).fill('goals')
            sources.get_by_role('textbox', name='Entry name', exact=True).blur()
            expect(sources.get_by_role('combobox', name='Component', exact=True)).to_have_value('input.BayesianRatingRun')
            sources.get_by_role('textbox', name='Path', exact=True).fill('saved-goals')
            sources.get_by_role('textbox', name='Path', exact=True).blur()

            features = page.locator('section.section').filter(has=page.get_by_role('heading', name='Named features', exact=True))

            def add_feature(name, component):
                features.get_by_role('button', name='+ Add', exact=True).click()
                row = features.locator('.map-row').last
                row.get_by_role('textbox', name='Entry name', exact=True).fill(name)
                row.get_by_role('textbox', name='Entry name', exact=True).blur()
                row.get_by_role('combobox', name='Component', exact=True).select_option(component)
                return row

            state = add_feature('state', 'features.Rating')
            state.get_by_role('combobox', name='Name', exact=True).select_option(label='goals')
            state.get_by_role('checkbox', name='Enable Fields', exact=True).check()
            state.get_by_role('checkbox', name='attack mean', exact=True).check()
            state.get_by_role('checkbox', name='defence vulnerability mean', exact=True).check()
            assert state.get_by_role('checkbox', name='rd', exact=True).count() == 0

            fixture = add_feature('fixture', 'features.BayesianFixture')
            fixture.get_by_role('checkbox', name='Enable Saved Bayesian run', exact=True).check()
            fixture.get_by_role('combobox', name='Saved Bayesian run', exact=True).select_option(label='goals')
            assert fixture.get_by_role('combobox', name='Saved Bayesian run', exact=True).locator('option').all_text_contents() == ['Select…', 'goals']
            fixture.get_by_role('checkbox', name='p draw', exact=True).check()

            legacy = add_feature('legacy', 'features.Rating')
            legacy.get_by_role('combobox', name='Name', exact=True).select_option(label='glicko')
            legacy.get_by_role('checkbox', name='Enable Fields', exact=True).check()
            legacy.get_by_role('checkbox', name='rating', exact=True).check()
            assert legacy.get_by_role('checkbox', name='rd', exact=True).count() == 1
            assert legacy.get_by_role('checkbox', name='attack mean', exact=True).count() == 0
            legacy.get_by_role('combobox', name='Name', exact=True).select_option(label='goals')
            expect(legacy.get_by_role('checkbox', name='attack mean', exact=True)).to_be_visible()
            assert legacy.get_by_role('checkbox', name='rd', exact=True).count() == 0
            expect(legacy.get_by_role('checkbox', name='Rating (saved)', exact=True)).to_be_checked()
            expect(legacy.get_by_role('alert')).to_have_text('Selected fields unavailable for this rating source: rating. Remove them or choose a compatible source.')
            legacy.get_by_role('checkbox', name='Rating (saved)', exact=True).click()
            expect(legacy.get_by_role('checkbox', name='Rating (saved)', exact=True)).to_have_count(0)
            expect(legacy.get_by_role('alert')).to_have_count(0)
            legacy.get_by_role('checkbox', name='attack mean', exact=True).check()
            legacy.get_by_role('combobox', name='Name', exact=True).select_option(label='glicko')
            expect(legacy.get_by_role('checkbox', name='rd', exact=True)).to_be_visible()
            expect(legacy.get_by_role('checkbox', name='Attack mean (saved)', exact=True)).to_be_checked()
            expect(legacy.get_by_role('alert')).to_contain_text('attack_mean')
            legacy.get_by_role('checkbox', name='Attack mean (saved)', exact=True).click()
            expect(legacy.get_by_role('checkbox', name='Attack mean (saved)', exact=True)).to_have_count(0)
            legacy.get_by_role('checkbox', name='rating', exact=True).check()

            saved = save_recipe(page, handle)
            assert saved['feature_options']['ratings'] == {'goals': node('input.BayesianRatingRun', path='saved-goals')}
            assert saved['features']['fixture']['params']['name'] == 'goals'
            for kind in ('Python', 'notebook'):
                with page.expect_download() as download:
                    page.get_by_role('button', name=f'Export {kind}', exact=True).click()
                payload = Path(download.value.path()).read_text()
                source = payload if kind == 'Python' else ''.join(json.loads(payload)['cells'][1]['source'])
                exported = export_payload(source)
                assert exported['features'] == saved['features']
                assert exported['feature_options']['ratings']['goals']['params']['path'] == str(run_path)
            page.get_by_role('button', name='Open recipe', exact=True).click()
            page.get_by_role('button', name='Open', exact=True).click()
            page.get_by_text('Recipe opened.', exact=True).wait_for()
            assert save_recipe(page, handle) == saved
            page.screenshot(path=str(tmp_path / 'saved-rating-controls.png'), full_page=True)
            with page.expect_response('**/api/prepare') as response:
                page.get_by_role('button', name='Inspect preparation', exact=True).click()
            job_id = response.value.json()['id']
            expect(page.locator('#job-status')).to_have_text('Completed', timeout=30000)
            prepared = handle.state.jobs[job_id]['object']
            expected = prepare_recipe(handle.state.recipe_paths(saved))
            pd.testing.assert_frame_equal(prepared.dataset.X, expected.dataset.X)
            assert any('attack_mean' in name for name in prepared.dataset.X)
            assert any('p_draw' in name for name in prepared.dataset.X)
            assert not errors
            browser.close()
    finally:
        handle.close()
    assert any(route == 'prepare' for route, _ in calls)
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run_path.iterdir()} == original_hashes
