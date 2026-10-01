"""Synthetic render/recipe IO checks. Never prepare data, fit, or predict."""

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from xdiyo_analytics.ui import default_recipe, launch_ui
from xdiyo_analytics.ui.recipe import node


def pinned(name):
    return node('input.OddsSeries', snapshot=f'/synthetic/{name}',
                crosswalk=f'/synthetic/mapping-{name}', snapshot_hash='a' * 64,
                crosswalk_hash='b' * 64, seasons=['25_26'], leagues=['Premier_League'],
                market='corners', selection='over', line=9.5, quote_type='closing',
                settlement_confirmed=True)


def recipe_fixture():
    recipe = default_recipe('/synthetic/native')
    option = node('labels.BetOption', source=deepcopy(recipe['labels']['corners']),
                  selection='over', line=9.5)
    a, b = pinned('A'), pinned('B')
    a['params']['future_setting'] = {'keep': [None, 'unchanged']}
    # Hidden optional values are preserved too, rather than cleaned on render.
    b['params'].update(market='1x2', selection='home', line=None, quote_type='opening')
    legacy = pinned('legacy')
    del legacy['params']['snapshot_hash']
    del legacy['params']['crosswalk_hash']
    current = deepcopy(legacy)
    current['params']['snapshot'] = 'data/odds'
    table = node('input.Table', path='/synthetic/prices.csv', column='odds')
    recipe['post_reporters'] = {
        'decisions': node('reporting.BetOutcomeReporter', type='overall', partition='test',
                          history={'ref': 'history'}, catalog={'ref': 'team_catalog'},
                          show_badges=True, offers={
                              name: node('evaluation.BetOffer', option=deepcopy(option), odds=odds)
                              for name, odds in [('A', a), ('legacy', legacy), ('current', current)]}),
        'performance': node('reporting.BetPerformanceReporter', type='overall', partition='test',
                           bets={name: node('evaluation.BetSpec', option=deepcopy(option), odds=odds,
                                            take=True) for name, odds in
                                 [('B', b), ('fixed', 1.85), ('table', table)]}),
    }
    recipe['config'] = {'unrelated': ['preserve', 7]}
    return recipe


@pytest.fixture
def browser_ui(tmp_path):
    api = pytest.importorskip('playwright.sync_api')
    handle = launch_ui(workspace=tmp_path, open_browser=False)
    original = handle.state.dispatch
    calls = []

    def dispatch(route, payload):
        calls.append((route, deepcopy(payload)))
        if route == 'odds-discover':
            return {'odds_crosswalks': [{'value': '/synthetic/new-mapping', 'label': 'New mapping'}]}
        if route == 'odds-crosswalk':
            return {'path': '/synthetic/built-mapping', 'status_counts': {'matched': 1},
                    'review_path': '/synthetic/review.parquet'}
        if route in ('discover', 'choices'):
            return {'publications': [], 'stats': [], 'teams': [], 'rounds': []}
        assert route in ('catalog', 'export', 'save', 'recipes', 'open'), f'Forbidden action: {route}'
        return original(route, payload)

    handle.state.dispatch = dispatch
    try:
        with api.sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('console', lambda message: errors.append(message.text) if message.type == 'error' else None)
            page.goto(handle.url)
            page.get_by_text('Data choices are ready.', exact=False).wait_for()
            yield page, handle, calls
            assert not errors
            assert not handle.state.jobs
            browser.close()
    finally:
        handle.close()


def import_recipe(page, recipe):
    page.locator('#import-file').set_input_files({
        'name': 'synthetic.json', 'mimeType': 'application/json',
        'buffer': json.dumps(recipe).encode()})
    page.get_by_text('Recipe imported.', exact=True).wait_for()
    page.get_by_role('link', name='Post-training analysis', exact=True).click()


def save_recipe(page, handle):
    with page.expect_response('**/api/save'):
        page.get_by_role('button', name='Save recipe', exact=True).click()
    page.get_by_text('Recipe saved to ', exact=False).wait_for()
    return json.loads(next(handle.state.recipes.glob('*.json')).read_text())


def export_payload(source):
    tree = ast.parse(source)
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'recipe')
    return json.loads(ast.literal_eval(assignment.value.args[0]))


def test_render_import_save_open_and_exports_preserve_sources(browser_ui, tmp_path):
    page, handle, calls = browser_ui
    recipe = recipe_fixture()
    import_recipe(page, recipe)
    assert save_recipe(page, handle) == recipe
    for _ in range(2):
        page.get_by_role('link', name='Data', exact=True).click()
        page.get_by_role('link', name='Post-training analysis', exact=True).click()
        with page.expect_response('**/api/odds-discover'):
            page.get_by_role('button', name='Refresh odds sources').first.click()
        assert save_recipe(page, handle) == recipe
    page.get_by_role('button', name='Open recipe', exact=True).click()
    page.get_by_role('button', name='Open', exact=True).click()
    page.get_by_text('Recipe opened.', exact=True).wait_for()
    assert save_recipe(page, handle) == recipe
    for fmt in ('Python', 'notebook'):
        with page.expect_download() as downloading:
            page.get_by_role('button', name=f'Export {fmt}', exact=True).click()
        text = Path(downloading.value.path()).read_text()
        source = text if fmt == 'Python' else ''.join(json.loads(text)['cells'][1]['source'])
        exported = export_payload(source)  # Parse only; do not execute the experiment.
        assert exported == handle.state.recipe_paths(recipe)
    assert page.title() == 'Football experiment builder'
    assert page.get_by_role('heading', name='Post-training analysis', exact=True).is_visible()
    page.screenshot(path=str(tmp_path / 'odds-roundtrip.png'))
    assert all(route not in ('prepare', 'run', 'predict') for route, _ in calls)


def test_edits_and_mapping_changes_are_local_and_new_nodes_use_defaults(browser_ui):
    page, handle, calls = browser_ui
    recipe = recipe_fixture()
    import_recipe(page, recipe)
    # Locate the nested odds editor rather than the option's own line control.
    editor = page.locator('select:has(> option[value="input.OddsSeries"]:checked)').locator('..').first
    editor.get_by_role('spinbutton', name='Line', exact=True).fill('10.5')
    editor.get_by_role('spinbutton', name='Line', exact=True).blur()
    recipe['post_reporters']['decisions']['params']['offers']['A']['params']['odds']['params']['line'] = 10.5
    assert save_recipe(page, handle) == recipe
    editor.get_by_role('combobox', name='Selection', exact=True).select_option(label='under')
    expected = recipe['post_reporters']['decisions']['params']['offers']['A']['params']['odds']['params']
    expected['selection'] = 'under'
    assert save_recipe(page, handle) == recipe
    editor.get_by_role('combobox', name='Fixture mapping', exact=True).select_option(label='New mapping')
    expected = recipe['post_reporters']['decisions']['params']['offers']['A']['params']['odds']['params']
    expected['crosswalk'] = '/synthetic/new-mapping'
    del expected['crosswalk_hash']
    assert save_recipe(page, handle) == recipe
    with page.expect_response('**/api/odds-crosswalk'):
        editor.get_by_role('button', name='Build fixture mapping', exact=True).click()
    expected['crosswalk'] = '/synthetic/built-mapping'
    assert save_recipe(page, handle) == recipe
    build = next(payload for route, payload in calls if route == 'odds-crosswalk')
    assert build['snapshot'] == '/synthetic/A' and build['snapshot_hash'] == 'a' * 64
    # A deliberate source replacement creates a fresh node without old mapping pins.
    modes = page.get_by_role('combobox', name='Odds source', exact=True)
    modes.first.select_option(label='Fixed decimal price')
    modes.first.select_option(label='Odds database')
    saved = save_recipe(page, handle)
    new = saved['post_reporters']['decisions']['params']['offers']['A']['params']['odds']['params']
    assert new['snapshot'] == 'data/odds'
    assert not new.get('snapshot_hash') and not new.get('crosswalk_hash') and not new.get('crosswalk')
    recipe['post_reporters']['decisions']['params']['offers']['A']['params']['odds']['params'] = new
    assert saved == recipe
    # Display-only analysis defaults must still support editing and removing options.
    page.get_by_role('textbox', name='Report title', exact=True).fill('Synthetic report')
    page.get_by_role('textbox', name='Report title', exact=True).blur()
    toggle = page.get_by_role('checkbox', name='Enable Choose test folds', exact=True)
    toggle.check()
    toggle.uncheck()
    assert save_recipe(page, handle)['analysis_options'] == {'post': {'title': 'Synthetic report'}}


def synthetic_source(root, name):
    """Real local manifests and mapping payload; no prices, outcomes or models."""
    import pandas as pd
    from xdiyo_analytics.odds import save_crosswalk
    from xdiyo_analytics.odds.crosswalk import MATCH_KEYS
    path = root / name
    path.mkdir(parents=True)
    identity = dict(adapter_version='1', source_files=[{'name': name}],
                    league_seasons=[], league_aliases={})
    manifest = dict(identity, snapshot=hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24])
    (path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    frame = pd.DataFrame([{**dict(zip(MATCH_KEYS, ['Synthetic', '25_26', '1', '2', '3'])),
                           'vendor_match_id': '4', 'status': 'matched',
                           'snapshot': manifest['snapshot']}])
    mapping = save_crosswalk(frame, snapshot=path, output_root=root / f'{name}-maps')
    return path, mapping


def test_real_manifest_pins_catalog_roundtrip_and_tamper_rejection(tmp_path):
    from dataclasses import replace
    from xdiyo_analytics.odds import OddsSeries, read_crosswalk
    from xdiyo_analytics.ui.recipe import catalog_for_ui
    source, mapping = synthetic_source(tmp_path, 'A')
    other, other_mapping = synthetic_source(tmp_path, 'B')
    odds = OddsSeries(str(source), str(mapping), ('25_26',), 'corners', 'over', line=9.5)
    catalog = catalog_for_ui()
    encoded = catalog.encode(odds)
    assert catalog.build(json.loads(json.dumps(encoded))) == odds
    with pytest.raises(ValueError, match='Crosswalk does not match'):
        replace(odds, crosswalk=str(other_mapping), crosswalk_hash=None)
    for field in ('snapshot_hash', 'crosswalk_hash'):
        with pytest.raises(ValueError, match='pinned'):
            replace(odds, **{field: 'wrong'})
    for directory in (source, mapping):
        path = directory / 'manifest.json'
        original = path.read_text()
        path.write_text(original + ' ')
        with pytest.raises(ValueError, match='pinned'):
            catalog.build(encoded)
        path.write_text(original)
    snapshot_id = json.loads((source / 'manifest.json').read_text())['snapshot']
    assert len(read_crosswalk(mapping, snapshot_id)) == 1
    payload = mapping / 'fixtures.parquet'
    payload.write_bytes(payload.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='modified'):
        read_crosswalk(mapping, snapshot_id)


def test_mapping_builder_honors_explicit_source_and_validates_pin(tmp_path, monkeypatch):
    from xdiyo_analytics.odds import crosswalk
    from xdiyo_analytics.odds.workbook import _hash
    from xdiyo_analytics.ui.recipe import catalog_for_ui
    from xdiyo_analytics.ui.server import BuilderState
    source, mapping = synthetic_source(tmp_path / 'outside-workspace', 'A')
    workspace = tmp_path / 'workspace'
    current, _ = synthetic_source(workspace / 'data/odds/database', 'current')
    state = BuilderState(workspace, catalog_for_ui())
    seen = []
    monkeypatch.setattr(crosswalk, 'native_fixture_metadata', lambda *a, **k: object())
    frame = crosswalk.read_crosswalk(mapping, json.loads((source / 'manifest.json').read_text())['snapshot'])
    def build(snapshot, native, **kwargs):
        seen.append(snapshot)
        return frame
    monkeypatch.setattr(crosswalk, 'build_fixture_crosswalk', build)
    monkeypatch.setattr(crosswalk, 'save_crosswalk', lambda *a, **k: mapping)
    request = dict(snapshot=str(source), snapshot_hash=_hash(source / 'manifest.json'),
                   root='/synthetic/native', seasons=['25_26'])
    try:
        state.dispatch('odds-crosswalk', request)
        assert seen == [source]
        with pytest.raises(ValueError, match='pinned'):
            state.dispatch('odds-crosswalk', dict(request, snapshot_hash='wrong'))
        assert seen == [source]
        state.dispatch('odds-crosswalk', dict(root='/synthetic/native', seasons=['25_26']))
        assert seen == [source, current]
    finally:
        state.executor.shutdown()
