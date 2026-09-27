"""Local HTTP API authorization, packaged assets, discovery, and recipe IO."""

import ast
import json
from pathlib import Path
import tomllib
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd
import pytest

from xdiyo_analytics.ui import launch_ui, default_recipe
from xdiyo_analytics.ui.server import _frame


@pytest.fixture
def builder(tmp_path):
    handle = launch_ui(workspace=tmp_path, open_browser=False)
    try:
        yield handle
    finally:
        handle.close()


def call(handle, route, payload=None, *, token=True, origin=None):
    base = handle.url.split('#')[0].rstrip('/')
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['X-Builder-Token'] = handle.state.token
    if origin:
        headers['Origin'] = origin
    request = Request(base + '/api/' + route, data=json.dumps(payload or {}).encode(), headers=headers)
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def test_api_requires_token_and_matching_origin(builder):
    for kwargs in ({'token': False}, {'origin': 'https://example.org'}):
        with pytest.raises(HTTPError) as error:
            call(builder, 'catalog', **kwargs)
        assert error.value.code == 403
    result = call(builder, 'catalog')
    assert result['components'] and result['stages']
    assert result['recipe']['version'] == 1


def test_static_assets_are_served_and_packaged(builder):
    base = builder.url.split('#')[0]
    for name in ('index.html', 'app.js', 'forms.js', 'feature-bundles.js', 'style.css', 'report.css'):
        with urlopen(base + name, timeout=10) as response:
            assert response.status == 200
            assert len(response.read()) > 50
            assert response.headers['X-Content-Type-Options'] == 'nosniff'
    with pytest.raises(HTTPError) as error:
        urlopen(base + '../catalog.py', timeout=10)
    assert error.value.code == 404
    project = Path(__file__).resolve().parents[2]
    config = tomllib.loads((project / 'pyproject.toml').read_text(encoding='utf-8'))
    included = config['tool']['setuptools']['package-data']['xdiyo_analytics.ui']
    assert {'static/*.html', 'static/*.js', 'static/*.css'} <= set(included)


def test_save_reopen_export_and_path_resolution(builder):
    recipe = default_recipe('data')
    recipe['name'] = 'Friendly saved recipe'
    saved = call(builder, 'save', {'recipe': recipe})
    saved_path = Path(saved['path'])
    assert saved_path.is_relative_to(builder.state.workspace)
    listing = call(builder, 'recipes')['recipes']
    assert listing == [{'file': saved_path.name, 'name': recipe['name']}]
    assert call(builder, 'open', {'file': saved_path.name})['recipe'] == recipe
    with pytest.raises(HTTPError) as error:
        call(builder, 'open', {'file': '../some.json'})
    assert error.value.code == 400
    exported = call(builder, 'export', {'recipe': recipe, 'format': 'python'})
    assert exported['filename'].endswith('.py')
    compile(exported['text'], 'exported_recipe.py', 'exec')
    assignment = next(n for n in ast.parse(exported['text']).body if isinstance(n, ast.Assign))
    exported_recipe = json.loads(ast.literal_eval(assignment.value.args[0]))
    assert exported_recipe['data']['data_root'] == str(builder.state.workspace / 'data')


def test_discovery_handles_league_names_containing_underscores(builder):
    root = builder.state.workspace / 'data'
    root.mkdir()
    for stem in ('Premier_League_24_25', 'La_Liga_23_24'):
        (root / (stem + '.manifest.json')).write_text('{}', encoding='utf-8')
    found = call(builder, 'discover', {'root': 'data'})['publications']
    assert found == [dict(stem='La_Liga_23_24', league='La_Liga', season='23_24'),
                     dict(stem='Premier_League_24_25', league='Premier_League', season='24_25')]


def test_preview_preserves_large_ids_and_missing_values():
    frame = pd.DataFrame({'event_id': pd.Series([2**63+19, 2**63+21], dtype='uint64'),
                          'value': [float('nan'), float('inf')]})
    result = _frame(frame)
    assert result['rows'] == [[str(2**63+19), None], [str(2**63+21), None]]
    json.dumps(result, allow_nan=False)


def test_saved_run_picker_offers_only_loadable_finals_including_legacy(builder):
    from dataclasses import replace
    from football_experiment_samples import prepared, ridge, post
    from model_selection_samples import plan
    from xdiyo_analytics.experiments import FootballExperiment
    from xdiyo_analytics.selection import ModelSelection
    from xdiyo_analytics.reporting import ExperimentLeaderboardReporter

    recipe = default_recipe('data')
    recipe['name'] = 'Saved search reports'
    paths = builder.state.recipe_paths(recipe)
    experiment = FootballExperiment(paths['name'], output_dir=paths['output_dir'])
    preparation = prepared(holdout=True)
    analysis = post()
    analysis.reporters['Board'] = ExperimentLeaderboardReporter(weights={'mse': 1.})
    result = experiment.run(preparation, model_selection=ModelSelection([ridge(.1), replace(ridge(1.), name='Ridge 1')], metrics='mse'),
                            selection_plan=plan(preparation.dataset), post_analysis=analysis)
    store = experiment.store
    legacy = store.save_run(result.training, result.post_report, name='Legacy final', config={})
    manifest = store.path / 'runs' / legacy['run_id'] / 'run.json'
    record = json.loads(manifest.read_text(encoding='utf-8'))
    record.pop('role')
    record.pop('run_group')
    manifest.write_text(json.dumps(record), encoding='utf-8')
    store.save_failure(name='Failed final', config={}, error='injected failure')
    store.save_failure(name='Failed trial', config={}, error='injected failure',
                       role='trial', run_group=result.record['run_group'])
    assert len(store.read_runs()) == 6
    assert len(store.read_runs(role='trial')) == 3

    listing = call(builder, 'runs', {'recipe': recipe})['runs']
    assert {item['id'] for item in listing} == {result.record['run_id'], legacy['run_id']}
    assert all(item['status'] == 'complete' for item in listing)
    for item in listing:
        response = call(builder, 'load_run', {'recipe': recipe, 'id': item['id']})
        assert 'Errors' in response['html']
        if item['id'] == result.record['run_id']:
            assert 'Experiment leaderboard' in response['html']
