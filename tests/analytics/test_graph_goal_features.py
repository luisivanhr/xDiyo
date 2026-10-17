"""Explicit edge predictors preserve canonical scores and legacy artifacts."""
from dataclasses import asdict, replace
import ast
from hashlib import sha256
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
from test_temporal_graph import sample, config, context
from test_ui_workflow import ui_recipe
from test_graph_integration import graph_recipe
from xdiyo_analytics.graphs import TemporalEventStore, TemporalGraphAdapter, TemporalGraphSerializer
from xdiyo_analytics.graphs.adapter import _Backend, torch_threads
from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor
from xdiyo_analytics.training import TrainingControl
from xdiyo_analytics.ui.recipe import catalog_for_ui, prepare_recipe, run_recipe, export_python


HALVES = tuple(f'{role}::{period}::Match overview::{stat}::value'
               for period in ('1ST', '2ND') for stat in ('goals', 'shotsOnGoal')
               for role in ('team', 'opponent'))
COVARIANCE = tuple(f'{role}::ALL::Spatial::covariance_xy{suffix}::value'
                   for suffix in ('', '_abs') for role in ('team', 'opponent'))
CHANNELS = HALVES + COVARIANCE


def period_store():
    original = sample()
    h = original.history
    for i, col in enumerate(CHANNELS[::2]):
        # Supplied synthetic observations, including genuine zeros and missing
        # periods. Do not infer period scores from canonical final scores.
        values = (np.arange(len(h)) + i) % 4 * 1.
        values[2:4] = np.nan
        h[col] = values
        h[CHANNELS[2*i+1]] = values.reshape(-1, 2)[:, ::-1].reshape(-1)
    return TemporalEventStore(h, original.team_seasons)


def exact(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: exact(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): exact(x, y)
    else:
        assert a == b


def legacy_version(backend):
    """Pre-option scalar fingerprint, independent of the new version method."""
    digest = sha256()
    for name, value in backend.model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    digest.update(json.dumps(backend.preprocessor.state(), sort_keys=True).encode())
    settings = asdict(backend.config)
    settings.pop('deterministic', None)
    settings.pop('boundary_memory', None)
    settings.pop('statistics_forecast', None)
    settings.pop('include_goal_features')
    settings.pop('heatmaps')
    settings.pop('execution', None)
    return sha256((digest.hexdigest()+json.dumps(settings, sort_keys=True)).encode()).hexdigest()


@pytest.mark.parametrize('value', [None, 0, 1, 'false', np.bool_(False)])
def test_requires_actual_boolean(value):
    with pytest.raises(ValueError, match='must be a boolean'):
        config(include_goal_features=value, statistics=CHANNELS)


def test_rejects_empty_goal_disabled_input():
    with pytest.raises(ValueError, match='requires at least one'):
        config(include_goal_features=False)


def test_default_matches_legacy_preparation_gradients_update_and_fingerprint(monkeypatch):
    import xdiyo_analytics.graphs.adapter as module
    store = period_store()
    ctx = context(store, [1, 2, 3]); ctx.validation = context(store, [5])
    c = config(statistics=CHANNELS)
    with torch_threads(1):
        current = _Backend(c, store, 17, 'cpu'); current.initialize(ctx)
        class LegacyPreprocessor(GraphPreprocessor):
            def __init__(self, settings):
                self.config = settings
                self.edge_columns = ('goals_for', 'goals_against', *settings.statistics)
                self.fixture_columns = ('neutral_venue', 'home_rest_days', 'away_rest_days',
                                        'home_support', 'away_support', *settings.fixture_features)
        monkeypatch.setattr(module, 'GraphPreprocessor', LegacyPreprocessor)
        old = _Backend(replace(c, include_goal_features=True), store, 17, 'cpu'); old.initialize(ctx)
        assert current.preprocessor.edge_columns == old.preprocessor.edge_columns
        exact(current.preprocessor.state(), old.preprocessor.state())
        for i in range(len(current.train_groups)):
            exact(current.train_groups[i], old.train_groups[i])
        exact(current.validation_groups[0], old.validation_groups[0])
        exact(current.step(), old.step())
        exact(current.model.state_dict(), old.model.state_dict())
        exact(current.optimizer.state_dict(), old.optimizer.state_dict())
        for a, b in zip(current.model.parameters(), old.model.parameters()): exact(a.grad, b.grad)
        current.refresh_version()
        assert current.model_version == legacy_version(current)


def test_half_columns_masks_and_score_independence():
    store = period_store(); before = store.history
    c = config(include_goal_features=False, statistics=CHANNELS)
    ctx = context(store, [1, 2, 3])
    with torch_threads(1):
        backend = _Backend(c, store, 8, 'cpu'); backend.initialize(ctx)
        p = backend.preprocessor
        assert p.edge_columns == CHANNELS
        assert not any('::ALL::' in x for x in p.edge_columns if x not in COVARIANCE)
        snapshot = store.snapshot(pd.Timestamp('2020-01-10', tz='UTC'), c)
        tensors = p.tensors(snapshot, 'cpu')
        values = snapshot.edges[list(CHANNELS)].to_numpy(dtype=float)
        n = len(CHANNELS)
        np.testing.assert_array_equal(tensors['x'][:, n:2*n], np.isnan(values))
        assert tensors['x'].shape[1] == 2*n + 1 + len(p.leagues) + 1
        # Pure tensor perturbation test, NOT a reconciled real observation.
        # Eligibility and explicit period observations are deliberately fixed.
        altered = snapshot.edges.copy()
        altered['goals_for'] = 100.; altered['goals_against'] = 200.
        exact(tensors, p.tensors(replace(snapshot, edges=altered), 'cpu'))
        pd.testing.assert_frame_equal(store.history, before)
        pd.testing.assert_frame_equal(context(store, [1, 2, 3]).y, ctx.y)
        old = GraphPreprocessor(config(statistics=CHANNELS))
        assert not np.array_equal(old.numeric_edges(snapshot.edges), old.numeric_edges(altered), equal_nan=True)
        with pytest.raises(ValueError, match='paired'):
            GraphPreprocessor(config(statistics=(CHANNELS[0],))).numeric_edges(
                snapshot.edges.drop(columns=[CHANNELS[1]]))


@pytest.mark.parametrize('enabled,head_width', [(True, 0), (False, 0), (False, 11)])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_native_persistence_and_legacy_key_omission(tmp_path, enabled, head_width, device):
    store = period_store()
    c = config(statistics=CHANNELS, include_goal_features=enabled,
               fixture_features=tuple(f'causal_fixture_{i}' for i in range(head_width)))
    def inputs(positions, fit=True):
        ctx = context(store, positions, fit=fit)
        for i, name in enumerate(c.fixture_features): ctx.X[name] = i * .1
        ctx.metadata['feature_cutoff_at'] = [c.history_cutoff(t) for t in store.issue_times(ctx.metadata, c)]
        return ctx
    model = TemporalGraphAdapter(c, store, device=device,
                                 control=TrainingControl(max_steps=1, seed=17))
    model.fit(inputs([1, 2, 3]))
    serializer = TemporalGraphSerializer(embed_event_store=True, device=device)
    serializer.save(model, tmp_path)
    path = tmp_path/'graph.json'; meta = json.loads(path.read_text())
    assert meta['config']['include_goal_features'] is enabled
    assert meta['edge_feature_order'] == list(model.backend_.preprocessor.edge_columns)
    if enabled:
        meta['config'].pop('include_goal_features')
        meta['model_version'] = legacy_version(model.backend_)
        path.write_text(json.dumps(meta))
    loaded = serializer.load(tmp_path)
    assert loaded.config.include_goal_features is enabled
    exact(loaded.backend_.model.state_dict(), model.backend_.model.state_dict())
    pred = inputs([7, 8], fit=False)
    # CUDA scatter reductions may differ by a float32 ULP between identical
    # forwards. Weights are exact above; CPU predictions remain bit-exact.
    pd.testing.assert_frame_equal(loaded.predict(pred)['predict_proba'], model.predict(pred)['predict_proba'],
                                  check_exact=device == 'cpu', rtol=1e-6, atol=1e-7)
    pd.testing.assert_frame_equal(loaded.event_store.history, store.history)
    # The serialized order still guards architecture/predictor disagreements.
    meta['config']['include_goal_features'] = not enabled
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='feature order'):
        serializer.load(tmp_path)
    meta['config']['include_goal_features'] = enabled
    meta['edge_feature_order'].reverse(); path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='feature order'):
        serializer.load(tmp_path)


def test_ui_recipe_disabled_goals_prepares_runs_and_exports(ui_recipe, tmp_path):
    recipe = graph_recipe(ui_recipe, tmp_path)
    settings = recipe['model']['params']['config']['params']
    settings.update(include_goal_features=False, statistics=[
        'team::ALL::Match overview::cornerKicks::value',
        'opponent::ALL::Match overview::cornerKicks::value'])
    prepared = prepare_recipe(recipe)
    result = run_recipe(recipe, prepared=prepared)
    for fold in result.training.folds:
        assert fold.model.config.include_goal_features is False
        assert fold.model.backend_.preprocessor.edge_columns == tuple(settings['statistics'])
    exported = export_python(recipe)
    assignment = next(n for n in ast.parse(exported).body if isinstance(n, ast.Assign) and n.targets[0].id == 'recipe')
    round_trip = json.loads(ast.literal_eval(assignment.value.args[0]))
    assert round_trip['model']['params']['config']['params']['include_goal_features'] is False
    compile(exported, 'recipe.py', 'exec')
    assert catalog_for_ui().build(json.loads(json.dumps(recipe['model']))).config.include_goal_features is False
    # The shipped inventory exposes the actual boolean default used by the UI.
    from pathlib import Path
    import xdiyo_analytics.ui as ui
    inventory = json.loads(Path(ui.__file__).with_name('inventory.json').read_text(encoding='utf-8'))
    field = next(f for f in inventory['components']['graphs.TemporalGraphConfig']['fields'] if f['name']=='include_goal_features')
    assert field['kind']=='boolean' and field['default'] is True
