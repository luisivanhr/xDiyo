from copy import deepcopy
import json
from pathlib import Path
import pytest
import pandas as pd
from xdiyo_analytics.experiments.frozen import FrozenExecution
from xdiyo_analytics.experiments.recovery import execution_key


def test_frozen_keys_match_mutable_reference_and_own_nested_cells(monkeypatch):
    frame = pd.DataFrame({'a': [1., float('nan')], 'nested': [{'values': [1]}, None]})
    frame.attrs['origin'] = {'keys': [7]}
    frozen = FrozenExecution(frame, ('a', 'b'))
    assert frozen.key(None, 3, None, shared_positions=((0, 0), (2, 1))) == execution_key(frame, 3, ('a', 'b'))
    frame.at[0, 'nested']['values'].append(4)
    frame.attrs['origin']['keys'].append(9)
    copy = frozen.copy(0)
    assert copy.at[0, 'nested'] == {'values': [1]}
    copy.at[0, 'nested']['values'].append(5)
    assert frozen.copy(0).at[0, 'nested'] == {'values': [1]}
    assert frozen.key(None, shared_positions=((0, 0),)) != execution_key(frame)
    original = Path.read_bytes
    def changed(path):
        raw = original(path)
        return raw + b'# mutation' if path.name == 'frozen.py' else raw
    monkeypatch.setattr(Path, 'read_bytes', changed)
    with pytest.raises(RuntimeError, match='Source changed'):
        frozen.key(None, shared_positions=((0, 0),))


def record(root, number, key='key'):
    folder = root / 'runs' / str(number)
    folder.mkdir(parents=True, exist_ok=True)
    data = dict(schema=1, experiment_id='e', run_id=str(number), created_at=str(number), status='complete',
                recovery_key=key, role='trial', artifacts={'recovery': 'recovery.json'})
    (folder / 'run.json').write_text(json.dumps(data), encoding='utf-8')
    return data


def test_index_rebuild_mutation_corruption_and_copy_isolation(tmp_path, monkeypatch):
    from xdiyo_analytics.experiments.run_index import RunIndex
    record(tmp_path, 1)
    index = RunIndex(tmp_path, 'e')
    first = index.read('trial', key='key', completed=True)
    first['artifacts'].clear()
    assert index.read('trial', key='key', completed=True)['artifacts']
    assert index.decoded == 1
    (tmp_path / 'run-index.json').write_text('corrupt')
    assert index.read('trial', key='key', completed=True)['run_id'] == '1'
    record(tmp_path, 2)
    assert index.read('trial', key='key', completed=True)['run_id'] == '2'
    record(tmp_path, 2, 'changed')
    assert index.read('trial', key='key', completed=True)['run_id'] == '1'
    (tmp_path / 'runs/1/run.json').unlink()
    assert index.read('trial', key='key', completed=True) is None
    record(tmp_path, '.pending-hidden')
    assert len(index.read()) == 1
    (tmp_path / 'runs/2/run.json').write_text('corrupt')
    with pytest.raises(ValueError): index.read()


def test_index_concurrent_instances_and_interrupted_hint_publication(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from xdiyo_analytics.experiments.run_index import RunIndex
    for i in range(20): record(tmp_path, i)
    with ThreadPoolExecutor(4) as executor:
        values = list(executor.map(lambda _: RunIndex(tmp_path, 'e').read(), range(8)))
    assert all(value == values[0] for value in values)
    index = RunIndex(tmp_path, 'e')
    (tmp_path / 'run-index.json').unlink()
    def fail(*args): raise OSError('interrupted')
    monkeypatch.setattr(Path, 'replace', fail)
    assert len(index.read()) == 20
    assert not list(tmp_path.glob('.run-index-*.tmp'))


def test_report_reuses_validated_document_and_invalidates_mutation(tmp_path, monkeypatch):
    from xdiyo_analytics.ui.server import BuilderState
    from xdiyo_analytics.reporting import AnalysisReport
    from football_experiment_samples import prepared, ridge, post
    from xdiyo_analytics.experiments import FootballExperiment
    result = FootballExperiment('render', output_dir=tmp_path).run(prepared(), model=ridge(), post_analysis=post())
    state = BuilderState(tmp_path, None)
    recipe = {'name': 'render', 'output_dir': str(tmp_path)}
    job = dict(object=result, recipe=recipe)
    calls = []
    native = AnalysisReport.to_html
    def tracked(self, *args, **kwargs):
        calls.append(1)
        return native(self, *args, **kwargs)
    monkeypatch.setattr(AnalysisReport, 'to_html', tracked)
    html = state._report(job)
    assert state._report(job) == html and len(calls) == 1
    result.record['name'] = 'Changed title'
    assert 'Changed title' in state._report(job) and len(calls) == 2
    state.executor.shutdown()


def test_completed_eviction_reload_export_and_mutation_guard(tmp_path):
    from xdiyo_analytics.ui.server import BuilderState
    from football_experiment_samples import prepared, ridge, post
    from xdiyo_analytics.experiments import FootballExperiment
    result = FootballExperiment('retained', output_dir=tmp_path).run(prepared(), model=ridge(), post_analysis=post())
    state = BuilderState(tmp_path, None, retention_bytes=0)
    recipe = {'name': 'retained', 'output_dir': str(tmp_path)}
    job = dict(object=result, recipe=recipe, status='complete', action='run')
    state.jobs['result'] = job
    original = state._report(job)
    state._trim_results()
    assert 'object' not in job, job.get('retention')
    assert state._report(job) == original
    state.base_url = 'http://127.0.0.1:9999'
    assert state._report(job)
    assert 'html' not in job and 'object' not in job
    export = state.dispatch('save_model', {'id': 'result', 'path': str(tmp_path / 'export'), 'fold_id': 0})
    assert Path(export['path']).is_dir()
    (result.path / 'run.json').write_text('{}')
    with pytest.raises(ValueError, match='files changed'): state._object(job)
    state.executor.shutdown()


def test_unsaved_and_active_payloads_stay_pinned(tmp_path):
    from xdiyo_analytics.ui.server import BuilderState
    state = BuilderState(tmp_path, None, retention_bytes=0)
    state.jobs = {'active': dict(object=[1], status='running', action='run'),
                  'unsaved': dict(object=[2], status='complete', action='prepare')}
    state._trim_results()
    assert all('object' in j for j in state.jobs.values())
    state.executor.shutdown()


def test_spatial_spool_keeps_fixture_requests_and_verifies_files(tmp_path):
    from xdiyo_analytics.reporting.heatmaps import SpatialFixtureData
    from xdiyo_analytics.reporting.spatial_live import SpatialStore
    data = SpatialFixtureData(pd.DataFrame({'mass': [1., 2.]}),
        [{'id': '0', 'home': 1, 'away': 2, 'rows': {'home': 0, 'away': 1}}],
        {'m': {'label': 'Mass', 'spec': {'kind': 'scalar'}, 'sides': {'team': {'columns': ['mass']}}, 'scale': None}}, {})
    store = SpatialStore(tmp_path)
    key = store.register(data)
    assert store.fetch({'id': key, 'fixture': '0', 'map': 'm'}) == data.panel_pair('0', 'm')
    assert store.data[key] is not data
    data.values.iloc[:] = 99
    assert store.fetch({'id': key, 'fixture': '0', 'map': 'm'})['panels'][0]['values'] == [1.]
    store.data[key].path.write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        store.fetch({'id': key, 'fixture': '0', 'map': 'm'})

def test_identity_registry_pickle_preserves_alias_guard_without_retention():
    import gc, weakref
    try:
        from joblib.externals import cloudpickle
    except ImportError:
        import cloudpickle  # joblib 1.6 uses the external dependency.
    from xdiyo_analytics._identity import IdentityRegistry
    class Item:
        pass
    value = Item()
    registry = IdentityRegistry(); registry.add(value)
    restored, shared = cloudpickle.loads(cloudpickle.dumps((registry, value)))
    assert shared in restored
    weak = weakref.ref(shared)
    del shared
    gc.collect()
    assert weak() is None and not restored._weak
    nonweak = []
    registry.add(nonweak)
    restored, shared = cloudpickle.loads(cloudpickle.dumps((registry, nonweak)))
    assert shared in restored


def test_frozen_dataclass_preserves_nonconstructor_state_and_nested_cells():
    from dataclasses import dataclass, field
    from xdiyo_analytics.experiments.frozen import isolate
    @dataclass(frozen=True)
    class State:
        rows: pd.DataFrame
        fitted: object = field(init=False, default=None)
    value = State(pd.DataFrame({'x': [{'a': [1]}]}))
    object.__setattr__(value, 'fitted', {'result': pd.DataFrame({'x': [{'b': [2]}]})})
    result = isolate(value)
    assert result.fitted is not None
    result.rows.at[0, 'x']['a'].append(3)
    result.fitted['result'].at[0, 'x']['b'].append(4)
    assert value.rows.at[0, 'x'] == {'a': [1]}
    assert value.fitted['result'].at[0, 'x'] == {'b': [2]}
