"""Concurrent requests and optional disk failures must not lose successful fits."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import pytest
from xdiyo_analytics.ui import launch_ui
from test_ui_server import call
from football_experiment_samples import prepared, ridge, post
from xdiyo_analytics.experiments import FootballExperiment


def completed(root, name):
    result = FootballExperiment(name, output_dir=root).run(prepared(), model=ridge(), post_analysis=post())
    return dict(object=result, recipe={'name': name, 'output_dir': str(root)}, status='complete', action='run')


@pytest.mark.parametrize('route', ['report', 'save_model'])
def test_request_snapshot_survives_concurrent_eviction(tmp_path, route):
    handle = launch_ui(workspace=tmp_path, open_browser=False, retention_bytes=0)
    state = handle.state
    lookup, attempted, evicted = threading.Event(), threading.Event(), threading.Event()
    class PausedLookup(dict):
        paused = False
        def get(self, key, default=None):
            value = super().get(key, default)
            if key == 'object' and not self.paused and threading.current_thread().name != 'trimmer':
                self.paused = True
                lookup.set()
                # Under the fix the trimmer must wait for the snapshot lock.
                # Without it, force actual eviction before the subsequent lookup.
                assert (attempted if state._retention_lock._is_owned() else evicted).wait(10)
            return value
    job = completed(tmp_path, 'race')
    expected = state._report(job)
    job = PausedLookup(job); state.jobs['result'] = job
    def trim():
        threading.current_thread().name = 'trimmer'
        assert lookup.wait(10)
        attempted.set()
        try: state._trim_results()
        finally: evicted.set()
    try:
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(trim)
            response = call(handle, route, {'id': 'result', 'path': str(tmp_path/'export'), 'fold_id': 0})
            pending.result(timeout=15)
        assert 'object' not in job and '_retained' in job
        if route == 'report': assert response['html'] == expected
        else: assert Path(response['path']).is_dir()
        assert call(handle, 'report', {'id': 'result'})['html'] == expected
    finally:
        handle.close()


@pytest.mark.parametrize('failure', ['write', 'read'])
def test_optional_document_failure_preserves_new_and_older_completed_jobs(tmp_path, monkeypatch, failure):
    import xdiyo_analytics.ui.server as server
    from xdiyo_analytics.ui.retention import RetainedDocument
    handle = launch_ui(workspace=tmp_path, open_browser=False, retention_bytes=0)
    state = handle.state
    old, new = completed(tmp_path, 'old'), completed(tmp_path, 'new')
    state.jobs['old'] = old
    original = state._report(old)
    monkeypatch.setattr(state, 'recipe_paths', lambda recipe: recipe)
    monkeypatch.setattr(server, 'run_recipe', lambda *a, **k: new['object'])
    try:
        with monkeypatch.context() as fault:
            if failure == 'write':
                write = Path.write_bytes
                def fail(path, data):
                    if '_ui_reports' in path.parts: raise OSError('disk full')
                    return write(path, data)
                fault.setattr(Path, 'write_bytes', fail)
            else:
                def fail(self): raise OSError('spool unreadable')
                fault.setattr(RetainedDocument, 'read', fail)
            new_id = state.start('run', new['recipe'])['id']
            state.jobs[new_id]['future'].result(timeout=30)
            for key in ['old', new_id]:
                job = state.jobs[key]
                assert job['status'] == 'complete'
                assert 'object' in job and 'html' in job
                assert '_retained' not in job and '_document' not in job
                assert job['retention'].startswith('pinned:')
                assert call(handle, 'report', {'id': key})['html']
                assert Path(call(handle, 'save_model', {'id': key, 'path': str(tmp_path/key/'export'), 'fold_id': 0})['path']).is_dir()
            assert state._report(old) == original
        state._trim_results()
        assert all('_retained' in state.jobs[k] for k in ['old', new_id])
    finally:
        handle.close()
