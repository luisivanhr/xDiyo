"""CPU-light runner contracts; never construct or train a graph model."""
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import time

import numpy as np
import pandas as pd
import pytest


def module():
    directory = Path(__file__).resolve().parents[2]/'tools/benchmarks'
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location('gat_dependency_matrix_test', directory/'gat_dependency_matrix.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_frozen_24_fit_pair_order_and_controls():
    runner = module()
    configs = {arm: {mode: {'execution': {'mode': mode}} for mode in ('native', 'dependency_packed')}
               for arm in runner.ARMS}
    fits = runner.planned_fits(configs)
    assert len(fits) == len({f['id'] for f in fits}) == 24
    assert {f['fold'] for f in fits} == {0, 2}
    assert {f['seed'] for f in fits} == {17, 29, 43}
    for start in range(0, 24, 2):
        pair = fits[start:start+2]
        assert len({(f['arm'], f['fold'], f['seed']) for f in pair}) == 1
        expected = ['dependency_packed', 'native'] if pair[0]['seed'] == 29 else ['native', 'dependency_packed']
        assert [f['mode'] for f in pair] == expected
        for fit in pair:
            assert fit['control'] == dict(max_steps=100, early_stopping=None, scheduler=None,
                restarts=0, seed=fit['seed'], monitor='validation_loss', direction='minimize', restore_best=True)


def test_canonical_parquet_restores_physical_unnamed_cohort_index(tmp_path):
    import pyarrow.parquet as pq
    runner = module()
    original = pd.DataFrame({'event_id': [17, 3, 91]}, index=pd.Index(np.array([901, 12, 501], dtype=np.int64)))
    path = tmp_path/'train.parquet'
    original.to_parquet(path, index=True)
    metadata = json.loads(pq.read_schema(path).metadata[b'pandas'])
    assert metadata['index_columns'] == ['__index_level_0__']
    assert 'cohort_position' not in original.columns
    actual = runner.load_training_population(path)
    assert actual.index.name == 'cohort_position'
    pd.testing.assert_frame_equal(actual.rename_axis(None), original, check_exact=True)
    assert actual.index.tolist() == [901, 12, 501]  # Neither sorting nor new row numbers.


@pytest.mark.parametrize('stored_index', [False, None])
def test_canonical_parquet_refuses_missing_or_virtual_cohort_positions(tmp_path, stored_index):
    runner = module()
    path = tmp_path/'train.parquet'
    pd.DataFrame({'event_id': [17, 3, 91]}).to_parquet(path, index=stored_index)
    with pytest.raises(ValueError, match='physical unnamed cohort index'):
        runner.load_training_population(path)


def test_split_keeps_kickoff_batches_and_purges_late_labels():
    from xdiyo_analytics.graphs import TemporalGraphConfig
    runner = module()
    times = pd.Series(pd.date_range('2020-01-01', periods=10, tz='UTC').repeat(2))
    frame = pd.DataFrame(dict(competition_id=1, season_id=1, event_id=np.arange(20),
        kickoff_at=times, round_issue_at=times-pd.Timedelta(days=1),
        result_available_at=times+pd.Timedelta(hours=2)))
    frame.index = pd.Index(np.arange(20)+100, name='cohort_position')
    frame.loc[100, 'result_available_at'] = pd.Timestamp('2020-01-09', tz='UTC')
    rows, info = runner.split_population(frame, TemporalGraphConfig('UTC'))
    assert rows.index.equals(frame.index)
    assert info['validation_batches'] == 2 and info['validation'] == 4
    assert rows.role.iloc[-4:].eq('validation').all()
    assert rows.loc[100, 'role'] == 'purged'
    assert rows.role.iloc[14:16].eq('purged').all()  # Jan8 labels unknown at Jan8 boundary.
    assert rows.loc[rows.role.eq('fit'), 'label_known_at'].le(pd.Timestamp(info['fitting_label_boundary'])).all()
    assert rows.groupby('kickoff_at').role.apply(lambda x: 'validation' in set(x) and len(set(x)) > 1).sum() == 0
    assert info['fit']+info['purged']+info['validation'] == 20


def test_manifest_paths_cannot_escape_and_started_runs_never_resume(tmp_path, monkeypatch):
    runner = module()
    with pytest.raises(ValueError, match='escapes'):
        runner.checked_relative(tmp_path, '../outside')
    runner.atomic_json(tmp_path/'status.json', dict(state='failed', completed=[]))
    monkeypatch.setattr(runner, 'verify_manifest', lambda path: (dict(output=str(tmp_path)), 'digest'))
    with pytest.raises(RuntimeError, match='implicit resume'):
        runner.run_matrix(tmp_path/'manifest.json')
    assert not (tmp_path/'matrix.lock').exists()


def test_atomic_progress_and_append_only_trajectory(tmp_path):
    runner = module()
    runner.atomic_json(tmp_path/'progress.json', {'step': 1})
    runner.atomic_json(tmp_path/'progress.json', {'step': 2})
    assert json.loads((tmp_path/'progress.json').read_text()) == {'step': 2}
    assert not (tmp_path/'progress.json.pending').exists()
    for step in (1, 2):
        runner.append_json(tmp_path/'events.jsonl', {'step': step})
    assert [json.loads(line)['step'] for line in (tmp_path/'events.jsonl').read_text().splitlines()] == [1, 2]


def test_gpu_gate_blocks_unknown_and_python_but_records_desktop(monkeypatch):
    from types import SimpleNamespace
    import psutil
    runner = module()
    monkeypatch.setattr(runner.subprocess, 'run', lambda *args, **kwargs:
        SimpleNamespace(stdout='1,unknown\n2,python.exe\n3,dwm.exe\n4,unknown\n5,lghub_system_tray.exe\n6,lghub_system_tray_extra.exe\n'))
    class Process:
        def __init__(self, pid):
            self.pid = pid
            if pid == 4: raise psutil.AccessDenied(pid)
        def name(self): return {1:'other.exe', 2:'python.exe', 3:'dwm.exe',
            5:'lghub_system_tray.exe', 6:'lghub_system_tray_extra.exe'}[self.pid]
        def exe(self):
            if self.pid == 3: raise psutil.AccessDenied(self.pid)
            if self.pid == 5: return 'C:/Program Files/LGHUB/system_tray/lghub_system_tray.exe'
            return 'C:/fake/'+self.name()
    monkeypatch.setattr(psutil, 'Process', Process)
    report = runner.gpu_processes()
    assert any(row.get('name') == 'python.exe' for row in report['blocking'])
    assert any(row.get('name') == 'lghub_system_tray_extra.exe' for row in report['blocking'])
    assert any('resolution_error' in row for row in report['blocking'])
    if runner.os.name == 'nt':
        assert [row['name'] for row in report['excluded_desktop']] == ['dwm.exe', 'lghub_system_tray.exe']
        assert report['excluded_desktop'][0]['path_resolution_limitation']
    else:
        assert not report['excluded_desktop']
    own = runner.gpu_processes(ignore_pids=(2,))
    assert [row['pid'] for row in own['ignored_own']] == [2]


def launcher_probe(tmp_path, mode, unrelated=False):
    """Launch the actual active interpreter, including Windows' venv redirector."""
    import psutil
    runner = module()
    script = tmp_path/'launcher_probe.py'
    script.write_text(
        'import sys,os,json,time,psutil\n'
        f'sys.path.insert(0,{str(runner.SOURCE/"tools/benchmarks")!r})\n'
        'from gat_dependency_matrix import validate_launcher_ancestry\n'
        'mode=sys.argv[1]; target=int(sys.argv[2]); created=float(sys.argv[3]); expected=json.loads(sys.argv[4])\n'
        'try:\n'
        ' chain=validate_launcher_ancestry(target,executable=sys.executable,actual_executable=psutil.Process().exe(),'
        'worker_command=psutil.Process().cmdline(),coordinator_command=expected,coordinator_created=created)\n'
        'except RuntimeError:\n'
        ' if mode=="reject": print(json.dumps({"rejected":True})); sys.exit(0)\n'
        ' raise\n'
        'if mode=="reject": raise AssertionError("unrelated ancestor accepted")\n'
        'result=dict(pid=os.getpid(),chain=chain)\n'
        'if mode=="wait":\n'
        ' from pathlib import Path\n'
        ' Path(sys.argv[5]).write_text(json.dumps(result))\n'
        ' time.sleep(30)\n'
        'else: print(json.dumps(result))\n', encoding='utf-8')
    parent = psutil.Process()
    command = [sys.executable, str(script), mode, str(parent.pid+(1000000 if unrelated else 0)),
               repr(parent.create_time()), json.dumps(parent.cmdline()), str(tmp_path/'ready.json')]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             env=dict(runner.os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1'))
    return runner, child, command


def test_actual_venv_launcher_ancestry_is_validated(tmp_path):
    runner, child, _ = launcher_probe(tmp_path, 'validate')
    stdout, stderr = child.communicate(timeout=15)
    assert child.returncode == 0, stderr
    report = json.loads(stdout)
    assert report['chain'][-1]['pid'] == runner.os.getpid()
    if report['pid'] != child.pid:
        assert report['chain'][0]['pid'] == child.pid
        assert runner._same_executable(report['chain'][0]['executable'], sys.executable)


def test_actual_venv_launcher_rejects_unrelated_ancestor(tmp_path):
    _, child, _ = launcher_probe(tmp_path, 'reject', unrelated=True)
    stdout, stderr = child.communicate(timeout=15)
    assert child.returncode == 0, stderr
    assert json.loads(stdout) == {'rejected': True}


def test_owned_launcher_cleanup_stops_actual_child_only(tmp_path):
    import psutil
    runner, child, command = launcher_probe(tmp_path, 'wait')
    created = psutil.Process(child.pid).create_time()
    ready = tmp_path/'ready.json'
    deadline = time.monotonic()+10
    while not ready.exists() and time.monotonic() < deadline and child.poll() is None:
        time.sleep(.02)
    assert ready.exists(), child.communicate(timeout=1)
    actual_pid = json.loads(ready.read_text())['pid']
    report = runner.terminate_owned_worker(child, command=command, executable=sys.executable,
        actual_executable=psutil.Process().exe(), created=created)
    assert actual_pid in report['terminated']
    assert child.poll() is not None
    assert not psutil.pid_exists(actual_pid)


def test_owned_launcher_cleanup_refuses_unrelated_command(tmp_path):
    import psutil
    runner, child, command = launcher_probe(tmp_path, 'validate')
    created = psutil.Process(child.pid).create_time()
    with pytest.raises(RuntimeError, match='unverified/reused'):
        runner.terminate_owned_worker(child, command=[sys.executable, 'unrelated.py'],
            executable=sys.executable, actual_executable=psutil.Process().exe(), created=created)
    stdout, stderr = child.communicate(timeout=15)
    assert child.returncode == 0, stderr
    assert json.loads(stdout)['chain'][-1]['pid'] == runner.os.getpid()


def test_lease_accepts_pinned_B_flag_and_relative_manifest(tmp_path, monkeypatch):
    import psutil
    runner = module()
    path = tmp_path/'manifest.json'
    executable = sys.executable
    actual = psutil.Process().exe()
    manifest = dict(executable=executable, runtime={'actual_executable': actual},
                    source={'sha256':'source'}, output=str(tmp_path))
    lease = dict(pid=123, created=456., actual_executable=actual, executable=executable,
        command=[executable, '-B', runner.os.path.relpath(runner.__file__, tmp_path), '--run', '--manifest', 'manifest.json'],
        cwd=str(tmp_path), manifest=str(path.resolve()), manifest_sha256='manifest', source_sha256='source', output=str(tmp_path))
    def validated(pid, **kwargs):
        assert pid == 123 and kwargs['coordinator_created'] == 456.
        assert kwargs['coordinator_command'] == lease['command']
        return [{'pid':123}]
    monkeypatch.setattr(runner, 'validate_launcher_ancestry', validated)
    assert runner.validate_worker_lease(lease, path, manifest, 'manifest', 'fit') == [{'pid':123}]
    lease['command'][2] = 'unrelated.py'
    with pytest.raises(RuntimeError, match='source-pinned matrix run'):
        runner.validate_worker_lease(lease, path, manifest, 'manifest', 'fit')


def test_cuda_selection_and_initialization_precede_peak_reset():
    from types import SimpleNamespace
    runner = module()
    calls = []
    cuda = SimpleNamespace(is_available=lambda: calls.append('available') or True,
        set_device=lambda device: calls.append(('select', device)),
        init=lambda: calls.append('initialize'),
        reset_peak_memory_stats=lambda device: calls.append(('reset', device)))
    runner.initialize_worker_device(SimpleNamespace(cuda=cuda), 'cuda:0')
    assert calls == ['available', ('select', 'cuda:0'), 'initialize', ('reset', 'cuda:0')]
    calls.clear()
    runner.initialize_worker_device(SimpleNamespace(cuda=cuda), 'cpu')
    assert not calls


def test_unavailable_cuda_never_selects_or_resets():
    from types import SimpleNamespace
    runner = module()
    calls = []
    cuda = SimpleNamespace(is_available=lambda: False,
        set_device=lambda device: calls.append('select'), init=lambda: calls.append('init'),
        reset_peak_memory_stats=lambda device: calls.append('reset'))
    with pytest.raises(RuntimeError, match='CUDA unavailable'):
        runner.initialize_worker_device(SimpleNamespace(cuda=cuda), 'cuda:0')
    assert not calls
