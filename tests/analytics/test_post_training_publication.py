"""Bounded Windows publication recovery and visibility of permanent failures."""
from pathlib import Path
import pytest
from xdiyo_analytics.experiments import ExperimentStore
from xdiyo_analytics.experiments import store as module
from test_post_training_experiments import computed_run


class Stage:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = []

    def rename(self, destination):
        self.calls.append(destination)
        if self.failures:
            raise self.failures.pop(0)

    def __str__(self):
        return 'bounded-test/.pending-run'


def windows_error(code):
    error = OSError('injected Windows publication error')
    error.winerror = code
    return error


@pytest.mark.parametrize('code', [5, 32, 33])
def test_transient_windows_errors_recover_with_declared_delays(code, monkeypatch):
    delays = []
    monkeypatch.setattr(module.time, 'sleep', delays.append)
    stage = Stage([windows_error(code), windows_error(code)])
    module._publish(stage, 'published')
    assert stage.calls == ['published']*3
    assert delays == [.05, .1]


@pytest.mark.parametrize('code', [None, 2, 13])
def test_ordinary_errors_propagate_immediately(code, monkeypatch):
    delays = []
    monkeypatch.setattr(module.time, 'sleep', delays.append)
    error = OSError('ordinary failure') if code is None else windows_error(code)
    stage = Stage([error])
    with pytest.raises(OSError) as caught:
        module._publish(stage, 'published')
    assert caught.value is error
    assert delays == [] and stage.calls == ['published']
    assert '.pending-run' in caught.value.__notes__[0]


@pytest.mark.parametrize('code', [5, 32, 33])
def test_persistent_windows_errors_have_exact_attempt_bound(code, monkeypatch):
    delays = []
    monkeypatch.setattr(module.time, 'sleep', delays.append)
    error = windows_error(code)
    stage = Stage([error]*6)
    with pytest.raises(OSError) as caught:
        module._publish(stage, 'published')
    assert caught.value is error
    assert len(stage.calls) == 6 and delays == [.05, .1, .2, .4, .8]
    assert sum(delays) == 1.55
    assert '.pending-run' in error.__notes__[0]


@pytest.mark.parametrize('save_kind', ['run', 'failure'])
def test_both_save_paths_use_atomic_publication_and_keep_permanent_stage_hidden(save_kind, tmp_path, monkeypatch):
    store = ExperimentStore(tmp_path, 'publication failure')
    training, report = computed_run()
    attempts, delays = [], []
    error = windows_error(5)
    original = Path.rename
    def fail_publication(self, destination):
        if self.name.startswith('.pending-'):
            attempts.append(self)
            raise error
        return original(self, destination)
    monkeypatch.setattr(Path, 'rename', fail_publication)
    monkeypatch.setattr(module.time, 'sleep', delays.append)
    with pytest.raises(OSError) as caught:
        if save_kind == 'run':
            store.save_run(training, report, name='unpublished', config={}, save_predictions=False)
        else:
            store.save_failure(name='unpublished', config={}, error='candidate failure')
    assert caught.value is error and len(attempts) == 6
    assert delays == [.05, .1, .2, .4, .8]
    assert store.read_runs() == []
    pending = list((store.path / 'runs').glob('.pending-*'))
    assert len(pending) == 1 and (pending[0] / 'run.json').is_file()
