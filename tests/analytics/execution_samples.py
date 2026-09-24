"""Synthetic process probes; device names in this module are simulated."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import multiprocessing
import os
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info
from xdiyo_analytics.training import TrainingEvent


class TraceAdapter:
    def __init__(self, *, folder=None, barrier=False, label='probe', delay=.02, fail=False):
        self.folder = None if folder is None else Path(folder)
        self.barrier = barrier
        self.label = label
        self.delay = delay
        self.fail = fail

    def fit(self, context):
        self.pid = os.getpid()
        parent = multiprocessing.parent_process()
        self.parent_pid = None if parent is None else parent.pid
        self.context = deepcopy(replace(context, observer=None))
        self.targets = tuple(context.y)
        self.means = context.y.mean()
        self.native_threads = threadpool_info()
        self.started = time.monotonic()
        if self.folder is not None:
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder / f'{self.label}-{context.fold_id}-start.json').write_text(json.dumps(
                {'pid': self.pid, 'fold': context.fold_id, 'time': self.started}))
        if self.barrier:
            deadline = time.monotonic() + 15
            while len(list(self.folder.glob(f'{self.label}-*-start.json'))) < 2:
                if time.monotonic() > deadline:
                    raise TimeoutError('Second fold did not start concurrently')
                time.sleep(.01)
        if context.observer is not None:
            context.observer(TrainingEvent(context.fold_id, 0, 1,
                {'train_loss': 2., 'worker_pid': self.pid, 'emitted_at': time.monotonic()}, 'train_loss'))
        time.sleep(self.delay if context.fold_id == 0 else min(self.delay, .01))
        if self.fail:
            raise RuntimeError('injected execution failure')
        self.finished = time.monotonic()
        if self.folder is not None:
            (self.folder / f'{self.label}-{context.fold_id}-done.json').write_text(json.dumps(
                {'pid': self.pid, 'fold': context.fold_id, 'time': self.finished}))
        if context.observer is not None:
            context.observer(TrainingEvent(context.fold_id, 0, 2,
                {'train_loss': 1., 'worker_pid': self.pid, 'emitted_at': self.finished}, 'train_loss', final=True))
        self.training_history_ = pd.DataFrame({'step': [1, 2], 'train_loss': [2., 1.]})
        self.training_summary_ = {'trace': self.label}

    def fit_controlled(self, context, control):
        self.control = control
        return self.fit(context)

    def predict(self, context):
        assert not hasattr(context, 'y')
        self.prediction_context = deepcopy(context)
        return {'predict': pd.DataFrame(np.tile(self.means.to_numpy(), (len(context.X), 1)),
            index=context.X.index, columns=self.targets)}


class ParentObserver:
    """Intentionally cannot be serialized, like an interactive display handle."""
    def __init__(self):
        self.calls = []

    def __getstate__(self):
        raise TypeError('This observer must remain in the parent')

    def __call__(self, event):
        self.calls.append((os.getpid(), time.monotonic(), event))


def configure_fake_device(adapter, device):
    adapter.simulated_device = device


def fake_devices():
    return ['cpu', 'cuda:0', 'cuda:2']
