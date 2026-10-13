"""Opt-in synchronized diagnostics; absent from model state and recipes."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import perf_counter

_active = ContextVar('gat_profile', default=None)


class GraphProfile:
    """Time native phases, including nested phases (do not sum overlapping totals).

    CUDA timings synchronize the selected device at both boundaries. This adds
    overhead and serializes outstanding work: compare an unprofiled wall-time
    run separately. Use ``phase('checkpoint_export')`` around external observers.
    No tensors, models or inputs are retained by the profile.
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.timings = {}

    def __enter__(self):
        self._token = _active.set(self)
        return self

    def __exit__(self, *exc):
        _active.reset(self._token)

    def synchronize(self):
        if str(self.device).startswith('cuda'):
            import torch
            torch.cuda.synchronize(self.device)


@contextmanager
def phase(name):
    profile = _active.get()
    if profile is None:
        yield
        return
    profile.synchronize()
    started = perf_counter()
    try:
        yield
    finally:
        profile.synchronize()
        record = profile.timings.setdefault(name, dict(calls=0, seconds=0.))
        record['calls'] += 1
        record['seconds'] += perf_counter() - started


def timed(name):
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            if _active.get() is None:
                return function(*args, **kwargs)
            with phase(name):
                return function(*args, **kwargs)
        return call
    return decorate
