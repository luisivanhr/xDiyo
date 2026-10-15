"""Operation-scoped, byte-bounded reuse of already verified source tables."""
from collections import OrderedDict
from contextvars import ContextVar
from functools import wraps
from threading import RLock

_active = ContextVar('xdiyo_odds_tables', default=None)
_LIMIT = 64*1024**2


class _Tables:
    def __init__(self, limit=_LIMIT):
        self.limit, self.bytes = limit, 0
        self.entries = OrderedDict()
        self._lock = RLock()

    def get(self, key, load):
        # Context propagation may share this operation with another thread.
        with self._lock:
            return self._get(key, load)

    def _get(self, key, load):
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key][0]
        frame = load()
        size = int(frame.memory_usage(index=True, deep=True).sum())
        if size <= self.limit:
            while self.entries and self.bytes+size > self.limit:
                _, (_, previous) = self.entries.popitem(last=False)
                self.bytes -= previous
            self.entries[key] = (frame, size)
            self.bytes += size
        return frame


def table(key, load):
    cache = _active.get()
    return load() if cache is None else cache.get(key, load)


def operation(function):
    """Nested calls share a cache; independent calls and threads do not."""
    @wraps(function)
    def call(*args, **kwargs):
        if _active.get() is not None:
            return function(*args, **kwargs)
        token = _active.set(_Tables())
        try:
            return function(*args, **kwargs)
        finally:
            _active.reset(token)
    return call
