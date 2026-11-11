"""Bounded, process-local residency for native immutable input snapshots.

Cache entries are private. Consumers lease independent tensor copies, including
transfers in the aggregate budget. No activation, gradient or model is cached.
CUDA allocator reserved memory is distinct from these live tensor bytes.
"""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
import weakref


@dataclass(frozen=True)
class InputCachePolicy:
    max_bytes: int = 0
    headroom_bytes: int = 1073741824
    admission: str = 'lru'

    def __post_init__(self):
        for name in ('max_bytes', 'headroom_bytes'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer.')
        if self.admission not in ('lru', 'retain'):
            raise ValueError("admission must be 'lru' or 'retain'.")


def _map(value, function):
    import torch
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, dict):
        return {k: _map(v, function) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_map(v, function) for v in value)
    if isinstance(value, list):
        return [_map(v, function) for v in value]
    # Positions and scalar metadata must also be isolated from consumers.
    from copy import deepcopy
    return deepcopy(value)


def tensor_bytes(value):
    count = 0
    def add(tensor):
        nonlocal count
        count += tensor.numel() * tensor.element_size()
        return tensor
    _map(value, add)
    return count


class _Pool:
    def __init__(self, device):
        self.device = device
        self.lock = RLock()
        self.owners = weakref.WeakKeyDictionary()
        self.entries = OrderedDict()
        self.resident = self.transient = self.hits = self.misses = 0

    def _purge(self):
        for key in list(self.entries):
            if key[0]() is None:
                _, size = self.entries.pop(key)
                self.resident -= size

    def _trim(self, needed, budget, protect=None):
        while self.entries and self.resident + self.transient + needed > budget:
            key = next((key for key in self.entries if key != protect), None)
            if key is None:
                break
            _, size = self.entries.pop(key)
            self.resident -= size

    def clear(self, owner):
        with self.lock:
            for key in list(self.entries):
                if key[0]() is owner or key[0]() is None:
                    _, size = self.entries.pop(key)
                    self.resident -= size
            self.owners.pop(owner, None)

    def _available(self, reusable=False):
        """Free driver bytes plus reusable blocks owned by our CUDA allocator."""
        import torch
        free, _ = torch.cuda.mem_get_info(self.device)
        if reusable and str(self.device).startswith('cuda'):
            free += max(0, torch.cuda.memory_reserved(self.device)
                        - torch.cuda.memory_allocated(self.device))
        return free

    def _trim_headroom(self, required, protect=None):
        while self.entries and self._available(reusable=True) < required:
            key = next((key for key in self.entries if key != protect), None)
            if key is None:
                break
            value, size = self.entries.pop(key)
            self.resident -= size
            del value

    @contextmanager
    def lease(self, owner, key, build_cpu):
        import torch
        reserved = 0
        value = None
        with self.lock:
            self._purge()
            self.owners[owner] = owner.policy
            budget = min(p.max_bytes for p in self.owners.values())
            headroom = max(p.headroom_bytes for p in self.owners.values())
            retain = owner.policy.admission == 'retain'
            token = (weakref.ref(owner), key)
            entry = self.entries.get(token)
            if entry is None:
                self.misses += 1
                cpu = build_cpu()
                size = tensor_bytes(cpu)
                # Two copies: retained input and the isolated consumer lease.
                if not retain:
                    self._trim(2 * size, budget)
                free = self._available(reusable=retain)
                cacheable = self.resident + self.transient + 2 * size <= budget and free >= headroom + 2 * size
                if cacheable:
                    fixed = _map(cpu, lambda t: t.to(self.device, copy=True))
                    self.entries[token] = (fixed, size)
                    self.resident += size
                    entry = (fixed, size)
                else:
                    if retain:
                        # A sequential scan larger than the cache must not
                        # displace every reusable input merely to admit misses.
                        self._trim(size, budget)
                        self._trim_headroom(headroom + size)
                    else:
                        # Preserve the original LRU streaming fallback.
                        self._trim(budget, budget)
                    value = _map(cpu, lambda t: t.to(self.device, copy=True))
                    reserved = size
            else:
                self.hits += 1
                self.entries.move_to_end(token)
                fixed, size = entry
                self._trim(size, budget, protect=token)
                if retain:
                    self._trim_headroom(headroom + size, protect=token)
                free = self._available(reusable=retain)
                if self.resident + self.transient + size > budget or free < headroom + size:
                    # Remove retained tensors before streaming a replacement.
                    entry = fixed = None
                    self.clear(owner)
                    cpu = build_cpu()
                    value = _map(cpu, lambda t: t.to(self.device, copy=True))
                    reserved = tensor_bytes(value)
            if value is None:
                fixed, size = entry
                value = _map(fixed, lambda t: t.clone())
                reserved = size
            cpu = fixed = entry = None
            self.transient += reserved
        try:
            yield value
        finally:
            value = entry = None
            with self.lock:
                self.transient -= reserved


_POOLS = {}
_POOLS_LOCK = RLock()


class InputResident:
    """Owner of one immutable generation. Native leases never escape updates."""
    def __init__(self, device, policy=None):
        self.policy = policy or InputCachePolicy()
        self.device = str(device)
        self.pool = None
        if self.policy.max_bytes and self.device.startswith('cuda'):
            import torch
            self.device = str(torch.device('cuda', torch.cuda.current_device()) if self.device == 'cuda' else torch.device(self.device))
            with _POOLS_LOCK:
                self.pool = _POOLS.setdefault(self.device, _Pool(self.device))
            with self.pool.lock:
                self.pool.owners[self] = self.policy

    @contextmanager
    def lease(self, key, build_cpu, build_default):
        if self.pool is None:
            yield build_default()
        else:
            with self.pool.lease(self, key, build_cpu) as value:
                yield value

    def close(self):
        if self.pool is not None:
            self.pool.clear(self)

    def __getstate__(self):
        return self.device, self.policy

    def __setstate__(self, state):
        self.__init__(*state)

    def __del__(self):
        if hasattr(self, 'pool'):
            self.close()
