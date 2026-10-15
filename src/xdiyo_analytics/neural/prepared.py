"""Owned numeric populations. Public reads cannot alias the retained buffers."""
from hashlib import sha256
import json
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, init=False, eq=False)
class OwnedPopulation:
    """A fit-local immutable snapshot; dropping it releases its entire generation.

    The backing bytes cannot be made writable. Array views and torch tensors are
    never returned publicly: iteration/indexing produce independent copies. Only
    native batching reads the private buffers, and it copies before torch can
    alias them. No parameter-dependent value is retained.
    """
    def __init__(self, arrays, *, binding, validate_kl=False):
        object.__setattr__(self, "_arrays", tuple(self._freeze(value) for value in arrays))
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "kl_validated", bool(validate_kl))
        if validate_kl:
            validate_kl_targets(*self._arrays[:2])
        digest = sha256(binding.encode())
        for array in self._arrays:
            digest.update(repr((array.dtype.str, array.shape)).encode())
            digest.update(array.tobytes())
        object.__setattr__(self, "generation", digest.hexdigest())

    @staticmethod
    def _freeze(value):
        value = np.asarray(value)
        if value.dtype.hasobject:
            raise TypeError('Owned neural populations require numeric arrays.')
        return np.frombuffer(value.tobytes(order='C'), dtype=value.dtype).reshape(value.shape)

    def __len__(self):
        return len(self._arrays)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(value.copy() for value in self._arrays[index])
        return self._arrays[index].copy()

    @property
    def rows(self):
        return len(self._arrays[3])

    @property
    def nbytes(self):
        return sum(value.nbytes for value in self._arrays)

    def batch(self, selected):
        return tuple(value[:, selected].copy() if i == 4 else value[selected].copy()
                     for i, value in enumerate(self._arrays))


def population_binding(spec, inputs, device):
    from dataclasses import asdict
    from ..experiments.recovery import pack
    return json.dumps(pack((asdict(spec), inputs.state(), str(device))), sort_keys=True)


def validate_kl_targets(maps, available):
    """Validate owned CPU targets before their device arithmetic is repeated."""
    target = np.where(available[:, :, None, None].astype(bool), maps, 0)
    if not np.isfinite(target).all() or (target < 0).any():
        raise ValueError('KL reconstruction needs finite nonnegative observed maps.')
    if ((target.max(axis=(2, 3)) == 0) & available.astype(bool)).any():
        raise ValueError('KL reconstruction needs positive mass in every observed map.')
