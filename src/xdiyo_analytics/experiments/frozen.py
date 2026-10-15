"""Explicit owned execution boundary for shared search inputs.

Mutable callers keep execution_key's full validation. A snapshot holds isolated
copies, never caller aliases, and provides fresh copies to each consumer. Source
content is verified before returning a key; paths/mtimes are not evidence.
"""
from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path
import hashlib
import sys
import numpy as np
import pandas as pd
from .._ownership import isolated_frame
from .recovery import signature, training_source_files
from .store import configuration_hash


def isolate(value):
    if isinstance(value, pd.DataFrame):
        return isolated_frame(value)
    if isinstance(value, pd.Series):
        result = isolated_frame(value.to_frame()).iloc[:, 0]
        result.name = deepcopy(value.name)
        result.attrs = deepcopy(value.attrs)
        return result
    if is_dataclass(value) and not isinstance(value, type):
        # Preserve init=False state and custom post-init results as well. Calling
        # the constructor again can silently recompute these from different data.
        result = deepcopy(value)
        for field in fields(value):
            object.__setattr__(result, field.name, isolate(getattr(value, field.name)))
        return result
    if isinstance(value, dict):
        return {deepcopy(k): isolate(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(isolate(v) for v in value)
    if isinstance(value, list):
        return [isolate(v) for v in value]
    return deepcopy(value)


class FrozenExecution:
    def __init__(self, *shared):
        self.__values = tuple(isolate(value) for value in shared)
        self.__signatures = tuple(signature(value) for value in self.__values)
        self.__root = Path(__file__).resolve().parents[1]
        self.__source = tuple((p, p.read_bytes()) for p in training_source_files(self.__root))
        digest = hashlib.sha256()
        for path, contents in self.__source:
            digest.update(str(path.relative_to(self.__root)).encode())
            digest.update(contents)
        self.__digest = digest.hexdigest()

    def copy(self, index):
        return isolate(self.__values[index])

    def key(self, *values, shared_positions=()):
        # Positions map an argument slot to a frozen input. Placeholder values
        # have no authority; shared values are precisely those copied above.
        current = training_source_files(self.__root)
        if current != [p for p, _ in self.__source] or any(p.read_bytes() != data for p, data in self.__source):
            raise RuntimeError('Source changed during frozen execution; rebuild the execution snapshot.')
        shared_positions = dict(shared_positions)
        inputs = [self.__signatures[shared_positions[i]] if i in shared_positions else signature(v)
                  for i, v in enumerate(values)]
        return configuration_hash(dict(source=self.__digest, python=sys.version,
                                      numpy=np.__version__, pandas=pd.__version__, inputs=inputs))
