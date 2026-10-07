"""Explicit output features with stable names; preprocessing belongs inside models."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OutputFeatures:
    names: tuple
    passthrough: tuple = ()
    mode: str = 'identity'
    clip: float = 1e-8

    def transform(self, frames, context):
        if not frames or any(not f.index.equals(context.X.index) for f in frames):
            raise ValueError('Output feature rows must match original context identities.')
        if self.mode not in {'identity', 'logit'} or not 0 < self.clip < .5:
            raise ValueError('Feature transform needs identity/logit and valid clipping.')
        values = np.column_stack([f.to_numpy(dtype=float) for f in frames])
        if self.mode == 'logit':
            if (values < 0).any() or (values > 1).any():
                raise ValueError('Logit transforms probabilities only.')
            values = np.clip(values, self.clip, 1-self.clip)
            values = np.log(values) - np.log1p(-values)
        if len(self.names) != values.shape[1] or len(set((*self.names, *self.passthrough))) != len(self.names) + len(self.passthrough):
            raise ValueError('Declare distinct names for every output and passthrough column.')
        return pd.concat([pd.DataFrame(values, index=context.X.index, columns=list(self.names)), context.X[list(self.passthrough)].copy()], axis=1)
