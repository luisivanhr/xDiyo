"""Typed fixed reducers; averaging parameters does not define a mixture."""
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Mean:
    weights: tuple = ()
    kind: str = 'regression'

    def validate_weights(self, count):
        w = np.asarray(self.weights if self.weights else [1.] * count, dtype=float)
        if not count or w.shape != (count,) or not np.isfinite(w).all() or (w < 0).any() or not w.max() > 0:
            raise ValueError('Weights must be finite, nonnegative and have positive total.')
        w = w / w.max()
        return w / w.sum()

    def reduce(self, frames, schemas):
        if not frames or any(s != schemas[0] for s in schemas):
            raise ValueError('Mean requires identical targets, classes, units, timing and conditioning schemas.')
        if self.kind not in {'regression', 'probability'} or schemas[0].kind != self.kind:
            raise ValueError('Mean supports regression values or event probabilities, not distribution parameters.')
        w = self.validate_weights(len(frames))
        first = frames[0]
        if any(not f.index.equals(first.index) or not f.columns.equals(first.columns) for f in frames):
            raise ValueError('Mean needs exactly aligned rows and columns.')
        return pd.DataFrame(sum(v * f.to_numpy(dtype=float) for v, f in zip(w, frames)), index=first.index, columns=first.columns)


@dataclass(frozen=True)
class HardVote:
    fractions: bool = False

    def reduce(self, frames, schemas):
        if not frames or any(s != schemas[0] for s in schemas) or schemas[0].kind != 'labels':
            raise ValueError('HardVote requires identically typed label outputs.')
        first, schema = frames[0], schemas[0]
        if any(not f.index.equals(first.index) for f in frames):
            raise ValueError('Vote row identities differ.')
        values = np.column_stack([f.iloc[:, 0].to_numpy() for f in frames])
        counts = np.column_stack([(values == c).sum(axis=1) for c in schema.classes]) / len(frames)
        if self.fractions:
            return pd.DataFrame(counts, index=first.index, columns=pd.MultiIndex.from_tuples(
                [(schema.target, c) for c in schema.classes], names=['target', 'class']))
        return pd.DataFrame({schema.target: np.asarray(schema.classes)[counts.argmax(axis=1)]}, index=first.index)


@dataclass(frozen=True)
class DistributionMixture:
    """Complete finite-support PMF mixture; parameter distributions need an event adapter."""
    weights: tuple = ()

    def reduce(self, frames, schemas):
        if not schemas or any(s != schemas[0] for s in schemas) or schemas[0].family != 'categorical_pmf':
            raise ValueError('Only complete finite-support PMFs can be mixed; convert parametric distributions to common events first.')
        if list(frames[0].columns) != list(schemas[0].support):
            raise ValueError('PMF columns must equal the complete declared support.')
        for f in frames:
            a = f.to_numpy(dtype=float)
            if not np.isfinite(a).all() or (a < 0).any() or not np.allclose(a.sum(axis=1), 1, atol=1e-10, rtol=0):
                raise ValueError('Incomplete or invalid PMF.')
        return Mean(self.weights, 'probability').reduce(frames, [replace(s, kind='probability', classes=s.support) for s in schemas])
