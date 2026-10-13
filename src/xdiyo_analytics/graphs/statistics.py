"""Training-only statistic transforms; labels stay outside neural inputs."""
from dataclasses import asdict
import numpy as np
import pandas as pd
from .config import MATCH_KEYS


def labels(store, metadata, config, known_at):
    source = store._history
    keys = [*MATCH_KEYS, 'side']
    for target in config.targets:
        if target.column not in source:
            raise ValueError(f'Statistics target {target.name}: missing native column {target.column}; no period substitution.')
    # Never copy heatmap arrays or unrelated edge columns to look up labels.
    columns = list(dict.fromkeys([*keys, *(t.column for t in config.targets)]))
    indexed = source.loc[:, columns].set_index(keys)
    row_lookup = pd.Series(np.arange(len(source)), index=indexed.index)
    output = []
    for target in config.targets:
        wanted = pd.MultiIndex.from_tuples([(*row, target.side) for row in metadata[list(MATCH_KEYS)].itertuples(index=False, name=None)], names=keys)
        rows = indexed.reindex(wanted)
        y = rows[target.column].to_numpy(dtype=float, na_value=np.nan, copy=True)
        # Do not train on an uncompleted or not-yet-released fixture label.
        loc = row_lookup.reindex(wanted)
        valid = loc.notna().to_numpy()
        positions = loc.fillna(0).to_numpy(dtype=int)
        valid &= store._finished[positions] & (store._available_ns[positions] != np.iinfo(np.int64).min) & (store._available_ns[positions] <= pd.Timestamp(known_at).value)
        y[~valid] = np.nan
        if np.isinf(y).any() or np.any(y[np.isfinite(y)] < 0):
            raise ValueError(f'Invalid support for {target.name}.')
        if target.proportion:
            if np.any(y[np.isfinite(y)] > 100):
                raise ValueError('Native possession targets must be percentages in [0,100].')
            y = y/100.
        elif np.any(y[np.isfinite(y)] != np.floor(y[np.isfinite(y)])):
            raise ValueError(f'Count target {target.name} must contain integers.')
        if np.any(np.isfinite(y) & (np.abs(y) > np.finfo(np.float32).max)):
            raise ValueError(f'Statistics target {target.name} exceeds neural float32 range.')
        output.append(y)
    return np.column_stack(output)


class StatisticsScale:
    def __init__(self, config):
        self.config = config

    def fit(self, values):
        transformed = self.transform(values)
        self.counts = np.isfinite(values).sum(0)
        if (self.counts == 0).any():
            missing = [t.name for t, n in zip(self.config.targets, self.counts) if n == 0]
            raise ValueError(f'No usable training observations for statistic targets: {missing}')
        self.mean = np.nanmean(transformed, axis=0)
        scale = np.nanstd(transformed, axis=0)
        self.scale = np.where(scale > 1e-12, scale, 1.)
        self.mask_shape = list(values.shape)
        self.mask_bits = np.packbits(np.isfinite(values), bitorder='little').tobytes().hex()
        return self

    def transform(self, values):
        return np.column_stack([values[:, i] if t.proportion else np.log1p(values[:, i])
                                for i, t in enumerate(self.config.targets)])

    def inverse(self, transformed):
        """Inverse of target transforms, returning native percentages/counts."""
        return np.column_stack([transformed[:, i]*100 if t.proportion else np.expm1(transformed[:, i])
                                for i, t in enumerate(self.config.targets)])

    def state(self):
        return dict(targets=[asdict(t) for t in self.config.targets], mean=self.mean.tolist(), scale=self.scale.tolist(),
                    counts=self.counts.tolist(), mask_shape=self.mask_shape, mask_bits=self.mask_bits,
                    transforms=['percentage_to_fraction' if t.proportion else 'log1p' for t in self.config.targets])

    @classmethod
    def from_state(cls, config, state):
        obj = cls(config)
        if state['targets'] != [asdict(t) for t in config.targets]:
            raise ValueError('Saved statistics target order/configuration mismatch.')
        for key in ('mean', 'scale', 'counts'):
            setattr(obj, key, np.asarray(state[key]))
        if len(obj.mean) != len(config.targets) or not np.isfinite(obj.mean).all() or not np.isfinite(obj.scale).all() or (obj.scale <= 0).any():
            raise ValueError('Invalid saved statistics transform.')
        obj.mask_shape, obj.mask_bits = state['mask_shape'], state['mask_bits']
        return obj
