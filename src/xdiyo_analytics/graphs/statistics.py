"""Training-only statistic transforms; labels stay outside neural inputs."""
from .extensions import target_schema, NumericStatisticTarget
import numpy as np
import pandas as pd
from .config import MATCH_KEYS


def labels(store, metadata, config, known_at):
    source = store._history
    keys = [*MATCH_KEYS, 'side']
    for target in config.targets:
        if target.column not in source:
            raise ValueError(f'Statistics target {target.name}: missing native column {target.column}; no period substitution.')
    if source.empty:
        return np.full((len(metadata), len(config.targets)), np.nan)
    # Never copy heatmap arrays or unrelated edge columns to look up labels.
    columns = list(dict.fromkeys([*keys, *(t.column for t in config.targets)]))
    indexed = source.loc[:, columns].set_index(keys)
    # Gather each fixture side once, rather than copying the full target frame
    # for every target (74 targets would otherwise do quadratic column work).
    row_lookup = pd.Series(np.arange(len(source)), index=indexed.index)
    requested = list(metadata[list(MATCH_KEYS)].itertuples(index=False, name=None))
    sides = {}
    for side in dict.fromkeys(t.side for t in config.targets):
        wanted = pd.MultiIndex.from_tuples([(*row, side) for row in requested], names=keys)
        loc = row_lookup.reindex(wanted)
        positions = loc.fillna(0).to_numpy(dtype=int)
        valid = loc.notna().to_numpy()
        valid &= store._finished[positions] & (store._available_ns[positions] != np.iinfo(np.int64).min) & (store._available_ns[positions] <= pd.Timestamp(known_at).value)
        sides[side] = (positions, valid)
    output = []
    for target in config.targets:
        column = source[target.column]
        if isinstance(target, NumericStatisticTarget) and (not pd.api.types.is_numeric_dtype(column.dtype) or pd.api.types.is_bool_dtype(column.dtype)):
            raise ValueError(f'Numeric target {target.name} requires a numeric source column.')
        positions, valid = sides[target.side]
        y = column.to_numpy(dtype=float, na_value=np.nan)[positions].copy()
        y[~valid] = np.nan
        finite = y[np.isfinite(y)]
        if isinstance(target, NumericStatisticTarget):
            invalid = np.isinf(y).any()
            if target.support in ('count', 'nonnegative_continuous'):
                invalid |= np.any(finite < 0)
            if target.support == 'count':
                invalid |= np.any(finite != np.floor(finite))
            if target.support == 'bounded_continuous':
                invalid |= np.any((finite < target.lower) | (finite > target.upper))
            if invalid:
                raise ValueError(f'Invalid support for {target.name}.')
        else:
            if np.isinf(y).any() or np.any(finite < 0):
                raise ValueError(f'Invalid support for {target.name}.')
            if target.proportion:
                if np.any(finite > 100):
                    raise ValueError('Native possession targets must be percentages in [0,100].')
                y = y/100.
            elif np.any(finite != np.floor(finite)):
                raise ValueError(f'Count target {target.name} must contain integers.')
        if np.any(np.isfinite(y) & (np.abs(y) > np.finfo(np.float32).max)):
            raise ValueError(f'Statistics target {target.name} exceeds neural float32 range.')
        output.append(y)
    return np.column_stack(output)


class StatisticsScale:
    def __init__(self, config):
        self.config = config

    def fit(self, values, weights=None):
        values = np.asarray(values, dtype=float).copy()
        if weights is not None:
            for i,t in enumerate(self.config.targets):
                if isinstance(t, NumericStatisticTarget):
                    values[np.asarray(weights) <= 0, i] = np.nan
        transformed = self.transform(values)
        self.counts = np.isfinite(values).sum(0)
        if (self.counts == 0).any():
            missing = [t.name for t, n in zip(self.config.targets, self.counts) if n == 0]
            raise ValueError(f'No usable training observations for statistic targets: {missing}')
        self.mean = np.nanmean(transformed, axis=0)
        scale = np.nanstd(transformed, axis=0)
        self.scale = np.where(scale > 1e-12, scale, 1.)
        for i,t in enumerate(self.config.targets):
            if isinstance(t, NumericStatisticTarget):
                if t.scaling == 'robust':
                    self.mean[i] = np.nanmedian(transformed[:, i])
                    lo, hi = np.nanquantile(transformed[:, i], [.25,.75])
                    self.scale[i] = hi-lo if hi-lo > 1e-12 else 1.
                elif t.scaling == 'none':
                    self.mean[i], self.scale[i] = 0., 1.
        self.mask_shape = list(values.shape)
        self.mask_bits = np.packbits(np.isfinite(values), bitorder='little').tobytes().hex()
        return self

    def transform(self, values):
        return np.column_stack([transform_target(t, values[:, i])
                                for i, t in enumerate(self.config.targets)])

    def inverse(self, transformed):
        """Inverse of target transforms, returning native percentages/counts."""
        return np.column_stack([inverse_target(t, transformed[:, i])
                                for i, t in enumerate(self.config.targets)])

    def state(self):
        state = dict(targets=[target_schema(t) for t in self.config.targets], mean=self.mean.tolist(), scale=self.scale.tolist(),
                    counts=self.counts.tolist(), mask_shape=self.mask_shape, mask_bits=self.mask_bits,
                    transforms=[t.transform if isinstance(t, NumericStatisticTarget) else 'percentage_to_fraction' if t.proportion else 'log1p' for t in self.config.targets])
        quantiles = [dict(target=t.name, levels=list(t.quantiles), parameterization=parameterization(t), support=t.support if isinstance(t, NumericStatisticTarget) else 'percentage_0_100' if t.proportion else 'nonnegative', outputs=len(t.quantiles)) for t in self.config.targets if t.family == 'quantile']
        if quantiles:
            state['quantile_schema'] = quantiles
        if any(isinstance(t, NumericStatisticTarget) for t in self.config.targets):
            from .extensions import quantile_column
            state['target_manifest'] = [
                dict(order=i, name=t.name, source_column=t.column, side=t.side,
                     units=t.units if isinstance(t, NumericStatisticTarget) else 'percent' if t.proportion else 'count',
                     schema=target_schema(t), fitting_observations=int(self.counts[i]),
                     center=float(self.mean[i]), scale=float(self.scale[i]),
                     outputs=[quantile_column(t,q) for q in t.quantiles] if t.family == 'quantile' else [])
                for i,t in enumerate(self.config.targets)]
        return state

    @classmethod
    def from_state(cls, config, state):
        obj = cls(config)
        if state['targets'] != [target_schema(t) for t in config.targets]:
            raise ValueError('Saved statistics target order/configuration mismatch.')
        for key in ('mean', 'scale', 'counts'):
            setattr(obj, key, np.asarray(state[key]))
        if len(obj.mean) != len(config.targets) or not np.isfinite(obj.mean).all() or not np.isfinite(obj.scale).all() or (obj.scale <= 0).any():
            raise ValueError('Invalid saved statistics transform.')
        obj.mask_shape, obj.mask_bits = state['mask_shape'], state['mask_bits']
        if obj.state().get('quantile_schema') != state.get('quantile_schema'):
            raise ValueError('Saved quantile schema/parameterization mismatch.')
        expected = obj.state()
        if 'target_manifest' in expected:
            for key in ('transforms','target_manifest'):
                if expected[key] != state.get(key):
                    raise ValueError('Saved numeric target schema/transform manifest mismatch.')
        return obj


def transform_target(target, values):
    """Shared NumPy/Torch monotone transform; no fitted state or imputations."""
    generic = isinstance(target, NumericStatisticTarget)
    kind = target.transform if generic else 'identity' if target.proportion else 'log1p'
    if kind == 'identity':
        return values
    if kind == 'affine':
        return (values-target.lower)/(target.upper-target.lower)
    is_numpy = isinstance(values, np.ndarray)
    if kind == 'log1p':
        return np.log1p(values) if is_numpy else values.log1p()
    if is_numpy:
        return np.sign(values)*np.log1p(np.abs(values))
    import torch
    return torch.where(values >= 0, values.clamp_min(0).log1p(), -(-values).clamp_min(0).log1p())


def inverse_target(target, values):
    if not isinstance(target, NumericStatisticTarget):
        return values*100 if target.proportion else np.expm1(values)
    if target.transform == 'identity':
        return values
    if target.transform == 'affine':
        return target.lower+(target.upper-target.lower)*values
    if target.transform == 'log1p':
        return np.expm1(values)
    return np.sign(values)*np.expm1(np.abs(values))


def parameterization(target):
    if isinstance(target, NumericStatisticTarget):
        return {'signed_continuous':'sort_logits_identity',
                'bounded_continuous':'sort_logits_affine_sigmoid'}.get(target.support, 'sort_logits_then_softplus')
    return 'sort_logits_then_sigmoid' if target.proportion else 'sort_logits_then_softplus'
