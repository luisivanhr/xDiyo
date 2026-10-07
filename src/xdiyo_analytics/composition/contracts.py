"""Typed, identity-checked prediction edges. No implicit conversions."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OutputSchema:
    target: str
    kind: str = 'regression'
    perspective: str = 'match'
    row_keys: tuple = ('event_id',)
    classes: tuple = ()
    units: str = 'count'
    link: str = 'identity'
    horizon: str = 'full_match'
    family: str | None = None
    support: tuple = ()
    parameters: tuple = ()
    conditioning: str = 'decision_features'
    availability_column: str = 'kickoff_at'

    def __post_init__(self):
        for name in ('row_keys', 'classes', 'support', 'parameters'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.target or not self.row_keys or len(set(self.row_keys)) != len(self.row_keys):
            raise ValueError('OutputSchema needs a target and distinct row keys.')
        if self.kind not in {'regression', 'probability', 'labels', 'vote_fraction', 'distribution', 'features'}:
            raise ValueError('Unsupported prediction kind.')
        if self.kind in {'probability', 'labels', 'vote_fraction'} and (not self.classes or len(set(self.classes)) != len(self.classes)):
            raise ValueError('Classification schemas require explicit distinct class order.')
        if self.kind == 'distribution' and (not self.family or not self.support or not self.parameters):
            raise ValueError('Distribution schema needs family, support and parameter meanings.')
        if self.kind == 'distribution':
            self._validate_distribution()

    def _validate_distribution(self):
        if self.family != 'categorical_pmf':
            raise ValueError('Unsupported distribution family validator; use a complete categorical_pmf.')
        if self.parameters != ('mass',) or self.units != 'probability' or self.link != 'identity':
            raise ValueError('categorical_pmf requires mass parameters, probability units and identity link.')
        if not self.support or len(set(self.support)) != len(self.support) or any(pd.isna(v) for v in self.support):
            raise ValueError('PMF support must be distinct and nonmissing.')

    def validate(self, frame, context):
        if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or frame.empty:
            raise ValueError('Prediction edges require nonempty DataFrames with distinct columns.')
        if not frame.index.is_unique or not frame.index.equals(context.X.index):
            raise ValueError('Prediction row identity/order differs from context.')
        keys = list(self.row_keys)
        if not set(keys) <= set(context.metadata) or context.metadata[keys].isna().any().any() or context.metadata.duplicated(keys).any():
            raise ValueError('Missing or duplicate prediction identities.')
        if not context.metadata.index.equals(context.X.index) or context.layout != self.perspective:
            raise ValueError('Prediction metadata identity or perspective differs.')
        if self.availability_column not in context.metadata or pd.to_datetime(context.metadata[self.availability_column], utc=True).isna().any():
            raise ValueError('Prediction schema requires nonmissing as-of timestamps.')
        if self.kind in {'probability', 'vote_fraction'}:
            if list(frame.columns) != [(self.target, c) for c in self.classes]:
                raise ValueError('Probability class mapping/order differs from declared schema.')
            a = frame.to_numpy(dtype=float)
            if not np.isfinite(a).all() or (a < 0).any() or (a > 1).any() or not np.allclose(a.sum(axis=1), 1, atol=1e-10, rtol=0):
                raise ValueError('Invalid probabilities or vote fractions.')
        elif self.kind == 'distribution':
            self._validate_distribution()  # also applies to restored pre-validation schemas
            if list(frame.columns) != list(self.support):
                raise ValueError('PMF columns must equal the complete declared support in order.')
            a = frame.to_numpy(dtype=float)
            if not np.isfinite(a).all() or (a < 0).any() or (a > 1).any() or not np.allclose(a.sum(axis=1), 1, atol=1e-10, rtol=0):
                raise ValueError('Incomplete or invalid PMF.')
        elif self.kind == 'labels':
            if list(frame.columns) != [self.target] or not frame[self.target].isin(self.classes).all():
                raise ValueError('Labels differ from target/class schema.')
        else:
            if self.kind == 'regression' and list(frame.columns) != [self.target]:
                raise ValueError('Regression target columns differ from schema.')
            if not np.isfinite(frame.to_numpy(dtype=float)).all():
                raise ValueError('Prediction values must be finite.')
        return frame


@dataclass(frozen=True)
class OutputRef:
    node: str
    output: str = 'predict'


@dataclass(frozen=True)
class TargetSpec:
    kind: str = 'ordinary'
    loss: str = 'squared_error'
    link: str = 'identity'

    def __post_init__(self):
        if self.kind not in {'ordinary', 'residual', 'oof_error', 'take_skip', 'allocation'}:
            raise ValueError('Unknown meta target kind.')
        if (self.loss, self.link) not in {('squared_error', 'identity'), ('log_loss', 'logit')}:
            raise ValueError('Supported targets use squared_error/identity or binary log_loss/logit.')


@dataclass
class PredictionBundle:
    frames: dict
    schemas: dict
    provenance: dict

    def public(self, refs):
        return {name: self.frames[(ref.node, ref.output)].copy(deep=True) for name, ref in refs.items()}
