"""Outcome-free, identity-aligned multi-stream decisions, independent of fitting."""
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
import pandas as pd

_FORBIDDEN = {'y', 'target', 'outcome', 'settlement', 'profit', 'payout', 'result', 'actual', 'label'}


def _safe_columns(columns):
    for column in columns:
        tokens = str(column).lower().replace('::', '_').split('_')
        if any(token in _FORBIDDEN for token in tokens):
            raise ValueError(f'Outcome-bearing field {column!r} is forbidden in decision/allocation inputs.')


@dataclass(frozen=True, init=False)
class FrozenTable:
    """Copies on input/output; no reference to a reporting context survives."""
    _columns: tuple
    _index: tuple
    _index_name: object
    _records: tuple

    def __init__(self, frame):
        if not isinstance(frame, pd.DataFrame) or not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError('Safe tables require distinct row keys and columns.')
        if frame.index.to_frame(index=False).isna().any().any():
            raise ValueError('Safe table identities cannot be missing.')
        _safe_columns(frame.columns)
        object.__setattr__(self, '_columns', tuple(frame.columns))
        object.__setattr__(self, '_index', tuple(frame.index))
        object.__setattr__(self, '_index_name', frame.index.name)
        object.__setattr__(self, '_records', tuple(tuple(deepcopy(v) for v in row) for row in frame.itertuples(index=False, name=None)))

    @property
    def frame(self):
        return pd.DataFrame(deepcopy(self._records), columns=self._columns, index=pd.Index(self._index, name=self._index_name))


@dataclass(frozen=True)
class DecisionContext:
    time: object
    candidates: FrozenTable
    features: FrozenTable | None = None

    def __post_init__(self):
        time = pd.to_datetime(self.time, utc=True, errors='raise')
        if pd.isna(time) or not isinstance(self.candidates, FrozenTable) or (self.features is not None and not isinstance(self.features, FrozenTable)):
            raise ValueError('Decision contexts require a time and copied safe tables.')
        object.__setattr__(self, 'time', time)
        table = self.candidates.frame
        for column in ('quote_at', 'decision_at'):
            if column not in table or pd.to_datetime(table[column], utc=True).isna().any() or (pd.to_datetime(table[column], utc=True) > time).any():
                raise ValueError(f'Decision candidates need available {column}.')


@dataclass
class DecisionResult:
    selected: pd.DataFrame
    audit: pd.DataFrame


@dataclass(frozen=True)
class DecisionLayer:
    """Gate identical candidates with each model's probability or complete-ticket EV."""
    models: tuple
    gate: str = 'and'
    metric: str = 'ev'
    threshold: float = 0.
    strict: bool = True
    missing: str = 'error'
    identity_columns: tuple = ('economic_key', 'quote_id', 'quote_at', 'decision_at', 'odds')

    def decide(self, predictions, context):
        if not self.models or len(set(self.models)) != len(self.models) or self.gate not in {'and', 'or'} or self.metric not in {'ev', 'probability'} or self.missing not in {'error', 'reject'} or not np.isfinite(self.threshold):
            raise ValueError('Invalid gate configuration.')
        candidates = context.candidates.frame
        if not set(self.identity_columns) <= set(candidates):
            raise ValueError('Candidates lack economic/quote/time identities.')
        if candidates[list(self.identity_columns)].isna().any().any():
            raise ValueError('Candidate economic identities must be nonmissing.')
        approvals, records = [], []
        for model in self.models:
            frame = predictions.get(model)
            if frame is None:
                if self.missing == 'error':
                    raise ValueError(f'Missing model {model}.')
                frame = pd.DataFrame(index=candidates.index, columns=[*self.identity_columns, 'probability'])
            if not frame.index.is_unique or len(frame.index.difference(candidates.index)):
                raise ValueError('Duplicate or unexpected prediction event keys.')
            frame = frame.reindex(candidates.index)
            missing = frame.isna().any(axis=1)
            if missing.any() and self.missing == 'error':
                raise ValueError('Exactly one available model prediction is required for every candidate.')
            present = ~missing
            for key in self.identity_columns:
                if key not in frame or not frame.loc[present, key].equals(candidates.loc[present, key]):
                    raise ValueError(f'Model {model} economic identity differs at {key}.')
            p = pd.to_numeric(frame.probability, errors='raise')
            if (present & (~np.isfinite(p) | ~p.between(0, 1))).any():
                raise ValueError('Model probability must be finite and in [0,1].')
            for key in ('artifact_vintage', 'trained_through', 'issued_at'):
                if key not in frame:
                    raise ValueError(f'Model valuations require {key} provenance.')
                t = pd.to_datetime(frame[key], utc=True, errors='raise')
                if (present & (t.isna() | (t > context.time) | (t > pd.to_datetime(candidates.decision_at, utc=True)))).any():
                    raise ValueError('Model evidence was unavailable at decision time.')
            odds = pd.to_numeric(candidates.odds, errors='raise')
            if not np.isfinite(odds).all() or (odds <= 1).any():
                raise ValueError('Decision EV needs decimal odds >1.')
            if self.metric == 'ev' and ('payoff' not in candidates or not candidates.payoff.eq('binary').all()):
                raise ValueError('EV gates currently require binary win/loss payoff.')
            value = p * odds - 1 if self.metric == 'ev' else p
            take = ((value > self.threshold) if self.strict else (value >= self.threshold)) & present
            approvals.append(take)
            records.extend(dict(candidate_id=k, model=model, probability=p.loc[k], value=value.loc[k], take=bool(take.loc[k]),
                                reason='missing_model' if missing.loc[k] else 'accepted' if take.loc[k] else 'threshold',
                                comparator='>' if self.strict else '>=') for k in candidates.index)
        votes = pd.concat(approvals, axis=1)
        accepted = votes.all(axis=1) if self.gate == 'and' else votes.any(axis=1)
        return DecisionResult(candidates.loc[accepted].copy(), pd.DataFrame(records))
