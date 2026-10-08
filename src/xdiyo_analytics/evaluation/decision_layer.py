"""Outcome-free, identity-aligned multi-stream decisions, independent of fitting."""
from dataclasses import dataclass
from copy import deepcopy
from datetime import datetime, date
import numpy as np
import pandas as pd
from .timestamps import timestamps
from .quote_availability import QuoteAvailability, QUOTE_IDENTITY, ASSUMPTION_FIELDS, _QuoteLegEvidence

_FORBIDDEN = {'y', 'target', 'outcome', 'settlement', 'profit', 'payout', 'result', 'actual', 'label'}
_OUTCOME_FIELDS = {'won','lost','status','is_awarded','gross_return','return_multiplier','net_return_per_unit','settled_at'}


def _outcome_column(column):
    tokens = str(column).lower().replace('::', '_').split('_')
    return any(part in _OUTCOME_FIELDS for part in str(column).lower().split('::')) or any(token in _FORBIDDEN for token in tokens)


def _safe_columns(columns):
    for column in columns:
        if _outcome_column(column):
            raise ValueError(f'Outcome-bearing field {column!r} is forbidden in decision/allocation inputs.')


def _freeze(value):
    if isinstance(value, _QuoteLegEvidence):
        return value
    if isinstance(value, tuple):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or value is pd.NA or isinstance(value, (str, int, float, bool, datetime, date)):
        return value
    raise TypeError('Safe tables allow immutable scalar values or tuples only, not arbitrary objects/references.')


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
        object.__setattr__(self, '_records', tuple(tuple(_freeze(v) for v in row) for row in frame.itertuples(index=False, name=None)))

    @property
    def frame(self):
        return pd.DataFrame(deepcopy(self._records), columns=self._columns, index=pd.Index(self._index, name=self._index_name))


@dataclass(frozen=True)
class DecisionContext:
    time: object
    candidates: FrozenTable
    features: FrozenTable | None = None
    quote_availability: QuoteAvailability = QuoteAvailability()

    def __post_init__(self):
        time = timestamps(self.time, utc=True, errors='raise')
        if pd.isna(time) or not isinstance(self.candidates, FrozenTable) or (self.features is not None and not isinstance(self.features, FrozenTable)):
            raise ValueError('Decision contexts require a time and copied safe tables.')
        object.__setattr__(self, 'time', time)
        table = self.candidates.frame
        if not isinstance(self.quote_availability, QuoteAvailability):
            raise TypeError('Use a typed QuoteAvailability contract.')
        self.quote_availability.validate(table, time)


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

    def decide(self, predictions, context, *, audit_level='full'):
        if audit_level not in ('full', 'summary'):
            raise ValueError('audit_level must be full or summary.')
        if not self.models or len(set(self.models)) != len(self.models) or self.gate not in {'and', 'or'} or self.metric not in {'ev', 'probability'} or self.missing not in {'error', 'reject'} or not np.isfinite(self.threshold):
            raise ValueError('Invalid gate configuration.')
        candidates = context.candidates.frame
        contract = context.quote_availability
        research = contract.mode == 'research_assumed'
        if research and self.missing != 'error':
            raise ValueError('Research quote decisions require missing="error" for complete model preflight.')
        extra = contract.identities(candidates)
        identity_columns = tuple(dict.fromkeys((*self.identity_columns, *(c for c in extra if research or c in candidates))))
        mandatory = set(QUOTE_IDENTITY) | set(self.identity_columns) | {'quote_legs'}
        if research:
            mandatory.update(ASSUMPTION_FIELDS)
            mandatory.discard('quote_at')
        nonmissing = [c for c in identity_columns if c in mandatory]
        if not set(identity_columns) <= set(candidates):
            raise ValueError('Candidates lack economic/quote/time identities.')
        if candidates[nonmissing].isna().any().any():
            raise ValueError('Candidate economic identities must be nonmissing.')
        approvals, records = [], []
        for model in self.models:
            frame = predictions.get(model)
            if frame is None:
                if self.missing == 'error':
                    raise ValueError(f'Missing model {model}.')
                frame = pd.DataFrame(index=candidates.index, columns=[*identity_columns, 'probability', 'artifact_vintage', 'trained_through', 'issued_at'])
            if not frame.index.is_unique or len(frame.index.difference(candidates.index)):
                raise ValueError('Duplicate or unexpected prediction event keys.')
            frame = frame.reindex(candidates.index)
            required = [*identity_columns, 'probability', 'artifact_vintage', 'trained_through', 'issued_at']
            if not set(required) <= set(frame):
                raise ValueError('Model valuations lack economic identities, probability or timing provenance.')
            missing = frame[[*nonmissing, 'probability', 'artifact_vintage', 'trained_through', 'issued_at']].isna().any(axis=1)
            if missing.any() and self.missing == 'error':
                raise ValueError('Exactly one available model prediction is required for every candidate.')
            present = ~missing
            if research:
                contract.validate(frame, context.time, complete=True)
            for key in identity_columns:
                left, right = frame.loc[present, key], candidates.loc[present, key]
                same = (left.eq(right) | (left.isna() & right.isna())).fillna(False).all() if research else left.equals(right)
                if present.any() and not same:
                    raise ValueError(f'Model {model} economic identity differs at {key}.')
            p = pd.to_numeric(frame.probability, errors='raise')
            if (p.notna() & (~np.isfinite(p) | ~p.between(0, 1))).any():
                raise ValueError('Model probability must be finite and in [0,1].')
            for key in ('artifact_vintage', 'trained_through', 'issued_at'):
                if key not in frame:
                    raise ValueError(f'Model valuations require {key} provenance.')
                t = timestamps(frame[key], utc=True, errors='raise')
                if (present & (t.isna() | (t > context.time) | (t > timestamps(candidates.decision_at, utc=True)))).any():
                    raise ValueError('Model evidence was unavailable at decision time.')
            issued = timestamps(frame.issued_at, utc=True)
            if (present & ((timestamps(frame.trained_through, utc=True) >= issued) | (timestamps(frame.artifact_vintage, utc=True) > issued))).any():
                raise ValueError('Model evidence was unavailable at its prediction issue time.')
            odds = pd.to_numeric(candidates.odds, errors='raise')
            if not np.isfinite(odds).all() or (odds <= 1).any():
                raise ValueError('Decision EV needs decimal odds >1.')
            if self.metric == 'ev' and ('payoff' not in candidates or not candidates.payoff.eq('binary').all()):
                raise ValueError('EV gates currently require binary win/loss payoff.')
            value = p * odds - 1 if self.metric == 'ev' else p
            take = ((value > self.threshold) if self.strict else (value >= self.threshold)) & present
            approvals.append(take)
            if audit_level == 'summary':
                # Compact computational ballots, without per-row provenance
                # dictionaries. All guards above are identical in both modes.
                records.append(pd.DataFrame(dict(candidate_id=candidates.index, model=model,
                    probability=p.to_numpy(), value=value.to_numpy(), take=take.to_numpy(),
                    reason=np.where(missing, 'missing_model', np.where(take, 'accepted', 'threshold')),
                    comparator='>' if self.strict else '>=')))
                continue
            records.extend(dict(candidate_id=k, model=model, probability=p.loc[k], value=value.loc[k], take=bool(take.loc[k]),
                                reason='missing_model' if missing.loc[k] else 'accepted' if take.loc[k] else 'threshold',
                                comparator='>' if self.strict else '>=',
                                **({**contract.labels, 'assumed_available_at':pd.Timestamp(candidates.loc[k,'assumed_available_at']).isoformat(),
                                    'quote_at':None if pd.isna(candidates.loc[k,'quote_at']) else pd.Timestamp(candidates.loc[k,'quote_at']).isoformat(),
                                    'quote_id':candidates.loc[k,'quote_id'],
                                    'decision_at':pd.Timestamp(candidates.loc[k,'decision_at']).isoformat(),
                                    **({'quote_legs':candidates.loc[k,'quote_legs']} if 'quote_legs' in candidates else {})} if research else {})) for k in candidates.index)
        votes = pd.concat(approvals, axis=1)
        accepted = votes.all(axis=1) if self.gate == 'and' else votes.any(axis=1)
        return DecisionResult(candidates.loc[accepted].copy(),
                              pd.concat(records, ignore_index=True) if audit_level == 'summary' else pd.DataFrame(records))


@dataclass(frozen=True)
class LearnedGate:
    """A fitted take/skip adapter with a declared cutoff and safe feature columns."""
    model: object
    feature_columns: tuple
    positive_class: object = 1
    threshold: float = .5

    def decide(self, predictions, context):
        if not np.isfinite(self.threshold) or not 0 <= self.threshold <= 1 or getattr(getattr(self.model,'target_spec',None),'kind',None) != 'take_skip':
            raise ValueError('Learned gates require an explicit fitted take_skip target adapter.')
        cutoff=getattr(self.model,'training_cutoff_',None)
        if cutoff is None or cutoff >= context.time:
            raise ValueError('Learned gate must be fitted strictly before the decision.')
        candidates=context.candidates.frame
        if context.features is None:
            raise ValueError('Supply explicit outcome-free gate features.')
        features=context.features.frame
        if not features.index.equals(candidates.index):
            raise ValueError('Gate feature identities differ from candidates.')
        from ..training.contracts import PredictionContext
        prediction=PredictionContext(features[list(self.feature_columns)].copy(),candidates.copy(),'match',('economic_key',),0)
        outputs=self.model.predict(prediction)
        p=outputs.get('predict_proba')
        columns=[c for c in p.columns if isinstance(c,tuple) and c[1]==self.positive_class] if p is not None else []
        if len(columns)!=1 or not p.index.equals(candidates.index):
            raise ValueError('Learned gate requires one explicit positive-class probability per candidate.')
        values=p[columns[0]]
        if not np.isfinite(values).all() or not values.between(0,1).all():
            raise ValueError('Invalid learned gate probabilities.')
        take=values>=self.threshold
        return DecisionResult(candidates.loc[take].copy(),pd.DataFrame({'probability':values,'take':take,'reason':'learned_gate'},index=candidates.index))
