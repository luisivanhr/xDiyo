"""Explicit research assumptions; never manufacture observed quote timestamps."""
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd


QUOTE_IDENTITY = ('quote_id', 'quote_at', 'decision_at', 'odds', 'market', 'selection',
                  'quote_snapshot_hash', 'quote_crosswalk_hash')
ASSUMPTION_FIELDS = ('quote_availability_mode', 'assumed_available_at',
                     'quote_assumption_id', 'quote_assumption_rationale', 'quote_assumption_reference')


@dataclass(frozen=True)
class QuoteAvailability:
    """Observed by default; research_assumed requires an explicit documented attestation.

    assumed_available_at is supplied per quote, not inferred from vendor schedule
    or download time. The contract does not certify historical tradability.
    """
    mode: str = 'observed'
    assumption_id: str | None = None
    rationale: str | None = None
    reference: str | None = None

    def __post_init__(self):
        if self.mode not in ('observed', 'research_assumed'):
            raise ValueError('Quote availability mode must be observed or research_assumed.')
        values = (self.assumption_id, self.rationale, self.reference)
        if self.mode == 'research_assumed':
            if any(not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError('Research quote assumptions require a nonempty identity, rationale and attestation reference.')
        elif any(v is not None for v in values):
            raise ValueError('Observed availability cannot carry a research assumption.')

    @property
    def labels(self):
        return dict(quote_availability_mode=self.mode, quote_assumption_id=self.assumption_id,
                    quote_assumption_rationale=self.rationale, quote_assumption_reference=self.reference)

    def annotate(self, frame):
        """Attach declarations only. Existing evidence must agree; timestamps stay intact."""
        result = frame.copy()
        for field, value in self.labels.items():
            if field in result:
                same = result[field].isna() if value is None else result[field].eq(value)
                if not same.fillna(False).all():
                    raise ValueError(f'Mixed or contradictory quote evidence: {field}.')
            result[field] = value
        if self.mode == 'observed' and 'assumed_available_at' not in result:
            result['assumed_available_at'] = pd.NaT
        return result

    def identities(self, frame):
        base = ('quote_legs', 'quote_id', 'quote_at', 'decision_at', 'odds') if 'quote_legs' in frame else QUOTE_IDENTITY
        # Preserve additional native provenance (period, line, vendor/source pins).
        return tuple(dict.fromkeys((*base, *ASSUMPTION_FIELDS,
            *(c for c in frame if str(c).startswith('quote_') and '::' not in str(c)),
            *(c for c in ('period', 'line') if c in frame))))

    def validate(self, frame, time, *, complete=False):
        """Validate every distinct leg before any aggregate maximum can hide evidence."""
        for field in ('decision_at', 'quote_at'):
            if field not in frame:
                raise ValueError(f'Decision candidates need available {field}.')
        decision = pd.to_datetime(frame.decision_at, utc=True, errors='raise')
        observed = pd.to_datetime(frame.quote_at, utc=True, errors='raise')
        if decision.isna().any() or decision.gt(time).any():
            raise ValueError('Decision candidates need available decision_at.')
        if (observed.notna() & observed.gt(decision)).any():
            raise ValueError('Observed quotes must be available at each candidate decision time.')
        if self.mode == 'observed':
            if observed.isna().any():
                raise ValueError('Observed mode requires nonmissing quote_at.')
            if 'quote_availability_mode' in frame and not frame.quote_availability_mode.eq('observed').fillna(False).all():
                raise ValueError('Research quote evidence requires explicit opt-in; mixed quote modes are forbidden.')
            for field in ASSUMPTION_FIELDS[1:]:
                if field in frame and frame[field].notna().any():
                    raise ValueError('Observed mode cannot carry assumed availability evidence.')
        else:
            for field, expected in self.labels.items():
                if field not in frame or not frame[field].eq(expected).fillna(False).all():
                    raise ValueError(f'Missing or contradictory research quote evidence: {field}.')
            if 'assumed_available_at' not in frame:
                raise ValueError('Research quotes need a separate assumed_available_at.')
            for value in frame.assumed_available_at:
                stamp = pd.Timestamp(value)
                if pd.isna(stamp) or stamp.tzinfo is None:
                    raise ValueError('Assumed availability needs an explicit timezone and nonmissing timestamp.')
            assumed = pd.to_datetime(frame.assumed_available_at, utc=True, errors='raise')
            if assumed.gt(decision).any() or (observed.notna() & observed.gt(assumed)).any():
                raise ValueError('Future or contradictory quote availability assumption.')
            complete = True
        if 'quote_legs' in frame:
            for row in frame.to_dict('records'):
                legs = pd.DataFrame(json.loads(row['quote_legs']))
                from .decision_layer import _safe_columns
                _safe_columns(legs.columns)
                if legs.empty or 'quote_legs' in legs or 'fixture_identity' not in legs or legs.fixture_identity.duplicated().any():
                    raise ValueError('Ticket quote evidence requires distinct unnested fixture legs.')
                if legs.fixture_identity.isna().any() or legs.fixture_identity.map(lambda v: not isinstance(v, str) or not v.strip()).any():
                    raise ValueError('Ticket quote evidence requires nonempty fixture identities.')
                self.validate(legs, pd.Timestamp(row['decision_at']), complete=complete)
                if not pd.to_datetime(legs.decision_at, utc=True).eq(pd.Timestamp(row['decision_at'])).all():
                    raise ValueError('Ticket quote decisions differ from their legs.')
                if float(np.prod(legs.odds)) != row['odds'] or json.dumps(legs.quote_id.tolist()) != row['quote_id']:
                    raise ValueError('Ticket price or quote identities differ from their legs.')
                for field in ('quote_at', 'assumed_available_at'):
                    if field not in legs:
                        continue
                    values = pd.to_datetime(legs[field], utc=True)
                    expected = values.max() if values.notna().all() else pd.NaT
                    actual = pd.to_datetime(row.get(field), utc=True)
                    if not (pd.isna(actual) and pd.isna(expected)) and actual != expected:
                        raise ValueError(f'Ticket {field} differs from its leg evidence.')
        elif complete:
            for field in QUOTE_IDENTITY:
                if field == 'quote_at':
                    continue
                if field not in frame or frame[field].isna().any() or frame[field].map(lambda v: isinstance(v, str) and not v.strip()).any():
                    raise ValueError(f'Quote evidence requires nonempty {field}.')
            prices = pd.to_numeric(frame.odds, errors='raise')
            if not np.isfinite(prices).all() or prices.le(1).any():
                raise ValueError('Quote prices must be finite decimal odds greater than one.')


def quote_leg_records(legs, contract, match_columns):
    """Stable outcome-free membership, including all per-leg source evidence."""
    fields = contract.identities(legs)
    from .decision_layer import _safe_columns
    _safe_columns(fields)
    records = []
    for row in legs.to_dict('records'):
        record = {field: (None if pd.isna(row.get(field)) else row.get(field)) for field in fields}
        record['fixture_identity'] = json.dumps([row[k] for k in match_columns], default=str)
        records.append(record)
    return json.dumps(records, sort_keys=True, default=str)


def validate_model_quotes(legs, models, contract, time):
    """Require independently supplied per-model quote references, never fabricate them."""
    fields = contract.identities(legs)
    for model in models:
        sources = {field: f'{model}::{field}' for field in fields}
        if any(source not in legs for source in sources.values()):
            raise ValueError(f'Model {model} must reference every quote identity and availability field.')
        evidence = legs[list(sources.values())].rename(columns={v: k for k, v in sources.items()})
        contract.validate(evidence, time, complete=True)
        for field in fields:
            left, right = evidence[field], legs[field] if field in legs else pd.Series(None, index=legs.index)
            equal = left.eq(right) | (left.isna() & right.isna())
            if not equal.fillna(False).all():
                raise ValueError(f'Model {model} quote identity differs at {field}.')
        times = {}
        for field in ('issued_at', 'trained_through', 'artifact_vintage'):
            source = f'{model}::{field}'
            if source not in legs:
                raise ValueError(f'Missing model timing evidence: {source}.')
            times[field] = pd.to_datetime(legs[source], utc=True, errors='raise')
            if times[field].isna().any() or times[field].gt(pd.to_datetime(legs.decision_at, utc=True)).any():
                raise ValueError('Every model leg must be available at its own decision time.')
        if (times['trained_through'].ge(times['issued_at']) | times['artifact_vintage'].gt(times['issued_at'])).any():
            raise ValueError('Model evidence was unavailable at its own prediction issue time.')
