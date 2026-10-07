"""Optional allocation on a final selected batch, with independent hard limits."""
from dataclasses import dataclass, field
from typing import Protocol
from math import fsum
import numpy as np
import pandas as pd
from .decision_layer import FrozenTable


@dataclass(frozen=True)
class StakeContext:
    bankroll: float
    available_cash: float
    currency: str
    time: object
    basis: str = 'realized_wealth'
    open_exposure: tuple = ()

    def __post_init__(self):
        if not self.currency or self.basis not in {'realized_wealth', 'available_cash'}:
            raise ValueError('Declare currency and bankroll basis.')
        if not np.isfinite([self.bankroll, self.available_cash]).all() or not 0 <= self.available_cash <= self.bankroll:
            raise ValueError('Require finite 0 <= available cash <= bankroll.')
        time = pd.to_datetime(self.time, utc=True, errors='raise')
        if pd.isna(time):
            raise ValueError('Stake context requires an issue time.')
        object.__setattr__(self, 'time', time)
        object.__setattr__(self, 'open_exposure', tuple(tuple(v) for v in self.open_exposure))
        for _, _, amount in self.open_exposure:
            if not np.isfinite(amount) or amount < 0:
                raise ValueError('Open exposure must be finite and nonnegative.')

    @property
    def capital(self):
        return self.bankroll if self.basis == 'realized_wealth' else self.available_cash


@dataclass
class AllocationResult:
    amounts: pd.Series
    audit: pd.DataFrame


class StakePolicy(Protocol):
    def allocate(self, selected_tickets: FrozenTable, context: StakeContext) -> AllocationResult: ...


def _amounts(batch, values, reason):
    frame = batch.frame
    amounts = pd.Series(values, index=frame.index, dtype=float, name='actual_stake')
    return AllocationResult(amounts, pd.DataFrame({'reason': reason, 'desired_stake': amounts}, index=frame.index))


@dataclass(frozen=True)
class FixedStake:
    amount: float
    currency: str

    def allocate(self, selected_tickets, context):
        if self.currency != context.currency or not np.isfinite(self.amount) or self.amount < 0:
            raise ValueError('FixedStake needs matching currency and finite nonnegative amount.')
        return _amounts(selected_tickets, self.amount, 'fixed')


@dataclass(frozen=True)
class FixedFraction:
    fraction: float
    currency: str
    basis: str = 'realized_wealth'

    def allocate(self, selected_tickets, context):
        if self.currency != context.currency or self.basis != context.basis or not np.isfinite(self.fraction) or not 0 <= self.fraction <= 1:
            raise ValueError('FixedFraction needs matching currency/basis and fraction in [0,1].')
        return _amounts(selected_tickets, context.capital * self.fraction, 'fixed_fraction')


@dataclass(frozen=True)
class ModelProbabilitySource:
    column: str = 'probability'
    provenance_column: str = 'probability_provenance'
    issued_at_column: str = 'probability_issued_at'
    assume_available_at_decision: bool = False

    def probabilities(self, batch, context):
        frame = batch.frame
        if self.provenance_column not in frame or frame[self.provenance_column].isna().any():
            raise ValueError('Model probabilities require explicit provenance.')
        if self.column not in frame:
            raise ValueError('Model probability output is unavailable.')
        if self.issued_at_column in frame:
            issued = pd.to_datetime(frame[self.issued_at_column], utc=True)
            if issued.isna().any() or (issued > context.time).any():
                raise ValueError('Model probabilities were unavailable at allocation time.')
        elif not self.assume_available_at_decision:
            raise ValueError('Provide probability issue timestamps or explicitly attest decision-time availability.')
        return pd.to_numeric(frame[self.column], errors='raise')


@dataclass
class HistoricalRateSource:
    """Fit on eligible strategy-selected outcomes; never an unconditional match rate."""
    selection_id: str
    cutoff: object
    strata: tuple = ()
    prior_probability: float = .5
    prior_strength: float = 2.
    lookback: object = None
    learned_selection: bool = True
    exchangeable_within_strata: bool = False

    def fit(self, history):
        cutoff = pd.to_datetime(self.cutoff, utc=True)
        if pd.isna(cutoff) or not self.selection_id or not 0 <= self.prior_probability <= 1 or not np.isfinite(self.prior_strength) or self.prior_strength < 0:
            raise ValueError('Invalid historical-rate prior, cutoff or selection identity.')
        required = {'selection_id', 'available_at', 'won', 'selected', *self.strata}
        if not required <= set(history):
            raise ValueError('Historical rates require selected ticket outcomes and availability.')
        frame = history.copy()
        available = pd.to_datetime(frame.available_at, utc=True)
        eligible = available.notna() & available.le(cutoff) & frame.selection_id.eq(self.selection_id) & frame.selected.eq(True)
        if self.lookback is not None:
            delta = pd.Timedelta(self.lookback)
            if delta <= pd.Timedelta(0):
                raise ValueError('lookback must be positive.')
            eligible &= available.ge(cutoff-delta)
        frame = frame.loc[eligible]
        if frame.empty or not frame.won.isin([True, False, 0, 1]).all() or frame[list(self.strata)].isna().any().any():
            raise ValueError('Historical rate has no valid eligible binary outcomes/strata.')
        if self.learned_selection:
            needed = {'issued_at', 'trained_through', 'prediction_origin'}
            if not needed <= set(frame) or not frame.prediction_origin.isin(['issued', 'chronological_oof']).all():
                raise ValueError('Learned selection requires historical issued or chronological OOF provenance.')
            issued = pd.to_datetime(frame.issued_at, utc=True)
            trained = pd.to_datetime(frame.trained_through, utc=True)
            if issued.isna().any() or trained.isna().any() or (trained >= issued).any() or (issued >= pd.to_datetime(frame.available_at, utc=True)).any():
                raise ValueError('Historical selection used unavailable outcomes.')
        groups = frame.groupby(list(self.strata), dropna=False, sort=False) if self.strata else [((), frame)]
        self.rates_, self.audit_ = {}, []
        for key, group in groups:
            key = key if isinstance(key, tuple) else (key,)
            n, wins = len(group), float(group.won.sum())
            self.rates_[key] = (wins+self.prior_strength*self.prior_probability)/(n+self.prior_strength)
            self.audit_.append(dict(stratum=key, samples=n, wins=wins, probability=self.rates_[key], cutoff=cutoff,
                                    selection_id=self.selection_id, prior_strength=self.prior_strength, lookback=self.lookback))
        self.cutoff_ = cutoff
        return self

    def probabilities(self, batch, context):
        if not hasattr(self, 'rates_') or self.cutoff_ > context.time:
            raise ValueError('Historical rate must be fitted and available before this allocation.')
        frame = batch.frame
        if not self.exchangeable_within_strata:
            raise ValueError('Declare exchangeable_within_strata=True only when the fitted strategy rate is applicable to these offers and odds.')
        if 'selection_id' not in frame or not frame.selection_id.eq(self.selection_id).all():
            raise ValueError('Historical rate selection identity differs.')
        keys = list(frame[list(self.strata)].itertuples(index=False, name=None)) if self.strata else [()] * len(frame)
        if any(k not in self.rates_ for k in keys):
            raise ValueError('No fitted historical rate for this stratum.')
        return pd.Series([self.rates_[k] for k in keys], index=frame.index)


@dataclass(frozen=True)
class FractionalKelly:
    currency: str
    alpha: float = .25
    max_fraction: float = .05
    probability_source: object = field(default_factory=ModelProbabilitySource)
    basis: str = 'realized_wealth'

    def allocate(self, selected_tickets, context):
        if self.currency != context.currency or self.basis != context.basis or not np.isfinite([self.alpha, self.max_fraction]).all() or not 0 <= self.alpha <= 1 or not 0 <= self.max_fraction <= 1:
            raise ValueError('Kelly needs matching currency/basis and alpha/max_fraction in [0,1].')
        frame = selected_tickets.frame
        if 'payoff' not in frame or not frame.payoff.eq('binary').all():
            raise ValueError('Kelly supports declared binary win/loss payoffs only; pushes/void/partial payouts need a full payoff model.')
        odds = pd.to_numeric(frame.odds, errors='raise')
        p = self.probability_source.probabilities(selected_tickets, context)
        if not p.index.equals(frame.index) or not np.isfinite(p).all() or not p.between(0, 1).all() or not np.isfinite(odds).all() or (odds <= 1).any():
            raise ValueError('Kelly requires aligned finite probabilities in [0,1] and odds >1.')
        fraction = ((p*odds-1)/(odds-1)).clip(lower=0) * self.alpha
        result = _amounts(selected_tickets, context.capital*fraction.clip(upper=self.max_fraction), 'fractional_kelly_approximation')
        result.audit['probability'] = p
        result.audit['uncapped_fraction'] = fraction
        return result


@dataclass(frozen=True)
class RiskLimits:
    """One conservative global scale obeys every cap; floor rounding leaves residual cash."""
    per_ticket: float | None = None
    exposure_caps: tuple = ()
    rounding: float = .01

    def project(self, amounts, batch, context):
        frame = batch.frame
        if not isinstance(amounts, pd.Series) or not amounts.index.is_unique or set(amounts.index) != set(frame.index):
            raise ValueError('Allocation IDs must exactly match selected ticket IDs.')
        amounts = pd.to_numeric(amounts.reindex(frame.index), errors='raise').astype(float)
        if not np.isfinite(amounts).all() or (amounts < 0).any() or not np.isfinite(self.rounding) or self.rounding <= 0:
            raise ValueError('Invalid allocations or rounding quantum.')
        caps = [(np.ones(len(frame), dtype=bool), context.available_cash, 'available_cash')]
        if self.per_ticket is not None:
            if not np.isfinite(self.per_ticket) or self.per_ticket < 0:
                raise ValueError('Invalid per-ticket cap.')
            caps += [(np.arange(len(frame)) == i, self.per_ticket, 'per_ticket') for i in range(len(frame))]
        for column, limit in self.exposure_caps:
            if column not in frame or not np.isfinite(limit) or limit < 0:
                raise ValueError('Invalid exposure cap.')
            keys = set(v for values in frame[column] for v in (values if isinstance(values, tuple) else (values,)))
            for key in keys:
                opened = sum(v for col, k, v in context.open_exposure if col == column and k == key)
                mask = frame[column].map(lambda values: key in values if isinstance(values, tuple) else values == key).to_numpy()
                caps.append((mask, max(0., limit-opened), f'{column}:{key}'))
        scale, reasons = 1., []
        for mask, limit, reason in caps:
            total = fsum(amounts.iloc[np.flatnonzero(mask)])
            if not np.isfinite(total):
                raise ValueError('Allocation total overflow.')
            if total > limit:
                scale = min(scale, limit/total)
                reasons.append(reason)
        actual = np.floor((amounts*scale)/self.rounding)*self.rounding
        for mask, limit, _ in caps:
            if fsum(actual.iloc[np.flatnonzero(mask)]) > limit + 1e-10:
                raise ValueError('Projected allocation violates risk limits.')
        return actual, scale, tuple(sorted(set(reasons)))


def allocate_batch(batch, policy, context, limits=None):
    if not isinstance(batch, FrozenTable) or not isinstance(context, StakeContext):
        raise TypeError('Allocation requires immutable final tickets and a validated StakeContext.')
    frame = batch.frame
    if 'decision_at' not in frame or not pd.to_datetime(frame.decision_at, utc=True).eq(context.time).all():
        raise ValueError('Allocate exactly one shared decision timestamp per context.')
    result = policy.allocate(batch, context)
    actual, scale, caps = (limits or RiskLimits()).project(result.amounts, batch, context)
    audit = result.audit.reindex(frame.index).copy()
    audit['nominal_stake'] = frame.nominal_stake
    audit['actual_stake'] = actual
    audit['projection_scale'] = scale
    audit['binding_caps'] = [caps]*len(frame)
    audit['funded'] = actual.gt(0)
    return AllocationResult(actual, audit)


@dataclass(frozen=True)
class LearnedAllocation:
    """Fitted native allocation-target adapter, predicting a bounded capital fraction."""
    model: object
    feature_columns: tuple
    currency: str
    basis: str = 'realized_wealth'

    def allocate(self, selected_tickets, context):
        if self.currency!=context.currency or self.basis!=context.basis or getattr(getattr(self.model,'target_spec',None),'kind',None)!='allocation':
            raise ValueError('LearnedAllocation requires matching currency/basis and a declared allocation target adapter.')
        cutoff=getattr(self.model,'training_cutoff_',None)
        if cutoff is None or cutoff>=context.time:
            raise ValueError('Allocation model must be fitted strictly before the batch.')
        frame=selected_tickets.frame
        from ..training.contracts import PredictionContext
        prediction=PredictionContext(frame[list(self.feature_columns)].copy(),frame.copy(),'match',('economic_key',),0)
        output=self.model.predict(prediction)['predict']
        if output.shape!=(len(frame),1) or not output.index.equals(frame.index):
            raise ValueError('Allocation model must predict one identity-aligned fraction.')
        values=output.iloc[:,0]
        if not np.isfinite(values).all() or not values.between(0,1).all():
            raise ValueError('Learned allocation fractions must be finite and in [0,1]; no silent clipping.')
        return _amounts(selected_tickets,context.capital*values,'learned_fraction')
