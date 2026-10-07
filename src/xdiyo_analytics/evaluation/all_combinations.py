"""Whole-group ticket expansion, counted completely before materialization."""

from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from itertools import combinations
from math import comb, isfinite
from numbers import Real
import hashlib
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True, kw_only=True)
class AllCombinations:
    """Every unordered k-event combination in each league/season/stage/round.

    Stake is per ticket. Size one means singles; undersized pools emit none.
    stage_column=None explicitly declares that there is only one stage per
    league-season. Missing stage identity otherwise raises. Duplicate identical
    selections collapse; conflicting selections for one event raise.
    min_ev=None disables ticket filtering. A finite min_ev requires independent
    win/loss probabilities and accepts only unit-stake EV strictly above it.
    """
    legs: int = 2
    grouping: str = 'league_round'
    stage_column: str | None = 'tournament_id'
    stake: float = 1.0
    max_tickets: int = 100000
    on_push: str = 'remove'
    on_void: str = 'remove'
    probability_mode: str = 'none'
    min_ev: float | None = None
    probability_columns: dict | None = None
    ticket_gate: object = None
    payoff: str = 'push_void'
    quote_availability: object = field(default=None, repr=False)

    def __post_init__(self):
        self.validate_probability_columns()
        if self.quote_availability is not None:
            from .quote_availability import QuoteAvailability
            if not isinstance(self.quote_availability, QuoteAvailability):
                raise TypeError('Use a typed QuoteAvailability contract.')
            if self.ticket_gate is None or self.ticket_gate.missing != 'error':
                raise ValueError('Explicit quote contracts require a ticket gate with missing="error".')
        if self.min_ev is not None:
            try:
                valid = (not isinstance(self.min_ev, (bool, np.bool_))
                         and isinstance(self.min_ev, Real) and isfinite(self.min_ev))
            except (OverflowError, TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError('Minimum ticket EV (min_ev) must be a finite real number or None to disable.')
            if self.probability_mode != 'independent':
                raise ValueError('Ticket EV filtering requires explicit probability_mode="independent"; disable min_ev otherwise.')

    def validate_probability_columns(self):
        """Validate source names before they can be relabelled as probabilities."""
        if self.probability_columns is not None:
            from .decision_layer import _safe_columns
            if not isinstance(self.probability_columns, dict) or any(
                not isinstance(c, str) or not c.strip() for c in self.probability_columns.values()
            ):
                raise ValueError('Probability source names must be nonempty column names.')
            _safe_columns(self.probability_columns.values())


def _identity(values):
    return json.dumps([str(v) for v in values], ensure_ascii=False, separators=(',', ':'))


def _stake_preview(terms):
    """Float for ordinary totals; exact finite Decimal beyond float range.

    Each term is (exact ticket count, per-ticket stake). Decimal precision is
    sized to the inputs so even previews far above the expansion cap are safe.
    """
    amounts = []
    for count, stake in terms:
        value = Decimal(str(stake))
        with localcontext() as ctx:
            ctx.prec = max(28, len(str(count)) + len(value.as_tuple().digits))
            amounts.append(Decimal(count) * value)
    if not amounts:
        return 0.0
    with localcontext() as ctx:
        ctx.prec = max(28, max(v.adjusted() for v in amounts)
                       - min(v.as_tuple().exponent for v in amounts)
                       + len(str(len(amounts))) + 2)
        total = sum(amounts, Decimal(0))
    number = float(total)
    return number if np.isfinite(number) else total


def prepare_pools(ledger, policy, match_columns, name, *, outcome_free=False):
    from .tickets import Parlay, _validate
    policy.validate_probability_columns()
    _validate(Parlay(size=policy.legs, stake=policy.stake, max_tickets=policy.max_tickets,
                     on_push=policy.on_push, on_void=policy.on_void,
                     probability_mode=policy.probability_mode))
    if policy.grouping != 'league_round':
        raise ValueError('AllCombinations grouping must be league_round (league/season/stage/round).')
    if policy.stage_column not in ('tournament_id', 'stage_id', 'stage', None):
        raise ValueError('stage_column must be tournament_id, stage_id, stage or None for explicitly single-stage seasons.')
    if policy.min_ev is not None and 'p_win' not in ledger:
        raise ValueError('Ticket EV filtering requires retained win probabilities in p_win; manual bets without probabilities cannot be valued.')
    if ledger.empty:
        return [], []
    data = ledger.copy()
    if policy.ticket_gate is not None:
        if not policy.probability_columns:
            raise ValueError('Multi-model gates need an explicit probability column mapping.')
        if set(policy.probability_columns) != set(policy.ticket_gate.models):
            raise ValueError('Probability columns must match every gate model.')
        if policy.payoff != 'binary':
            raise ValueError('Complete-ticket EV gates require an explicitly binary payoff.')
    keys = list(match_columns)
    if not keys or any(k not in data for k in keys):
        raise ValueError('AllCombinations needs the full fixture identity columns.')
    groups = ['fold_id']
    for options in [('competition_id', 'source_league'), ('season_id', 'source_season')]:
        key = next((k for k in options if k in data), None)
        if key is None:
            raise ValueError('AllCombinations needs league and season identities.')
        groups.append(key)
    if policy.stage_column is not None:
        groups.append(policy.stage_column)
    groups.append('round')
    if any(k not in data for k in groups) or data[list(dict.fromkeys(groups + keys))].isna().any().any():
        raise ValueError('AllCombinations requires nonmissing league/season/stage/round and fold identities; '
                         'supply stage_column, or explicitly set None for single-stage seasons.')
    if data['take'].isna().any() or not data['take'].map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise ValueError('Ticket decisions must be explicit booleans.')
    data['kickoff_at'] = pd.to_datetime(data.kickoff_at, utc=True, errors='raise')
    if data.kickoff_at.isna().any():
        raise ValueError('Ticket legs require kickoff times.')
    data['odds'] = pd.to_numeric(data.odds, errors='raise')
    valid_price = data.odds.notna() & np.isfinite(data.odds) & data.odds.gt(1)
    if policy.quote_availability is not None:
        from .quote_availability import validate_model_quotes
        contract = policy.quote_availability
        data = contract.annotate(data)
        eligible = data.loc[data['take']]
        if len(eligible):
            time = pd.to_datetime(eligible.get('decision_at', pd.Series(pd.NaT, index=eligible.index)), utc=True).max()
            contract.validate(eligible, time, complete=True)
            validate_model_quotes(eligible, policy.ticket_gate.models, contract, time)
            # Preflight every eligible fixture, even undersized pools: an absent
            # stream must never vanish through combination construction.
            for model, column in policy.probability_columns.items():
                p = pd.to_numeric(eligible.get(column, pd.Series(np.nan, index=eligible.index)), errors='raise')
                if p.isna().any() or not np.isfinite(p).all() or not p.between(0, 1).all():
                    raise ValueError(f'Complete valid model stream required for every eligible fixture: {model}.')
    # Column iteration preserves uint64 IDs; row-wise apply can coerce mixed
    # signed/unsigned numeric identities to float and merge distinct fixtures.
    data['_group'] = [_identity(r) for r in zip(*(data[k] for k in groups))]
    data['_event'] = [_identity(r) for r in zip(*(data[k] for k in keys))]
    # An event in two stage/round pools is ambiguous, even if the selections agree.
    if data.groupby(['fold_id', '_event'])._group.nunique().gt(1).any():
        raise ValueError('One fixture has conflicting stage/round grouping identities.')
    data['_price_valid'] = valid_price
    previews, pools = [], []
    for group_id, all_rows in data.groupby('_group', sort=True):
        offered = all_rows.loc[all_rows['take'] & all_rows._price_valid].copy()
        if policy.min_ev is not None or policy.probability_mode == 'independent':
            for field in (('p_win', 'p_push') if policy.min_ev is not None else ('p_win',)):
                if field not in offered:
                    continue
                supplied = offered[field].dropna()
                if not supplied.map(lambda v: isinstance(v, Real) and not isinstance(v, (bool, np.bool_))).all():
                    raise ValueError(f'Ticket EV {field} probabilities must be numeric, finite and between 0 and 1.')
                numeric = supplied.astype(float)
                if (~np.isfinite(numeric) | ~numeric.between(0, 1)).any():
                    raise ValueError(f'Ticket EV {field} probabilities must be numeric, finite and between 0 and 1.')
                if field == 'p_push' and numeric.ne(0).any():
                    raise ValueError('Ticket EV supports win/loss probabilities only; nonzero p_push is unsupported.')
                # Use the validated numeric values during expansion, including
                # object/nullable columns. Missing values must remain abstentions.
                offered[field] = offered[field].map(lambda v: float(v) if pd.notna(v) else np.nan)
        duplicates = 0
        # Compare all semantic evidence. Row positions are occurrence bookkeeping;
        # derived accounting values must never choose a leg.
        compare = [c for c in offered if c not in {'row_position', 'stake', 'profit', 'payout',
                                                   'accounting_status', 'cumulative_known_profit'}]
        if outcome_free or policy.ticket_gate is not None:
            from .decision_layer import _outcome_column
            compare = [c for c in compare if not _outcome_column(c)]
        for _, repeated in offered.groupby('_event', sort=False):
            if len(repeated) > 1 and len(repeated[compare].drop_duplicates()) != 1:
                raise ValueError('Conflicting selections or evidence for one fixture; select exactly one option per event.')
            if (outcome_free or policy.ticket_gate is not None) and len(repeated) > 1:
                # Resolve contradictory retrospective evidence as unknown only
                # after validating decision evidence; never let it choose a leg.
                for column in [c for c in offered if _outcome_column(c)]:
                    if repeated[column].nunique(dropna=False) > 1:
                        offered.loc[offered._event.eq(repeated._event.iloc[0]),column] = 'missing' if column == 'settlement' else np.nan
        duplicates = len(offered) - offered._event.nunique()
        order = ['kickoff_at', '_event', 'bet'] + (['row_position'] if 'row_position' in offered else [])
        offered = offered.sort_values(order, kind='stable').drop_duplicates('_event')
        n = len(offered)
        count = comb(n, policy.legs) if n >= policy.legs else 0
        unknown = all_rows.get('decision_reason', pd.Series('', index=all_rows.index)).eq('Probability unavailable')
        unknown = all_rows.get('probability_abstention', unknown).fillna(False).astype(bool)
        record = {k: all_rows[k].iloc[0] for k in groups}
        record.update(template=name, eligible_events=n, legs=policy.legs, ticket_count=count,
                      expected_stake=_stake_preview([(count, policy.stake)]), input_events=all_rows._event.nunique(),
                      not_selected_events=all_rows.loc[~all_rows['take'], '_event'].nunique(),
                      missing_price_events=all_rows.loc[~all_rows._price_valid, '_event'].nunique(),
                      missing_probability_events=all_rows.loc[unknown, '_event'].nunique(),
                      duplicate_rows_removed=duplicates)
        previews.append(record)
        pools.append((group_id, groups, offered))
    return pools, previews


def expand_pools(pools, policy, name, *, settle=True):
    from .tickets import _settle
    rows, members, decisions = [], [], []
    for group_id, groups, offered in pools:
        for positions in combinations(range(len(offered)), policy.legs):
            legs = offered.iloc[list(positions)]
            identity = json.dumps([name, group_id, sorted(legs._event.tolist())], separators=(',', ':'))
            ticket_id = name + ':' + hashlib.sha256(identity.encode()).hexdigest()
            with np.errstate(over='ignore'):
                odds = float(np.prod(legs.odds))
            if not np.isfinite(odds):
                raise ValueError('Combined odds overflow; reduce ticket size.')
            probability = np.nan
            if policy.probability_mode == 'independent' and 'p_win' in legs and legs.p_win.notna().all():
                if (~np.isfinite(legs.p_win) | ~legs.p_win.between(0, 1)).any():
                    raise ValueError('Leg probabilities must be between 0 and 1.')
                probability = float(np.prod(legs.p_win.astype(float)))
                if policy.min_ev is not None and probability < np.finfo(float).tiny and legs.p_win.gt(0).all():
                    raise ValueError('Ticket probability product underflow; valuation is unrepresentable. Reduce ticket size.')
            ev = np.nan
            selected, reason = True, ''
            if policy.min_ev is not None:
                if pd.isna(probability):
                    selected, reason = False, 'missing_probability'
                else:
                    ev = probability * odds - 1
                    if not np.isfinite(ev):
                        raise ValueError('Ticket EV overflow; valuation is unrepresentable.')
                    selected = bool(ev > policy.min_ev)
                    reason = '' if selected else 'ev_not_above_threshold'
            decisions.append(dict(template=name, group_id=group_id, ticket_id=ticket_id,
                                  event_membership=json.dumps(sorted(legs._event.tolist())),
                                  probability=probability, odds=odds, expected_profit=ev,
                                  min_ev=policy.min_ev, take=selected, rejection_reason=reason,
                                  filter_enabled=policy.min_ev is not None, comparator='>',
                                  probability_assumption=policy.probability_mode,
                                  **{k: legs[k].iloc[0] for k in groups}))
            if not selected:
                continue
            state, payout = _settle(legs, policy, policy.stake) if settle else ('missing', np.nan)
            rows.append(dict(ticket_id=ticket_id, bet=name, kind='single' if policy.legs == 1 else 'parlay',
                             n_legs=policy.legs, kickoff_at=legs.kickoff_at.min(), last_kickoff_at=legs.kickoff_at.max(),
                             take=True, stake=policy.stake, odds=odds, probability=probability,
                             probability_assumption=policy.probability_mode, settlement=state,
                             accounting_status='unresolved' if pd.isna(payout) else 'settled',
                             payout=payout, profit=payout-policy.stake,
                             **({'expected_profit': ev} if policy.min_ev is not None else {}),
                             **{k: legs[k].iloc[0] for k in groups}))
            for position, (_, leg) in enumerate(legs.iterrows(), 1):
                member = leg.drop(labels=['_group', '_event', '_price_valid']).to_dict()
                members.append(dict(member, ticket_id=ticket_id, leg_number=position, template=name))
    return rows, members, decisions


def preview_combinations(ledger, composition, *, match_columns=('event_id',)):
    """Exact candidate counts after leg eligibility, BEFORE ticket EV filtering.

    Rows describe AllCombinations templates only. Exclusion counts can overlap.
    Totals and per-template limit status are in DataFrame.attrs. Counts/stakes
    are hypothetical candidate exposure, not selected bets. Never expands tickets.
    """
    from .tickets import BetSlip
    templates = composition.tickets if isinstance(composition, BetSlip) else {'tickets': composition}
    rows, limits, stake_terms = [], {}, []
    for name, policy in templates.items():
        if isinstance(policy, AllCombinations):
            _, preview = prepare_pools(ledger, policy, match_columns, name)
            rows.extend(preview)
            stake_terms.extend((int(r['ticket_count']), policy.stake) for r in preview)
            limits[name] = sum(int(r['ticket_count']) for r in preview) > policy.max_tickets
    # Object columns prevent pandas from coercing enormous integers to floats;
    # ordinary previews retain their usual numeric dtypes.
    object_columns = {key for key in ('ticket_count', 'expected_stake')
                      if any(isinstance(r[key], Decimal) or
                             (key == 'ticket_count' and r[key] > 2**63-1) for r in rows)}
    frame = pd.DataFrame({key: pd.Series([r.get(key, np.nan) for r in rows],
        dtype=object if key in object_columns else None)
        for key in dict.fromkeys(k for row in rows for k in row)}) if rows else pd.DataFrame(columns=[
        'template', 'eligible_events', 'legs', 'ticket_count', 'expected_stake',
        'input_events', 'not_selected_events', 'missing_price_events', 'missing_probability_events', 'duplicate_rows_removed'])
    frame.attrs.update(total_tickets=sum(int(r['ticket_count']) for r in rows),
                       total_stake=_stake_preview(stake_terms), exceeded_limits=limits)
    return frame
