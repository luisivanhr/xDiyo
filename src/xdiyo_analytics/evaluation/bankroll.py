"""Closed research accounting. No external execution or automatic policy updates."""
from .temporal_provenance import _validation_session
from dataclasses import dataclass, field
import math
import numpy as np
import pandas as pd
from .decision_layer import FrozenTable
from .stake_policy import StakeContext, allocate_batch


def _settlement_support(proof,ticket):
    from .temporal_provenance import CERT, _open, _safe, _dump, _json
    if proof.get('ticket_id') != _json(ticket.get('ticket_id')):
        raise ValueError('Settlement evidence names another ticket.')
    if not isinstance(ticket.get(CERT),str):
        return False
    packet=_open(ticket[CERT],{})
    expected=packet.get('legs',[])
    members=proof.get('membership',[])
    if not members or not expected:
        return False
    keys=set().union(*(r.keys() for r in expected))
    if any(not keys <= set(r) for r in members):
        return False
    actual=[{k:_safe(r).get(k) for k in keys} for r in members]
    if sorted(map(_dump,actual)) != sorted(map(_dump,expected)):
        raise ValueError('Settlement evidence differs from exact ticket legs.')
    return proof.get('explicit_availability_timezone') is True


@dataclass
class BankrollLedger:
    initial_capital: float
    currency: str
    open_tickets: dict = field(default_factory=dict, init=False)
    events: list = field(default_factory=list, init=False)
    realized_net_pnl: float = field(default=0.,init=False)
    _seen: set = field(default_factory=set,init=False)
    _temporal: dict = field(default_factory=dict,init=False)

    def __post_init__(self):
        if not self.currency or not np.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError('Closed bankroll requires finite positive initial capital and currency.')

    @property
    def wealth(self):
        return self.initial_capital + self.realized_net_pnl

    @property
    def reserve(self):
        return math.fsum(record['stake'] for record in self.open_tickets.values())

    @property
    def cash(self):
        return self.wealth-self.reserve

    def _record(self,time,kind,ticket_id,amount):
        time = pd.to_datetime(time,utc=True)
        if pd.isna(time) or (self.events and time < self.events[-1]['time']):
            raise ValueError('Bankroll events must be chronological UTC timestamps.')
        wealth, reserve = self.wealth, self.reserve
        cash = wealth-reserve
        if not np.isfinite([wealth,reserve,cash]).all() or min(reserve,cash) < -1e-9:
            raise ValueError('Bankroll conservation or solvency violated.')
        self.events.append(dict(time=time,kind=kind,ticket_id=ticket_id,amount=amount,wealth=wealth,
                                reserve=reserve,cash=cash,realized_net_pnl=self.realized_net_pnl,currency=self.currency,
                                **self._temporal.get(ticket_id, {'temporal_status':'unverified','comparison_eligible':False})))
        from .temporal_provenance import ledger_event_hash
        self.events[-1]['temporal_event_hash']=ledger_event_hash(self.events[-1])

    def context(self,time,*,basis='realized_wealth'):
        exposure = []
        for record in self.open_tickets.values():
            for column,values in record['exposures'].items():
                exposure.extend((column,key,record['stake']) for key in values)
        return StakeContext(self.wealth,max(0.,self.cash),self.currency,time,basis,tuple(exposure))

    @_validation_session
    def place(self,tickets,amounts,time,*,exposure_columns=()):
        frame = tickets.frame
        from .temporal_provenance import validate_frame, CERT, POOL, STATUS, _dump, _records
        status, _ = validate_frame(frame, placement_at=time)
        if not isinstance(amounts,pd.Series) or not amounts.index.is_unique or set(amounts.index) != set(frame.index):
            raise ValueError('Placement amounts must exactly match selected ticket IDs.')
        amounts = amounts.reindex(frame.index)
        from .temporal_provenance import dependency_hash, _policy_status
        allocation=amounts.attrs.get('temporal_allocation')
        if allocation is None:
            status='unverified'
        elif (allocation['policy_hash']!=dependency_hash(allocation['policy']) or
              allocation['amounts_hash']!=dependency_hash(list(zip(amounts.index,amounts.tolist())))):
            raise ValueError('Placement amounts or policy differ from their allocation evidence.')
        elif _policy_status(allocation['policy'])=='unverified':
            status='unverified'
        if set(frame.index) & self._seen or not np.isfinite(amounts).all() or (amounts<0).any() or float(amounts.sum()) > self.cash+1e-10:
            raise ValueError('Duplicate ticket, invalid stake or insufficient available cash.')
        stamp = pd.to_datetime(time,utc=True)
        if pd.isna(stamp) or (self.events and stamp < self.events[-1]['time']):
            raise ValueError('Placement time must be chronological.')
        if any(c not in frame for c in exposure_columns):
            raise ValueError('Missing placement exposure columns.')
        offers = dict(zip(frame.index,_records(frame)))
        for key in sorted(frame.index,key=repr):
            self._temporal[key] = {k:frame.at[key,k] for k in (CERT,POOL) if k in frame}
            self._temporal[key]['temporal_offer'] = _dump({k:v for k,v in offers[key].items() if k not in (CERT,POOL)})
            if allocation is not None:
                self._temporal[key]['temporal_allocation_policy']=_dump(allocation['policy'])
                self._temporal[key]['temporal_allocation_policy_hash']=allocation['policy_hash']
            self._temporal[key].update(temporal_status=status,comparison_eligible=status=='verified')
            self._seen.add(key)
            amount = float(amounts.loc[key])
            if amount > 0:
                exposures = {c:frame.at[key,c] if isinstance(frame.at[key,c],tuple) else (frame.at[key,c],) for c in exposure_columns}
                self.open_tickets[key] = dict(stake=amount,placed_at=stamp,exposures=exposures)
            self._record(stamp,'place' if amount else 'unfunded',key,amount)

    @_validation_session
    def settle(self,ticket_id,*,gross_return,time,available_at,fee=0.,evidence=None):
        if ticket_id not in self.open_tickets:
            raise ValueError('Only outstanding funded tickets can settle.')
        stamp,available = pd.to_datetime(time,utc=True),pd.to_datetime(available_at,utc=True)
        record = self.open_tickets[ticket_id]
        if pd.isna(stamp) or pd.isna(available) or stamp < available or available < record['placed_at'] or (self.events and stamp < self.events[-1]['time']):
            raise ValueError('Settlement precedes official availability, placement or ledger time.')
        if not np.isfinite([gross_return,fee]).all() or gross_return < 0 or fee < 0 or fee > gross_return+self.cash:
            raise ValueError('Invalid gross return or fee exceeds available resources.')
        if evidence is not None:
            import json
            proof = json.loads(evidence)
            bounds = pd.to_datetime(proof['available_at'],utc=True)
            values = np.asarray(proof['multipliers'],dtype=float)
            if (pd.isna(bounds).any() or not np.isfinite(values).all() or available != bounds.max()
                    or gross_return != record['stake']*float(np.prod(values))):
                raise ValueError('Settlement differs from complete leg availability/payoff evidence.')
            original=json.loads(self._temporal[ticket_id]['temporal_offer'])
            original.update({k:v for k,v in self._temporal[ticket_id].items() if k in ('temporal_certificate','temporal_pool')})
            supported=_settlement_support(proof,original)
        net = float(gross_return)-record['stake']-float(fee)
        proposed_pnl = self.realized_net_pnl + net
        proposed_wealth = self.initial_capital + proposed_pnl
        proposed_reserve = math.fsum(r['stake'] for k,r in self.open_tickets.items() if k != ticket_id)
        proposed_cash = proposed_wealth-proposed_reserve
        if not np.isfinite([net,proposed_pnl,proposed_wealth,proposed_reserve,proposed_cash]).all() or min(proposed_reserve,proposed_cash) < -1e-9:
            raise ValueError('Settlement would violate finite bankroll conservation or solvency.')
        del self.open_tickets[ticket_id]
        self.realized_net_pnl = proposed_pnl
        if evidence is None or not supported:
            self._temporal[ticket_id].update(temporal_status='unverified',comparison_eligible=False)
        if evidence is not None:
            self._temporal[ticket_id]['settlement_evidence'] = evidence
            from .temporal_provenance import dependency_hash
            self._temporal[ticket_id]['settlement_evidence_hash'] = dependency_hash(json.loads(evidence))
        self._record(stamp,'settle',ticket_id,net)

    def summary(self):
        wealth = np.array([self.initial_capital,*[e['wealth'] for e in self.events]])
        peak = np.maximum.accumulate(wealth)
        return dict(ending_wealth=self.wealth,available_cash=self.cash,outstanding_reserve=self.reserve,
                    return_fraction=self.wealth/self.initial_capital-1,realized_profit=self.realized_net_pnl,
                    log_growth=math.log(self.wealth/self.initial_capital) if self.wealth>0 else float('-inf'),
                    max_drawdown=float(np.max((peak-wealth)/peak)),
                    unfunded=sum(e['kind']=='unfunded' for e in self.events),
                    observed_insolvency_events=sum(e['wealth']<=0 for e in self.events),currency=self.currency,
                    temporal_status=max([r.get('temporal_status','unverified') for r in self._temporal.values()] or ['unverified'],
                                        key={'verified':0,'conditional_research':1,'unverified':2}.get),
                    comparison_eligible=bool(self._temporal) and all(r.get('temporal_status')=='verified' for r in self._temporal.values()))


@dataclass
class BankrollResult:
    ledger: pd.DataFrame
    allocations: pd.DataFrame
    summary: dict

    def report(self):
        """Native artifacts backed by this ledger, not a stock portfolio proxy."""
        from ..reporting.contracts import StudyResult, Artifact
        tables = {'bankroll_ledger': self.ledger.copy(), 'allocation_audit': self.allocations.copy(),
                  'bankroll_summary': pd.DataFrame([self.summary])}
        return StudyResult('Closed research bankroll', tables=tables, artifacts=[Artifact('table', table, name.replace('_', ' ').title())
                                                     for name, table in tables.items()])


@_validation_session
def settlements_from_legs(membership, leg_outcomes):
    """Conservative multiplicative-payoff settlement at the latest leg availability.

    membership has ticket_id and leg_id. leg_outcomes is uniquely indexed by leg_id
    with available_at and gross return_multiplier per staked unit. A returned/void
    leg has multiplier 1, a loss 0. All legs must be officially available, even if
    an earlier loss already determines the final return. Unknown legs stay pending.
    Custom early-settlement and nonmultiplicative payoff rules need their own table.
    """
    from .temporal_provenance import validate_frame, dependency_hash, _dump, _safe
    validate_frame(membership)
    if not {'ticket_id','leg_id'} <= set(membership) or membership[['ticket_id','leg_id']].isna().any().any() or membership.duplicated(['ticket_id','leg_id']).any():
        raise ValueError('Settlement membership requires distinct complete ticket/leg pairs.')
    if not leg_outcomes.index.is_unique or not {'available_at','return_multiplier'} <= set(leg_outcomes):
        raise ValueError('Leg outcomes require distinct identities, availability and gross multipliers.')
    records = []
    for ticket, rows in membership.groupby('ticket_id', sort=False):
        legs = leg_outcomes.reindex(rows.leg_id)
        times = pd.to_datetime(legs.available_at, utc=True)
        values = pd.to_numeric(legs.return_multiplier, errors='raise')
        if (values.dropna() < 0).any() or np.isinf(values).any():
            raise ValueError('Leg return multipliers must be finite and nonnegative.')
        pending = times.isna().any() or values.isna().any()
        multiplier = np.nan if pending else float(np.prod(values))
        if not pending and not np.isfinite(multiplier):
            raise ValueError('Ticket return multiplier overflow.')
        evidence = dict(leg_ids=rows.leg_id.tolist(), available_at=times.tolist(), multipliers=values.tolist(),
                        ticket_id=ticket,membership=[_safe(r) for r in rows.to_dict('records')],
                        explicit_availability_timezone=all(pd.notna(v) and pd.Timestamp(v).tzinfo is not None for v in legs.available_at))
        records.append(dict(ticket_id=ticket, available_at=pd.NaT if pending else times.max(), return_multiplier=multiplier,
            settlement_evidence=_dump(evidence), settlement_evidence_hash=dependency_hash(evidence)))
    return pd.DataFrame(records, columns=['ticket_id','available_at','return_multiplier','settlement_evidence','settlement_evidence_hash']).set_index('ticket_id')


@_validation_session
def replay_bankroll(selected_tickets, settlements, policy, *, initial_capital, currency, risk_limits=None, basis='realized_wealth'):
    """Settle available outcomes before same-time decisions; never reinvest unknown returns.

    settlements is a separate outcome table indexed by ticket_id with available_at,
    return_multiplier (gross returned per staked unit) and optional fee. Latest
    required leg availability must already have been resolved by the payoff adapter.
    Policies see only a FrozenTable and current cash, never this outcome table.
    """
    from copy import deepcopy
    policy = deepcopy(policy)  # fitted policy state is frozen for the replay
    table = selected_tickets.frame
    from .temporal_provenance import validate_frame, dependency_hash
    import json
    validate_frame(table)
    if not settlements.index.is_unique or len(settlements.index.difference(table.index)):
        raise ValueError('Settlement IDs must be unique selected ticket IDs.')
    if not {'available_at','return_multiplier'} <= set(settlements):
        raise ValueError('Replay needs official settlement availability and gross return multipliers.')
    decision = pd.to_datetime(table.decision_at,utc=True)
    available = pd.to_datetime(settlements.available_at,utc=True)
    if decision.isna().any():
        raise ValueError('Missing decision time.')
    for key in settlements.index:
        if 'settlement_evidence' in settlements:
            evidence = json.loads(settlements.at[key,'settlement_evidence'])
            if dependency_hash(evidence) != settlements.at[key,'settlement_evidence_hash']:
                raise ValueError(f'Settlement evidence changed for ticket {key}.')
            _settlement_support(evidence,{**table.loc[key].to_dict(),'ticket_id':key})
            bounds = pd.to_datetime(evidence['available_at'],utc=True)
            values = np.array(evidence['multipliers'],dtype=float)
            pending = pd.isna(bounds).any() or np.isnan(values).any()
            if (pending and pd.notna(available.loc[key])) or (not pending and (available.loc[key] != bounds.max() or settlements.at[key,'return_multiplier'] != float(np.prod(values)))):
                raise ValueError(f'Settlement for ticket {key} differs from complete leg availability/payoff evidence.')
        if pd.notna(available.loc[key]) and available.loc[key] < decision.loc[key]:
            raise ValueError('Settlement availability precedes decision time.')
    ledger = BankrollLedger(initial_capital,currency)
    allocations = []
    settled = set()
    times = sorted(set(decision) | set(available.dropna()))
    columns = tuple(c for c,_ in (risk_limits.exposure_caps if risk_limits else ()))
    # Every availability timestamp is an event. A ticket cannot be placed later
    # than its availability (validated above); unfunded tickets never reopen.
    by_time = {}
    for key, stamp in available.items():
        if pd.notna(stamp):
            by_time.setdefault(stamp, []).append(key)
    for time in times:
        due = by_time.get(time, ())
        for key in sorted(due,key=repr):
            if key in ledger.open_tickets:
                stake = ledger.open_tickets[key]['stake']
                multiplier = float(settlements.at[key,'return_multiplier'])
                fee = float(settlements.at[key,'fee']) if 'fee' in settlements else 0.
                ledger.settle(key,gross_return=stake*multiplier,time=time,available_at=available.loc[key],fee=fee,
                              evidence=settlements.at[key,'settlement_evidence'] if 'settlement_evidence' in settlements else None)
                settled.add(key)
        batch = table.loc[decision.eq(time)]
        if len(batch):
            snapshot = FrozenTable(batch)
            allocation = allocate_batch(snapshot,policy,ledger.context(time,basis=basis),risk_limits)
            ledger.place(snapshot,allocation.amounts,time,exposure_columns=columns)
            allocations.append(allocation.audit.assign(decision_at=time))
            # A zero-duration ticket may settle after placement at the same instant.
            for key in due:
                if key in ledger.open_tickets and key not in settled:
                    stake = ledger.open_tickets[key]['stake']
                    ledger.settle(key,gross_return=stake*float(settlements.at[key,'return_multiplier']),time=time,
                                  available_at=available.loc[key],fee=float(settlements.at[key,'fee']) if 'fee' in settlements else 0.,
                                  evidence=settlements.at[key,'settlement_evidence'] if 'settlement_evidence' in settlements else None)
                    settled.add(key)
    summary = ledger.summary()
    if 'settlement_evidence' not in settlements:
        summary.update(temporal_status='unverified',comparison_eligible=False,
                       temporal_limitation='Settlement table lacks complete leg-availability evidence.')
    return BankrollResult(pd.DataFrame(ledger.events),pd.concat(allocations) if allocations else pd.DataFrame(),summary)
