"""Closed research accounting. No external execution or automatic policy updates."""
from dataclasses import dataclass, field
import math
import numpy as np
import pandas as pd
from .decision_layer import FrozenTable
from .stake_policy import StakeContext, allocate_batch


@dataclass
class BankrollLedger:
    initial_capital: float
    currency: str
    open_tickets: dict = field(default_factory=dict, init=False)
    events: list = field(default_factory=list, init=False)
    realized_net_pnl: float = field(default=0.,init=False)
    _seen: set = field(default_factory=set,init=False)

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
        if not np.isfinite([self.wealth,self.reserve,self.cash]).all() or min(self.reserve,self.cash) < -1e-9:
            raise ValueError('Bankroll conservation or solvency violated.')
        self.events.append(dict(time=time,kind=kind,ticket_id=ticket_id,amount=amount,wealth=self.wealth,
                                reserve=self.reserve,cash=self.cash,realized_net_pnl=self.realized_net_pnl,currency=self.currency))

    def context(self,time,*,basis='realized_wealth'):
        exposure = []
        for record in self.open_tickets.values():
            for column,values in record['exposures'].items():
                exposure.extend((column,key,record['stake']) for key in values)
        return StakeContext(self.wealth,max(0.,self.cash),self.currency,time,basis,tuple(exposure))

    def place(self,tickets,amounts,time,*,exposure_columns=()):
        frame = tickets.frame
        if not isinstance(amounts,pd.Series) or not amounts.index.is_unique or set(amounts.index) != set(frame.index):
            raise ValueError('Placement amounts must exactly match selected ticket IDs.')
        amounts = amounts.reindex(frame.index)
        if set(frame.index) & self._seen or not np.isfinite(amounts).all() or (amounts<0).any() or float(amounts.sum()) > self.cash+1e-10:
            raise ValueError('Duplicate ticket, invalid stake or insufficient available cash.')
        stamp = pd.to_datetime(time,utc=True)
        if pd.isna(stamp) or (self.events and stamp < self.events[-1]['time']):
            raise ValueError('Placement time must be chronological.')
        if any(c not in frame for c in exposure_columns):
            raise ValueError('Missing placement exposure columns.')
        for key in sorted(frame.index,key=repr):
            self._seen.add(key)
            amount = float(amounts.loc[key])
            if amount > 0:
                exposures = {c:frame.at[key,c] if isinstance(frame.at[key,c],tuple) else (frame.at[key,c],) for c in exposure_columns}
                self.open_tickets[key] = dict(stake=amount,placed_at=stamp,exposures=exposures)
            self._record(stamp,'place' if amount else 'unfunded',key,amount)

    def settle(self,ticket_id,*,gross_return,time,available_at,fee=0.):
        if ticket_id not in self.open_tickets:
            raise ValueError('Only outstanding funded tickets can settle.')
        stamp,available = pd.to_datetime(time,utc=True),pd.to_datetime(available_at,utc=True)
        record = self.open_tickets[ticket_id]
        if pd.isna(stamp) or pd.isna(available) or stamp < available or available < record['placed_at'] or (self.events and stamp < self.events[-1]['time']):
            raise ValueError('Settlement precedes official availability, placement or ledger time.')
        if not np.isfinite([gross_return,fee]).all() or gross_return < 0 or fee < 0 or fee > gross_return+self.cash:
            raise ValueError('Invalid gross return or fee exceeds available resources.')
        net = float(gross_return)-record['stake']-float(fee)
        del self.open_tickets[ticket_id]
        self.realized_net_pnl += net
        self._record(stamp,'settle',ticket_id,net)

    def summary(self):
        wealth = np.array([self.initial_capital,*[e['wealth'] for e in self.events]])
        peak = np.maximum.accumulate(wealth)
        return dict(ending_wealth=self.wealth,available_cash=self.cash,outstanding_reserve=self.reserve,
                    return_fraction=self.wealth/self.initial_capital-1,realized_profit=self.realized_net_pnl,
                    log_growth=math.log(self.wealth/self.initial_capital) if self.wealth>0 else float('-inf'),
                    max_drawdown=float(np.max((peak-wealth)/peak)),
                    unfunded=sum(e['kind']=='unfunded' for e in self.events),
                    observed_insolvency_events=sum(e['wealth']<=0 for e in self.events),currency=self.currency)


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


def settlements_from_legs(membership, leg_outcomes):
    """Conservative multiplicative-payoff settlement at the latest leg availability.

    membership has ticket_id and leg_id. leg_outcomes is uniquely indexed by leg_id
    with available_at and gross return_multiplier per staked unit. A returned/void
    leg has multiplier 1, a loss 0. All legs must be officially available, even if
    an earlier loss already determines the final return. Unknown legs stay pending.
    Custom early-settlement and nonmultiplicative payoff rules need their own table.
    """
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
        records.append(dict(ticket_id=ticket, available_at=pd.NaT if pending else times.max(), return_multiplier=multiplier))
    return pd.DataFrame(records, columns=['ticket_id','available_at','return_multiplier']).set_index('ticket_id')


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
    if not settlements.index.is_unique or len(settlements.index.difference(table.index)):
        raise ValueError('Settlement IDs must be unique selected ticket IDs.')
    if not {'available_at','return_multiplier'} <= set(settlements):
        raise ValueError('Replay needs official settlement availability and gross return multipliers.')
    decision = pd.to_datetime(table.decision_at,utc=True)
    available = pd.to_datetime(settlements.available_at,utc=True)
    if decision.isna().any():
        raise ValueError('Missing decision time.')
    for key in settlements.index:
        if pd.notna(available.loc[key]) and available.loc[key] < decision.loc[key]:
            raise ValueError('Settlement availability precedes decision time.')
    ledger = BankrollLedger(initial_capital,currency)
    allocations = []
    settled = set()
    times = sorted(set(decision) | set(available.dropna()))
    columns = tuple(c for c,_ in (risk_limits.exposure_caps if risk_limits else ()))
    for time in times:
        due = [key for key in settlements.index if key not in settled and pd.notna(available.loc[key]) and available.loc[key] <= time]
        for key in sorted(due,key=repr):
            if key in ledger.open_tickets:
                stake = ledger.open_tickets[key]['stake']
                multiplier = float(settlements.at[key,'return_multiplier'])
                fee = float(settlements.at[key,'fee']) if 'fee' in settlements else 0.
                ledger.settle(key,gross_return=stake*multiplier,time=time,available_at=available.loc[key],fee=fee)
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
                                  available_at=available.loc[key],fee=float(settlements.at[key,'fee']) if 'fee' in settlements else 0.)
                    settled.add(key)
    return BankrollResult(pd.DataFrame(ledger.events),pd.concat(allocations) if allocations else pd.DataFrame(),ledger.summary())
