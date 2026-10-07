"""Outcome-free ticket selection/allocation followed by explicit retrospective settlement."""
import json
import numpy as np
import pandas as pd
from .decision_layer import FrozenTable, DecisionContext
from .stake_policy import allocate_batch
from .quote_availability import quote_leg_records, ASSUMPTION_FIELDS


def _index_members(members):
    """Index exact ticket IDs once, retaining every row and its original order.

    No fixture/quote deduplication: conflicting leg evidence must still reach
    the existing validation, and repeated legs must not disappear from checks.
    """
    return {key: members.iloc[positions] for key, positions in
            members.groupby('ticket_id', sort=False, dropna=False, observed=True).indices.items()}


def ticket_batch(tickets, members, templates, match_columns, *, context=None, _members_by_ticket=None):
    indexed = _index_members(members) if _members_by_ticket is None else _members_by_ticket
    records = []
    for row in tickets.to_dict('records'):
        legs = indexed.get(row['ticket_id'])
        if legs is None:
            legs = members.iloc[:0]
        policy = templates[row['bet']]
        contract = getattr(policy, 'quote_availability', None)
        if policy.payoff == 'binary' and 'p_push' in legs and legs.p_push.fillna(0).ne(0).any():
            raise ValueError('Binary ticket valuation cannot include nonzero push probability.')
        if 'decision_at' in legs:
            stamps = pd.to_datetime(legs.decision_at,utc=True)
            if stamps.nunique() != 1 or stamps.isna().any():
                raise ValueError('Every ticket leg must share one decision timestamp.')
            time = stamps.iloc[0]
        elif context is not None:
            time = context.time
        else:
            raise ValueError('Ticket decisions need explicit decision_at or a manual StakeContext.')
        if time > pd.to_datetime(legs.kickoff_at,utc=True).min():
            raise ValueError('Ticket issue time follows a leg kickoff.')
        for key in ('issued_at','quote_at'):
            if key == 'quote_at' and contract is not None:
                contract.validate(legs, time, complete=True)
                continue
            if key in legs and (pd.to_datetime(legs[key],utc=True).isna().any() or (pd.to_datetime(legs[key],utc=True)>time).any()):
                raise ValueError('A leg prediction or quote is unavailable at ticket issue time.')
        fixtures = tuple(sorted((tuple(v) for v in legs[list(match_columns)].itertuples(index=False,name=None)),key=repr))
        identities = []
        for leg in legs.to_dict('records'):
            identities.append({k:leg.get(k) for k in (*match_columns,'bet','market','selection','line','odds','quote_id','quote_at')})
        record = dict(ticket_id=row['ticket_id'],nominal_stake=row['stake'],odds=row['odds'],probability=row['probability'],
                      decision_at=time,fixture_keys=fixtures,
                      league_keys=tuple(sorted(set(legs.get('competition_id',legs.get('source_league',[]))),key=repr)),
                      round_keys=tuple(sorted(set(zip(legs.get('competition_id',legs.get('source_league',[None]*len(legs))),legs.get('season_id',legs.get('source_season',[None]*len(legs))),legs.get('round',[None]*len(legs)))),key=repr)),
                      payoff=policy.payoff,selection_id=row['bet'],probability_provenance=f"retained_leg_probabilities:{row['probability_assumption']}",
                      economic_key=json.dumps([identities,policy.on_push,policy.on_void],sort_keys=True,default=str))
        if 'quote_id' in legs and 'quote_at' in legs:
            record['quote_id'] = json.dumps(legs.quote_id.tolist(),default=str)
            record['quote_at'] = pd.to_datetime(legs.quote_at,utc=True).max() if legs.quote_at.notna().all() else pd.NaT
        if contract is not None:
            record.update(contract.labels)
            if contract.mode == 'research_assumed':
                record['assumed_available_at'] = pd.to_datetime(legs.assumed_available_at, utc=True).max()
            record['quote_legs'] = quote_leg_records(legs, contract, match_columns)
            record['economic_key'] = json.dumps([record['economic_key'], record['quote_legs']], sort_keys=True)
        if 'issued_at' in legs:
            record['probability_issued_at'] = pd.to_datetime(legs.issued_at,utc=True).max()
        records.append(record)
    return pd.DataFrame(records).set_index('ticket_id') if records else pd.DataFrame(columns=['nominal_stake','decision_at'])


def finalize_tickets(tickets,members,templates,match_columns,policy,context,limits, *, _audit_sink=None):
    from .tickets import _settle
    if tickets.empty:
        return tickets,members
    indexed = _index_members(members)
    batch = ticket_batch(tickets,members,templates,match_columns,context=context,
                         _members_by_ticket=indexed)
    for name, template in templates.items():
        contract = getattr(template, 'quote_availability', None)
        if contract is not None:
            selected = tickets.bet.eq(name)
            for field in (*ASSUMPTION_FIELDS, 'quote_at', 'quote_legs'):
                if field in batch:
                    tickets.loc[selected, field] = tickets.loc[selected, 'ticket_id'].map(batch[field])
    audit = []
    rejected = set()
    for name,template in templates.items():
        gate = getattr(template,'ticket_gate',None)
        if gate is None:
            continue
        template.validate_probability_columns()
        mapping = template.probability_columns
        if not mapping or set(mapping) != set(gate.models) or template.probability_mode != 'independent':
            raise ValueError('Ticket gates need an explicit model -> leg probability column mapping and independent joint assumption.')
        subset = batch.loc[batch.selection_id.eq(name)]
        for time,offers in subset.groupby('decision_at',sort=True):
            predictions = {}
            for model,column in mapping.items():
                valuations = offers.copy()
                probabilities = []
                provenance = {field: [] for field in ('issued_at','trained_through','artifact_vintage')}
                for ticket_id in offers.index:
                    legs = indexed[ticket_id]
                    p = pd.to_numeric(legs.get(column,pd.Series(np.nan,index=legs.index)),errors='raise').astype(float)
                    present = p.notna()
                    if (present & (~np.isfinite(p) | ~p.between(0,1))).any():
                        raise ValueError('Supplied model leg probabilities must be finite and in [0,1].')
                    times = {}
                    for field in provenance:
                        source = f'{model}::{field}'
                        times[field] = pd.to_datetime(legs.get(source,pd.Series(pd.NaT,index=legs.index)),utc=True)
                        if (present & (times[field].isna() | times[field].gt(time))).any():
                            raise ValueError(f'Every valued model leg needs nonmissing available {source} provenance.')
                    if (present & (times['trained_through'].ge(times['issued_at']) | times['artifact_vintage'].gt(times['issued_at']))).any():
                        raise ValueError('Model leg evidence was unavailable at its own prediction issue time.')
                    complete = present.all()
                    probabilities.append(float(np.prod(p)) if complete else np.nan)
                    for field in provenance:
                        provenance[field].append(times[field].max() if complete else pd.NaT)
                valuations['probability'] = probabilities
                for field, values in provenance.items():
                    valuations[field] = values
                predictions[model] = valuations
            contract = getattr(template, 'quote_availability', None)
            kwargs = {'quote_availability':contract} if contract is not None else {}
            result = gate.decide(predictions,DecisionContext(time,FrozenTable(offers), **kwargs))
            audit.extend(result.audit.to_dict('records'))
            rejected.update(set(offers.index)-set(result.selected.index))
    tickets = tickets.loc[~tickets.ticket_id.isin(rejected)].copy()
    members = members.loc[~members.ticket_id.isin(rejected)].copy()
    batch = batch.loc[tickets.ticket_id]
    output_attrs = {'decision_policy_audit': audit}
    if policy is not None and len(batch):
        if context is None:
            raise ValueError('Static allocation requires a manually supplied StakeContext; use replay_bankroll for chronological compounding.')
        allocation = allocate_batch(FrozenTable(batch),policy,context,limits)
        tickets['nominal_stake'] = tickets.stake
        tickets['stake'] = tickets.ticket_id.map(allocation.amounts)
        tickets['funded'] = tickets.stake.gt(0)
        output_attrs['allocation_audit'] = allocation.audit.reset_index().to_dict('records')
    for idx,row in tickets.iterrows():
        legs = indexed[row.ticket_id]
        if policy is not None and row.stake == 0:
            tickets.loc[idx,['settlement','payout','profit','accounting_status']] = ['missing',0.,0.,'unfunded']
            continue
        state,payout = _settle(legs,templates[row.bet],row.stake)
        tickets.loc[idx,['settlement','payout','profit','accounting_status']] = [state,payout,payout-row.stake,'unresolved' if pd.isna(payout) else 'settled']
    # pandas propagates attrs to row Series and slices via deepcopy. Publish
    # complete audits only after internal settlement and caller summaries.
    (tickets.attrs if _audit_sink is None else _audit_sink).update(output_attrs)
    return tickets,members


def allocate_singles(ledger,policy,context,limits=None,*,match_columns=('event_id',),payoff='push_void'):
    """Preserve resolved numeric/Series leg stakes as nominal amounts before replacement."""
    from .tickets import Parlay, ticket_metrics
    selected=ledger.loc[ledger['take']].copy()
    if selected.empty:
        from .tickets import compose_bets
        return compose_bets(ledger,Parlay(size=1))
    if selected.duplicated(['fold_id','row_position','bet']).any():
        raise ValueError('Single allocations require unique observation/offer identities.')
    selected['ticket_id']=[json.dumps([str(row[k]) for k in ('fold_id','row_position','bet')]) for row in selected.to_dict('records')]
    rows=[]
    if payoff not in {'push_void','binary'}:
        raise ValueError('Singles need an explicit binary or push_void payoff.')
    templates={name:Parlay(size=1,payoff=payoff) for name in selected.bet.unique()}
    for record in selected.to_dict('records'):
        rows.append({k:record[k] for k in ('ticket_id','bet','fold_id','kickoff_at','take','stake','odds')} | dict(
            kind='single',n_legs=1,last_kickoff_at=record['kickoff_at'],probability=record.get('p_win',np.nan),
            probability_assumption='single',settlement='missing',accounting_status='unresolved',payout=np.nan,profit=np.nan))
    members=selected.assign(leg_number=1,template=selected.bet)
    output_attrs = {}
    tickets,members=finalize_tickets(pd.DataFrame(rows),members,templates,match_columns,policy,context,limits,
                                   _audit_sink=output_attrs)
    from .tickets import BetSlip
    metrics = ticket_metrics(tickets,BetSlip(templates),members)
    tickets.attrs.update(output_attrs)
    return tickets,members,metrics
