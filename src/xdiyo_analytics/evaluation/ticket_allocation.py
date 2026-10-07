"""Outcome-free ticket selection/allocation followed by explicit retrospective settlement."""
import json
import hashlib
from collections.abc import Mapping
import numpy as np
import pandas as pd
from .decision_layer import FrozenTable, DecisionContext
from .stake_policy import allocate_batch
from .quote_availability import QuoteAvailability, quote_leg_records, ASSUMPTION_FIELDS


class _MembershipIndex(Mapping):
    """One positional index, not a retained DataFrame per candidate ticket."""
    def __init__(self, members):
        self.members = members
        self.positions = members.groupby('ticket_id', sort=False, dropna=False, observed=True).indices

    def __getitem__(self, key):
        return self.members.iloc[self.positions[key]]

    def __iter__(self):
        return iter(self.positions)

    def __len__(self):
        return len(self.positions)


def _index_members(members):
    """Preserve exact IDs, every conflicting/duplicate row and original order."""
    return _MembershipIndex(members)


def ticket_batch(tickets, members, templates, match_columns, *, context=None, _members_by_ticket=None):
    indexed = _index_members(members) if _members_by_ticket is None else _members_by_ticket
    checked_contracts = set()
    if isinstance(indexed, _MembershipIndex) and 'decision_at' in members:
        for name, offered in tickets.groupby('bet', sort=False):
            contract = getattr(templates[name], 'quote_availability', None)
            if type(contract) is not QuoteAvailability:
                continue
            pieces = [indexed.positions[key] for key in offered.ticket_id if key in indexed]
            if not pieces:
                continue
            evidence = members.iloc[np.concatenate(pieces)]
            try:
                time = pd.to_datetime(evidence.decision_at, utc=True).max()
                # validate compares each quote with its own original decision,
                # not merely this outer maximum. No deduplication or cross-call
                # approval cache; every consumed membership row is checked.
                contract.validate(evidence, time, complete=True)
            except ValueError:
                # Preserve per-ticket parsing for differing valid string
                # formats, and original errors for invalid evidence.
                continue
            checked_contracts.add(name)
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
                if row['bet'] not in checked_contracts:
                    contract.validate(legs, time, complete=True)
                continue
            if key in legs and (pd.to_datetime(legs[key],utc=True).isna().any() or (pd.to_datetime(legs[key],utc=True)>time).any()):
                raise ValueError('A leg prediction or quote is unavailable at ticket issue time.')
        fixtures = tuple(sorted((tuple(v) for v in legs[list(match_columns)].itertuples(index=False,name=None)),key=repr))
        identities = []
        identity_fields = (*match_columns,'bet','market','selection','line','odds','quote_id','quote_at')
        identity_columns = list(dict.fromkeys(k for k in identity_fields if k in legs))
        for leg in legs[identity_columns].to_dict('records'):
            identities.append({k:leg.get(k) for k in identity_fields})
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
            compact = getattr(policy, 'audit_level', 'full') == 'summary'
            record['quote_legs'] = quote_leg_records(legs, contract, match_columns, compact=compact)
            evidence_key = hashlib.sha256(repr(record['quote_legs'].records).encode()).hexdigest() if compact else record['quote_legs']
            record['economic_key'] = json.dumps([record['economic_key'], evidence_key], sort_keys=True)
        if 'issued_at' in legs:
            record['probability_issued_at'] = pd.to_datetime(legs.issued_at,utc=True).max()
        records.append(record)
    return pd.DataFrame(records).set_index('ticket_id') if records else pd.DataFrame(columns=['nominal_stake','decision_at'])


def finalize_tickets(tickets,members,templates,match_columns,policy,context,limits, *, _audit_sink=None):
    from .tickets import _settle
    if tickets.empty:
        return tickets,members
    levels = {getattr(template, 'audit_level', 'full') for template in templates.values()}
    if len(levels) != 1 or not levels <= {'full', 'summary'}:
        raise ValueError('All templates in one composition must use the same valid audit_level.')
    compact = levels == {'summary'}
    # This helper is also called directly: do not trust a prior compose_bets
    # preflight when retained members may have been edited since that call.
    from .quote_availability import validate_model_quotes
    for name, template in templates.items():
        contract = getattr(template, 'quote_availability', None)
        if contract is not None:
            legs = members.loc[members.template.eq(name)]
            if len(legs):
                time = pd.to_datetime(legs.decision_at, utc=True, errors='raise').max()
                contract.validate(legs, time, complete=True)
                validate_model_quotes(legs, template.ticket_gate.models, contract, time)
    indexed = _index_members(members)
    settlement_members = members[['settlement', 'odds']]
    batch = ticket_batch(tickets,members,templates,match_columns,context=context,
                         _members_by_ticket=indexed)
    for name, template in templates.items():
        contract = getattr(template, 'quote_availability', None)
        if contract is not None:
            selected = tickets.bet.eq(name)
            for field in (*ASSUMPTION_FIELDS, 'quote_at', 'quote_legs'):
                if compact and field == 'quote_legs':
                    continue
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
        # Parse each original model stream once for this template. Keep original
        # positional keys even when caller-supplied row labels are duplicated.
        positions = (np.sort(np.concatenate([indexed.positions[key] for key in subset.index]))
                     if len(subset) else np.array([], dtype=np.intp))
        source = members.iloc[positions]
        source_times = pd.Series(source.ticket_id.map(batch.decision_at).array, index=positions)
        position_to_source = np.full(len(members), -1, dtype=np.intp)
        position_to_source[positions] = np.arange(len(positions))
        model_inputs = {}
        for model, column in mapping.items():
            p = pd.to_numeric(source.get(column, pd.Series(np.nan, index=source.index)), errors='raise').astype(float)
            times = {}
            for field in ('issued_at', 'trained_through', 'artifact_vintage'):
                raw = source.get(f'{model}::{field}', pd.Series(pd.NaT, index=source.index))
                try:
                    times[field] = pd.to_datetime(raw, utc=True)
                except ValueError:
                    # A template can contain tickets with different valid
                    # timestamp string formats. Retain the original per-ticket
                    # parser in that case, including its rejection behavior.
                    positional = pd.Series(raw.array, index=positions)
                    parts = [pd.to_datetime(positional.loc[indexed.positions[key]], utc=True)
                             for key in subset.index]
                    times[field] = pd.concat(parts).loc[positions]
            p = pd.Series(p.array, index=positions)
            times = {field: pd.Series(values.array, index=positions) for field, values in times.items()}
            present = p.notna()
            if (present & (~np.isfinite(p) | ~p.between(0,1))).any():
                raise ValueError('Supplied model leg probabilities must be finite and in [0,1].')
            for field, values in times.items():
                if (present & (values.isna() | values.gt(source_times))).any():
                    raise ValueError(f'Every valued model leg needs nonmissing available {model}::{field} provenance.')
            if (present & (times['trained_through'].ge(times['issued_at']) | times['artifact_vintage'].gt(times['issued_at']))).any():
                raise ValueError('Model leg evidence was unavailable at its own prediction issue time.')
            # Keep ordered original values. Validation remains per original leg
            # and its own ticket cutoff; only pandas lookup work is batched.
            model_inputs[model] = (p.to_numpy(), {field: values.array for field, values in times.items()})
        for time,offers in subset.groupby('decision_at',sort=True):
            predictions = {}
            for model,column in mapping.items():
                valuations = offers.copy()
                probabilities = []
                provenance = {field: [] for field in ('issued_at','trained_through','artifact_vintage')}
                for ticket_id in offers.index:
                    positions = position_to_source[indexed.positions[ticket_id]]
                    model_p, model_times = model_inputs[model]
                    p = model_p[positions]
                    complete = not np.isnan(p).any()
                    probabilities.append(float(np.prod(p)) if complete else np.nan)
                    for field in provenance:
                        provenance[field].append(model_times[field].take(positions).max() if complete else pd.NaT)
                valuations['probability'] = probabilities
                for field, values in provenance.items():
                    valuations[field] = values
                predictions[model] = valuations
            contract = getattr(template, 'quote_availability', None)
            kwargs = {'quote_availability':contract} if contract is not None else {}
            result = gate.decide(predictions,DecisionContext(time,FrozenTable(offers), **kwargs),
                                 **({'audit_level': 'summary'} if compact else {}))
            if compact:
                audit.append(result.audit.assign(template=name))
            else:
                audit.extend(result.audit.to_dict('records'))
            rejected.update(set(offers.index)-set(result.selected.index))
    tickets = tickets.loc[~tickets.ticket_id.isin(rejected)].copy()
    members = members.loc[~members.ticket_id.isin(rejected)].copy()
    batch = batch.loc[tickets.ticket_id]
    if compact:
        ballots = pd.concat(audit, ignore_index=True) if audit else pd.DataFrame(columns=[
            'candidate_id', 'model', 'probability', 'value', 'take', 'reason', 'comparator', 'template'])
        reasons = ballots.groupby(['template', 'model', 'reason'], sort=False).size().reset_index(name='count')
        selected_ballots = ballots.loc[ballots.candidate_id.isin(tickets.ticket_id)]
        output_attrs = {'gate_summary': reasons.to_dict('records'),
                        'selected_model_values': {'columns': list(selected_ballots), 'data': selected_ballots.values.tolist()},
                        'gate_counts': {'candidates': ballots.candidate_id.nunique(),
                                        'selected': selected_ballots.candidate_id.nunique(), 'rejected': len(rejected)}}
    else:
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
        legs = settlement_members.iloc[indexed.positions[row.ticket_id]]
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
