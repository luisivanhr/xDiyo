"""Guarded, call-local ticket execution; private backend of compose_bets.

The ordinary engine owns pool validation and final metrics/audit publication.
Representation rules follow the supplied columnar evidence, checked against the
repaired reference. No cached approvals, monkeypatches or prototype dependency.
"""
from itertools import combinations, islice
from math import fsum
import hashlib
import json
import numpy as np
import pandas as pd
from .all_combinations import AllCombinations
from .decision_layer import DecisionLayer, FrozenTable
from .quote_availability import QuoteAvailability, ASSUMPTION_FIELDS
from .timestamps import timestamps
from .ticket_numeric import checked_product


def supports_bulk(ledger, templates, match_columns, stake_policy=None, stake_context=None, risk_limits=None):
    """Capability only, never an evidence approval. No exception fallback."""
    if list(templates) != ['tickets'] or any(v is not None for v in (stake_policy, stake_context, risk_limits)):
        return False
    p = templates['tickets']; gate = getattr(p, 'ticket_gate', None)
    if (type(p) is not AllCombinations or type(gate) is not DecisionLayer
        or type(p.quote_availability) is not QuoteAvailability or p.audit_level != 'full'
        or p.stake != 1. or p.on_push != 'remove' or p.on_void != 'remove'
        or p.min_ev is not None or p.payoff != 'binary' or p.probability_mode != 'independent'
        or gate.gate != 'and' or gate.metric != 'ev' or gate.threshold != 0.
        or gate.strict is not True or gate.missing != 'error'
        or gate.identity_columns != DecisionLayer.__dataclass_fields__['identity_columns'].default
        or len(gate.models) != 2 or len(set(gate.models)) != 2
        or tuple(match_columns) != ('event_id',) or 'quote_legs' in ledger):
        return False
    if not {'event_id','odds','decision_at','settlement'} <= set(ledger):
        return False
    if ledger.event_id.duplicated().any() or ledger.event_id.isna().any() or not isinstance(ledger.index, pd.RangeIndex):
        return False
    dtype = ledger.odds.to_numpy().dtype
    return dtype == np.dtype('float64') or dtype.kind in 'iu'


def _readonly(values, dtype=None):
    result = np.array(values, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _kernel(odds, probabilities, decisions, kickoffs, members):
    cutoffs = decisions[members]
    if (cutoffs != cutoffs[:, :1]).any() or (cutoffs[:,0] > kickoffs[members].min(axis=1)).any():
        raise ValueError('Every ticket needs one shared decision timestamp no later than kickoff.')
    prices = checked_product(odds[members], axis=1).astype(float)
    p = np.prod(probabilities[members], axis=1)
    ev = p * prices[:, None] - 1.
    if not np.isfinite(ev).all():
        raise ValueError('Ticket EV overflow.')
    return p, (ev > 0.).all(axis=1)


def _settlement(odds, outcomes, members):
    states = outcomes[members]
    if not np.isin(states, ['win','loss','missing','push','void']).all():
        raise ValueError('Unknown leg settlement.')
    loss = (states == 'loss').any(axis=1)
    missing = (states == 'missing').any(axis=1) & ~loss
    active = states == 'win'
    state = np.where(loss, 'loss', np.where(missing, 'missing',
        np.where(~active.any(axis=1), np.where((states == 'void').all(axis=1), 'void', 'push'), 'win')))
    # Compute payouts only for resolved, non-losing tickets, as the reference.
    payout = np.full(len(members), np.nan)
    payout[loss] = 0.
    resolved = ~(loss | missing)
    payout[resolved] = checked_product(np.where(active[resolved], odds[members[resolved]], 1), axis=1)
    return state, payout

_EMPTY_TICKET_COLUMNS = [
    'ticket_id', 'bet', 'fold_id', 'kind', 'n_legs', 'kickoff_at',
    'last_kickoff_at', 'take', 'stake', 'odds', 'probability',
    'probability_assumption', 'settlement', 'accounting_status', 'payout', 'profit',
]


def _source_records(pools, policy, contract):
    """Box each participating fixture once, as native iterrows/to_dict does."""
    records, events, group_for_position, quote_json = {}, {}, {}, {}
    for group_id, columns, offered in pools:
        if len(offered) < policy.legs:
            continue
        for position, leg in offered.iterrows():
            record = leg.drop(labels=['_group', '_event', '_price_valid']).to_dict()
            records[position] = record
            events[position] = leg['_event']
            group_for_position[position] = (group_id, columns)
    # Native membership construction re-infers dtypes from the boxed records.
    # Quote evidence is serialized from that inferred table, not original rows:
    # e.g. an object column containing integer 1 and float 1.5 becomes 1.0/1.5.
    prototype = pd.DataFrame(list(records.values()), index=list(records))
    fields = contract.identities(prototype)
    quote_ids = {}
    for position, record in zip(records, prototype.to_dict('records')):
        evidence = {field: None if pd.isna(record.get(field)) else record.get(field)
                    for field in fields}
        evidence['fixture_identity'] = json.dumps([record['event_id']], default=str)
        # Joining these once-serialized records reproduces json.dumps(list)
        # exactly, without reserializing each repeated fixture occurrence.
        quote_json[position] = json.dumps(evidence, sort_keys=True, default=str)
        quote_ids[position] = record['quote_id']
    return records, events, group_for_position, quote_json, quote_ids, prototype


def _membership(prototype, member_positions, selected, ticket_ids, data):
    """Materialize selected legs only; infer native dtypes from all candidates."""
    if not len(member_positions):
        return pd.DataFrame(columns=[*data.columns, 'ticket_id', 'leg_number', 'template'])
    take_positions = member_positions[selected].ravel()
    members = prototype.loc[take_positions].copy()
    width = member_positions.shape[1]
    # Native finalize_tickets filters an already expanded table, retaining its
    # original candidate/member row indices even when most candidates fail EV.
    members.index = pd.Index((np.flatnonzero(selected)[:, None] * width
                             + np.arange(width)).ravel(), dtype=np.int64)
    members['ticket_id'] = pd.Series(np.repeat(np.asarray(ticket_ids, dtype=object)[selected], width),
                                     index=members.index, dtype=pd.Series(ticket_ids).dtype)
    members['leg_number'] = np.tile(np.arange(1, width + 1, dtype=np.int64), int(selected.sum()))
    members['template'] = 'tickets'
    return members


def execute_bulk(data, prepared, policy):
    # Called only after native preflight in this invocation. Its private
    # intermediate arrays never leave the call or act as reusable credentials.
    contract, models = policy.quote_availability, policy.ticket_gate.models
    pools, previews = prepared['tickets']
    # Box once with exactly the reference's membership inference/order.
    records, events, group_for_position, quote_json, quote_ids, source = _source_records(pools, policy, contract)
    keys = list(records)
    position = {key: i for i, key in enumerate(keys)}
    fields = contract.identities(source)
    if len(source):
        FrozenTable(source[list(fields)])
    norm_odds = _readonly(source.odds if len(source) else np.empty(0), data.odds.to_numpy().dtype)
    probabilities = _readonly(np.column_stack([source[policy.probability_columns[m]].astype(float) for m in models]) if len(source) else np.empty((0, 2)))
    decisions = _readonly(timestamps(source.decision_at).array if len(source) else [], object)
    kickoffs = _readonly(timestamps(source.kickoff_at).array if len(source) else [], object)
    if len(source) and 'p_push' in source and source.p_push.fillna(0).ne(0).any():
        raise ValueError('Binary ticket valuation cannot include nonzero push probability.')
    # Outcomes are read only after the decision arrays are frozen.
    outcomes = source.settlement.to_numpy(copy=True) if len(source) else np.empty(0)
    memberships, probs, takes, states, payouts = [], [], [], [], []
    for _, _, offered in pools:
        iterator = combinations(offered.index, policy.legs)
        while chunk := list(islice(iterator, 4096)):
            membership = np.array([[position[key] for key in row] for row in chunk], dtype=np.intp)
            prob, take = _kernel(norm_odds, probabilities, decisions, kickoffs, membership)
            state, payout = np.full(len(chunk), 'missing', dtype=object), np.full(len(chunk), np.nan)
            if take.any():
                state[take], payout[take] = _settlement(norm_odds, outcomes, membership[take])
            memberships.extend(chunk); probs.append(prob); takes.append(take); states.append(state); payouts.append(payout)
    member_positions = np.asarray(memberships, dtype=object).reshape(-1, policy.legs)
    model_probability = np.concatenate(probs) if probs else np.empty((0, 2))
    selected = np.concatenate(takes) if takes else np.empty(0, dtype=bool)
    settlement = np.concatenate(states) if states else np.empty(0, dtype=object)
    payout = np.concatenate(payouts) if payouts else np.empty(0)
    member_positions = np.asarray(member_positions, dtype=np.intp)
    model_probability = np.asarray(model_probability)
    selected = np.asarray(selected, dtype=bool)
    settlement = np.asarray(settlement, dtype=object)
    payout = np.asarray(payout, dtype=float)
    labels = contract.labels
    member_prototype = source
    rows, candidate_audit, ticket_ids, quote_metadata = [], [], [], []
    ticket_times = []
    candidate_groups = []

    # Timestamp parsing stays fixture-linear. Raw fixture representations are
    # retained separately for exact quote JSON and membership evidence.
    if records:
        time_source = pd.DataFrame([records[p] for p in records], index=list(records))
        parsed = {field: timestamps(time_source[field], utc=True)
                  for field in ('decision_at', 'quote_at')}
        if contract.mode == 'research_assumed':
            parsed['assumed_available_at'] = timestamps(time_source.assumed_available_at, utc=True)
        stamp_lookup = {field: values.to_dict() for field, values in parsed.items()}
    else:
        stamp_lookup = {}

    for i, positions in enumerate(member_positions):
        legs = [records[p] for p in positions]
        first = legs[0]
        group_id, columns = group_for_position[positions[0]]
        event_membership = sorted(events[p] for p in positions)
        identity = json.dumps(['tickets', group_id, event_membership], separators=(',', ':'))
        ticket_id = 'tickets:' + hashlib.sha256(identity.encode()).hexdigest()
        ticket_ids.append(ticket_id)
        candidate_groups.append(group_id)
        odds = float(checked_product(np.asarray([leg['odds'] for leg in legs], dtype=norm_odds.dtype)))
        probability = np.nan
        if 'p_win' in first and all(pd.notna(leg['p_win']) for leg in legs):
            probability = float(np.prod([float(leg['p_win']) for leg in legs]))
        grouping = {column: first[column] for column in columns}
        rows.append(dict(ticket_id=ticket_id, bet='tickets',
                         kind='single' if policy.legs == 1 else 'parlay', n_legs=policy.legs,
                         kickoff_at=min(leg['kickoff_at'] for leg in legs),
                         last_kickoff_at=max(leg['kickoff_at'] for leg in legs),
                         take=True, stake=policy.stake, odds=odds, probability=probability,
                         probability_assumption=policy.probability_mode,
                         settlement='missing', accounting_status='unresolved',
                         payout=np.nan, profit=np.nan, **grouping))
        candidate_audit.append(dict(template='tickets', group_id=group_id, ticket_id=ticket_id,
                                    event_membership=json.dumps(event_membership),
                                    probability=probability, odds=odds, expected_profit=np.nan,
                                    min_ev=None, take=True, rejection_reason='', filter_enabled=False,
                                    comparator='>', probability_assumption=policy.probability_mode,
                                    **grouping))
        observed = [stamp_lookup['quote_at'][p] for p in positions]
        quote_at = max(observed) if all(pd.notna(t) for t in observed) else pd.NaT
        metadata = dict(labels, quote_at=quote_at,
                        quote_legs='[' + ', '.join(quote_json[p] for p in positions) + ']')
        if contract.mode == 'research_assumed':
            metadata['assumed_available_at'] = max(stamp_lookup['assumed_available_at'][p] for p in positions)
        metadata['quote_id'] = json.dumps([quote_ids[p] for p in positions], default=str)
        quote_metadata.append(metadata)
        ticket_times.append(stamp_lookup['decision_at'][positions[0]])

    tickets = pd.DataFrame(rows) if rows else pd.DataFrame(columns=_EMPTY_TICKET_COLUMNS)
    members = _membership(member_prototype, member_positions, selected, ticket_ids, data)
    decision_audit = []
    if len(tickets):
        # Construct the same batch column inference as native ticket_batch; in
        # particular all-NaT observed quote times are timezone-naive datetime64.
        metadata = pd.DataFrame(quote_metadata)
        for field in (*ASSUMPTION_FIELDS, 'quote_at', 'quote_legs'):
            if field in metadata:
                tickets.loc[tickets.bet.eq('tickets'), field] = metadata[field]
        values = model_probability * tickets.odds.to_numpy()[:, None] - 1.
        times = pd.Series(ticket_times)
        for _, positions in times.groupby(times, sort=True).indices.items():
            for model_index, model in enumerate(models):
                for i in positions:
                    take = bool(values[i, model_index] > 0.)
                    record = dict(candidate_id=ticket_ids[i], model=model,
                                  probability=float(model_probability[i, model_index]),
                                  value=float(values[i, model_index]), take=take,
                                  reason='accepted' if take else 'threshold', comparator='>')
                    if contract.mode == 'research_assumed':
                        evidence = quote_metadata[i]
                        record.update(labels,
                                      assumed_available_at=evidence['assumed_available_at'].isoformat(),
                                      quote_at=None if pd.isna(evidence['quote_at']) else evidence['quote_at'].isoformat(),
                                      quote_id=evidence['quote_id'], decision_at=ticket_times[i].isoformat(),
                                      quote_legs=evidence['quote_legs'])
                    decision_audit.append(record)
        tickets = tickets.loc[selected].copy()
        if selected.any():
            tickets['settlement'] = settlement[selected]
            tickets['payout'] = payout[selected]
            tickets['profit'] = payout[selected] - policy.stake
            tickets['accounting_status'] = np.where(np.isnan(payout[selected]), 'unresolved', 'settled')

    summaries = []
    for (group_id, _, _), preview in zip(pools, previews):
        summaries.append(dict(preview, group_id=group_id,
            candidate_tickets=preview['ticket_count'], candidate_stake=preview['expected_stake'],
            selected_tickets=preview['ticket_count'], selected_stake=preview['expected_stake'],
            rejected_by_ev=0, rejected_missing_probability=0, min_ev=None,
            comparator='>', filter_enabled=False, probability_assumption=policy.probability_mode))
    for field in ('stake', 'payout', 'profit'):
        try:
            total = fsum(float(v) for v in tickets[field] if pd.notna(v))
        except OverflowError as exc:
            raise ValueError('Selected ticket accounting total overflow.') from exc
        if not np.isfinite(total):
            raise ValueError('Selected ticket accounting total overflow.')
    attrs = {'decision_policy_audit': decision_audit} if len(member_positions) else {}
    return tickets, members, candidate_audit, summaries, attrs
