"""Compose selected single bets into auditable tickets without fitting a model."""

from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
from math import comb, fsum
import hashlib
import json

import numpy as np
from .ticket_numeric import checked_product
import pandas as pd
from .all_combinations import AllCombinations, prepare_pools, expand_pools, preview_combinations


@dataclass(frozen=True, kw_only=True)
class Parlay:
    """Disjoint tickets of size legs from each prediction-only group.

    Incomplete batches are omitted. Never combines two selections from the same
    fixture or different fold occurrences. Odds multiply; probabilities multiply
    only with the explicit independent assumption. Stake is per ticket.
    """
    size: int = 2
    grouping: str = "league_round"
    order_by: str = "kickoff"
    stake: float = 1.0
    on_push: str = "remove"
    on_void: str = "remove"
    probability_mode: str = "none"
    max_tickets: int = 1000
    payoff: str = 'push_void'


@dataclass(frozen=True, kw_only=True)
class MultiBet(Parlay):
    """System bet: all requested k-leg combinations within each size-leg batch.

    For example size=3, sizes=(2,3) produces three doubles and one treble.
    total stake is divided equally across that batch's generated tickets.
    """
    size: int = 3
    sizes: tuple[int, ...] = (2, 3)
    stake_mode: str = "per_ticket"


@dataclass(frozen=True)
class BetSlip:
    """Named Parlay/MultiBet templates, settled and staked independently.

    size=1 creates singles. Reusing a leg across templates places additional
    stakes; no netting or bankroll simulation is performed.
    """
    tickets: dict


def _validate(policy):
    for key in ("size", "max_tickets"):
        value = getattr(policy, key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if policy.grouping not in {"league_round", "round", "day"}:
        raise ValueError("grouping must be league_round, round across leagues, or day (UTC).")
    if policy.order_by not in {"kickoff", "probability", "expected_profit"}:
        raise ValueError("order_by must be kickoff, probability or expected_profit.")
    if not np.isfinite(policy.stake) or policy.stake < 0:
        raise ValueError("Ticket stake must be finite and nonnegative.")
    if policy.on_push not in {"remove", "refund", "loss"} or policy.on_void not in {"remove", "refund", "loss"}:
        raise ValueError("Push/void rules must be remove, refund or loss.")
    if policy.probability_mode not in {"none", "independent"}:
        raise ValueError("probability_mode must be none or independent.")
    sizes = policy.sizes if isinstance(policy, MultiBet) else (policy.size,)
    if not sizes or len(set(sizes)) != len(sizes) or any(
        isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= policy.size for k in sizes
    ):
        raise ValueError("System sizes must be distinct integers from 1 to pool size.")
    if isinstance(policy, MultiBet) and policy.stake_mode not in {"per_ticket", "total"}:
        raise ValueError("stake_mode must be per_ticket or total.")
    return sizes


def _settle(legs, policy, stake):
    states = legs.settlement.tolist()
    if any(s not in {"win", "loss", "push", "void", "missing"} for s in states):
        raise ValueError("Unknown leg settlement.")
    losing = "loss" in states or any(s in states and getattr(policy, "on_" + s) == "loss" for s in ("push", "void"))
    refunds = [s for s in ("void", "push") if s in states and getattr(policy, "on_" + s) == "refund"]
    if losing:
        return "loss", 0.0
    if refunds:
        return refunds[0], stake
    if "missing" in states:
        return "missing", np.nan
    active = legs.loc[legs.settlement.eq("win"), "odds"]
    if not len(active):
        return ("void" if all(s == "void" for s in states) else "push"), stake
    if active.isna().any():
        return "win", np.nan
    payout = stake * float(checked_product(active))
    if not np.isfinite(payout):
        raise ValueError("Ticket payout overflow; inspect quotes and ticket size.")
    return "win", payout


def compose_bets(ledger, composition, *, match_columns=("event_id",), stake_policy=None, stake_context=None, risk_limits=None):
    """Return ticket ledger, ticket-leg membership and ticket-level metrics.

    Only selected legs enter composition. Grouping/ranking never consults
    outcomes, profits or settlement status. Retained predictions must have been
    available together to interpret tickets as a prospective betting simulation.
    """
    templates = composition.tickets if isinstance(composition, BetSlip) else {"tickets": composition}
    deferred = stake_policy is not None or any(getattr(p, 'ticket_gate', None) is not None for p in templates.values())
    if not isinstance(templates, dict) or not templates:
        raise ValueError("A BetSlip needs at least one named ticket template.")
    levels = {getattr(p, 'audit_level', 'full') for p in templates.values()}
    if len(levels) != 1 or not levels <= {'full', 'summary'}:
        raise ValueError('All templates in one composition must use the same valid audit_level.')
    compact = levels == {'summary'}
    rows, members = [], []
    prepared, decisions, summaries = {}, [], []
    # Complete preflight for every whole-group template before any expansion.
    for name, policy in templates.items():
        if isinstance(policy, AllCombinations):
            pools, preview = prepare_pools(ledger, policy, match_columns, name, outcome_free=deferred)
            count = sum(int(r['ticket_count']) for r in preview)
            if count > policy.max_tickets:
                raise ValueError(f'{name}: requested {count} tickets exceeds max_tickets={policy.max_tickets}; '
                                 'reduce the eligible population/legs or deliberately increase max_tickets. '
                                 'Use preview_combinations to inspect group counts.')
            prepared[name] = (pools, preview)
    from .bulk_tickets import supports_bulk, execute_bulk
    if supports_bulk(ledger, templates, match_columns, stake_policy, stake_context, risk_limits):
        tickets, membership, decisions, summaries, output_attrs = execute_bulk(ledger, prepared, templates['tickets'], match_columns)
        return _finish_composition(tickets, membership, composition, templates, prepared, decisions,
                                   summaries, output_attrs, True, False, ledger,
                                   stake_policy, stake_context, risk_limits)
    for name, policy in templates.items():
        if not isinstance(name, str) or not name or not isinstance(policy, (Parlay, MultiBet, AllCombinations)):
            raise TypeError("Name each BetSlip entry and use a Parlay, MultiBet or AllCombinations template.")
        if isinstance(policy, AllCombinations):
            pools, preview = prepared[name]
            compact_counts = {} if compact else None
            new_rows, new_members, audit = expand_pools(pools, policy, name, settle=not deferred, _summary=compact_counts)
            rows.extend(new_rows)
            members.extend(new_members)
            decisions.extend(audit)
            counts = compact_counts if compact else {}
            for candidate in audit:
                tally = counts.setdefault(candidate['group_id'], [0, 0, 0])
                tally[0 if candidate['take'] else 2 if candidate['rejection_reason'] == 'missing_probability' else 1] += 1
            for (group_id, _, _), record in zip(pools, preview):
                selected, rejected_ev, rejected_missing = counts.get(group_id, [0, 0, 0])
                summaries.append(dict(record, group_id=group_id,
                    candidate_tickets=record['ticket_count'], candidate_stake=record['expected_stake'],
                    selected_tickets=selected, selected_stake=selected * policy.stake,
                    rejected_by_ev=rejected_ev, rejected_missing_probability=rejected_missing,
                    min_ev=policy.min_ev, comparator='>', filter_enabled=policy.min_ev is not None,
                    probability_assumption=policy.probability_mode))
            continue
        sizes = _validate(policy)
        if ledger.empty:
            continue
        selected = ledger.loc[ledger['take']].copy()
        if selected.empty:
            continue
        match_keys = list(match_columns)
        if not match_keys or any(k not in selected for k in match_keys):
            raise ValueError("Ticket composition needs the fixture identity columns.")
        selected["kickoff_at"] = pd.to_datetime(selected.kickoff_at, utc=True, errors="raise")
        if selected.kickoff_at.isna().any() or selected[match_keys].isna().any().any():
            raise ValueError("Ticket legs require fixture identities and kickoff times.")
        quotes = pd.to_numeric(selected.odds, errors="raise")
        if (quotes.notna() & (~np.isfinite(quotes) | quotes.le(1))).any():
            raise ValueError("Leg odds must be finite decimal odds greater than 1.")
        selected['odds'] = quotes
        groups = ['fold_id']
        if policy.grouping == "day":
            selected["ticket_day"] = selected.kickoff_at.dt.strftime("%Y-%m-%d")
            groups += ["ticket_day"]
        else:
            # League-specific season IDs cannot identify a shared cross-league
            # season. Require the export's common season label in that mode.
            grouping_keys = (("source_season",), ("round",)) if policy.grouping == "round" else (
                ("competition_id", "source_league"), ("season_id", "source_season"), ("round",))
            for choices in grouping_keys:
                key = next((key for key in choices if key in selected), None)
                if key is None:
                    raise ValueError("Round tickets need season and round metadata (source_season across leagues); league_round also needs league metadata.")
                groups.append(key)
        if selected[groups].isna().any().any():
            raise ValueError("Ticket grouping metadata must be nonmissing.")
        rank = {"kickoff": "kickoff_at", "probability": "p_win", "expected_profit": "expected_profit"}[policy.order_by]
        if rank not in selected or selected[rank].isna().any():
            raise ValueError(f"Ticket ordering needs nonmissing {rank}; use kickoff ordering for manual bets.")
        selected["_fixture_order"] = selected[match_keys].astype(str).agg("|".join, axis=1)
        selected = selected.sort_values([rank, "kickoff_at", "_fixture_order", "bet"],
                                        ascending=[policy.order_by == "kickoff", True, True, True], kind="stable")
        count = 0
        for group, offered in selected.groupby(groups, sort=False):
            # Team-match layouts can select two sides; retain the highest ranked
            # selection per fixture rather than silently assume independence.
            offered = offered.drop_duplicates(match_keys)
            for start in range(0, len(offered) - policy.size + 1, policy.size):
                batch = offered.iloc[start:start + policy.size]
                n_tickets = sum(comb(policy.size, k) for k in sizes)
                if count + n_tickets > policy.max_tickets:
                    raise ValueError(f"{name}: ticket count exceeds max_tickets; reduce pool size/system sizes.")
                stake = policy.stake / n_tickets if isinstance(policy, MultiBet) and policy.stake_mode == "total" else policy.stake
                for size in sizes:
                    for positions in combinations(range(len(batch)), size):
                        legs = batch.iloc[list(positions)]
                        count += 1
                        ticket_id = f"{name}:{count}"
                        odds = float(checked_product(legs.odds)) if legs.odds.notna().all() else np.nan
                        if not pd.isna(odds) and not np.isfinite(odds):
                            raise ValueError("Combined odds overflow; reduce ticket size.")
                        state, payout = _settle(legs, policy, stake) if not deferred else ('missing', np.nan)
                        probability = np.nan
                        if policy.probability_mode == "independent" and 'p_win' in legs and legs.p_win.notna().all():
                            if (~np.isfinite(legs.p_win) | ~legs.p_win.between(0, 1)).any():
                                raise ValueError("Leg probabilities must be between 0 and 1.")
                            probability = float(np.prod(legs.p_win))
                        record = dict(ticket_id=ticket_id, bet=name, fold_id=legs.fold_id.iloc[0],
                                      kind="single" if size == 1 else "parlay", n_legs=size,
                                      kickoff_at=legs.kickoff_at.min(), last_kickoff_at=legs.kickoff_at.max(),
                                      take=True, stake=stake, odds=odds, probability=probability,
                                      probability_assumption=policy.probability_mode, settlement=state,
                                      accounting_status="unresolved" if pd.isna(payout) else "settled",
                                      payout=payout, profit=payout-stake)
                        for key in groups:
                            record[key] = legs[key].iloc[0]
                        rows.append(record)
                        for position, (_, leg) in enumerate(legs.iterrows(), 1):
                            member = leg.drop(labels=['_fixture_order']).to_dict()
                            members.append(dict(member, ticket_id=ticket_id, leg_number=position, template=name))
    # Check the combined slip, including mixed templates, before metrics or curves.
    try:
        for field in ('stake', 'payout', 'profit'):
            if not np.isfinite(fsum(r[field] for r in rows if pd.notna(r[field]))):
                raise OverflowError
    except OverflowError as exc:
        raise ValueError('Selected ticket accounting total overflow; reduce stake or ticket population.') from exc
    columns = ['ticket_id', 'bet', 'fold_id', 'kind', 'n_legs', 'kickoff_at', 'last_kickoff_at', 'take',
               'stake', 'odds', 'probability', 'probability_assumption', 'settlement', 'accounting_status', 'payout', 'profit']
    tickets = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    membership = pd.DataFrame(members) if members else pd.DataFrame(columns=[*ledger.columns, 'ticket_id', 'leg_number', 'template'])
    output_attrs = {}
    if deferred:
        from .ticket_allocation import finalize_tickets
        tickets, membership = finalize_tickets(tickets, membership, templates, match_columns,
                                               stake_policy, stake_context, risk_limits, _audit_sink=output_attrs)
    return _finish_composition(tickets, membership, composition, templates, prepared, decisions,
                               summaries, output_attrs, deferred, compact, ledger,
                               stake_policy, stake_context, risk_limits)


def _finish_composition(tickets, membership, composition, templates, prepared, decisions,
                        summaries, output_attrs, deferred, compact, ledger,
                        stake_policy, stake_context, risk_limits):
    if prepared:
        if deferred:
            # Group positions once; preserve each original ordered Series.sum
            # instead of changing floating-point reduction order via groupby.sum.
            summary_groups = {name: frame.groupby('group_id', sort=False).indices
                              for name, frame in tickets.groupby('bet', sort=False)
                              if 'group_id' in frame}
            summary_templates = {name: frame for name, frame in tickets.groupby('bet', sort=False)}
            for record in summaries:
                selected = summary_templates.get(record['template'], tickets.iloc[:0])
                if 'group_id' in selected:
                    selected = selected.iloc[summary_groups.get(record['template'], {}).get(record['group_id'], [])]
                else:
                    for key in ('fold_id','competition_id','source_league','season_id','source_season','round','tournament_id','stage_id','stage'):
                        if key in record and key in selected:
                            selected = selected.loc[selected[key].eq(record[key])]
                record.update(selected_tickets=len(selected), selected_stake=_finite_accounting_sum(selected.stake),
                              funded_tickets=int(selected.stake.gt(0).sum()), unfunded_tickets=int(selected.stake.eq(0).sum()))
        # Internal transport only; reporters promote these records to exportable tables.
        if not compact:
            output_attrs['ticket_candidates'] = _audit_records(decisions)
        output_attrs.update(ticket_selection_summary=_audit_records(summaries),
                             ticket_ev_enabled=any(p.min_ev is not None for p in templates.values() if isinstance(p, AllCombinations)))
    metrics = ticket_metrics(tickets, composition, membership)
    declarations = []
    for name, policy in templates.items():
        contract = getattr(policy, 'quote_availability', None)
        if contract is None:
            continue
        labels = contract.labels
        declarations.append(dict(template=name, **labels))
        for frame, key in ((tickets, 'bet'), (membership, 'template'), (metrics, 'target')):
            if key in frame:
                for field, value in labels.items():
                    if field not in frame:
                        frame[field] = pd.Series(index=frame.index, dtype=object)
                    frame.loc[frame[key].eq(name), field] = value
        for key in ('ticket_candidates', 'ticket_selection_summary'):
            for record in output_attrs.get(key, []):
                if record['template'] == name:
                    record.update(labels)
    if declarations:
        output_attrs['quote_assumptions'] = declarations
        membership.attrs['quote_assumptions'] = declarations
        metrics.attrs['quote_assumptions'] = declarations
    if compact:
        from .audit_storage import summary_manifest
        manifest = summary_manifest(ledger, templates, tickets, output_attrs.get('ticket_selection_summary', []),
                                    output_attrs, stake_policy, stake_context, risk_limits)
        output_attrs['audit_manifest'] = manifest
        membership.attrs['audit_manifest'] = manifest
        metrics.attrs['audit_manifest'] = manifest
    tickets.attrs.update(output_attrs)
    return tickets, membership, metrics


def _audit_records(records):
    """JSON-safe transport, retaining exact integer IDs and all rejection evidence.

    Unrepresentably large candidate stake previews remain exact decimal text;
    selected accounting totals must be finite. Missing evidence is JSON null.
    """
    def scalar(value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, Decimal):
            return str(value)
        return None if pd.isna(value) else value
    return [{key: scalar(value) for key, value in record.items()} for record in records]


def _finite_accounting_sum(values):
    # Retain pandas' ordinary reduction and rounding. Only repair a nonfinite
    # reduction of finite amounts; fsum can represent totals pandas overflows.
    with np.errstate(over='ignore', invalid='ignore'):
        total = float(values.sum())
    if not np.isfinite(total):
        try:
            total = fsum(float(v) for v in values if pd.notna(v))
        except (OverflowError, ValueError) as exc:
            raise ValueError('Selected ticket metric total overflow.') from exc
        if not np.isfinite(total):
            raise ValueError('Selected ticket metric total overflow.')
    return total


def ticket_metrics(tickets, composition, membership):
    """Same metric contract as single bets; each ticket counts once."""
    result = []
    templates = composition.tickets if isinstance(composition, BetSlip) else {'tickets':composition}
    for name in templates:
        group = tickets.loc[tickets.bet.eq(name)]
        settled = group.accounting_status.eq('settled')
        stakes = _finite_accounting_sum(group.loc[settled, 'stake'])
        profit = _finite_accounting_sum(group.loc[settled, 'profit'])
        roi = profit/stakes if stakes else np.nan
        if stakes and not np.isfinite(roi):
            raise ValueError('Selected ticket ROI overflow.')
        pending = int(group.accounting_status.eq('unresolved').sum())
        values = dict(profit=profit, roi=roi, bets_placed=len(group),
                      settled_bets=int(settled.sum()), unresolved_bets=pending, settled_stakes=stakes,
                      payout=_finite_accounting_sum(group.loc[settled, 'payout']))
        values.update({f'{s}_count': int((settled & group.settlement.eq(s)).sum()) for s in ('win','loss','push','void')})
        if 'funded' in group:
            values.update(selected_tickets=len(group), funded_tickets=int(group.funded.sum()), unfunded_tickets=int((~group.funded).sum()))
        evidence = membership.loc[membership.template.eq(name)].drop(columns=['profit', 'payout'], errors='ignore')
        digest = hashlib.sha256(evidence.to_json(date_format='iso').encode()).hexdigest()
        for metric, value in values.items():
            status = 'undefined' if not np.isfinite(value) else 'partial' if pending and metric in {'profit','roi'} else 'ok'
            result.append(dict(metric=metric, calculation=metric, target=name, output='bets', value=value,
                               direction='maximize' if metric in {'profit','roi'} else None,
                               n=int(settled.sum()), n_total=len(group), n_missing=pending, status=status,
                               parameters=json.dumps({'composition':repr(composition), 'roi_denominator':'settled_ticket_stakes'}),
                               sample_hash=digest))
    return pd.DataFrame(result, columns=['metric','calculation','target','output','value','direction','n','n_total','n_missing','status','parameters','sample_hash'])
