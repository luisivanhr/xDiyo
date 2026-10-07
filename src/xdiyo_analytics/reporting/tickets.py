"""Ticket presentation layered on the retained selected-leg ledger."""

from html import escape
import json
import pandas as pd

from ..evaluation.tickets import compose_bets, preview_combinations
from .contracts import Artifact
from .teams import team_key


def add_quote_disclosure(result, tickets):
    """Carry the declaration into saved/reused reports and independent table exports."""
    declarations = tickets.attrs.get('quote_assumptions')
    if not declarations:
        return
    result.tables['quote_assumptions'] = pd.DataFrame(declarations)
    result.artifacts.append(Artifact('table', result.tables['quote_assumptions'], 'Quote availability declarations'))
    for name, table in result.tables.items():
        if name != 'quote_assumptions' and 'quote_availability_mode' not in table:
            table['quote_availability_declarations'] = json.dumps(declarations, sort_keys=True)
    if any(r['quote_availability_mode'] == 'research_assumed' for r in declarations):
        result.notes.append('Research simulation only: quote availability is assumed, not observed or certified historically tradable. See quote availability declarations and per-leg evidence.')


def add_tickets(result, composition, context, *, fixture_table=None, teams=None, stake_policy=None, stake_context=None, risk_limits=None, single_payoff='push_void'):
    legs = result.tables['ledger']
    alternatives = result.tables.get('alternatives')
    if alternatives is not None:
        fields = ['fold_id', 'row_position', 'bet', 'p_win', 'p_push', 'p_loss', 'expected_profit', 'description', 'reason']
        if 'probability_abstention' in alternatives:
            fields.append('probability_abstention')
        evidence = alternatives[fields].rename(columns={'reason':'decision_reason'})
        keys = ['fold_id', 'row_position', 'bet']
        # Probability evidence comes from the retained decision output. Metadata
        # may contain equally named columns; do not create ambiguous _x/_y fields.
        legs = legs.drop(columns=[c for c in evidence if c not in keys], errors='ignore').merge(
            evidence, on=keys, how='left', validate='one_to_one')
    if fixture_table is not None and len(fixture_table):
        fields = ['fold_id', 'row_position', 'home_id', 'away_id', 'home', 'away', 'result']
        display = fixture_table[fields]
        legs = legs.drop(columns=[f for f in fields[2:] if f in legs]).merge(
            display, on=fields[:2], how='left', validate='many_to_one')
    if composition is None:
        from ..evaluation.ticket_allocation import allocate_singles
        tickets, membership, metrics = allocate_singles(legs, stake_policy, stake_context, risk_limits, match_columns=context.match_columns, payoff=single_payoff)
    else:
        tickets, membership, metrics = compose_bets(legs, composition, match_columns=context.match_columns,
                                                   stake_policy=stake_policy, stake_context=stake_context, risk_limits=risk_limits)
    result.tables.update(leg_ledger=legs, ledger=tickets, tickets=tickets, ticket_legs=membership, bet_metrics=metrics)
    manifest = tickets.attrs.get('audit_manifest')
    if manifest is not None:
        result.tables['audit_manifest'] = pd.DataFrame([{'audit_level': manifest['audit_level'],
                                                       'manifest': json.dumps(manifest, sort_keys=True)}])
        result.artifacts.append(Artifact('table', result.tables['audit_manifest'], 'Audit manifest and omissions'))
        values = tickets.attrs.get('selected_model_values', {'columns': [], 'data': []})
        result.tables['selected_model_values'] = pd.DataFrame(values['data'], columns=values['columns'])
        result.tables['gate_summary'] = pd.DataFrame(manifest.get('gate_reasons', []))
        result.notes.append(f"Summary audit: {manifest['candidate_count']} candidates, {manifest['selected_count']} selected, "
                            f"{manifest['rejected_count']} rejected. Rejected row-level evidence and repeated quote payloads were omitted. "
                            'This is compact traceability, not a complete row audit. See the manifest for fingerprints, policies, counts and omissions.')
    for key in ('allocation_audit', 'decision_policy_audit'):
        if key in tickets.attrs:
            result.tables[key] = pd.DataFrame(tickets.attrs[key])
            result.artifacts.append(Artifact('table', result.tables[key], key.replace('_',' ').title()))
    empty_columns = {
        'ticket_candidates': ['template', 'group_id', 'ticket_id', 'event_membership', 'probability', 'odds',
                              'expected_profit', 'min_ev', 'take', 'rejection_reason', 'filter_enabled',
                              'comparator', 'probability_assumption'],
        'ticket_selection_summary': ['template', 'group_id', 'eligible_events', 'input_events', 'legs',
                                     'ticket_count', 'expected_stake', 'candidate_tickets', 'candidate_stake',
                                     'selected_tickets', 'selected_stake', 'rejected_by_ev',
                                     'rejected_missing_probability', 'min_ev', 'comparator',
                                     'filter_enabled', 'probability_assumption'],
    }
    for name in empty_columns:
        if name in tickets.attrs:
            records = tickets.attrs[name]
            result.tables[name] = pd.DataFrame(records) if records else pd.DataFrame(columns=empty_columns[name])
    if tickets.attrs.get('ticket_ev_enabled'):
        result.artifacts.append(Artifact('table', result.tables['ticket_selection_summary'], 'Ticket selection: candidates and placed bets'))
        result.notes.append('Ticket EV filter: strict EV > minimum, per unit stake, using independent win/loss probabilities. '
                            + ('Rejected candidates are counted in the summary manifest and incur no stake. ' if manifest else
                               'Rejected candidates are audited separately and incur no stake. ')
                            + 'Overlapping tickets share risk.')
    from ..evaluation.tickets import Parlay
    preview = preview_combinations(legs, composition or Parlay(size=1), match_columns=context.match_columns)
    if preview.attrs['exceeded_limits']:
        result.tables['combination_preview'] = preview
        result.artifacts.append(Artifact('table', preview, 'Candidate combinations before ticket EV filtering'))
        result.notes.append(f"Whole-group combinations: {preview.attrs['total_tickets']} candidate tickets; "
                            f"hypothetical candidate stake {preview.attrs['total_stake']:g}. Selected counts/stakes are in the selection summary. Exclusion counts may overlap. "
                            'Each combination is staked once; shared events across tickets are intentional.')
    result.notes.extend([
        'Ticket stakes replace leg stakes. Each BetSlip template places separate bets. Parlay/MultiBet omit incomplete batches; AllCombinations uses whole eligible groups.',
        'Probabilities stay blank unless independence is explicitly selected. Quoted odds multiply; removed push/void legs contribute odds 1 at settlement.',
        'All leg predictions must be available before the first kickoff for prospective use. Groups stay within fold occurrences; calendar days use UTC.',
    ])
    result.artifacts.insert(0, Artifact('html', ticket_html(tickets, membership, teams or {}), 'Bet tickets'))
    add_quote_disclosure(result, tickets)
    return result


def ticket_html(tickets, membership, teams):
    declarations = tickets.attrs.get('quote_assumptions', [])
    warning = ''.join('<p><strong>Research simulation only: assumed quote availability.</strong> '
                      + escape(str(r['quote_assumption_id'])) + ' · '
                      + escape(str(r['quote_assumption_rationale'])) + ' · Reference: '
                      + escape(str(r['quote_assumption_reference']))
                      + '. Historical tradability is unverified.</p>'
                      for r in declarations if r['quote_availability_mode'] == 'research_assumed')
    if tickets.attrs.get('audit_manifest', {}).get('audit_level') == 'summary':
        warning += '<p><strong>Summary audit.</strong> Detailed rejected-candidate provenance was omitted. Counts and omission declarations are in the audit manifest.</p>'
    def fmt(value):
        return '—' if pd.isna(value) else escape(f'{value:g}' if isinstance(value, (float, int)) else str(value))

    def team(row, side):
        key = team_key(row.get(side + '_id', ''))
        display = teams.get(key, {})
        name = display.get('name') or row.get(side) or row.get(side + '_name') or f'Team {key}'
        badge = display.get('badge')
        image = f'<img src="{escape(badge, quote=True)}" alt="" width="24" height="24"> ' if badge else ''
        return image + escape(str(name))

    if tickets.empty:
        if tickets.attrs.get('ticket_ev_enabled'):
            evidence = 'audit manifest and selection summary' if tickets.attrs.get('audit_manifest') else 'candidate decisions and selection summary'
            return warning + f'<p>No tickets placed. The ticket EV filter requires a scorable probability and EV strictly above the minimum. See {evidence} for missing probabilities, rejected EVs, or undersized groups.</p>'
        return warning + '<p>No complete tickets in the selected groups. Reduce the leg count or select more fixtures.</p>'
    # Only display values are needed below. Preserve caller attrs and copy them
    # once, rather than propagating a full audit/manifest to every leg slice.
    tickets = tickets.copy(deep=False)
    membership = membership.copy(deep=False)
    tickets.attrs = {}
    membership.attrs = {}
    display_fields = ('ticket_id', 'home_id', 'home', 'home_name', 'away_id', 'away',
                      'away_name', 'bet', 'description', 'p_win', 'result', 'settlement', 'odds')
    membership = membership[[c for c in display_fields if c in membership]]
    panels = []
    membership_positions = membership.groupby('ticket_id', sort=False).indices
    for record in tickets.to_dict('records'):
        legs = membership.iloc[membership_positions[record['ticket_id']]]
        color = {'win':'#58c7b2', 'loss':'#ed7975'}.get(record['settlement'], '#9eb0c5')
        header = (f"{escape(record['ticket_id'])} · {record['n_legs']} leg(s) · {escape(record['settlement'])}"
                  f" · Stake {fmt(record['stake'])}")
        if pd.notna(record['odds']):
            header += f" · Quoted odds {fmt(record['odds'])}"
        if record['stake'] > 0 and record['settlement'] in ('win', 'push', 'void') and pd.notna(record['payout']):
            settled_odds = record['payout'] / record['stake']
            if pd.isna(record['odds']) or settled_odds != record['odds']:
                header += f" · Settled multiplier {fmt(settled_odds)}"
        if pd.notna(record['probability']):
            header += f" · P(all win, independent) {record['probability']:.1%}"
        if pd.notna(record.get('expected_profit', float('nan'))):
            header += f" · Expected profit per unit stake {fmt(record['expected_profit'])}"
        if pd.notna(record['profit']):
            header += f" · Net profit {fmt(record['profit'])}"
        rows = []
        for row in legs.to_dict('records'):
            rows.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in (
                team(row, 'home'), team(row, 'away'), escape(str(row.get('description', row['bet']))),
                fmt(row.get('p_win', float('nan'))), fmt(row.get('result', float('nan'))),
                escape(row['settlement']), fmt(row['odds']))) + '</tr>')
        panels.append(f'<details class="bet-ticket" style="border-left:3px solid {color};padding:12px;margin:8px 0">'
                      f'<summary>{header}</summary><div class="table-scroll"><table><thead><tr>'
                      + ''.join(f'<th>{label}</th>' for label in ('Home','Away','Selected bet','Probability','Actual result','Settlement','Odds'))
                      + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div></details>')
    return warning + ''.join(panels)
