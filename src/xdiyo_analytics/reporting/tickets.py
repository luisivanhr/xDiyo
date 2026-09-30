"""Ticket presentation layered on the retained selected-leg ledger."""

from html import escape
import pandas as pd

from ..evaluation.tickets import compose_bets
from .contracts import Artifact
from .teams import team_key


def add_tickets(result, composition, context, *, fixture_table=None, teams=None):
    legs = result.tables['ledger']
    alternatives = result.tables.get('alternatives')
    if alternatives is not None:
        fields = ['fold_id', 'row_position', 'bet', 'p_win', 'p_push', 'p_loss', 'expected_profit', 'description']
        legs = legs.merge(alternatives[fields], on=['fold_id', 'row_position', 'bet'], how='left', validate='one_to_one')
    if fixture_table is not None and len(fixture_table):
        fields = ['fold_id', 'row_position', 'home_id', 'away_id', 'home', 'away', 'result']
        display = fixture_table[fields]
        legs = legs.drop(columns=[f for f in fields[2:] if f in legs]).merge(
            display, on=fields[:2], how='left', validate='many_to_one')
    tickets, membership, metrics = compose_bets(legs, composition, match_columns=context.match_columns)
    result.tables.update(leg_ledger=legs, ledger=tickets, tickets=tickets, ticket_legs=membership, bet_metrics=metrics)
    result.notes.extend([
        'Ticket stakes replace leg stakes. Each BetSlip template places separate bets; incomplete batches are omitted.',
        'Probabilities stay blank unless independence is explicitly selected. Quoted odds multiply; removed push/void legs contribute odds 1 at settlement.',
        'All leg predictions must be available before the first kickoff for prospective use. Groups stay within fold occurrences; calendar days use UTC.',
    ])
    result.artifacts.insert(0, Artifact('html', ticket_html(tickets, membership, teams or {}), 'Bet tickets'))
    return result


def ticket_html(tickets, membership, teams):
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
        return '<p>No complete tickets in the selected groups. Reduce the leg count or select more fixtures.</p>'
    panels = []
    for record in tickets.to_dict('records'):
        legs = membership.loc[membership.ticket_id.eq(record['ticket_id'])]
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
    return ''.join(panels)
