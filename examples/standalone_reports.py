"""Create small synthetic sharing examples. No fitting or betting calculations.

python examples/standalone_reports.py --output PATH
Requires the reporting extra for the synthetic Plotly runtime/figure.
"""
import argparse
import csv
import json
from pathlib import Path

from xdiyo_analytics.reporting.templates import build_tickets, build_curves, build_portfolio


def create_inputs(root, count=61):
    """Deterministic audited-shaped fixtures, including long literal decimals."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    headers = ['ticket_id', 'league', 'result', 'first_kickoff_utc', 'legs_json',
               'stake_units', 'payout_units', 'profit_units', 'details', 'extra_literal']
    with (root / 'tickets.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        for i in range(count):
            legs = []
            for j in range(i % 3 + 1):
                legs.append(dict(event_id=str(2**63+i*3+j), kickoff_utc='2025-01-01T12:00:00Z',
                                 market='Total corners' if j == 0 else 'Match outcome',
                                 selection='Under 7.5' if j == 0 else 'Home',
                                 home_id='0001', home_name='North "City", Atlético',
                                 away_id='0002', away_name='South United',
                                 odds='2.25000000000000000001', probability='0.80000000000000000001'))
            if i == 2:
                legs[-1] = dict(event_id='non-team', kickoff_utc='2025-01-01T12:00:00Z',
                                market='Custom market', selection='Outcome A', label='Non-team example')
            writer.writerow([f'{i:04d}', 'Example league' if i%2 else 'Second league', 'win',
                             '2025-01-01T12:00:00Z', json.dumps(legs), '1.00', '2.25', '1.25',
                             'Synthetic evidence only\nNo selection or settlement performed.', '00001'])
    # Exact strings in supplied summaries, not computed by the rendering layer.
    from decimal import Decimal
    summary = dict(rows=count, stake_units=str(count), payout_units=str(Decimal('2.25')*count),
                   profit_units=str(Decimal('1.25')*count), roi_percent='125.00')
    fields = {key: key for key in headers if key not in ('legs_json', 'extra_literal')}
    fields['legs'] = 'legs_json'
    ticket_manifest = dict(schema_version='1.0', title='Synthetic ticket explorer',
                           subtitle='Singles, doubles and three-leg examples', notice='Synthetic fixture. No real bets.',
                           footer='Independent example policies; do not sum them.',
                           policies=[dict(id='example/'+str(i), label='Policy '+str(i), group='Example study',
                                          csv='tickets.csv', fields=fields, summary=summary) for i in range(2)])
    (root / 'tickets.json').write_text(json.dumps(ticket_manifest, indent=2), encoding='utf-8')
    return ticket_manifest


def create_curves(root):
    """Saved line specifications; cumulative values and endpoints are supplied."""
    from plotly.offline import get_plotlyjs
    from plotly.io import templates
    import re
    root = Path(root)
    runtime = get_plotlyjs()
    version = re.search(r'plotly\.js v([0-9.]+)', runtime[:600])[1]
    (root / 'plotly.min.js').write_text(runtime, encoding='utf-8')
    data = [dict(type='scatter', mode='lines', name=f'Policy {i+1}',
                 x=['2025-01-01T00:00:00Z','2025-01-02T00:00:00Z','2025-01-03T00:00:00Z'],
                 y=['0','0','1.25'], line={'color':'#58c7b2', 'dash':'solid' if i%2 else 'dash'},
                 hovertemplate='%{x}<br>%{y} units<extra>%{fullData.name}</extra>', visible=i==0) for i in range(25)]
    figure = dict(data=data, layout=dict(template=templates['plotly_dark'].to_plotly_json(), paper_bgcolor='#17202b',
        plot_bgcolor='#17202b', font=dict(family='system-ui, sans-serif', color='#e8eef7'),
        height=400, margin=dict(l=65,r=25,t=75,b=65), legend=dict(orientation='h',y=-.25),
        xaxis={'title':{'text':'UTC day'}}, yaxis={'title':{'text':'Supplied cumulative P&L · units'}},
        updatemenus=[dict(buttons=[dict(label=f'Policy {i+1}', method='update',
                                     args=[dict(visible=[j==i for j in range(25)])]) for i in range(25)])]))
    (root / 'curve.plotly.json').write_text(json.dumps(figure), encoding='utf-8')
    (root / 'curve.csv').write_text('date_utc,cumulative\n2025-01-01T00:00:00Z,0\n2025-01-02T00:00:00Z,0\n2025-01-03T00:00:00Z,1.25\n', encoding='utf-8')
    m = dict(schema_version='1.0', kind='curves', title='Synthetic supplied curves',
        scope_label='Synthetic exported evidence; no match population', timing_note='Supplied UTC calendar; no settlement proxy inferred.',
        units='Stake and profit in units.', aggregation_note='Alternative policies are displayed separately.',
        plotly_js='plotly.min.js', plotly_js_version=version,
        sections=[dict(title='Policy comparison', figures=[dict(title='25 supplied policies', json='curve.plotly.json',
            producer_revision='synthetic-example-v1', plotly_js_version=version, endpoints=['1.25']*25,
            curve_tables=[dict(trace=0,csv='curve.csv',x='date_utc',y='cumulative',
                calendar=dict(start='2025-01-01T00:00:00Z',end='2025-01-03T00:00:00Z',initial_zero=True))])],
            tables=[dict(title='Supplied calendar values',csv='curve.csv')])])
    (root / 'curves.json').write_text(json.dumps(m, indent=2), encoding='utf-8')
    p = dict(m, kind='portfolio', title='Synthetic pre-aggregated portfolio',
             aggregation_note='This example treats each supplied path as a separate portfolio scenario; nothing is summed by the renderer.')
    (root / 'portfolio.json').write_text(json.dumps(p, indent=2), encoding='utf-8')
    return m


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    root = parser.parse_args().output
    inputs = root / 'inputs'
    create_inputs(inputs)
    create_curves(inputs)
    for kind, build in [('tickets',build_tickets),('curves',build_curves),('portfolio',build_portfolio)]:
        print(build(inputs / f'{kind}.json', root / kind))
