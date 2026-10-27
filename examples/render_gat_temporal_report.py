"""Render the frozen temporal comparison receipts without fitting or selection."""
import argparse
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {'snapshot':'#65758b','time_features':'#1776c7','historical':'#008575'}


def render(run):
    result = json.loads((run/'results.json').read_text(encoding='utf-8'))
    seeds = result['protocol']['seeds']
    modes = list(result['models'])
    curves = make_subplots(rows=1,cols=2,subplot_titles=['Training cross-entropy','Validation cross-entropy'])
    for mode in modes:
        trajectories = [pd.read_csv(run/'fit'/f'{mode}-{s}'/'training.csv') for s in seeds]
        for column,metric in enumerate(('train_loss','validation_loss'),1):
            values = [f.loc[f.metric.eq(metric)].set_index('step').value.sort_index() for f in trajectories]
            # A stopped seed has no later observations. Restrict the mean to the
            # common prefix instead of averaging only surviving runs or padding.
            for seed,ys in zip(seeds,values):
                curves.add_trace(go.Scatter(x=ys.index,y=ys,mode='lines',line=dict(color=COLORS[mode],width=1),
                    opacity=.35,name=f'{mode}, seed {seed}',legendgroup=mode,showlegend=False),row=1,col=column)
            mean = pd.concat(values,axis=1,join='inner').mean(axis=1)
            curves.add_trace(go.Scatter(x=mean.index,y=mean,mode='lines',line=dict(color=COLORS[mode],width=3),
                name=mode,legendgroup=mode,showlegend=column==1),row=1,col=column)
    curves.update_xaxes(title_text='Full-training-set updates')
    curves.update_yaxes(title_text='Cross-entropy')
    curves.update_layout(template='plotly_white',height=420,title='Learning curves: bold = all-seed mean where all are active; thin = individual seeds')

    heldout = make_subplots(rows=1,cols=2,subplot_titles=['Test log loss by seed','Paired test log-loss differences'])
    for mode in modes:
        values = [r['log_loss'] for r in result['models'][mode]['per_seed_test']]
        heldout.add_trace(go.Scatter(x=[mode]*len(values),y=values,mode='markers',
            marker=dict(color=COLORS[mode],size=10),text=[f'seed {s}' for s in seeds],name=mode,showlegend=False),row=1,col=1)
    heldout.add_hline(y=result['rolling_logistic']['log_loss'],line_dash='dash',line_color='#a65c00',
                     annotation_text='Rolling logistic',row=1,col=1)
    for name,contrast in result['paired_log_loss'].items():
        label=name.replace('_minus_',' − ')
        heldout.add_trace(go.Scatter(x=[contrast['lower_95'],contrast['upper_95']],y=[label,label],
            mode='lines',line=dict(color='#294052',width=3),showlegend=False),row=1,col=2)
        heldout.add_trace(go.Scatter(x=[contrast['mean']],y=[label],mode='markers',
            marker=dict(color='#294052',size=9),showlegend=False),row=1,col=2)
    heldout.add_vline(x=0,line_dash='dash',line_color='#aaa',row=1,col=2)
    heldout.update_yaxes(title_text='Log loss (lower is better)',row=1,col=1)
    heldout.update_xaxes(title_text='Candidate − reference (negative favors candidate)',row=1,col=2)
    heldout.update_layout(template='plotly_white',height=430,title='Retrospective test; intervals use paired four-round blocks')

    rows = []
    for mode in modes:
        m,p = result['models'][mode],result['profiles'][mode]
        steps = [x['attempts'][0]['steps'] for x in m['training_summaries']]
        rows.append(dict(mode=mode,parameters=p['parameters'],
            epochs_min=min(steps),epochs_max=max(steps),
            validation_log_loss_mean=np.mean(m['validation_losses']),test_log_loss_mean=m['mean_test_log_loss'],
            test_log_loss_seed_sd=m['sd_test_log_loss'],test_brier_mean=m['mean_test_brier'],test_ece_mean=m['mean_test_ece'],
            profile_first_update_seconds=p['update_with_validation_seconds'][0],
            profile_second_update_seconds=p['update_with_validation_seconds'][1],
            profile_peak_rss_gib=p['memory']['lifetime_peak_rss_bytes']/2**30,
            profile_peak_cuda_allocated_gib=p['memory']['peak_cuda_allocated_bytes']/2**30,
            profile_peak_cuda_reserved_gib=p['memory']['peak_cuda_reserved_bytes']/2**30,
            fit_seconds_mean=np.mean(m['fit_seconds']),predict_seconds_mean=np.mean(m['predict_seconds'])))
    table = pd.DataFrame(rows)
    table.to_csv(run/'comparison.csv',index=False)
    pd.DataFrame(result['paired_log_loss']).T.to_csv(run/'paired_log_loss.csv')
    tables = table.round(6).to_html(index=False,border=0)
    class_rows = []
    for mode in modes:
        for seed,metrics in zip(seeds,result['models'][mode]['per_seed_test']):
            for name,calibration in metrics['class_calibration'].items():
                class_rows.append(dict(mode=mode,seed=seed,outcome=name,ece_10=calibration['ece_10'],
                    mean_probability=calibration['mean_probability'],frequency=calibration['frequency']))
    pd.DataFrame(class_rows).to_csv(run/'class_calibration.csv',index=False)
    limits = ''.join(f'<li>{escape(x)}</li>' for x in result['protocol']['limitations'])
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Temporal GAT comparison</title>'
    html += '<style>body{font:16px system-ui;max-width:1400px;margin:36px auto;padding:0 24px;color:#243746}h1{font-size:32px}table{border-collapse:collapse;font-size:12px}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:right}.scroll{overflow:auto}li{margin:8px 0}</style>'
    policy = result['protocol'].get('early_stopping')
    budget = f"maximum {result['protocol']['max_steps']} updates"
    budget += f"; validation patience {policy['patience']}, min_delta {policy['min_delta']}" if policy else '; no early stopping'
    html += '<h1>Temporal GAT: controlled CUDA comparison</h1><p>Training 2022/23 · validation 2023/24 · test 2024/25 · Bundesliga targets · seeds 17 / 42 / 73 · '+escape(budget)+'.</p>'
    html += '<p>All three modes share the same archive, cutoffs, splits and execution settings. The paired intervals are exploratory and conditional on the three fitted seeds and this season.</p>'
    html += curves.to_html(full_html=False,include_plotlyjs=True)
    html += heldout.to_html(full_html=False,include_plotlyjs=False)
    if result['protocol'].get('profile_reference'):
        html += '<p>The two-update resource profile is reused from the original experiment, with unchanged model and execution settings. Full-fit and test timings below come from this follow-up.</p>'
    html += '<h2>Measurements</h2><p>First and second update times include validation; the second historical update was only partially cached. Peak allocated and reserved VRAM are different quantities. Timing samples are descriptive; the profile contains two updates.</p><div class="scroll">'+tables+'</div>'
    html += '<h2>Interpretation limits</h2><ul>'+limits+'</ul></html>'
    (run/'report.html').write_text(html,encoding='utf-8')
    print(table.round(6).to_string(index=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    render(parser.parse_args().run)
