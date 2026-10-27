"""Render completed diagnostic evidence; no fitting or model selection."""
import argparse
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS={'snapshot':'#65758b','time_features':'#1776c7','historical':'#008575',
        'rolling_logistic':'#af6b00','training_frequency':'#9565b3','uniform':'#a5a5a5'}


def render(run):
    protocol=json.loads((run/'protocol.json').read_text())
    scores=json.loads((run/'scores.json').read_text())
    probes={m:json.loads((run/'probe'/m/'result.json').read_text()) for m in ('snapshot','time_features','historical')}
    matched={a:json.loads((run/'matched'/a/'result.json').read_text()) for a in ('zero_time','learn_time')}
    summary=pd.DataFrame(scores['mean_scores']).T
    summary.index.name='model'
    summary.to_csv(run/'draw_winner_scores.csv')
    features=json.loads((run/'feature_distributions.json').read_text())
    feature_rows=[]
    for split,values in features.items():
        for name,value in values.items():
            feature_rows.append(dict(split=split,channel=name,count=value['count'],mean=value['mean'],std=value['std'],
                **{f'q_{q}':v for q,v in value['quantiles'].items()}))
    pd.DataFrame(feature_rows).to_csv(run/'time_feature_ranges.csv',index=False)
    projection_rows=[]
    gradient_rows=[]
    probability_rows=[]
    for model,rows in scores['per_seed'].items():
        for r in rows:
            for outcome,value in r['probabilities'].items():
                probability_rows.append(dict(model=model,seed=r.get('seed'),outcome=outcome,mean=value['mean'],std=value['std'],
                    **{f'q_{q}':v for q,v in value['quantiles'].items()}))
    pd.DataFrame(probability_rows).to_csv(run/'probability_distributions.csv',index=False)
    for model,probe in probes.items():
        for stage,state in probe['states'].items():
            for name,value in state['activations']['projections'].items():
                projection_rows.append(dict(model=model,stage=stage,projection=name,**{k:v for k,v in value.items() if k!='sample_quantiles'},
                    **{f'q_{q}':v for q,v in value['sample_quantiles'].items()}))
            for name,value in state['gradient']['projections'].items():
                gradient_rows.append(dict(model=model,stage=stage,projection=name,
                    weight_rms=value['weight']['rms'],gradient_rms=value['gradient']['rms'],
                    gradient_max_abs=max(abs(value['gradient']['quantiles']['0']),abs(value['gradient']['quantiles']['1']))))
    pd.DataFrame(projection_rows).to_csv(run/'projection_measurements.csv',index=False)
    pd.DataFrame(gradient_rows).to_csv(run/'gradient_measurements.csv',index=False)
    dependencies=[]
    for name,value in probes['historical']['dependency_feature_distributions'].items():
        dependencies.append(dict(channel=name,**{k:v for k,v in value.items() if k!='sample_quantiles'},
            **{f'q_{q}':v for q,v in value['sample_quantiles'].items()}))
    pd.DataFrame(dependencies).to_csv(run/'historical_dependency_feature_ranges.csv',index=False)
    paired_rows=[]
    for comparison,metrics in scores['paired'].items():
        for metric,value in metrics.items():
            paired_rows.append(dict(comparison=comparison,metric=metric,**value))
    pd.DataFrame(paired_rows).to_csv(run/'paired_diagnostics.csv',index=False)
    matched_rows=[]
    for arm,r in matched.items():
        matched_rows.append(dict(arm=arm,epochs=r['training_summary']['attempts'][0]['steps'],selected_epoch=r['training_summary']['retained_step'],
            training_log_loss=r['selected']['train']['log_loss'],validation_log_loss=r['selected']['validation']['log_loss'],
            validation_draw_log_loss=r['selected']['validation']['draw_log_loss'],
            validation_winner_log_loss=r['selected']['validation']['conditional_winner_log_loss']))
    pd.DataFrame(matched_rows).to_csv(run/'matched_summary.csv',index=False)

    decomposition=go.Figure()
    for metric,label,color in [('draw_log_loss','Draw / not draw','#4c91c5'),('winner_contribution','Non-draw-weighted winner component','#bd9260')]:
        decomposition.add_bar(x=list(scores['mean_scores']),y=[r[metric] for r in scores['mean_scores'].values()],name=label,marker_color=color)
    decomposition.update_layout(template='plotly_white',barmode='stack',height=410,title='Exact decomposition of test log loss',yaxis_title='Mean loss; lower is better')

    uncertainty=make_subplots(rows=1,cols=2,subplot_titles=['Draw log loss: candidate minus snapshot','Three-way log loss: candidate minus logistic'])
    for column,reference,metric,candidates in [(1,'snapshot','draw_log_loss',['time_features','historical']),
                                              (2,'rolling_logistic','log_loss',['snapshot','time_features','historical'])]:
        for candidate in candidates:
            v=scores['paired'][candidate+'_minus_'+reference][metric]
            uncertainty.add_trace(go.Scatter(x=[v['lower_95'],v['upper_95']],y=[candidate,candidate],mode='lines',showlegend=False,line_color=COLORS[candidate]),row=1,col=column)
            uncertainty.add_trace(go.Scatter(x=[v['mean']],y=[candidate],mode='markers',marker=dict(size=10,color=COLORS[candidate]),showlegend=False),row=1,col=column)
        uncertainty.add_vline(x=0,line_dash='dash',line_color='#999',row=1,col=column)
    uncertainty.update_layout(template='plotly_white',height=320,title='Exploratory paired four-round bootstrap intervals; conditional on this season and fitted models')

    distribution=go.Figure()
    base=Path(protocol['base'])
    for model in probes:
        values=np.concatenate([pd.read_parquet(base/'fit'/f'{model}-{seed}'/'predictions.parquet').p_draw for seed in (17,42,73)])
        distribution.add_trace(go.Box(y=values,name=model,boxpoints=False,marker_color=COLORS[model]))
    distribution.add_hline(y=scores['training_class_frequencies']['draw'],line_dash='dash',annotation_text='Training draw frequency')
    distribution.add_hline(y=1/3,line_dash='dot',annotation_text='Uniform three-class forecast')
    distribution.update_layout(template='plotly_white',height=380,title='Test draw-probability distributions across three separately fitted seeds',yaxis_title='P(draw)')

    curves=make_subplots(rows=1,cols=2,subplot_titles=['Training loss before each update','Validation loss after each update'])
    for arm,color in [('zero_time','#65758b'),('learn_time','#1776c7')]:
        progress=json.loads((run/'matched'/arm/'progress.json').read_text())
        for column,metric in [(1,'train_loss'),(2,'validation_loss')]:
            x=[p['step']-(column==1) for p in progress]
            y=[p[metric] for p in progress]
            if column==2:
                x=[0]+x; y=[matched[arm]['initial']['validation']['log_loss']]+y
            curves.add_trace(go.Scatter(x=x,y=y,mode='lines',name=arm,legendgroup=arm,showlegend=column==1,line_color=color),row=1,col=column)
    curves.update_layout(template='plotly_white',height=410,title='Matched initial function, seed 17: no new test evaluation')
    curves.update_xaxes(title_text='Completed updates at measurement')
    curves.update_yaxes(title_text='Cross-entropy')

    projection=make_subplots(rows=1,cols=2,subplot_titles=['Message-value component RMS','Mean local tanh derivative'])
    keys=[(m,layer) for m in probes for layer in [1,2]]
    labels=[f'{m} / layer {layer}' for m,layer in keys]
    for metric,label,color in [('value.edge_base','Other edge inputs','#65758b'),('value.edge_time','Time inputs','#1776c7'),('value.sender','Sender state','#008575')]:
        projection.add_trace(go.Bar(x=labels,y=[probes[m]['states']['selected']['activations']['projections'][f'layer_{layer}.{metric}']['rms'] for m,layer in keys],name=label,marker_color=color),row=1,col=1)
    projection.add_trace(go.Bar(x=labels,y=[probes[m]['states']['selected']['activations']['projections'][f'layer_{layer}.value.pre_tanh']['mean_tanh_derivative'] for m,layer in keys],name='Derivative',marker_color='#bd9260',showlegend=False),row=1,col=2)
    projection.update_layout(template='plotly_white',height=490,barmode='group',title='Retained seed-17 models: root-message measurements at six declared issuances')

    html='<!doctype html><html lang="en"><meta charset="utf-8"><title>Temporal GAT diagnostics</title>'
    html+='<style>body{font:16px system-ui;color:#243746;max-width:1400px;margin:36px auto;padding:0 24px}table{border-collapse:collapse;font-size:13px}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:right}.scroll{overflow:auto}p{max-width:1050px;line-height:1.5}</style>'
    html+='<h1>Temporal GAT diagnostics after patience 15</h1><p>Saved test predictions: all three seeds. Mechanism probes and two matched-function training/validation arms: seed 17. Original model and experiment artifacts remain unchanged.</p>'
    html+='<p>The temporal models show a small draw-score gain over snapshot, offset by worse conditional winner scores. Draw AUC is near 0.5 and no GAT improves on the training-frequency draw baseline by point estimate. The paired intervals do not establish superiority over logistic.</p>'
    for i,figure in enumerate([decomposition,uncertainty,distribution,curves,projection]):
        html+=figure.to_html(full_html=False,include_plotlyjs=(i==0))
    html+='<p>Projection RMS is an unweighted description of root-edge coordinates, including old edges with small attention. It is not a feature-importance measure. Native attention and projection outputs were observed without changing model calculations. Moments are exact over the declared root probes; large-array quantiles use bounded deterministic samples.</p>'
    html+='<h2>Scores</h2><div class="scroll">'+summary.round(6).to_html()+'</div>'
    html+='<h2>Matched-function diagnostic</h2><div class="scroll">'+pd.DataFrame(matched_rows).round(6).to_html(index=False)+'</div>'
    html+='<p>Matching initialization did not remove the validation deficit in this seed: learning time connections reduced selected training loss slightly but increased selected validation loss. This is a mechanism diagnostic using one seed and the already-used validation season, with no claim of confirmatory generalization.</p>'
    html+='<h2>Full temporal feature ranges</h2><div class="scroll">'+pd.DataFrame(feature_rows).round(6).to_html(index=False)+'</div>'
    html+='<p>All comparisons remain exploratory on previously inspected seasons. No no-opponent encoder, new architecture, or binary-target refit was introduced.</p></html>'
    (run/'report.html').write_text(html,encoding='utf-8')
    print(json.dumps(dict(html=str(run/'report.html'),tables=8,matched=matched_rows)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    render(parser.parse_args().run.resolve())
