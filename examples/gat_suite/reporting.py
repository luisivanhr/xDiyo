"""Saved-prediction audits, prespecified paired contrasts, and standalone reports."""
from pathlib import Path
import json
import re
import time
import numpy as np
import pandas as pd
from scipy.special import gammaln
from . import core,metrics
from compare_gat_temporal import write_json,digest,frame_hash,validate_training_progress


def html(markdown,path):
    from markdown_it import MarkdownIt
    body=MarkdownIt('commonmark',{'html':True}).enable('table').render(markdown)
    path.write_text('<!doctype html><html><head><meta charset="utf-8"><title>xDiyo four-study suite</title>'
        '<style>body{font:16px/1.55 system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#243545}'
        'table{border-collapse:collapse;display:block;overflow:auto;font-size:12px}td,th{padding:6px 9px;border:1px solid #cdd8df;text-align:right}'
        'th{background:#e8f0f5}img{max-width:100%}h1,h2{color:#15485e}code{background:#eef3f5}details{margin:16px 0}</style>'
        '</head><body>'+body+'</body></html>',encoding='utf-8')


def table(frame,digits=6):
    if not len(frame):return 'No applicable observations.'
    def value(x):
        if isinstance(x,(float,np.floating)):return '' if not np.isfinite(x) else f'{x:.{digits}f}'
        return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(value(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)])


def audited_model(run,job):
    folder=run/job['path'];result=json.loads((folder/'result.json').read_text());curve=pd.read_csv(folder/'training.csv',float_precision='round_trip')
    validate_training_progress(curve.to_dict('records'),result['summary'],dict(max_steps=200,early_stopping=dict(patience=15,min_delta=0.)))
    calibration=json.loads((folder/'calibration.json').read_text())
    frames=core.frames(run,job['fold']);selected=result['summary']['best_step'];predictions={}
    for split in ('train','inner','outer'):
        expected=frames['recent' if job['arm']=='A_recent' else 'expanded'] if split=='train' else frames[split]
        pred=pd.read_parquet(folder/f'{split}_predictions.parquet')
        assert frame_hash(pred)==frame_hash(expected) and len(pred)==len(expected)
        np.testing.assert_array_equal(pred[['goals_home','goals_away']],expected[['goals_for','goals_against']])
        p=pred[metrics.P_COLUMNS].to_numpy();q=pred[metrics.C_COLUMNS].to_numpy()
        np.testing.assert_allclose(q,metrics.calibrate(p,calibration['intercept']),rtol=1e-13,atol=1e-13)
        for view,cols in [('raw',metrics.P_COLUMNS),('calibrated',metrics.C_COLUMNS)]:
            scored=metrics.outcome_rows(pred[cols],pred.y)
            for name,value in scored.items():np.testing.assert_allclose(pred[f'{view}_{name}'],value,rtol=1e-12,atol=1e-12)
        score=metrics.table_metrics(pred)
        for view in ('raw','calibrated'):
            for name in metrics.ADDITIVE:
                np.testing.assert_allclose(score[view][name],result['metrics'][split][view][name],rtol=1e-12,atol=1e-12)
        family=core.config(job['mode'],job['arm'],tuple(json.loads((run/'MASTER_PROTOCOL.json').read_text())['competitions'])).goal_auxiliary.likelihood
        independent=metrics.count_rows(pred[['lambda_home','lambda_away']],pred[['goals_home','goals_away']],family,result['alpha'])
        for key,value in independent.items():np.testing.assert_allclose(pred[key],value,rtol=1e-12,atol=1e-12)
        if split!='outer':
            prefix='train' if split=='train' else 'validation'
            row=curve.set_index('step').loc[selected]
            np.testing.assert_allclose(score['raw']['log_loss'],row[f'{prefix}_loss'],rtol=1e-5,atol=1e-6)
            constant=gammaln(pred[['goals_home','goals_away']].to_numpy()+1).sum(1).mean() if job['arm'] in core.ARMS[:3] else 0.
            np.testing.assert_allclose(score['goals']['goal_joint_nll'],row[f'{prefix}_goal_loss']+constant,rtol=1e-5,atol=1e-6)
            beta=0. if job['arm'] in core.ARMS[:2] else .1
            np.testing.assert_allclose(curve[f'{prefix}_objective'],curve[f'{prefix}_loss']+beta*curve[f'{prefix}_goal_loss']+curve[f'{prefix}_regularization'],rtol=1e-12,atol=1e-12)
        predictions[split]=pred
    refit=metrics.draw_calibration(predictions['inner'][metrics.P_COLUMNS],predictions['inner'].y)
    np.testing.assert_allclose(calibration['intercept'],refit['intercept'],atol=1e-12,rtol=1e-12)
    assert calibration['population_sha256']==frame_hash(predictions['inner'])
    # Verify exact-continuation checkpoint refers to the last update, before restoration.
    import torch
    attempt=folder/f"attempt_{result['attempts']:02d}"
    receipt=json.loads((attempt/'last_training.json').read_text())
    assert receipt['identity']==result['identity'] and receipt['sha256']==digest(attempt/'last_training.pt')
    last=torch.load(attempt/'last_training.pt',map_location='cpu',weights_only=True)
    assert last['active']['step']==len(curve) and last['active']['best_step']==selected
    assert len(last['active']['backend']['optimizer']['state'])>0
    assert result['reload_exact'] and result['public_boundaries_verified']
    return result,predictions,curve


def contrasts(study):
    pairs=[]
    def add(label,terms,category='primary'):pairs.append(dict(label=label,terms=terms,category=category))
    modes=('snapshot',) if study==1 else core.MODES
    if study<=2:
        for m in modes:
            add(f'{m}: B-A',[(f'{m}/B_expanded',1),(f'{m}/A_recent',-1)])
            add(f'{m}: C-B',[(f'{m}/C_flexible',1),(f'{m}/B_expanded',-1)])
    if study==2:
        for arm in core.ARMS[:3]:
            for a,b in [('time_features','snapshot'),('historical','time_features'),('historical','snapshot')]:
                add(f'{arm}: {a}-{b}',[(f'{a}/{arm}',1),(f'{b}/{arm}',-1)],'primary' if b!='snapshot' or a=='time_features' else 'descriptive')
        for a,b in [('time_features','snapshot'),('historical','time_features')]:
            add(f'auxiliary interaction: {a}-{b}',[(f'{a}/C_flexible',1),(f'{a}/B_expanded',-1),(f'{b}/C_flexible',-1),(f'{b}/B_expanded',1)])
    if study==3:
        for m in modes:
            for other in ('C_flexible','B_expanded'):
                add(f'{m}: S-{other[0]}',[(f'{m}/S_structured_poisson',1),(f'{m}/{other}',-1)])
    if study==4:
        for m in modes:
            for a,b in [('N_independent_nb','S_structured_poisson'),('G_shared_gamma','N_independent_nb'),('G_shared_gamma','S_structured_poisson')]:
                add(f'{m}: {a[0]}-{b[0]}',[(f'{m}/{a}',1),(f'{m}/{b}',-1)])
    arms=core.ARMS[:3] if study<=2 else core.ARMS[:4] if study==3 else core.ARMS
    for m in modes:
        for arm in arms:
            window='recent' if arm=='A_recent' else 'expanded'
            for baseline in ('rolling_logistic','frequency'):
                add(f'{m}/{arm} - {window} {baseline}',[(f'{m}/{arm}',1),(f'{window}/{baseline}',-1)],'practical_baseline')
            if arm in core.ARMS[2:]:
                add(f'{m}/{arm} raw - calibrated B',[(f'{m}/{arm}',1),(f'{m}/B_expanded',-1)],'raw_vs_calibrated_control')
    return pairs


def aggregate_sample(run,fold,key,view,entries,length):
    """Average separately scored seed losses before block resampling."""
    folder=run/'shared/bootstrap/results';folder.mkdir(exist_ok=True)
    signature={'fold':fold,'key':key,'view':view,'block':length,'files':[digest(path) for _,path in entries]}
    from compare_gat_goals import digest_state
    cache=folder/(digest_state(signature)+'.npz')
    if cache.exists():
        z=np.load(cache);return z['point'],z['samples'],z['per_seed'],z['fixture']
    predictions=[p for p,_ in entries]
    first=predictions[0];rounds=json.loads((run/'shared/bootstrap'/fold/'rounds.json').read_text())
    count=np.load(run/'shared/bootstrap'/fold/f'blocks_{length}.npy')
    ix=np.array([rounds.index(int(r)) for r in first['round']])
    weights=count[:,ix].astype(float);denominator=weights.sum(1)
    names=metrics.ADDITIVE
    fixture=np.mean([p[[f'{view}_{n}' for n in names]].to_numpy() for p in predictions],axis=0)
    samples=weights@fixture/denominator[:,None]
    cols=metrics.P_COLUMNS if view=='raw' else metrics.C_COLUMNS
    ranks=[metrics.rank_bootstrap(p[cols[1]].to_numpy(),p.y.to_numpy()==1,weights) for p in predictions]
    conditional=samples[:,names.index('winner_contribution')]/((weights@(first.y.to_numpy()!=1))/denominator)
    samples=np.column_stack((samples,conditional,np.mean([r[0] for r in ranks],axis=0),np.mean([r[1] for r in ranks],axis=0)))
    allnames=[*names,'conditional_winner_log_loss','draw_auc','draw_ap']
    per_seed=np.array([[metrics.outcome_metrics(p[cols],p.y)[name] for name in allnames] for p in predictions])
    point=per_seed.mean(0)
    np.savez_compressed(cache,point=point,samples=samples,per_seed=per_seed,fixture=fixture)
    return point,samples,per_seed,fixture


def comparisons(run,study,models):
    rows=[];seed_rows=[];fixtures=[];common_cache={}
    names=[*metrics.ADDITIVE,'conditional_winner_log_loss','draw_auc','draw_ap']
    for contrast in contrasts(study):
        views=('raw',) if contrast['category']=='raw_vs_calibrated_control' else ('raw','calibrated')
        for view in views:
            for length in (4,2,8):
                combined=[]
                for fold in ('fold_1','fold_2'):
                    parts=[]
                    for i,(key,coefficient) in enumerate(contrast['terms']):
                        actual_view='calibrated' if contrast['category']=='raw_vs_calibrated_control' and i else view
                        cachekey=(fold,key,actual_view,length)
                        if cachekey not in common_cache:common_cache[cachekey]=aggregate_sample(run,fold,key,actual_view,models[fold,key],length)
                        parts.append((coefficient,common_cache[cachekey]))
                    estimate=sum(c*p[0] for c,p in parts);samples=sum(c*p[1] for c,p in parts)
                    n=len(models[fold,contrast['terms'][0][0]][0][0]);combined.append((n,estimate,samples))
                    for j,name in enumerate(names):
                        lo,hi=np.quantile(samples[:,j],[.025,.975])
                        rows.append(dict(contrast=contrast['label'],category=contrast['category'],fold=fold,view=view,block_rounds=length,metric=name,
                            difference=estimate[j],lower=lo,upper=hi))
                    if length==4:
                        for seed_index,seed in enumerate(core.SEEDS):
                            difference=sum(c*p[2][seed_index if len(p[2])==3 else 0] for c,p in parts)
                            for j,name in enumerate(names):seed_rows.append(dict(contrast=contrast['label'],fold=fold,view=view,seed=seed,metric=name,difference=difference[j]))
                        frame=models[fold,contrast['terms'][0][0]][0][0]
                        local=frame[[*core.MATCH_KEYS,'round','home_id','away_id']].copy()
                        local['fold']=fold;local['contrast']=contrast['label'];local['view']=view
                        per_fixture=sum(c*p[3] for c,p in parts)
                        for j,name in enumerate(metrics.ADDITIVE):local[name]=per_fixture[:,j]
                        fixtures.append(local)
                total=sum(n for n,_,_ in combined)
                estimate=sum(n*p for n,p,_ in combined)/total
                samples=sum(n*s for n,_,s in combined)/total
                for j,name in enumerate(names):
                    lo,hi=np.quantile(samples[:,j],[.025,.975])
                    rows.append(dict(contrast=contrast['label'],category=contrast['category'],fold='combined_fixture_weighted',view=view,block_rounds=length,
                        metric=name,difference=estimate[j],lower=lo,upper=hi))
        print(json.dumps(dict(report_study=study,contrast=contrast['label'],status='scored')),flush=True)
    return pd.DataFrame(rows),pd.DataFrame(seed_rows),pd.concat(fixtures,ignore_index=True)


def diagnostics(pred,store,cfg,identity):
    rows=[];goals=[];slices=[]
    for view,cols in [('raw',metrics.P_COLUMNS),('calibrated',metrics.C_COLUMNS)]:
        q=pred[cols[1]].to_numpy();bins=np.minimum((q*10).astype(int),9)
        for b in range(10):
            take=bins==b
            rows.append(dict(**identity,view=view,bin=b,count=int(take.sum()),predicted=float(q[take].mean()) if take.any() else None,
                observed=float((pred.y.to_numpy()[take]==1).mean()) if take.any() else None))
    if 'lambda_home' in pred:
        for side in ('home','away'):
            rate=pred[f'lambda_{side}'];truth=pred[f'goals_{side}']
            # Constant forecasts have one occupied diagnostic bin.
            bins=pd.qcut(rate,10,duplicates='drop',labels=False) if rate.nunique()>1 else pd.Series(0,index=rate.index)
            for b in sorted(bins.dropna().unique()):
                take=bins.eq(b)
                goals.append(dict(**identity,side=side,bin=int(b),count=int(take.sum()),predicted=float(rate[take].mean()),observed=float(truth[take].mean())))
    for venue in ('home','away'):
        states=[]
        for _,row in pred.iterrows():
            available=store._memberships[(store._memberships.team_id==row[f'{venue}_id']) & (store._memberships.entry_at<=row.issuance_at)]
            state=available.iloc[-1].movement if len(available) and int(available.iloc[-1].competition_id)==int(row.competition_id) else 'unknown'
            states.append(state if state in ('promoted','retained') else 'other_or_unknown')
        for status in ('promoted','retained','other_or_unknown'):
            take=np.array(states)==status
            for view in ('raw','calibrated'):
                values=pred.loc[take]
                slices.append(dict(**identity,venue=venue,status=status,view=view,count=len(values),
                    observed_home=float(values.y.eq(0).mean()) if len(values) else None,observed_draw=float(values.y.eq(1).mean()) if len(values) else None,
                    observed_away=float(values.y.eq(2).mean()) if len(values) else None,
                    log_loss=float(values[f'{view}_log_loss'].mean()) if len(values) else None,
                    draw_component=float(values[f'{view}_draw_log_loss'].mean()) if len(values) else None,
                    winner_component=float(values[f'{view}_winner_contribution'].mean()) if len(values) else None))
    return rows,goals,slices


def plots(folder,records,curves,outcomes,goals,paired,study):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    modes=sorted({r['job']['mode'] for r in records},key=core.MODES.index)
    colors=dict(zip(core.ARMS,['#68717a','#165b99','#e48b23','#27834a','#8c53aa','#c34855']))
    for which in ('outcome','goal'):
        fig,axes=plt.subplots(2*len(modes),2,figsize=(14,5.2*len(modes)),squeeze=False)
        for r in records:
            job=r['job'];curve=curves[job['job_id']];m=modes.index(job['mode']);f=int(job['fold'][-1])-1
            if which=='goal' and job['arm'] in core.ARMS[:2]:continue
            for col,split in enumerate(('train','validation')):
                name=f'{split}_loss' if which=='outcome' else f'{split}_goal_loss'
                values=curve[name].copy()
                if which=='goal' and job['arm']=='C_flexible':
                    population=core.frames(folder.parent,job['fold'])['expanded' if split=='train' else 'inner']
                    values+=gammaln(population[['goals_for','goals_against']].to_numpy(float)+1).sum(1).mean()
                ax=axes[2*m+f,col]
                ax.plot(curve.step,values,color=colors[job['arm']],alpha=.65,linewidth=1,label=job['arm'] if job['seed']==17 else None)
                ax.set(title=f"{job['mode']} {job['fold']} / {split}",xlabel='Completed full-population updates',ylabel='Outcome log loss' if which=='outcome' else 'Full count score per fixture')
        for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(folder/f'{which}_curves.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    sample=outcomes[(outcomes.seed==17)&outcomes['count'].gt(0)]
    for (mode,arm,fold,view),group in sample.groupby(['mode','arm','fold','view']):
        axes[0].plot(group.predicted,group.observed,'o-',alpha=.35,color=colors.get(arm,'black'))
    axes[0].plot([0,1],[0,1],'k--');axes[0].set(title='Draw calibration, seed 17 (raw and calibrated)',xlabel='Mean predicted draw probability',ylabel='Observed draw fraction')
    if len(goals):
        for (mode,arm,fold,side),group in goals[goals.seed.eq(17)&~goals.arm.isin(core.ARMS[:2])].groupby(['mode','arm','fold','side']):
            axes[1].plot(group.predicted,group.observed,'o-',alpha=.4,color=colors.get(arm,'black'))
    axes[1].plot([0,5],[0,5],'k--');axes[1].set(title='Goal means by rate quantile, seed 17',xlabel='Mean predicted goals',ylabel='Mean observed goals')
    fig.tight_layout();fig.savefig(folder/'calibration.png',dpi=130);plt.close(fig)
    primary=paired[(paired.category=='primary')&(paired.metric=='log_loss')&(paired.block_rounds==4)&(paired.fold=='combined_fixture_weighted')]
    fig,ax=plt.subplots(figsize=(12,max(4,len(primary)*.3)))
    labels=[]
    for i,row in enumerate(primary.itertuples()):
        ax.errorbar(row.difference,i,xerr=[[max(0,row.difference-row.lower)],[max(0,row.upper-row.difference)]],fmt='o',color='#165b99' if row.view=='raw' else '#27834a')
        labels.append(f'{row.contrast} / {row.view}')
    ax.axvline(0,color='gray');ax.set_yticks(range(len(labels)),labels,fontsize=8);ax.set_xlabel('Paired outer log-loss difference (95% descriptive block interval)')
    fig.tight_layout();fig.savefig(folder/'paired_differences.png',dpi=130);plt.close(fig)
    resources=pd.DataFrame([dict(mode=r['job']['mode'],arm=r['job']['arm'],seconds=r['fit_seconds'],vram=r['cuda_allocated_peak']/2**30) for r in records])
    fig,axes=plt.subplots(1,2,figsize=(13,5));mean=resources.groupby(['mode','arm']).mean()
    mean.seconds.plot.bar(ax=axes[0],title='Mean fit seconds');mean.vram.plot.bar(ax=axes[1],title='Mean peak allocated CUDA GiB')
    fig.tight_layout();fig.savefig(folder/'resources.png',dpi=120);plt.close(fig)
    if study>=3:
        points=[]
        for r in records:
            if r['job']['arm'] not in core.ARMS[3:]:continue
            pred=pd.read_parquet(folder.parent/r['job']['path']/'outer_predictions.parquet')
            for side in ('home','away'):
                for team,g in pred.groupby(f'{side}_id'):
                    points.append(dict(**r['job'],team=team,venue=side,attack=g[f'attack_{side}'].mean(),defence=g[f'defence_{side}'].mean()))
        points=pd.DataFrame(points);points.to_csv(folder/'centered_attack_defence.csv',index=False)
        fig,ax=plt.subplots(figsize=(9,6))
        for arm,g in points[points.seed.eq(17)].groupby('arm'):ax.scatter(g.attack,g.defence,alpha=.5,label=arm,s=18)
        ax.set(xlabel='Appearance-averaged centered attack (log-rate units)',ylabel='Centered defence',title='Outer pre-match summaries, seed 17; descriptive')
        ax.legend();fig.tight_layout();fig.savefig(folder/'attack_defence.png',dpi=130);plt.close(fig)
    if study==4:
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for r in records:
            job=r['job'];curve=curves[job['job_id']]
            if 'alpha' not in curve:continue
            for ax,name in zip(axes,('alpha','kappa')):ax.plot(curve.step,curve[name],color=colors[job['arm']],alpha=.5);ax.set(xlabel='Update',ylabel=name)
        fig.tight_layout();fig.savefig(folder/'dispersion.png',dpi=130);plt.close(fig)


def report(run,study):
    run=Path(run)
    if study is None:return summary(run)
    protocol,store=core.load(run,frozen=True);folder=run/core.STUDIES[study-1]
    jobs=[r for r in protocol['matrix'] if r['study']<=study]
    records=[];curves={};models={};outcome=[];goal=[];slices=[];scores=[];rate_ranges=[]
    for job in jobs:
        result,predictions,curve=audited_model(run,job);records.append(result);curves[job['job_id']]=curve
        key=(job['fold'],f"{job['mode']}/{job['arm']}");models.setdefault(key,[]).append((predictions['outer'],run/job['path']/'outer_predictions.parquet'))
        identity=dict(fold=job['fold'],mode=job['mode'],arm=job['arm'],seed=job['seed'])
        for split,pred in predictions.items():
            scored=metrics.table_metrics(pred)
            for view in ('raw','calibrated'):
                item=dict(**identity,split=split,view=view,**scored[view]);item.pop('confusion')
                item.update(scored['goals']);scores.append(item)
            for side in ('home','away'):
                for quantile,value in pred[f'lambda_{side}'].quantile([0,.01,.05,.25,.5,.75,.95,.99,1]).items():
                    rate_ranges.append(dict(**identity,split=split,side=side,quantile=quantile,value=value))
        o,g,s=diagnostics(predictions['outer'],store,core.config(job['mode']),identity);outcome+=o;goal+=g;slices+=s
    baseline_receipt=json.loads((run/'shared/baselines/COMPLETE.json').read_text());baseline_rows=[]
    for entry in baseline_receipt['entries']:
        folder_b=run/entry['path'];result=json.loads((folder_b/'result.json').read_text())
        if entry['kind']=='outcome':
            pred=pd.read_parquet(folder_b/'outer_predictions.parquet')
            models[entry['fold'],f"{entry['window']}/{entry['method']}"]=[(pred,folder_b/'outer_predictions.parquet')]
            for view,value in metrics.table_metrics(pred).items():
                item=dict(fold=entry['fold'],window=entry['window'],method=entry['method'],view=view,**value);item.pop('confusion');baseline_rows.append(item)
        else:
            baseline_rows.append(dict(fold=entry['fold'],window=entry['window'],method=entry['method'],view='goal',**result['metrics']['outer']))
    paired,seeds,fixture=comparisons(run,study,models)
    paired.to_csv(folder/'paired_intervals.csv',index=False);seeds.to_csv(folder/'paired_seed_differences.csv',index=False)
    fixture.to_parquet(folder/'paired_fixture_differences.parquet',index=False)
    grouping=['contrast','view','fold']
    round_table=fixture.groupby([*grouping,'round'])[metrics.ADDITIVE].mean().reset_index();round_table.to_csv(folder/'paired_round_differences.csv',index=False)
    team_tables=[];loro=[];extremes=[]
    for key,g in fixture.groupby(grouping):
        for r in sorted(g['round'].unique()):
            keep=g['round']!=r
            loro.append(dict(zip(grouping,key),omitted_round=int(r),difference=float(g.loc[keep,'log_loss'].mean()),fixtures=int(keep.sum())))
        extremes.append(pd.concat([g.nsmallest(5,'log_loss'),g.nlargest(5,'log_loss')]))
        for venue in ('home','away'):
            per=g.groupby(f'{venue}_id')[metrics.ADDITIVE].mean().reset_index().rename(columns={f'{venue}_id':'team_id'})
            for name,value in zip(grouping,key):per[name]=value
            per['venue']=venue;team_tables.append(per)
    pd.DataFrame(loro).to_csv(folder/'leave_one_round_out.csv',index=False);pd.concat(team_tables).to_csv(folder/'per_team_differences.csv',index=False)
    pd.concat(extremes).to_csv(folder/'largest_fixture_contributions_descriptive.csv',index=False)
    scores=pd.DataFrame(scores);scores.to_csv(folder/'metrics_by_seed.csv',index=False)
    means=scores.groupby(['fold','mode','arm','split','view']).mean(numeric_only=True).drop(columns='seed').reset_index();means.to_csv(folder/'mean_separately_scored_seeds.csv',index=False)
    outcome=pd.DataFrame(outcome);goal=pd.DataFrame(goal)
    outcome.to_csv(folder/'outcome_calibration_bins.csv',index=False);goal.to_csv(folder/'goal_rate_calibration_bins.csv',index=False)
    pd.DataFrame(slices).to_csv(folder/'promoted_retained_slices.csv',index=False);pd.DataFrame(rate_ranges).to_csv(folder/'rate_quantiles.csv',index=False)
    pd.DataFrame(baseline_rows).to_csv(folder/'baseline_metrics.csv',index=False)
    resources=pd.DataFrame([dict(fold=r['job']['fold'],mode=r['job']['mode'],arm=r['job']['arm'],seed=r['job']['seed'],
        selected=r['summary']['best_step'],completed=r['summary']['attempts'][0]['steps'],fit_seconds=r['fit_seconds'],
        preprocessing_seconds=r['preprocessing_seconds'],outer_prediction_seconds=r['outer_prediction_seconds'],
        warm_update_seconds=r['warm_update_seconds'],
        rss_peak_gib=r['memory']['sampled_peak_rss_bytes']/2**30,allocated_cuda_gib=r['cuda_allocated_peak']/2**30,
        reserved_cuda_gib=r['cuda_reserved_peak']/2**30,parameters=r['parameters'],goal_parameters=r['goal_parameters'],
        shared_parameters=r['shared_parameters'],alpha=r['alpha']) for r in records])
    resources.to_csv(folder/'resources.csv',index=False)
    grad=[]
    for r in records:
        for point,value in r['gradients'].items():grad.append(dict(job=r['job']['job_id'],checkpoint=point,**value))
    write_json(folder/'shared_encoder_gradients.json',grad)
    plots(folder,records,curves,outcome,goal,paired,study)
    primary=paired[(paired.category=='primary')&(paired.metric=='log_loss')&(paired.block_rounds==4)]
    primary_names=set(primary.contrast)
    seed_table=seeds[seeds.contrast.isin(primary_names)&seeds.metric.eq('log_loss')].pivot(index=['contrast','fold','view'],columns='seed',values='difference').reset_index()
    components=paired[(paired.category=='primary')&paired.metric.isin(['draw_log_loss','conditional_winner_log_loss'])&paired.block_rounds.eq(4)]
    raw=primary[(primary.fold=='combined_fixture_weighted')&(primary.view=='raw')]
    improved=int((raw.difference<0).sum());excluded=int((raw.upper<0).sum())
    questions={1:'Does expanded fitting supervision help, and does flexible goal supervision help on that population?',
        2:'Do bounded time features help, and do historical sender states add value with identical time encoding?',
        3:'Does structured attack/defence goal supervision improve the outcome task?',
        4:'Does overdispersion help, and does shared match dependence add value beyond independent NB marginals?'}
    md=[f'# Experiment {study}: {questions[study]}',
        f'All {sum(j["study"]==study for j in jobs)} new fits for this study are complete. Earlier controls are linked by their original suite paths; {len(jobs)} total fitted models are represented in this report. No prior pilot fit was claimed as exact reuse.',
        f'Of {len(raw)} prespecified primary combined raw comparisons, {improved} have a negative mean outcome-log-loss difference and {excluded} have a four-round descriptive interval wholly below zero. The complete fold-specific raw/calibrated estimates below determine which interventions those are; this count is not a superiority test or a winner-selection rule.',
        '## Primary outcome comparisons',table(primary[['contrast','fold','view','difference','lower','upper']]),
        'Negative differences favor the first model. Each model uses its own lowest raw-inner-outcome-loss checkpoint. All seed probabilities were scored separately before averaging. No probability ensemble was substituted.',
        '## Matched-seed consistency',table(seed_table),
        '## Draw and conditional-winner differences',table(components[['contrast','fold','view','metric','difference','lower','upper']]),
        '![Paired differences](paired_differences.png)',
        '## Outer outcomes by encoder and arm',table(means[means.split.eq('outer')][['fold','mode','arm','view','log_loss','brier','draw_log_loss','draw_brier','draw_auc','draw_ap','conditional_winner_log_loss','predicted_draw','observed_draw']]),
        'The draw and conditional-winner components are exported per fixture. Their exact decomposition is verified. Any draw gain must also be assessed after the separately fitted inner draw-intercept calibration; calibration preserves conditional winner probabilities and draw ranking.',
        '## Practical baselines',table(pd.DataFrame(baseline_rows)),
        'The frequency and rolling logistic comparators use fitting labels only. Both windows share the recent-window imputation/scaling transformation. Full per-model contrasts against these baselines and against calibrated outcome-only B are in `paired_intervals.csv`.',
        '## Count forecasts',table(means[(means.split=='outer')&(means.view=='raw')&~means.arm.isin(core.ARMS[:2])][['fold','mode','arm','goal_joint_nll','goal_home_nll','goal_away_nll','poisson_mean_nll','lambda_home','goals_home','lambda_away','goals_away','home_absolute_error','away_absolute_error','residual_covariance','predicted_covariance']]),
        'For C, the count score is the sum of marginal Poisson scores and establishes no empirical independence claim. For S/N it is the independent product log score; G uses its integrated joint log score. G marginal NB scores are separate. The common Poisson mean score evaluates means and is not G’s native likelihood. Untrained A/B goal heads are excluded as goal baselines. MAE is supporting evidence, not the sole criterion for conditional means.',
        '![Calibration](calibration.png)',
        '## Learning trajectories',
        'Each thin line is an individual seed on its observed updates only. Train and inner losses are recomputed after the same update. No stopped curve is extrapolated. Count curves include factorial constants; combined objectives at different beta values are not performance metrics.',
        '![Outcome trajectories](outcome_curves.png)','![Goal trajectories](goal_curves.png)',
        '## Cost and optimization',table(resources.groupby(['mode','arm']).mean(numeric_only=True).drop(columns='seed').reset_index()),
        '![Resources](resources.png)',
        'Actual completed and selected updates, fitting/preprocessing/prediction wall times, host RSS, live allocated CUDA memory and allocator-reserved memory are retained for every seed. Disposable warm-update profiles are linked in `../preflight/`. Shared-encoder task gradients, parameter names/counts and zero-norm handling are in `shared_encoder_gradients.json`. Both heads, intercepts, dispersion and regularization are excluded; positive cosine is not proof of generalization and AdamW transforms the gradients.',
        '## Uncertainty and diagnostic boundaries',
        'Primary descriptive intervals use 20,000 paired circular four-round blocks (seed 20261026); two- and eight-round sensitivities are all in the interval export. Per-seed fixture losses are averaged before additive resampling. Rank metrics are recomputed with sampled fixture multiplicities. Identical draws are used across matched models, seeds and interaction terms. Combined estimates weight the two distinct outer populations by fixture count, with resampling confined within each season.',
        'These are retrospective development evaluations of already inspected seasons. Training/inner roles overlap across folds. Intervals condition on selected fitted models and do not correct prior inspection, checkpoint selection, suite-wide multiplicity or future-season uncertainty. Inner data serve both stopping and scalar draw calibration; outer labels do neither. July 1 membership and retrospective schedule assumptions remain. All venue flags are unknown in this archive and use the declared ordinary home effect.',
        'The common preprocessor is fitted from the recent fitting covariates and their permitted history, then reused across all supervision arms before caches are constructed. It is applied to earlier expanded fitting examples as a fixed training transformation; those in-sample predictions are not historical real-time forecasts. Outer model parameters and transformations remain frozen, while earlier outer results may enter later causal histories once available.',
        'The promoted/retained/other-or-unknown slices, per-team/per-round differences, leave-one-round-out results and largest fixture contributions are descriptive. No fixture was removed and no parameter was retuned from these diagnostics.',
        '## Evidence and verification',
        'Every fit has an initial-state reference, separate selected and resumable last-update checkpoints, optimizer/RNG/patience state, model export, calibration receipt, train/inner/outer predictions and scalar curves. The last-update optimizer is paired with last-update weights before best restoration. Saved-prediction metrics, alignment, class order, normalization, decomposition, stopping and reloads are checked independently of the training logger.',
        '[Master protocol](../MASTER_PROTOCOL.json) · [Run manifest](../RUN_MANIFEST.csv) · [Source archive](../frozen_source.zip) · [Baseline archive](../shared/baselines/COMPLETE.json)',
        'Mathematical construction reference: [Fabio, Paula and de Castro (2011)](https://arxiv.org/abs/1105.2072). Selection limitations: [Cawley and Talbot (2010)](https://jmlr.org/papers/v11/cawley10a.html). Protocol choices are prespecified engineering choices, not literature-established optima.']
    if study>=3:md+=['## Centered attack and defence','![Centered effects](attack_defence.png)',
        'Offsets come only from all fitting home/away pre-match appearances, using their own causal cutoffs. Equivalent centered rates are verified; no prediction-batch or outer-team centering occurs. These are supervised summaries, not uniquely identified causal measures of skill. C versus S changes structure, link, inputs and initialization together.']
    if study==4:md+=['## Dispersion and retrospective shocks','![Dispersion trajectories](dispersion.png)',
        'N−S compares marginal overdispersion; G−N compares the additional shared-effect family. Independently fitted means and dispersion can differ. G permits positive within-match dependence tied to marginal overdispersion, not arbitrary low-score dependence. Post-score posterior shock summaries in exports are explicitly retrospective diagnostics, never forecast inputs or state carried to another fixture. Near-initial or boundary dispersion is reported by the trajectories, not treated as exhaustive family optimization.']
    text='\n\n'.join(md)+'\n';(folder/'report.md').write_text(text,encoding='utf-8');html(text,folder/'report.html')
    write_json(folder/'protocol.json',dict(master_sha256=digest(run/'MASTER_PROTOCOL.json'),study=study,new_jobs=[j for j in jobs if j['study']==study],
        control_paths=[j['path'] for j in jobs if j['study']<study]))
    write_json(folder/'VERIFIED.json',dict(status='complete',new_fits=sum(j['study']==study for j in jobs),controls=len(jobs)-sum(j['study']==study for j in jobs),
        report_sha256=digest(folder/'report.md'),prediction_audits=len(jobs),paired_contrasts=len(contrasts(study))))


def summary(run):
    run=Path(run);protocol,_=core.load(run,frozen=True)
    parts=[];consistency=[];component_parts=[];cost_parts=[]
    for i,name in enumerate(core.STUDIES,1):
        assert (run/name/'VERIFIED.json').exists()
        frame=pd.read_csv(run/name/'paired_intervals.csv');frame=frame[(frame.category=='primary')&(frame.metric=='log_loss')&(frame.block_rounds==4)&(frame.fold=='combined_fixture_weighted')]
        frame['study']=i;parts.append(frame)
        seeds=pd.read_csv(run/name/'paired_seed_differences.csv')
        primary_names=set(frame.contrast)
        for key,g in seeds[seeds.contrast.isin(primary_names)&seeds.metric.eq('log_loss')].groupby(['contrast','fold','view']):
            consistency.append(dict(study=i,contrast=key[0],fold=key[1],view=key[2],improving_seeds=int((g.difference<0).sum()),
                mean_difference=float(g.difference.mean()),largest_harm=float(g.difference.max())))
        allmetrics=pd.read_csv(run/name/'paired_intervals.csv')
        take=allmetrics.category.eq('primary')&allmetrics.block_rounds.eq(4)&allmetrics.fold.eq('combined_fixture_weighted')&allmetrics.metric.isin(['draw_log_loss','conditional_winner_log_loss'])
        comp=allmetrics[take].copy();comp['study']=i;component_parts.append(comp)
        if i==4:cost_parts=pd.read_csv(run/name/'resources.csv').groupby(['mode','arm']).mean(numeric_only=True).drop(columns='seed').reset_index()
    all=pd.concat(parts);all.to_csv(run/'summary_primary_comparisons.csv',index=False)
    text='# Four-study GAT suite: completed retrospective development evaluation\n\n'
    text+='All 108 planned GAT fits completed as new fits, with no claimed exact reuse. Both chronological folds and all three seeds are present. Every study retained its prespecified configuration regardless of preceding results.\n\n'
    questions=[('Expanded supervision and encoder differences',1,'B-A'),('Flexible goal supervision beyond calibrated controls',1,'C-B'),
        ('Bounded temporal features and historical sender states',2,None),('Structured attack/defence supervision',3,None),('Overdispersion and shared random effects',4,None)]
    for title,study,match in questions:
        rows=all[all.study.eq(study)]
        if match:rows=rows[rows.contrast.str.contains(match,regex=False)]
        text+=f'## {title}\n\n'+table(rows[['contrast','view','difference','lower','upper']])+'\n\n'
    text+='## Consistency, calibration and cost\n\n'+table(pd.DataFrame(consistency))+'\n\n'
    text+='These matched-seed signs distinguish consistency across initializations and outer periods. Small gains, intervals spanning zero and changes in direction across folds remain preliminary. Raw draw improvements that disappear after calibration do not establish richer draw information.\n\n'
    text+=table(pd.concat(component_parts)[['study','contrast','view','metric','difference','lower','upper']])+'\n\n'
    text+='Goal-likelihood gains alone do not establish improved classification. Historical computation must be judged against the corresponding time-features and calibrated outcome-only controls using these measured costs:\n\n'+table(cost_parts)+'\n\n'
    text+='## Practical baselines\n\n'+table(pd.read_csv(run/core.STUDIES[-1]/'baseline_metrics.csv'))+'\n\n'
    text+='## Most defensible next step\n\nReview the paired fold/seed evidence and computational costs before adopting added model complexity. Any stronger generalization claim requires a separately declared chronological evaluation on genuinely unused data; this suite does not choose a universal winner or launch another search. The optional auxiliary features remain disabled by default.\n\n'
    text+='## Reports and evidence\n\n'+ '\n'.join(f'- [Experiment {i+1}]({name}/report.html)' for i,name in enumerate(core.STUDIES))+'\n\n'
    text+='The full matrix, source/data/initialization freeze, failed preflight attempts, training continuation checkpoints, baseline states, per-fixture predictions, calibration and paired bootstrap definitions are under this root. Sources remain uncommitted in the existing research checkout. Outer periods are already-inspected retrospective development data; the schedule/membership archive is not a historical-publication vintage. No later unused period was evaluated.\n'
    (run/'summary.md').write_text(text,encoding='utf-8');html(text,run/'summary.html')
    (run/'README.md').write_text('# Completed four-study GAT suite\n\n'
        'All 108 main jobs completed as new fits. No old fit was claimed as exact reuse. '
        'The four independently readable reports and the cross-study summary are complete. '
        'See [summary.html](summary.html), [MASTER_PROTOCOL.json](MASTER_PROTOCOL.json), '
        '[RUN_MANIFEST.csv](RUN_MANIFEST.csv), [ENVIRONMENT.json](ENVIRONMENT.json) and [COMPLETE.json](COMPLETE.json).\n\n'
        'Execution used `examples/run_gat_suite.py prepare`, then sequential `profile` and `pairing` checks, '
        '`freeze`, and `run`, each with `--run experiments/gat_four_study_suite_20261026_01` and '
        'the authorized `.misc314/Scripts/python.exe` CUDA environment. `run` resumes only matching identities. '
        'The frozen command source is included in `frozen_source.zip`.\n\n'
        'These are retrospective development evaluations, not untouched confirmatory holdouts. '
        'No main model was selected by outer scores. No new search, publication, merge or deployment was launched. '
        'The source changes remain uncommitted. Preflight engineering events are in `preflight/engineering_events.json`.\n',encoding='utf-8')


def audit(run):
    run=Path(run);p,_=core.load(run,frozen=True)
    for job in p['matrix']:audited_model(run,job)
    for name in core.STUDIES:
        assert (run/name/'VERIFIED.json').exists()
        verified=json.loads((run/name/'VERIFIED.json').read_text())
        assert verified['report_sha256']==digest(run/name/'report.md')
        text=(run/name/'report.md').read_text()
        for reference in re.findall(r'!\[[^]]*\]\(([^)]+)\)',text):assert (run/name/reference).exists()
    protected=json.loads((run/'preflight/preserved_artifacts.json').read_text())
    for name,expected in protected.items():assert digest(core.ROOT/name)==expected,name
    assert (run/'summary.md').exists() and (run/'shared/baselines/COMPLETE.json').exists()
    manifest=pd.read_csv(run/'RUN_MANIFEST.csv');assert len(manifest)==108 and manifest.status.eq('completed').all()
    mutable={'COMPLETE.json','PROGRESS.json','audit.log','orchestrator.log'}
    write_json(run/'COMPLETE.json',dict(status='complete',new_fits=108,reused_fits=0,failed_jobs=0,blocked_jobs=0,
        excluded_mutable_operational_files=sorted(mutable),
        protected_files=len(protected),files={f.relative_to(run).as_posix():digest(f) for f in sorted(run.rglob('*')) if f.is_file() and f.relative_to(run).as_posix() not in mutable and not f.name.endswith('.tmp')},
        completed_at=pd.Timestamp.now(tz='UTC').isoformat()))
