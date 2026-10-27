"""Independent saved-prediction scoring and prespecified transfer comparisons."""
from pathlib import Path
import json
import re
import hashlib
import numpy as np
import pandas as pd
from scipy.special import gammaln
from . import core,bootstrap
from gat_suite import metrics
from gat_suite.reporting import table,diagnostics
from compare_gat_temporal import digest,write_json,frame_hash,validate_training_progress

NAMES=[*metrics.ADDITIVE,'conditional_winner_log_loss','draw_auc','draw_ap']


def same_fixture_keys(left,right):
    """Compare ordered identities across Arrow and NumPy integer storage."""
    return np.array_equal(left[list(core.MATCH_KEYS)].to_numpy(dtype=np.int64),
                          right[list(core.MATCH_KEYS)].to_numpy(dtype=np.int64))


def html(text,path):
    from markdown_it import MarkdownIt
    content=MarkdownIt('commonmark',{'html':True}).enable('table').render(text)
    path.write_text('<!doctype html><html><head><meta charset="utf-8"><title>xDiyo cross-league experiment</title>'
        '<style>body{font:16px/1.55 system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#243545}'
        'table{border-collapse:collapse;display:block;overflow:auto;font-size:12px}td,th{padding:6px 9px;border:1px solid #cdd8df;text-align:right}'
        'th{background:#e8f0f5}img{max-width:100%}h1,h2{color:#15485e}details{margin:16px 0}</style>'
        '</head><body>'+content+'</body></html>',encoding='utf-8')


def publish(folder,name,text):
    (folder/f'{name}.md').write_text(text,encoding='utf-8');html(text,folder/f'{name}.html')


def contrasts():
    result=[('time_C pooled-local',[('pooled/time_C',1),('local/time_C',-1)]),
            ('snapshot_B pooled-local',[('pooled/snapshot_B',1),('local/snapshot_B',-1)]),
            ('time_B pooled-local',[('pooled/time_B',1),('local/time_B',-1)])]
    for arm in core.ARMS:
        for a,b in [('time_C','time_B'),('time_C','snapshot_B'),('time_B','snapshot_B')]:
            result.append((f'{arm}: {a}-{b}',[(f'{arm}/{a}',1),(f'{arm}/{b}',-1)]))
    result.extend([
        ('population-by-auxiliary interaction',[('pooled/time_C',1),('pooled/time_B',-1),('local/time_C',-1),('local/time_B',1)]),
        ('pooled time_C - Bundesliga logistic',[('pooled/time_C',1),('local_logistic',-1)]),
        ('pooled time_C - pooled logistic',[('pooled/time_C',1),('pooled_logistic',-1)]),
        ('pooled logistic - Bundesliga logistic',[('pooled_logistic',1),('local_logistic',-1)])])
    return result


def secondary_contrasts():
    result=[(f'{cfg} - {baseline}',[(cfg,1),(baseline,-1)]) for cfg in core.CONFIGS
            for baseline in ('league_logistic','pooled_logistic','league_frequency')]
    result.extend((f'{a} - {b}',[(a,1),(b,-1)]) for a,b in [('time_C','time_B'),('time_C','snapshot_B'),('time_B','snapshot_B')])
    return result


def audit_predictions(pred,expected,calibrators):
    assert frame_hash(pred)==frame_hash(expected) and len(pred)==len(expected)
    for key in ('home_id','away_id','goals_for','goals_against'):
        np.testing.assert_array_equal(pred[key].to_numpy(),expected[key].to_numpy())
    np.testing.assert_array_equal(pred.y,core.truth(expected))
    for view,cols in [('raw',metrics.P_COLUMNS),('calibrated',metrics.C_COLUMNS)]:
        rows=metrics.outcome_rows(pred[cols],pred.y)
        for key,value in rows.items():np.testing.assert_allclose(value,pred[f'{view}_{key}'],rtol=1e-12,atol=1e-12)
    for league,group in pred.groupby('competition_id'):
        intercept=calibrators.get(str(league),{'intercept':0.})['intercept']
        np.testing.assert_allclose(metrics.calibrate(group[metrics.P_COLUMNS],intercept),group[metrics.C_COLUMNS],rtol=1e-13,atol=1e-13)
    if 'lambda_home' in pred:
        ys=pred[['goals_for','goals_against']].to_numpy(float)
        rates=pred[['lambda_home','lambda_away']].to_numpy(float)
        # Direct factorial-constant calculation, independent of training loss.
        np.testing.assert_allclose((rates-ys*np.log(rates)+gammaln(ys+1)).sum(1),pred.goal_joint_nll,rtol=1e-12,atol=1e-12)


def audited_fit(run,job):
    folder=run/job['path'];result=json.loads((folder/'result.json').read_text())
    assert result['job']==job and result['identity']['protocol']==digest(run/'PROTOCOL.json')
    assert result['identity']['freeze']==digest(run/'FREEZE.json')
    curve=pd.read_csv(folder/'training.csv',float_precision='round_trip')
    validate_training_progress(curve.to_dict('records'),result['summary'],dict(max_steps=200,early_stopping=dict(patience=15,min_delta=0.)))
    pop=core.populations(run,job['fold'],job['arm']);cal=json.loads((folder/'calibration.json').read_text());predictions={}
    for phase,expected in pop.items():
        pred=pd.read_parquet(folder/f'{phase}_predictions.parquet');audit_predictions(pred,expected,cal)
        actual=metrics.table_metrics(pred)
        for view in ('raw','calibrated'):
            for name in metrics.ADDITIVE:
                np.testing.assert_allclose(actual[view][name],result['metrics'][phase][view][name],rtol=1e-12,atol=1e-12)
        predictions[phase]=pred
    best=curve.set_index('step').loc[result['summary']['best_step']]
    inner=predictions['inner'];bl=inner[inner.competition_id.eq(35)]
    np.testing.assert_allclose(bl.raw_log_loss.mean(),best.validation_loss,atol=1e-6,rtol=1e-5)
    np.testing.assert_allclose(predictions['fit'].raw_log_loss.mean(),best.train_loss,atol=1e-6,rtol=1e-5)
    # Refit only the deterministic inner intercept for the audit; outer absent.
    for league,group in inner.groupby('competition_id'):
        check=metrics.draw_calibration(group[metrics.P_COLUMNS],group.y)
        np.testing.assert_allclose(check['intercept'],cal[str(league)]['intercept'],atol=1e-12,rtol=1e-12)
        assert cal[str(league)]['population_sha256']==frame_hash(group)
    assert result['reload_exact'] and result['public_boundaries_verified']
    return result,predictions,curve


def primary_samples(run,fold,key,view,frames,length):
    first=frames[0];rounds=json.loads((run/'shared/bootstrap/bundesliga'/fold/'rounds.json').read_text())
    count=np.load(run/'shared/bootstrap/bundesliga'/fold/f'blocks_{length}.npy')
    ix=np.array([rounds.index(int(r)) for r in first['round']]);weight=count[:,ix].astype(float)
    denominator=weight.sum(1)
    fixture=np.mean([f[[f'{view}_{n}' for n in metrics.ADDITIVE]].to_numpy() for f in frames],axis=0)
    samples=weight@fixture/denominator[:,None]
    non=weight@(first.y.to_numpy()!=1)
    conditional=np.divide(weight@fixture[:,metrics.ADDITIVE.index('winner_contribution')],non,
                          out=np.full(len(weight),np.nan),where=non>0)
    cols=metrics.P_COLUMNS if view=='raw' else metrics.C_COLUMNS
    # Calibration preserves Bundesliga ranking, so rank caches can be shared.
    rank_cache=run/'04_comparisons/rank_cache';rank_cache.mkdir(exist_ok=True)
    content=hashlib.sha256(count.tobytes())
    for frame in frames:
        content.update(frame[['p_draw','y']].to_numpy(dtype=np.float64).tobytes())
    signature=f'{fold}__{key.replace("/","_")}__{length}__{content.hexdigest()[:16]}'
    cached=rank_cache/f'{signature}.npz'
    if cached.exists():
        ranks=np.load(cached);auc,ap=ranks['auc'],ranks['ap']
    else:
        with np.errstate(divide='ignore',invalid='ignore'):
            ranks=[metrics.rank_bootstrap(f.p_draw.to_numpy(),f.y.to_numpy()==1,weight) for f in frames]
        auc=np.mean([r[0] for r in ranks],axis=0);ap=np.mean([r[1] for r in ranks],axis=0)
        np.savez_compressed(cached,auc=auc,ap=ap)
    samples=np.column_stack((samples,conditional,auc,ap))
    per_seed=np.array([[metrics.outcome_metrics(f[cols],f.y)[name] for name in NAMES] for f in frames],dtype=float)
    return per_seed.mean(0),samples,per_seed,fixture


def primary(run,models):
    rows=[];seeds=[];fixtures=[];cache={}
    for label,terms in contrasts():
        for view in ('raw','calibrated'):
            for length in (4,2,8):
                combined=[]
                for fold in core.FOLDS:
                    parts=[]
                    reference=None
                    for key,coefficient in terms:
                        fs=models[fold,key]
                        if reference is None:reference=fs[0]
                        assert same_fixture_keys(reference,fs[0])
                        ck=(fold,key,view,length)
                        if ck not in cache:cache[ck]=primary_samples(run,fold,key,view,fs,length)
                        parts.append((coefficient,cache[ck]))
                    point=sum(c*x[0] for c,x in parts);samples=sum(c*x[1] for c,x in parts)
                    count=len(reference);combined.append((count,point,samples))
                    for name,value,col in zip(NAMES,point,samples.T):
                        valid=np.isfinite(col);lo,hi=np.quantile(col[valid],[.025,.975]) if valid.any() else (np.nan,np.nan)
                        rows.append(dict(contrast=label,fold=fold,view=view,block_rounds=length,metric=name,difference=value,
                                         lower=lo,upper=hi,undefined_replicates=int((~valid).sum())))
                    if length==4:
                        for i,seed in enumerate(core.SEEDS):
                            point_seed=sum(c*x[2][i if len(x[2])==3 else 0] for c,x in parts)
                            for name,value in zip(NAMES,point_seed):seeds.append(dict(contrast=label,fold=fold,view=view,seed=seed,metric=name,difference=value))
                        f=models[fold,terms[0][0]][0][[*core.MATCH_KEYS,'round','home_id','away_id']].copy()
                        f['fold'],f['contrast'],f['view']=fold,label,view
                        differences=sum(c*x[3] for c,x in parts)
                        for name,col in zip(metrics.ADDITIVE,differences.T):f[name]=col
                        fixtures.append(f)
                total=sum(n for n,_,_ in combined)
                point=sum(n*x for n,x,_ in combined)/total;samples=sum(n*s for n,_,s in combined)/total
                for name,value,col in zip(NAMES,point,samples.T):
                    valid=np.isfinite(col);lo,hi=np.quantile(col[valid],[.025,.975]) if valid.any() else (np.nan,np.nan)
                    rows.append(dict(contrast=label,fold='combined',view=view,block_rounds=length,metric=name,difference=value,
                        lower=lo,upper=hi,undefined_replicates=int((~valid).sum())))
        print(json.dumps(dict(primary_contrast=label,status='scored')),flush=True)
    directory=run/'04_comparisons';result=pd.DataFrame(rows);seed_table=pd.DataFrame(seeds);fixture=pd.concat(fixtures)
    result.to_csv(directory/'paired_comparisons.csv',index=False);seed_table.to_csv(directory/'paired_seed_differences.csv',index=False)
    fixture.to_parquet(directory/'per_fixture_differences.parquet',index=False)
    fixture.groupby(['contrast','fold','view','round'])[metrics.ADDITIVE].agg(['mean','count']).to_csv(directory/'per_round_differences.csv')
    return result,seed_table


def secondary(run,p,models):
    rows=[];league_rows=[]
    for view in ('raw','calibrated'):
        for length in (4,2,8):
            samples_available=all((run/'shared/bootstrap/calendar'/fold/f'blocks_{length}.npy').exists() for fold in core.FOLDS)
            by_model={}
            for key in [*core.CONFIGS,'league_logistic','pooled_logistic','league_frequency']:
                fold_points=[];fold_samples=[];original_counts=[]
                for fold in core.FOLDS:
                    expected=core.populations(run,fold)['outer'];indices=core.keys(expected)
                    frames=[f.set_index(list(core.MATCH_KEYS)).reindex(indices).reset_index() for f in models[fold,key]]
                    assert all(f.y.notna().all() for f in frames)
                    fixture=np.mean([f[[f'{view}_{n}' for n in metrics.ADDITIVE]].to_numpy() for f in frames],axis=0)
                    base=run/'shared/bootstrap/calendar'/fold
                    week=np.load(base/'fixture_week.npy');count=np.load(base/'counts_by_league.npy')
                    weights=np.load(base/f'blocks_{length}.npy') if (base/f'blocks_{length}.npy').exists() else None
                    points=[];draws=[]
                    for j,league in enumerate(p['competitions']):
                        selected=expected.competition_id.eq(league).to_numpy()
                        points.append(fixture[selected].mean(0) if selected.any() else np.full(len(metrics.ADDITIVE),np.nan))
                        sums=np.zeros((len(count),len(metrics.ADDITIVE)))
                        np.add.at(sums,week[selected],fixture[selected])
                        if weights is not None and count[:,j].sum()>0:
                            draws.append((weights@sums)/(weights@count[:,j])[:,None])
                        else:draws.append(np.full((20000,len(metrics.ADDITIVE)),np.nan))
                    fold_points.append(np.stack(points));fold_samples.append(np.stack(draws,axis=1));original_counts.append(count.sum(0))
                counts=np.array(original_counts);totals=counts.sum(0)
                fixed=np.divide(counts,totals,out=np.zeros_like(counts,dtype=float),where=totals>0)
                point=sum(np.nan_to_num(x)*fixed[i,:,None] for i,x in enumerate(fold_points))
                sample=sum(np.nan_to_num(x)*fixed[i,None,:,None] for i,x in enumerate(fold_samples))
                by_model[key]=(point,sample,counts,fold_points,fold_samples)
            for label,terms in secondary_contrasts():
                point=sum(c*by_model[k][0] for k,c in terms);sample=sum(c*by_model[k][1] for k,c in terms)
                counts=by_model[terms[0][0]][2];totals=counts.sum(0)
                for j,league in enumerate(p['competitions']):
                    for m,name in enumerate(metrics.ADDITIVE):
                        lo,hi=np.quantile(sample[:,j,m],[.025,.975]) if totals[j]>0 and samples_available else (np.nan,np.nan)
                        league_rows.append(dict(contrast=label,competition_id=league,view=view,block_weeks=length,metric=name,
                            difference=point[j,m] if totals[j]>0 else np.nan,lower=lo,upper=hi))
                for aggregation,league_weights in [('macro',np.full(len(totals),1/len(totals))),('micro',totals/totals.sum())]:
                    valid=(totals>0).all()
                    estimate=league_weights@point;distribution=np.einsum('rlm,l->rm',sample,league_weights)
                    for name,value,col in zip(metrics.ADDITIVE,estimate,distribution.T):
                        lo,hi=np.quantile(col,[.025,.975]) if valid and samples_available else (np.nan,np.nan)
                        rows.append(dict(contrast=label,aggregation=aggregation,view=view,block_weeks=length,metric=name,
                            difference=value if valid else np.nan,lower=lo,upper=hi,all_13_supported=bool(valid)))
    pd.DataFrame(rows).to_csv(run/'04_comparisons/secondary_intervals.csv',index=False)
    pd.DataFrame(league_rows).to_csv(run/'04_comparisons/per_league_intervals.csv',index=False)
    return pd.DataFrame(rows)


def plots(run,curves,primary_table,league_scores,resources,calibration):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(snapshot_B='#68717a',time_B='#165b99',time_C='#e48b23')
    destination=run/'04_comparisons'
    for which in ('outcome','goal'):
        fig,axes=plt.subplots(2,3,figsize=(15,8),squeeze=False)
        for job,curve in curves:
            if which=='goal' and job['configuration']!='time_C':continue
            row=int(job['fold'][-1])-1
            names=['train_loss','bundesliga_train_loss','validation_loss'] if which=='outcome' else ['train_goal_loss','bundesliga_train_goal_loss','validation_goal_loss']
            for column,name in enumerate(names):
                axes[row,column].plot(curve.step,curve[name],color=colors[job['configuration']],
                    linestyle='-' if job['arm']=='pooled' else '--',alpha=.65,
                    label=f'{job["configuration"]}/{job["arm"]}' if job['seed']==17 else None)
                axes[row,column].set(title=f'{job["fold"]}: {name}',xlabel='Full-population updates',ylabel='Outcome CE' if which=='outcome' else 'Goal loss (constants omitted)')
        for ax in axes.flat:ax.legend(fontsize=7);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(destination/f'{which}_curves.png',dpi=140);plt.close(fig)
    rows=primary_table[primary_table.fold.eq('combined')&primary_table.metric.eq('log_loss')&primary_table.block_rounds.eq(4)]
    fig,ax=plt.subplots(figsize=(13,11))
    for i,r in enumerate(rows.itertuples()):
        color='#165b99' if r.view=='raw' else '#27834a'
        ax.hlines(i,r.lower,r.upper,color=color)
        ax.plot(r.difference,i,'o',color=color)
    ax.set_yticks(range(len(rows)),[f'{r.contrast} / {r.view}' for r in rows.itertuples()],fontsize=8)
    ax.axvline(0,color='gray');ax.set_xlabel('Bundesliga log-loss difference; 95% descriptive round interval')
    fig.tight_layout();fig.savefig(destination/'paired_differences.png',dpi=140);plt.close(fig)
    means=league_scores[(league_scores.split=='outer')&(league_scores.view=='raw')]
    summary=means.groupby(['league','model']).log_loss.mean().unstack()
    summary.plot.bar(figsize=(16,7),title='Outer league CE: fold/seed means (precise weighted tables supplied)')
    plt.ylabel('Three-class log loss');plt.tight_layout();plt.savefig(destination/'league_comparison.png',dpi=140);plt.close()
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    cost=resources.groupby(['configuration','arm'])[['fit_seconds','allocated_gib']].mean()
    cost.fit_seconds.plot.bar(ax=axes[0],title='Mean fit seconds');cost.allocated_gib.plot.bar(ax=axes[1],title='Mean peak allocated CUDA GiB')
    fig.tight_layout();fig.savefig(destination/'resources.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    cal=calibration[(calibration.competition_id==35)&(calibration.seed==17)]
    for i,view in enumerate(('raw','calibrated')):
        for (model,fold),group in cal[cal.view==view].groupby(['model','fold']):
            axes[i].plot(group.predicted,group.observed,'o-',alpha=.6,label=f'{model}/{fold}')
        axes[i].plot([0,1],[0,1],'k--');axes[i].set(title=f'Bundesliga draw calibration, seed 17 / {view}',xlabel='Predicted draw fraction',ylabel='Observed draw fraction')
        axes[i].legend(fontsize=6)
    fig.tight_layout();fig.savefig(destination/'calibration.png',dpi=140);plt.close(fig)


def render(run):
    p,store=core.load(run,frozen=True);model_tables={};secondary_models={};records=[];curves=[];scores=[];calibration=[];goal_cal=[];slices=[]
    assert json.loads(json.dumps(contrasts()))==p['primary_contrasts']
    assert json.loads(json.dumps(secondary_contrasts()))==p['secondary_contrasts']
    league_names={v['competition_id']:k for k,v in p['catalog'].items()}
    def score_table(table,fold,key,seed,split):
        for league,group in table.groupby('competition_id'):
            result=metrics.table_metrics(group)
            for view in ('raw','calibrated'):
                row=dict(fold=fold,model=key,seed=seed,split=split,competition_id=int(league),league=league_names[int(league)],view=view)
                row.update({k:v for k,v in result[view].items() if k!='confusion'})
                if 'goals' in result:row.update(result['goals'])
                scores.append(row)
            if split=='outer':
                identity=dict(fold=fold,model=key,seed=seed,competition_id=int(league))
                cal,gcal,sub=diagnostics(group,store,core.config(),identity)
                calibration.extend(cal);goal_cal.extend(gcal);slices.extend(sub)
    for job in p['matrix']:
        result,pred,curve=audited_fit(run,job);records.append(result);curves.append((job,curve))
        key=f'{job["arm"]}/{job["configuration"]}'
        for split,frame in pred.items():score_table(frame,job['fold'],key,job['seed'],split)
        bl=pred['outer'][pred['outer'].competition_id.eq(35)].reset_index(drop=True)
        model_tables.setdefault((job['fold'],key),[]).append(bl)
        if job['arm']=='pooled':secondary_models.setdefault((job['fold'],job['configuration']),[]).append(pred['outer'])
    baseline_receipt=json.loads((run/'03_baselines/COMPLETE.json').read_text());baseline_audits=0
    for fold in core.FOLDS:
        pop=core.populations(run,fold);by_method={}
        for spec in [r for r in baseline_receipt['models'] if r['fold']==fold]:
            folder=run/spec['path'];cal=json.loads((folder/'calibration.json').read_text())
            for split in ('fit','inner','outer'):
                expected=pop[split] if spec['league']=='pooled' else pop[split][pop[split].competition_id.eq(int(spec['league']))].reset_index(drop=True)
                if not len(expected):continue
                frame=pd.read_parquet(folder/f'{split}_predictions.parquet');audit_predictions(frame,expected,cal);baseline_audits+=1
                key='pooled_logistic' if spec['league']=='pooled' else f'league_{spec["method"]}'
                score_table(frame,fold,key,17,split)
                if split=='outer':by_method.setdefault(key,[]).append(frame)
        for key,frames in by_method.items():
            merged=pd.concat(frames,ignore_index=True).set_index(list(core.MATCH_KEYS)).reindex(core.keys(pop['outer'])).reset_index()
            assert merged.y.notna().all()
            secondary_models[fold,key]=[merged]
            if key in ('pooled_logistic','league_logistic'):
                model_tables[fold,'pooled_logistic' if key=='pooled_logistic' else 'local_logistic']=[merged[merged.competition_id.eq(35)].reset_index(drop=True)]
    destination=run/'04_comparisons'
    goal_references=[];goal_comparisons=[];rate_distributions=[]
    for fold in core.FOLDS:
        for league in p['competitions']:
            folder=run/'03_baselines'/fold/f'goal_mean_{league}'
            state=json.loads((folder/'model.json').read_text())
            for split in ('inner','outer'):
                path=folder/f'{split}_predictions.parquet'
                if not path.exists():continue
                ref=pd.read_parquet(path)
                goal_references.append(dict(fold=fold,competition_id=league,league=league_names[league],split=split,count=len(ref),
                    goal_nll=ref.goal_joint_nll.mean(),home_mae=ref.home_absolute_error.mean(),away_mae=ref.away_absolute_error.mean(),
                    observed_home=ref.goals_for.mean(),observed_away=ref.goals_against.mean(),
                    predicted_home=state['predictive_rates'][0],predicted_away=state['predictive_rates'][1]))
                for job in [j for j in p['matrix'] if j['fold']==fold and j['configuration']=='time_C' and (j['arm']=='pooled' or league==35)]:
                    pred=pd.read_parquet(run/job['path']/f'{split}_predictions.parquet')
                    pred=pred[pred.competition_id.eq(league)].reset_index(drop=True)
                    assert same_fixture_keys(pred,ref)
                    goal_comparisons.append(dict(fold=fold,competition_id=league,league=league_names[league],seed=job['seed'],arm=job['arm'],split=split,
                        goal_nll=pred.goal_joint_nll.mean(),reference_nll=ref.goal_joint_nll.mean(),difference=pred.goal_joint_nll.mean()-ref.goal_joint_nll.mean(),
                        home_mae=pred.home_absolute_error.mean(),away_mae=pred.away_absolute_error.mean(),
                        predicted_home=pred.lambda_home.mean(),predicted_away=pred.lambda_away.mean(),
                        observed_home=pred.goals_home.mean(),observed_away=pred.goals_away.mean()))
                    for side in ('home','away'):
                        quantiles=pred[f'lambda_{side}'].quantile([0,.01,.05,.25,.5,.75,.95,.99,1])
                        rate_distributions.append(dict(fold=fold,competition_id=league,seed=job['seed'],arm=job['arm'],split=split,side=side,
                            **{f'q_{q}':v for q,v in quantiles.items()}))
    pd.DataFrame(goal_references).to_csv(destination/'goal_reference_metrics.csv',index=False)
    pd.DataFrame(goal_comparisons).to_csv(destination/'goal_comparisons.csv',index=False)
    pd.DataFrame(rate_distributions).to_csv(destination/'rate_distributions.csv',index=False)
    score_df=pd.DataFrame(scores);score_df.to_csv(destination/'metrics_by_league_seed.csv',index=False)
    pd.DataFrame(calibration).to_csv(destination/'calibration_bins.csv',index=False)
    pd.DataFrame(goal_cal).to_csv(destination/'goal_rate_calibration.csv',index=False)
    pd.DataFrame(slices).to_csv(destination/'entry_slices.csv',index=False)
    primary_table,seed_table=primary(run,model_tables)
    secondary_table=secondary(run,p,secondary_models)
    resources=pd.DataFrame([dict(**r['job'],selected=r['summary']['best_step'],completed=r['summary']['attempts'][0]['steps'],
        fit_seconds=r['fit_seconds'],total_seconds=r['total_seconds'],warm_update_seconds=r['warm_update_seconds'],outer_seconds=r['outer_seconds'],
        allocated_gib=None if r['cuda_allocated_peak'] is None else r['cuda_allocated_peak']/2**30,
        reserved_gib=None if r['cuda_reserved_peak'] is None else r['cuda_reserved_peak']/2**30,
        rss_gib=None if r['memory']['sampled_peak_rss_bytes'] is None else r['memory']['sampled_peak_rss_bytes']/2**30,
        original_training_peak_available=r.get('original_training_peak_available',True),
        fit_seconds_source=r.get('fit_seconds_source','Internal synchronized timer'),
        recovery_seconds=r.get('recovery_seconds',0.),engineering_active_elapsed_seconds=r.get('engineering_active_elapsed_seconds',0.),
        parameters=r['parameters']) for r in records])
    resources.to_csv(destination/'resources.csv',index=False)
    # Explicit fold-weighted per-league and macro/micro point estimates, for
    # separately scored seeds. Baselines are single fitted models, not ensembles.
    aggregates=[]
    for (key,seed,view),group in score_df[score_df.split.eq('outer')].groupby(['model','seed','view']):
        per_league=[]
        for league,part in group.groupby('competition_id'):
            values={m:float(np.average(part[m],weights=part['count'])) for m in NAMES if part[m].notna().all()}
            item=dict(model=key,seed=seed,view=view,competition_id=league,league=league_names[league],count=int(part['count'].sum()),**values)
            aggregates.append(dict(aggregation='league',**item));per_league.append(item)
        local=pd.DataFrame(per_league)
        if len(local)==13:
            for kind,weights in [('macro',np.ones(13)),('micro',local['count'].to_numpy())]:
                values={m:float(np.average(local[m],weights=weights)) for m in NAMES if local[m].notna().all()}
                aggregates.append(dict(model=key,seed=seed,view=view,aggregation=kind,competition_id=None,league='all13',count=int(local['count'].sum()),**values))
    pd.DataFrame(aggregates).to_csv(destination/'combined_metrics.csv',index=False)
    plots(run,curves,primary_table,score_df,resources,pd.DataFrame(calibration))
    common='These are previously inspected development periods. Checkpoints use raw Bundesliga inner CE only. '
    common+='Seeds are scored separately. Calibration uses each evaluation league\'s inner observations only. For losses, negative differences favor the first model; positive differences favor it for AUC/AP and accuracy. Predicted draw frequency is descriptive.\n\n'
    focus=primary_table[primary_table.metric.eq('log_loss')&primary_table.block_rounds.eq(4)]
    text='# Bundesliga transfer: cross-league supervision\n\n'+common
    text+='The primary contrast is pooled minus local time_C on the unchanged 306/305 Bundesliga outer fixtures. Combined results weight those periods by 306/611 and 305/611.\n\n'
    text+=table(focus[['contrast','fold','view','difference','lower','upper']])
    text+='\n\n## Matched seeds\n\n'+table(seed_table[seed_table.metric.eq('log_loss')])
    text+='\n\n## Draw and winner decomposition\n\n'+table(primary_table[primary_table.metric.isin(['draw_log_loss','winner_contribution'])&primary_table.block_rounds.eq(4)&primary_table.fold.eq('combined')])
    for name in ('paired_differences','outcome_curves','goal_curves','calibration','resources'):
        text+=f'\n\n![{name}]({name}.png)'
    text+='\n\nCurves show observed updates only; solid lines pooled, dashed local, three seeds. Goal training losses omit the factorial constants. All detailed [metrics](metrics_by_league_seed.csv), [paired contrasts](paired_comparisons.csv), [per-round differences](per_round_differences.csv), [resource measurements](resources.csv) and [entry slices](entry_slices.csv) remain available. No stopped suffix is extrapolated.\n'
    publish(destination,'bundesliga_transfer',text)
    summary_metrics=pd.DataFrame(aggregates)
    text='# All-league secondary evaluation\n\n'+common
    text+='All pooled GATs are Bundesliga-selected models. Within each league, fold scores are combined with original outer fixture-count weights. Macro averages the 13 league means equally; micro weights by original league counts. Calendar-week intervals preserve full league-round groups and use common multiplicities across leagues, models and seeds; fold/league mixture weights remain fixed.\n\n'
    text+=table(summary_metrics[summary_metrics.aggregation.isin(['macro','micro'])])
    text+='\n\n## Every league\n\n'+table(summary_metrics[summary_metrics.aggregation.eq('league')&summary_metrics.view.eq('raw')])
    text+='\n\n## Paired calendar-week comparisons\n\n'+table(secondary_table[secondary_table.metric.eq('log_loss')&secondary_table.block_weeks.eq(4)])
    text+='\n\n## Goal supervision diagnostics\n\n'+table(pd.DataFrame(goal_comparisons).query("split == 'outer' and arm == 'pooled'"))
    text+='\n\n![League comparison](league_comparison.png)\n\n[All secondary intervals](secondary_intervals.csv) · [Per-league intervals](per_league_intervals.csv) · [Goal rate calibration](goal_rate_calibration.csv) · [Goal references](goal_reference_metrics.csv) · [Rate distributions](rate_distributions.csv). Foreign predictions from local GATs are not treated as trained league controls. Goal improvements alone do not establish outcome improvement.\n'
    publish(destination,'all_leagues',text)
    for arm,directory in core.DIRECTORIES.items():
        subset=score_df[score_df.model.str.startswith(arm+'/')&score_df.split.eq('outer')]
        text=f'# {arm.capitalize()} supervision: 18 new GAT fits\n\n'+common
        text+=table(subset[['fold','model','seed','league','view','count','log_loss','brier','draw_log_loss','draw_auc','draw_ap','conditional_winner_log_loss']])
        text+='\n\n'+table(resources[resources.arm.eq(arm)])
        text+='\n\n[Bundesliga comparisons](../04_comparisons/bundesliga_transfer.html) · [All leagues](../04_comparisons/all_leagues.html) · [Frozen protocol](../PROTOCOL.json).\n'
        publish(run/directory,'report',text)
    publish(run/'03_baselines','report','# Fitting-only baselines\n\n'+common+table(score_df[score_df.model.isin(['league_logistic','pooled_logistic','league_frequency'])&score_df.split.eq('outer')])+
        '\n\n54 outcome models and 26 closed-form goal references. Numeric imputer/scaler values come from the parent recent-Bundesliga transform. Competition indicators are unscaled; Bundesliga is omitted. Fallbacks and solver extensions are explicit in each model.json.\n')
    primary_result=focus[focus.contrast.eq('time_C pooled-local')&focus.fold.eq('combined')&focus.view.eq('raw')].iloc[0]
    summary='# Cross-league supervision: completed fixed experiment\n\n'+common
    summary+=f'All 36 fresh GAT fits and 54 outcome baselines completed. The primary combined raw Bundesliga log-loss difference is **{primary_result.difference:+.6f}**, with descriptive four-round interval **[{primary_result.lower:+.6f}, {primary_result.upper:+.6f}]**.\n\n'
    summary+=table(focus[focus.fold.eq('combined')][['contrast','view','difference','lower','upper']])
    summary+='\n\n## Cohort and fitting mixture\n\n'+table(pd.read_csv(run/'league_inventory.csv').query("phase == 'fit'")[["fold","league","competition_id","count","share"]])
    summary+='\n\n## Resources\n\n'+table(resources.groupby(['configuration','arm'])[['fit_seconds','warm_update_seconds','allocated_gib','reserved_gib','completed','selected']].mean().reset_index())
    missing=int((~resources.original_training_peak_available).sum())
    if missing:
        summary+=f'\n\nOriginal peak-memory telemetry is available for {len(resources)-missing}/{len(resources)} fits. One completed-checkpoint export recovery has separately recorded replay memory; its original fit wall time is reconstructed from durable receipt timestamps. Missing training peaks remain unavailable. See [numerical repair](SCORING_REPAIR.json) and the resource table for the scope and engineering overhead.\n'
    summary+='\n\n[Bundesliga transfer](04_comparisons/bundesliga_transfer.html) · [All-league results](04_comparisons/all_leagues.html) · [Local fits](01_local_supervision/report.html) · [Pooled fits](02_all13_supervision/report.html) · [Baselines](03_baselines/report.html).\n\n'
    summary+='Optional features remain disabled by default. The new matched local controls, using the enlarged common context and category registry, are the statistical reference. Previous selected fits were not reused. Stop for numerical review; no subsequent experiment is launched.\n'
    publish(run,'summary',summary)
    write_json(run/'REPORT_AUDIT.json',dict(main_fits=len(records),baseline_prediction_tables=baseline_audits,
        primary_contrasts=len(contrasts()),secondary_contrasts=len(secondary_contrasts()),completed_at=pd.Timestamp.now(tz='UTC').isoformat()))


def audit_all(run):
    p,_=core.load(run,frozen=True)
    for job in p['matrix']:audited_fit(run,job)
    previous=json.loads((run/'shared/checks/protected_artifacts.json').read_text())
    for path,expected in previous.items():assert digest(core.ROOT/path)==expected,path
    assert len(p['matrix'])==36 and json.loads((run/'03_baselines/COMPLETE.json').read_text())['outcome_models']==54
    for relative in ['summary','01_local_supervision/report','02_all13_supervision/report','03_baselines/report','04_comparisons/bundesliga_transfer','04_comparisons/all_leagues']:
        assert (run/f'{relative}.md').exists() and (run/f'{relative}.html').exists()
    mutable={'COMPLETE.json','PROGRESS.json','STATUS.md','orchestrator.log','audit.log'}
    write_json(run/'COMPLETE.json',dict(status='complete',new_fits=36,reused_fits=0,failed_jobs=0,blocked_jobs=0,
        outcome_baselines=54,goal_references=26,protected_files=len(previous),excluded_mutable=sorted(mutable),
        files={f.relative_to(run).as_posix():digest(f) for f in sorted(run.rglob('*')) if f.is_file() and f.relative_to(run).as_posix() not in mutable and not f.name.endswith('.tmp')},
        completed_at=pd.Timestamp.now(tz='UTC').isoformat()))
