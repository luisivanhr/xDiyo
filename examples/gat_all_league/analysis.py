"""Predeclared scoring and paired fixed-mixture calendar-week inference."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from . import core,metrics,selection

MODELS=(*core.CONFIGS,'validation_selected','pooled_logistic','league_logistic','league_frequency')
NUMERIC=(*metrics.NAMES,'non_draw_fraction','observed_draw')

def contrasts():
    return [(f'{a} - {b}',[(a,1),(b,-1)]) for a in ('validation_selected',*core.CONFIGS)
        for b in ('pooled_logistic','league_logistic')]+[(f'{a} - {b}',[(a,1),(b,-1)])
        for a,b in [('time_C','time_B'),('time_B','snapshot_B'),('time_C','snapshot_B')]]

def align(frame,expected):
    assert core.keys(frame).is_unique and len(frame)==len(expected)
    ix=core.keys(frame).get_indexer(core.keys(expected));assert (ix>=0).all()
    result=frame.iloc[ix].reset_index(drop=True);assert core.same_keys(result,expected)
    return result

def load_models(run,phase):
    run=Path(run);p,_=core.load(run,frozen=True);gate=selection.verify_gate(run);models={}
    choices={r['fold']:r['configuration'] for r in gate['fold_choices']}
    for fold in core.FOLDS:
        expected=core.populations(run,fold)[phase]
        for name in core.CONFIGS:
            jobs=[j for j in p['matrix'] if j['fold']==fold and j['configuration']==name]
            jobs=sorted(jobs,key=lambda j:j['seed']);assert [j['seed'] for j in jobs]==list(core.SEEDS)
            models[fold,name]=[align(pd.read_parquet(run/j['path']/f'{phase}_predictions.parquet'),expected) for j in jobs]
        models[fold,'validation_selected']=models[fold,choices[fold]]
        root=run/'02_baselines'/fold
        models[fold,'pooled_logistic']=[align(pd.read_parquet(root/'logistic_pooled'/f'{phase}_predictions.parquet'),expected)]
        for name,method in [('league_logistic','logistic'),('league_frequency','frequency')]:
            frame=pd.concat([pd.read_parquet(root/f'{method}_{league}'/f'{phase}_predictions.parquet') for league in p['competitions']],ignore_index=True)
            models[fold,name]=[align(frame,expected)]
    return models

def score_models(models,phase):
    """Score each seed's probabilities; concatenate rows across folds, never seeds."""
    rows=[]
    for name in MODELS:
        seed_count=len(models['fold_1',name]);assert seed_count==len(models['fold_2',name])
        for i in range(seed_count):
            seed=core.SEEDS[i] if seed_count==3 else 0
            frames={fold:models[fold,name][i] for fold in core.FOLDS}
            frames['combined']=pd.concat(list(frames.values()),ignore_index=True)
            for fold,frame in frames.items():
                for view in ('raw','calibrated'):
                    per_league=[]
                    for league,part in frame.groupby('competition_id'):
                        values=metrics.scores(part,view);per_league.append(values)
                        rows.append(dict(model=name,seed=seed,fold=fold,split=phase,view=view,aggregation='league',competition_id=league,
                            count=len(part),**{k:values[k] for k in NUMERIC}))
                    assert len(per_league)==13,'Macro denominator must remain all 13 leagues'
                    micro=metrics.scores(frame,view)
                    macro={k:float(np.mean([v[k] for v in per_league])) for k in NUMERIC}
                    for aggregation,values in [('micro',micro),('macro',macro)]:
                        rows.append(dict(model=name,seed=seed,fold=fold,split=phase,view=view,aggregation=aggregation,competition_id=0,
                            count=len(frame),**{k:values[k] for k in NUMERIC}))
    return pd.DataFrame(rows)

def stratum_samples(losses,leagues,week,weights,competitions):
    """Mean seed losses first; recompute within-league denominators per resample."""
    losses=np.asarray(losses,float);week=np.asarray(week,int);leagues=np.asarray(leagues)
    points=[];samples=[];counts=[]
    for league in competitions:
        keep=leagues==league;count=np.bincount(week[keep],minlength=weights.shape[1])
        sums=np.bincount(week[keep],weights=losses[keep],minlength=weights.shape[1])
        denominator=weights@count
        if not keep.any() or (denominator<=0).any():raise ValueError('Unsupported original or resampled league')
        points.append(float(losses[keep].mean()));samples.append((weights@sums)/denominator);counts.append(int(keep.sum()))
    return np.array(points),np.stack(samples,axis=1),np.array(counts)

def combine_strata(points,samples,counts):
    """Preserve each original within-league fold weight and total league weight."""
    counts=np.asarray(counts);totals=counts.sum(0);fixed=counts/totals
    point=np.sum(np.asarray(points)*fixed,axis=0)
    sample=np.sum(np.asarray(samples)*fixed[:,None,:],axis=0)
    return point,sample,totals

def paired_intervals(run,models):
    run=Path(run);p,_=core.load(run,frozen=True);rows=[]
    for view in ('raw','calibrated'):
        for length in (4,2,8):
            cached={}
            for name in MODELS:
                if name=='league_frequency':continue
                parts=[]
                for fold in core.FOLDS:
                    frame=models[fold,name][0];base=run/'shared/bootstrap'/fold
                    week=np.load(base/'fixture_week.npy');weights=np.load(base/f'blocks_{length}.npy').astype(float)
                    loss=np.mean([f[f'{view}_log_loss'].to_numpy() for f in models[fold,name]],axis=0)
                    point,sample,count=stratum_samples(loss,frame.competition_id,week,weights,p['competitions'])
                    cached[fold,name]=(point,sample,count);parts.append((point,sample,count))
                cached['combined',name]=combine_strata(*zip(*parts))
            for label,terms in contrasts():
                for fold in (*core.FOLDS,'combined'):
                    point=sum(sign*cached[fold,name][0] for name,sign in terms)
                    sample=sum(sign*cached[fold,name][1] for name,sign in terms)
                    count=cached[fold,terms[0][0]][2]
                    for aggregation,mix in [('micro',count/count.sum()),('macro',np.full(len(count),1/len(count)))]:
                        distribution=sample@mix;lower,upper=np.quantile(distribution,[.025,.975])
                        rows.append(dict(contrast=label,fold=fold,view=view,block_weeks=length,metric='log_loss',aggregation=aggregation,
                            difference=float(point@mix),lower=float(lower),upper=float(upper),replicates=len(distribution),undefined_replicates=0))
    result=pd.DataFrame(rows);result.to_csv(run/'03_comparisons/paired_intervals.csv',index=False);return result

def paired_details(run,models,scores):
    run=Path(run);fixtures=[];seeds=[]
    for label,terms in contrasts():
        for fold in core.FOLDS:
            first=models[fold,terms[0][0]][0]
            for view in ('raw','calibrated'):
                frame=first[[*core.MATCH_KEYS,'round_stage','round','home_id','away_id','issuance_at']].copy()
                frame['fold']=fold;frame['contrast']=label;frame['view']=view
                values=sum(sign*np.mean([f[[f'{view}_{k}' for k in metrics.ADDITIVE]].to_numpy() for f in models[fold,name]],axis=0) for name,sign in terms)
                frame[metrics.ADDITIVE]=values;fixtures.append(frame)
        for fold in (*core.FOLDS,'combined'):
            for view in ('raw','calibrated'):
                for aggregation in ('micro','macro'):
                    for seed in core.SEEDS:
                        result={k:0. for k in NUMERIC}
                        for name,sign in terms:
                            choose=seed if len(models['fold_1',name])==3 else 0
                            part=scores[(scores.model==name)&(scores.seed==choose)&(scores.fold==fold)&(scores.view==view)&(scores.aggregation==aggregation)]
                            assert len(part)==1
                            for k in NUMERIC:result[k]+=sign*part[k].iloc[0]
                        seeds.append(dict(contrast=label,fold=fold,view=view,aggregation=aggregation,seed=seed,**result))
    fixture=pd.concat(fixtures,ignore_index=True);fixture.to_parquet(run/'03_comparisons/per_fixture_differences.parquet',index=False)
    fixture.groupby(['contrast','fold','view',*core.ROUND_KEYS],observed=True)[metrics.ADDITIVE].agg(['mean','count']).to_csv(run/'03_comparisons/per_round_differences.csv')
    result=pd.DataFrame(seeds);result.to_csv(run/'03_comparisons/paired_seed_differences.csv',index=False);return result

def goal_diagnostics(run,phase,models):
    run=Path(run);p,_=core.load(run,frozen=True);rows=[];calibration=[];quantiles=[]
    refs={fold:pd.concat([pd.read_parquet(run/'02_baselines'/fold/f'goal_mean_{league}'/f'{phase}_predictions.parquet')
        for league in p['competitions']],ignore_index=True) for fold in core.FOLDS}
    for name in ('time_C','goal_mean'):
        for index,seed in enumerate(core.SEEDS if name=='time_C' else [0]):
            frames={fold:(models[fold,name][index] if name=='time_C' else refs[fold]) for fold in core.FOLDS}
            frames['combined']=pd.concat(list(frames.values()),ignore_index=True)
            for fold,frame in frames.items():
                groups=[(0,frame),*frame.groupby('competition_id')]
                for league,part in groups:
                    identity=dict(model=name,seed=seed,fold=fold,split=phase,competition_id=league)
                    row=dict(**identity,count=len(part),poisson_nll=float(part.goal_marginal_nll.mean()),
                        home_mae=float(part.home_absolute_error.mean()),away_mae=float(part.away_absolute_error.mean()))
                    for side in ('home','away'):
                        rate=part[f'lambda_{side}'];observed=part[f'goals_{side}']
                        row[f'mean_predicted_{side}']=float(rate.mean());row[f'mean_observed_{side}']=float(observed.mean())
                        for q,value in rate.quantile([0,.01,.1,.25,.5,.75,.9,.99,1]).items():quantiles.append(dict(**identity,side=side,quantile=q,rate=value))
                        bins=pd.qcut(rate,10,labels=False,duplicates='drop') if rate.nunique()>1 else pd.Series(0,index=rate.index)
                        for b in sorted(bins.dropna().unique()):
                            keep=bins.eq(b);calibration.append(dict(**identity,side=side,bin=b,count=int(keep.sum()),
                                predicted=float(rate[keep].mean()),observed=float(observed[keep].mean())))
                    rows.append(row)
    return pd.DataFrame(rows),pd.DataFrame(calibration),pd.DataFrame(quantiles)
