"""Final-only analysis. No fitting, publication, or analysis on import.

Input: results/ANALYSIS_MANIFEST.json, created after COMPUTE_COMPLETE.json.
Manifest has stages (ordered), seeds, families, folds, original13 (competition IDs),
jobs [{stage,family,fold,seed,predictions,sha256}], and optional odds path + hash.
Paths are relative to results. Probabilities are ordered home/draw/away.
"""
from pathlib import Path
import hashlib,json,itertools
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
KEYS=['competition_id','season_id','event_id']
GROUP=['stage','family','fold','seed']
METRICS=['ce','rps','accuracy','draw_ce','brier']

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def read_bound(base,record,key='predictions'):
    path=(base/record[key]).resolve()
    require(path.is_relative_to(base.resolve()),'Input escaped results directory')
    require(digest(path)==record['sha256'],'Input hash mismatch: '+str(path))
    return pd.read_parquet(path)

def build_manifest(odds_path,base=ROOT/'results'):
    """Bind completed runner outputs and copy exact verified odds bytes locally.

    Caller supplies the provenance-verified opening-odds table. This function
    does not select a quote source or infer opening prices from latest quotes.
    """
    import shutil
    base=Path(base);require((base/'COMPUTE_COMPLETE.json').is_file(),'Compute incomplete')
    stages=['original13','first_divisions','first_and_second_divisions']
    families=['time_B/N','time_C/E1','time_B/CD_DW'];folds=['fold_1','fold_2','fold_3','fold_4'];seeds=[17,42,73]
    jobs=[]
    for stage,family,fold,seed in itertools.product(stages,families,folds,seeds):
        path=base/'refit'/stage/family/str(seed)/fold/'outer.parquet'
        require((path.parent/'evaluation.json').is_file(),'Missing evaluation receipt')
        jobs.append(dict(stage=stage,family=family,fold=fold,seed=seed,predictions=path.relative_to(base).as_posix(),sha256=digest(path)))
    target=base/'analysis_inputs'/'financial_inputs.parquet';target.parent.mkdir(exist_ok=True)
    if target.exists():require(digest(target)==digest(odds_path),'Existing financial input differs')
    else:shutil.copyfile(odds_path,target)
    manifest=dict(stages=stages,families=families,folds=folds,seeds=seeds,original13=[8,17,18,23,34,35,36,37,38,44,53,54,182],jobs=jobs,odds=dict(path=target.relative_to(base).as_posix(),sha256=digest(target)))
    path=base/'ANALYSIS_MANIFEST.json'
    if path.exists():require(json.loads(path.read_text())==manifest,'Existing manifest differs')
    else:path.write_text(json.dumps(manifest,indent=2))
    return path

def fixture_losses(d,family):
    d=d.copy();p=d[['p_home','p_draw','p_away']].to_numpy(float);y=d.y.to_numpy(int)
    require(np.isfinite(p).all() and (p>0).all() and (p<=1).all(),'Invalid probabilities; no silent clipping')
    require(np.allclose(p.sum(1),1,atol=1e-6,rtol=0),'Probability sums')
    require(np.isin(y,[0,1,2]).all(),'Invalid outcome class')
    require(np.isfinite(d[['goals_for','goals_against']]).all().all(),'Missing settled goals')
    actual=np.where(d.goals_for>d.goals_against,0,np.where(d.goals_for==d.goals_against,1,2))
    require(np.array_equal(actual,y),'Outcome target disagrees with settled goals')
    truth=np.eye(3)[y]
    d['ce']=-np.log(p[np.arange(len(p)),y])
    d['rps']=np.square(np.cumsum(p,axis=1)[:,:2]-np.cumsum(truth,axis=1)[:,:2]).sum(1)/2
    d['accuracy']=(p.argmax(1)==y).astype(float)
    d['draw_ce']=-np.where(y==1,np.log(p[:,1]),np.log1p(-p[:,1]))
    d['brier']=np.square(p-truth).sum(1)
    if family=='time_C/E1':
        z=d[['lambda_home','lambda_away','goals_for','goals_against']].to_numpy(float)
        require(np.isfinite(z).all() and (z[:,:2]>=0).all(),'Missing C/E1 goal predictions')
        d['goal_mse']=np.square(z[:,:2]-z[:,2:]).mean(1)
    else:d['goal_mse']=np.nan
    require(np.isfinite(d[METRICS]).all().all(),'Nonfinite scores')
    return d

def summarize(d,by):
    g=d.groupby(by,dropna=False,sort=True)
    result=g[METRICS+['goal_mse']].mean();result['fixtures']=g.size()
    result['goal_rmse']=np.sqrt(result.pop('goal_mse'))
    return result.reset_index()

def paired_bootstrap(d,seeds,weeks=4,reps=20000):
    """Joint calendar block resampling; fixed original fold/league mixture.

    Average independently scored seed losses before resampling. Calendar draws
    are shared across leagues within a fold. Empty-denominator draws are rejected
    jointly, never silently replaced by league-specific calendars.
    """
    identity=['fold',*KEYS]
    require(d.groupby(identity).seed.nunique().eq(len(seeds)).all(),'Incomplete seed pairing')
    avg=d.groupby(identity+['kickoff_at'],as_index=False)[METRICS].mean()
    draws=np.zeros((reps,len(METRICS)));rng=np.random.default_rng(20261107)
    for fold,f in avg.groupby('fold',sort=True):
        dates=pd.to_datetime(f.kickoff_at,utc=True).dt.tz_convert('Europe/Berlin').dt.tz_localize(None).dt.normalize()
        monday=dates.min()-pd.Timedelta(days=dates.min().weekday())
        week=((dates-monday).dt.days//7).to_numpy();nweeks=int(week.max())+1
        # Circular moving calendar blocks, truncate to the original week count.
        strata=[]
        for league,g in f.groupby('competition_id',sort=True):
            mask=f.index.isin(g.index);idx=week[mask]
            count=np.bincount(idx,minlength=nweeks)
            sums=np.stack([np.bincount(idx,weights=g[m],minlength=nweeks) for m in METRICS],axis=1)
            strata.append((count,sums,len(g)/len(avg)))
        accepted=[];attempts=0
        while sum(len(x) for x in accepted)<reps:
            attempts+=1;require(attempts<1000,'Sparse bootstrap strata; inspect protocol')
            n=min(512,reps-sum(len(x) for x in accepted))
            starts=rng.integers(0,nweeks,(n,int(np.ceil(nweeks/weeks))))
            picks=((starts[:,:,None]+np.arange(weeks))%nweeks).reshape(n,-1)[:,:nweeks]
            weights=np.stack([np.bincount(row,minlength=nweeks) for row in picks])
            keep=np.ones(n,dtype=bool);values=np.zeros((n,len(METRICS)))
            for counts,sums,weight in strata:
                denom=weights@counts;keep &= denom>0
                values+=np.divide(weights@sums,denom[:,None],out=np.zeros_like(values),where=denom[:,None]>0)*weight
            accepted.append(values[keep])
        draws+=np.concatenate(accepted)[:reps]
    return np.quantile(draws,[.025,.975],axis=0)

def finances(predictions,odds,seeds):
    """Frozen historical draw strategy; never places bets.

    Opening draw decimal odds, p_draw>=.20, STRICT p_draw*odds>1, flat1u.
    Missing/invalid odds excluded explicitly. Only settled original13 fixtures.
    Sharpe-like statistic is daily unit PnL, NOT bankroll-normalized return.
    """
    require(not odds.duplicated(KEYS).any(),'Duplicate odds')
    require({'draw_opening_odds','result_available_at'}<=set(odds),'Odds/settlement fields absent')
    d=predictions[predictions.original13].drop(columns=['draw_opening_odds','result_available_at'],errors='ignore').merge(odds[KEYS+['draw_opening_odds','result_available_at']],on=KEYS,how='left',validate='many_to_one')
    d['price_valid']=np.isfinite(d.draw_opening_odds)&d.draw_opening_odds.gt(1)
    d['selected']=d.price_valid&d.p_draw.ge(.2)&(d.p_draw*d.draw_opening_odds).gt(1)
    require(d.loc[d.selected,'result_available_at'].notna().all(),'Selected unsettled fixtures')
    d['stake']=d.selected.astype(float)
    d['profit']=np.where(d.selected,np.where(d.y.eq(1),d.draw_opening_odds-1,-1),0.)
    d['settlement_day']=pd.to_datetime(d.result_available_at,utc=True).dt.floor('D')
    valid_days=d.settlement_day.dropna();require(len(valid_days)>0,'No settlement calendar')
    calendar=pd.date_range(valid_days.min()-pd.Timedelta(days=1),valid_days.max(),freq='D')
    rows=[];daily=[]
    for (stage,family,seed),g in d.groupby(['stage','family','seed']):
        s=g.groupby('settlement_day').profit.sum().reindex(calendar,fill_value=0.)
        cumulative=s.cumsum();std=s.std(ddof=1);n=int(g.selected.sum());total=float(g.stake.sum())
        rows.append(dict(stage=stage,family=family,seed=seed,tickets=n,stake=total,profit=float(g.profit.sum()),roi=float(g.profit.sum()/total) if total else np.nan,win_rate=float(g.loc[g.selected,'y'].eq(1).mean()),missing_or_invalid_odds=int((~g.price_valid).sum()),daily_pnl_sharpe=float(np.sqrt(365)*s.mean()/std) if std>0 else np.nan,max_drawdown_units=float((cumulative.cummax()-cumulative).max())))
        daily.append(pd.DataFrame(dict(day=calendar,stage=stage,family=family,seed=seed,profit=s.to_numpy(),cumulative_profit=cumulative.to_numpy())))
    return d,pd.DataFrame(rows),pd.concat(daily,ignore_index=True)

def analyze(base=ROOT/'results'):
    base=Path(base);require((base/'COMPUTE_COMPLETE.json').exists(),'All computation must finish before analysis')
    manifest=json.loads((base/'ANALYSIS_MANIFEST.json').read_text());jobs=manifest['jobs']
    expected=set(itertools.product(manifest['stages'],manifest['families'],manifest['folds'],manifest['seeds']))
    actual=[tuple(j[k] for k in GROUP) for j in jobs]
    require(len(actual)==len(expected) and set(actual)==expected,'Incomplete/duplicate experiment matrix')
    frames=[]
    for j in jobs:
        d=read_bound(base,j);require(not d.duplicated(KEYS).any(),'Duplicate prediction identity')
        require(len(d)>0,'Empty test population');d=fixture_losses(d,j['family'])
        d=d.assign(**{k:j[k] for k in GROUP});d['original13']=d.competition_id.isin(manifest['original13'])
        require(d.original13.any(),'Original13 ID dtype mismatch or missing population');frames.append(d)
    data=pd.concat(frames,ignore_index=True);out=base/'analysis';out.mkdir(exist_ok=True)
    for _,g in data.groupby(['stage','family','fold']):
        require(g.groupby(KEYS).seed.nunique().eq(len(manifest['seeds'])).all(),'Seed populations differ')
    data.to_parquet(out/'fixture_losses.parquet',index=False)
    scope=pd.concat([data.assign(population='all_tested'),data[data.original13].assign(population='original13')])
    summarize(scope,GROUP+['population']).to_csv(out/'scores_fold_seed.csv',index=False)
    summarize(scope,GROUP+['population','competition_id']).to_csv(out/'scores_league_fold_seed.csv',index=False)
    pooled=summarize(scope,['stage','family','seed','population']);pooled.to_csv(out/'scores_pooled_seed.csv',index=False)
    pooled.groupby(['stage','family','population'])[METRICS+['goal_rmse']].mean().reset_index().to_csv(out/'scores_seed_mean.csv',index=False)
    fold=summarize(scope,GROUP+['population'])
    fold.groupby(['stage','family','population'])[METRICS+['goal_rmse']].mean().reset_index().to_csv(out/'scores_equal_fold_mean.csv',index=False)
    pairs=[]
    for a,b in itertools.combinations(manifest['stages'],2):
        for family in manifest['families']:
            aa=data[(data.stage==a)&(data.family==family)&data.original13]
            bb=data[(data.stage==b)&(data.family==family)&data.original13]
            on=['family','fold','seed',*KEYS]
            z=aa.merge(bb,on=on,how='outer',suffixes=('_a','_b'),indicator=True,validate='one_to_one')
            require(z._merge.eq('both').all(),'Original13 paired populations differ')
            require(z.y_a.eq(z.y_b).all() and z.kickoff_at_a.eq(z.kickoff_at_b).all(),'Pair identity/outcome mismatch')
            x=z[on].copy();x['kickoff_at']=z.kickoff_at_a;x['contrast']=b+' minus '+a
            for metric in METRICS:x[metric]=z[metric+'_b']-z[metric+'_a']
            pairs.append(x)
    differences=pd.concat(pairs,ignore_index=True);differences.to_parquet(out/'paired_fixture_differences.parquet',index=False)
    for label,keys in [('seed',['contrast','family','seed']),('fold_seed',['contrast','family','fold','seed']),('league_seed',['contrast','family','competition_id','seed'])]:
        differences.groupby(keys)[METRICS].mean().reset_index().to_csv(out/('paired_'+label+'.csv'),index=False)
    intervals=[]
    for (contrast,family),g in differences.groupby(['contrast','family']):
        for weeks in (2,4,8):
            ci=paired_bootstrap(g,manifest['seeds'],weeks,manifest.get('bootstrap_reps',20000))
            for i,m in enumerate(METRICS):intervals.append(dict(contrast=contrast,family=family,weeks=weeks,metric=m,difference=float(g[m].mean()),lower=ci[0,i],upper=ci[1,i]))
    pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv',index=False)
    require('odds' in manifest,'Financial output requires sealed odds input')
    odds=read_bound(base,manifest['odds'],key='path')
    ledger,financial,daily=finances(data,odds,manifest['seeds'])
    ledger.to_parquet(out/'simulated_decisions.parquet',index=False);financial.to_csv(out/'financial_seed.csv',index=False);daily.to_parquet(out/'financial_daily.parquet',index=False)
    (out/'ANALYSIS_COMPLETE.json').write_text(json.dumps(dict(status='ANALYSIS_ONLY_NOT_RELEASED',manifest_sha256=digest(base/'ANALYSIS_MANIFEST.json'),files={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='ANALYSIS_COMPLETE.json'}),indent=2))
    return out

if __name__=='__main__':analyze()
