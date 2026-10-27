"""Frozen fitting-only practical comparators; shared by all encoder studies."""
from pathlib import Path
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from . import core,metrics
from compare_gat_temporal import write_json,frame_hash


def rolling_features(store,cfg,frame):
    output=np.empty((len(frame),6))
    issue=store.issue_times(frame,cfg)
    for when,positions in issue.groupby(issue,sort=True).indices.items():
        # Explicit chronology precedes tail(10); source row order is irrelevant.
        eligible=store.eligible(cfg.history_cutoff(when)).sort_values(['kickoff_at','event_id','side'],kind='stable')
        tail=eligible.groupby('team_id',sort=False).tail(10)
        means=tail.groupby('team_id')[['goals_for','goals_against']].mean()
        targets=frame.iloc[positions]
        home=means.reindex(targets.home_id.to_numpy()).to_numpy(float)
        away=means.reindex(targets.away_id.to_numpy()).to_numpy(float)
        output[positions]=np.column_stack((home,away,home-away))
    return output


def centered_poisson_fit(frame):
    teams=sorted(set(frame.home_id)|set(frame.away_id)); mapping={int(t):i for i,t in enumerate(teams)}; n=len(teams)
    h=np.array([mapping[int(t)] for t in frame.home_id]);a=np.array([mapping[int(t)] for t in frame.away_id])
    y=frame[['goals_for','goals_against']].to_numpy(float)
    venue=1-frame.get('neutral_venue',pd.Series(np.nan,index=frame.index)).eq(1).to_numpy(float)
    means=np.maximum(y.mean(0),1e-6); initial=np.r_[np.log(means[1]),np.log(means[0]/means[1]),np.zeros(2*n)]
    def objective(theta):
        attack=theta[2:2+n]-theta[2:2+n].mean();defence=theta[2+n:]-theta[2+n:].mean()
        eta=np.column_stack((theta[0]+venue*theta[1]+attack[h]-defence[a],theta[0]+attack[a]-defence[h]))
        rate=np.exp(eta);residual=(rate-y)/len(y)
        value=(rate-y*eta).sum()/len(y)+.001*(np.dot(attack,attack)+np.dot(defence,defence))
        ga=np.bincount(h,weights=residual[:,0],minlength=n)+np.bincount(a,weights=residual[:,1],minlength=n)+.002*attack
        gd=-np.bincount(a,weights=residual[:,0],minlength=n)-np.bincount(h,weights=residual[:,1],minlength=n)+.002*defence
        gradient=np.r_[residual.sum(),np.dot(venue,residual[:,0]),ga-ga.mean(),gd-gd.mean()]
        return value,gradient
    fitted=minimize(objective,initial,jac=True,method='L-BFGS-B',options=dict(maxiter=2000,gtol=1e-8,ftol=1e-13,maxls=50))
    if not fitted.success and np.max(np.abs(fitted.jac))>1e-5: raise RuntimeError(f'Poisson baseline failed: {fitted.message}')
    theta=fitted.x;attack=theta[2:2+n]-theta[2:2+n].mean();defence=theta[2+n:]-theta[2+n:].mean()
    return dict(teams=teams,attack=attack.tolist(),defence=defence.tolist(),scoring=float(theta[0]),home_effect=float(theta[1]),
        convergence=dict(success=bool(fitted.success),message=str(fitted.message),iterations=int(fitted.nit),gradient_max=float(np.max(np.abs(fitted.jac)))),
        penalty=.001,penalty_convention='mean NLL plus .001 times sum of squared centered team effects',new_team_effect=0.)


def centered_poisson_predict(state,frame):
    attack=dict(zip(state['teams'],state['attack']));defence=dict(zip(state['teams'],state['defence']))
    ah=np.array([attack.get(int(t),0.) for t in frame.home_id]);aa=np.array([attack.get(int(t),0.) for t in frame.away_id])
    dh=np.array([defence.get(int(t),0.) for t in frame.home_id]);da=np.array([defence.get(int(t),0.) for t in frame.away_id])
    venue=1-frame.get('neutral_venue',pd.Series(np.nan,index=frame.index)).eq(1).to_numpy(float)
    return np.exp(np.column_stack((state['scoring']+venue*state['home_effect']+ah-da,state['scoring']+aa-dh)))


def constant_dispersion(means,y,family):
    rates=np.tile(means,(len(y),1))
    objective=lambda a:metrics.count_rows(rates,y,family,float(a))['goal_joint_nll'].mean()
    low,high=-20.,10.
    fitted=minimize_scalar(lambda loga:objective(np.exp(loga)),bounds=(low,high),method='bounded',options={'xatol':1e-10})
    if not fitted.success or abs(fitted.x-high)<.01: raise RuntimeError('Constant dispersion solver did not locate an interior optimum or Poisson boundary.')
    alpha=float(np.exp(fitted.x));value=objective(alpha);poisson=objective(0.)
    boundary=poisson<=value+1e-10
    return dict(alpha=0. if boundary else alpha,kappa=None if boundary else 1/alpha,poisson_boundary=bool(boundary),
        mean_objective=float(poisson if boundary else value),converged=bool(fitted.success),numerical_log_alpha_interval=[low,high])


def run_baselines(run):
    run=Path(run);protocol,store=core.load(run,frozen=True);cfg=core.config(competitions=tuple(protocol['competitions']))
    done=run/'shared/baselines/COMPLETE.json'
    if done.exists(): return
    summaries=[]
    for fold in ('fold_1','fold_2'):
        populations=core.frames(run,fold)
        features={k:rolling_features(store,cfg,v) for k,v in populations.items()}
        # One fixed transform from the recent fitting population across both windows.
        imputer=SimpleImputer(add_indicator=True,keep_empty_features=True)
        imputer.fit(features['recent']);scaler=StandardScaler().fit(imputer.transform(features['recent']))
        root=run/'shared/baselines'/fold;root.mkdir(parents=True,exist_ok=True)
        joblib.dump(dict(imputer=imputer,scaler=scaler),root/'rolling_transform.joblib')
        write_json(root/'transform.json',dict(fitting_population=frame_hash(populations['recent']),
            all_missing_columns=np.flatnonzero(np.isnan(features['recent']).all(0)).tolist(),
            feature_order=['home_for','home_against','away_for','away_against','difference_for','difference_against']))
        for window in ('recent','expanded'):
            fitting=populations[window];truth=lambda f:np.where(f.goals_for>f.goals_against,0,np.where(f.goals_for==f.goals_against,1,2))
            y=truth(fitting)
            for method in ('frequency','rolling_logistic'):
                start=time.perf_counter();folder=root/f'{window}_{method}';folder.mkdir(exist_ok=True)
                if method=='frequency':
                    frequencies=np.bincount(y,minlength=3)/len(y)
                    state=dict(frequencies=frequencies.tolist(),fit_identity=frame_hash(fitting))
                    predict=lambda key:np.tile(frequencies,(len(populations[key]),1))
                else:
                    model=LogisticRegression(C=1.,max_iter=500,random_state=17)
                    with warnings.catch_warnings(record=True) as recorded:
                        model.fit(scaler.transform(imputer.transform(features[window])),y)
                    if model.n_iter_.max()>=model.max_iter:
                        raise RuntimeError('Logistic iteration cap reached; numerical repair required before publication.')
                    state=dict(C=1.,max_iter=500,random_state=17,iterations=model.n_iter_.tolist(),classes=model.classes_.tolist(),
                        warnings=[str(w.message) for w in recorded],fit_identity=frame_hash(fitting),converged=True)
                    joblib.dump(model,folder/'model.joblib')
                    predict=lambda key:model.predict_proba(scaler.transform(imputer.transform(features[key])))[:,[list(model.classes_).index(i) for i in (0,1,2)]]
                fit_seconds=time.perf_counter()-start
                probabilities={key:predict(key) for key in (window,'inner','outer')}
                cal=metrics.draw_calibration(probabilities['inner'],truth(populations['inner']));cal['population_sha256']=frame_hash(populations['inner'])
                write_json(folder/'calibration.json',cal);write_json(folder/'model.json',state)
                result={}
                for split,key in [('train',window),('inner','inner'),('outer','outer')]:
                    frame=populations[key];goals=frame[['goals_for','goals_against']].to_numpy(float)
                    table=metrics.prediction_table(frame,probabilities[key],None,goals,cal)
                    issue=store.issue_times(frame,cfg);table['issuance_at']=issue.to_numpy();table['cutoff_at']=[cfg.history_cutoff(t) for t in issue]
                    table.to_parquet(folder/f'{split}_predictions.parquet',index=False)
                    result[split]=metrics.table_metrics(table)
                write_json(folder/'result.json',dict(kind='outcome_baseline',fold=fold,window=window,method=method,metrics=result,fit_seconds=fit_seconds))
                summaries.append(dict(kind='outcome',fold=fold,window=window,method=method,path=str(folder.relative_to(run))))
            means=fitting[['goals_for','goals_against']].to_numpy(float).mean(0);floors=(means==0);means=np.where(floors,1e-6,means)
            methods=['constant_poisson']+(['attack_defence_poisson','constant_independent_nb','constant_shared_gamma'] if window=='expanded' else [])
            for method in methods:
                folder=root/f'{window}_{method}';folder.mkdir(exist_ok=True);start=time.perf_counter()
                family='independent_nb' if method=='constant_independent_nb' else 'shared_gamma' if method=='constant_shared_gamma' else 'poisson'
                state=dict(means=means.tolist(),zero_mean_floor=floors.tolist(),fit_identity=frame_hash(fitting),alpha=0.)
                if method=='attack_defence_poisson': state.update(centered_poisson_fit(fitting))
                elif family!='poisson': state.update(constant_dispersion(means,fitting[['goals_for','goals_against']].to_numpy(float),family))
                fit_seconds=time.perf_counter()-start;result={}
                for split,key in [('train',window),('inner','inner'),('outer','outer')]:
                    frame=populations[key];ys=frame[['goals_for','goals_against']].to_numpy(float)
                    rate=centered_poisson_predict(state,frame) if method=='attack_defence_poisson' else np.tile(means,(len(frame),1))
                    table=frame.copy();table[['lambda_home','lambda_away']]=rate;table[['goals_home','goals_away']]=ys
                    for name,value in metrics.count_rows(rate,ys,family,state['alpha']).items():table[name]=value
                    issue=store.issue_times(frame,cfg);table['issuance_at']=issue.to_numpy();table['cutoff_at']=[cfg.history_cutoff(t) for t in issue]
                    table.to_parquet(folder/f'{split}_predictions.parquet',index=False)
                    result[split]={k:float(table[k].mean()) for k in ('goal_joint_nll','goal_home_nll','goal_away_nll','home_absolute_error','away_absolute_error')}
                write_json(folder/'model.json',state);write_json(folder/'result.json',dict(kind='goal_baseline',fold=fold,window=window,method=method,metrics=result,fit_seconds=fit_seconds))
                summaries.append(dict(kind='goal',fold=fold,window=window,method=method,path=str(folder.relative_to(run))))
        print(json.dumps(dict(baselines=fold,status='complete')),flush=True)
    write_json(done,dict(status='complete',entries=summaries))
