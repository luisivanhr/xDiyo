"""54 fitting-only outcome comparators and 26 closed-form goal references."""
from pathlib import Path
import json
import shutil
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from . import core
from .execution import prediction_table,calibration_by_league,per_league_metrics
from gat_suite import metrics
from gat_suite.baselines import rolling_features
from compare_gat_temporal import digest,write_json,frame_hash


def frequencies(y):
    counts=np.bincount(y,minlength=3)
    smoothing=.5 if (counts==0).any() else 0.
    return (counts+smoothing)/(len(y)+3*smoothing),smoothing


def design(features,frame,transform,competitions):
    numeric=transform['scaler'].transform(transform['imputer'].transform(features))
    categorical=np.column_stack([frame.competition_id.eq(k).to_numpy(float) for k in competitions if k!=35])
    return np.column_stack((numeric,categorical))


def fit_baselines(run):
    run=Path(run);p,store=core.load(run,frozen=True);cfg=core.config();summaries=[]
    root=run/'03_baselines'
    if (root/'COMPLETE.json').exists():return
    for fold in core.FOLDS:
        folder=root/fold;folder.mkdir(parents=True,exist_ok=True)
        population=core.populations(run,fold)
        transform_path=core.PARENT/'shared/baselines'/fold/'rolling_transform.joblib'
        transform=joblib.load(transform_path)
        shutil.copyfile(transform_path,folder/'rolling_transform.joblib')
        features={}
        for phase,frame in population.items():
            path=folder/f'{phase}_rolling.npy'
            if path.exists():values=np.load(path)
            else:
                values=rolling_features(store,cfg,frame);np.save(path,values)
            features[phase]=design(values,frame,transform,p['competitions'])
        write_json(folder/'feature_identity.json',dict(parent_numeric_transform_sha256=digest(transform_path),
            continuous_features=['home_for','home_against','away_for','away_against','difference_for','difference_against'],
            unscaled_indicators=[k for k in p['competitions'] if k!=35],omitted_category=35,
            population_hashes={k:frame_hash(f) for k,f in population.items()},
            feature_hashes={k:digest(folder/f'{k}_rolling.npy') for k in population}))
        specs=[(method,str(league),league) for league in p['competitions'] for method in ('logistic','frequency')]
        specs.append(('logistic','pooled',None))
        for method,label,league in specs:
            destination=folder/f'{method}_{label}';destination.mkdir(exist_ok=True)
            if (destination/'result.json').exists():
                summaries.append(dict(fold=fold,method=method,league=label,path=str(destination.relative_to(run))));continue
            selections={phase:np.flatnonzero(np.ones(len(f),bool) if league is None else f.competition_id.eq(league)) for phase,f in population.items()}
            fitting=population['fit'].iloc[selections['fit']];y=core.truth(fitting)
            if not len(y):raise ValueError(f'Unavailable baseline fitting population: {fold}/{label}')
            class_counts=np.bincount(y,minlength=3);fallback=method=='logistic' and (class_counts==0).any()
            start=time.perf_counter();probabilities={}
            state=dict(method=method,league=label,class_counts=class_counts.tolist(),fit_count=len(y),
                       fitting_identity=frame_hash(fitting),fallback=fallback)
            if method=='frequency' or fallback:
                rates,smoothing=frequencies(y)
                if fallback:
                    rates=(class_counts+.5)/(len(y)+1.5);smoothing=.5
                state.update(probabilities=rates.tolist(),smoothing=smoothing)
                probabilities={phase:np.tile(rates,(len(ix),1)) for phase,ix in selections.items()}
            else:
                model=LogisticRegression(C=1.,max_iter=500,random_state=17)
                with warnings.catch_warnings(record=True) as recorded:
                    model.fit(features['fit'][selections['fit']],y)
                extension=model.n_iter_.max()>=500
                if extension:
                    model=LogisticRegression(C=1.,max_iter=2000,random_state=17)
                    with warnings.catch_warnings(record=True) as retry:
                        model.fit(features['fit'][selections['fit']],y)
                    recorded.extend(retry)
                if model.n_iter_.max()>=model.max_iter:raise RuntimeError(f'Logistic failed to converge: {fold}/{label}')
                joblib.dump(model,destination/'model.joblib')
                state.update(C=1.,max_iter=model.max_iter,n_iter=model.n_iter_.tolist(),iteration_extension=bool(extension),
                    warnings=[str(w.message) for w in recorded],classes=model.classes_.tolist())
                assert list(model.classes_)==[0,1,2]
                probabilities={phase:model.predict_proba(features[phase][ix]) for phase,ix in selections.items()}
            seconds=time.perf_counter()-start
            inner=population['inner'].iloc[selections['inner']].reset_index(drop=True)
            calibrators=calibration_by_league(inner,probabilities['inner']) if len(inner) else {}
            for comp in p['competitions']:
                calibrators.setdefault(str(comp),dict(intercept=0.,available=False,count=0,population='missing inner',boundary_solution=False))
            write_json(destination/'calibration.json',calibrators)
            write_json(destination/'model.json',state)
            scores={}
            for phase,ix in selections.items():
                frame=population[phase].iloc[ix].reset_index(drop=True)
                if not len(frame):scores[phase]={};continue
                table=prediction_table(frame,probabilities[phase],None,calibrators,store,cfg)
                table.to_parquet(destination/f'{phase}_predictions.parquet',index=False)
                scores[phase]=per_league_metrics(table)
            write_json(destination/'result.json',dict(fold=fold,method=method,league=label,by_league=scores,
                fitting_seconds=seconds,state=state,completed_at=pd.Timestamp.now(tz='UTC').isoformat()))
            summaries.append(dict(fold=fold,method=method,league=label,path=str(destination.relative_to(run))))
        for league in p['competitions']:
            fit=population['fit'][population['fit'].competition_id.eq(league)]
            mean=fit[['goals_for','goals_against']].to_numpy(float).mean(0)
            predictive=np.maximum(mean,1e-6)
            destination=folder/f'goal_mean_{league}';destination.mkdir(exist_ok=True)
            write_json(destination/'model.json',dict(observed_mean=mean.tolist(),predictive_rates=predictive.tolist(),
                floor_applied=(mean==0).tolist(),fit_count=len(fit),fitting_identity=frame_hash(fit)))
            for phase,frame in population.items():
                table=frame[frame.competition_id.eq(league)].copy()
                if table.empty:continue
                rates=np.tile(predictive,(len(table),1));table[['lambda_home','lambda_away']]=rates
                goals=table[['goals_for','goals_against']].to_numpy(float)
                for key,value in metrics.count_rows(rates,goals).items():table[key]=value
                table.to_parquet(destination/f'{phase}_predictions.parquet',index=False)
        print(json.dumps(dict(baselines=fold,outcome_models=len(specs),goal_references=len(p['competitions']))),flush=True)
    assert len(summaries)==54
    write_json(root/'COMPLETE.json',dict(outcome_models=54,logistic_models=28,frequency_models=26,goal_references=26,
        models=summaries,completed_at=pd.Timestamp.now(tz='UTC').isoformat()))
