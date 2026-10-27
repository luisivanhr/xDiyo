"""Update six outcome and two goal references; verify and reuse the remainder."""
from pathlib import Path
import json,shutil,time,warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from . import core,metrics,selection
from .stages import attach_schedule
from gat_suite.baselines import rolling_features
from gat_cross_league.baselines import design,frequencies
from gat_cross_league.execution import calibration_by_league,prediction_table

def probabilities(folder,state,features):
    if state['method']=='logistic' and not state.get('fallback',False) in (True,'True'):
        return joblib.load(folder/'model.joblib').predict_proba(features)
    return np.tile(state['probabilities'],(len(features),1))

def prepare(run):
    run=Path(run);protocol,store=core.load(run,frozen=True);root=run/'02_baselines'
    if (root/'PREPARED.json').exists():return
    records=[];reuse=[];feature_checks=[];start=time.perf_counter()
    for fold in core.FOLDS:
        destination=root/fold;destination.mkdir(exist_ok=True)
        source=core.PARENT/'03_baselines'/fold
        shutil.copyfile(source/'rolling_transform.joblib',destination/'rolling_transform.joblib')
        transform=joblib.load(destination/'rolling_transform.joblib');pop=core.populations(run,fold);features={}
        for phase,frame in pop.items():
            raw=rolling_features(store,core.config(),frame)
            np.save(destination/f'{phase}_rolling.npy',raw)
            features[phase]=design(raw,frame,transform,protocol['competitions']);np.save(destination/f'{phase}_design.npy',features[phase])
            old_frame=pd.read_parquet(core.PARENT/'shared/populations'/f'{fold}_pooled_{phase}.parquet')
            old_raw=np.load(source/f'{phase}_rolling.npy')
            unaffected=np.flatnonzero(frame.competition_id.ne(38));old_ix=core.keys(old_frame).get_indexer(core.keys(frame.iloc[unaffected]))
            assert (old_ix>=0).all()
            assert len(unaffected)==int(old_frame.competition_id.ne(38).sum())
            np.testing.assert_array_equal(raw[unaffected],old_raw[old_ix])
            for column in ('home_id','away_id','goals_for','goals_against'):
                np.testing.assert_array_equal(frame.iloc[unaffected][column].to_numpy(),old_frame.iloc[old_ix][column].to_numpy())
            feature_checks.append(dict(fold=fold,phase=phase,unchanged_non_belgian_rows=len(unaffected),exact_rolling_inputs=True))
        specs=[(method,str(league),league) for league in protocol['competitions'] for method in ('logistic','frequency')]+[('logistic','pooled',None)]
        for method,label,league in specs:
            folder=destination/f'{method}_{label}';folder.mkdir(exist_ok=True)
            old=source/f'{method}_{label}';updated=league is None or league==38
            subsets={phase:np.flatnonzero(np.ones(len(frame),bool) if league is None else frame.competition_id.eq(league)) for phase,frame in pop.items()}
            fitting=pop['fit'].iloc[subsets['fit']];y=core.truth(fitting)
            began=time.perf_counter()
            if updated:
                counts=np.bincount(y,minlength=3);fallback=bool(method=='logistic' and (counts==0).any())
                state=dict(method=method,league=label,class_counts=counts.tolist(),fit_count=len(y),fitting_identity=core.frame_hash(fitting),fallback=fallback)
                if method=='frequency' or fallback:
                    values,smoothing=frequencies(y)
                    if fallback:values=(counts+.5)/(len(y)+1.5);smoothing=.5
                    state.update(probabilities=values.tolist(),smoothing=smoothing)
                else:
                    model=LogisticRegression(C=1.,max_iter=500,random_state=17)
                    with warnings.catch_warnings(record=True) as notices:model.fit(features['fit'][subsets['fit']],y)
                    extended=bool(model.n_iter_.max()>=500)
                    if extended:
                        model=LogisticRegression(C=1.,max_iter=2000,random_state=17)
                        with warnings.catch_warnings(record=True) as more:model.fit(features['fit'][subsets['fit']],y)
                        notices.extend(more)
                    assert model.n_iter_.max()<model.max_iter and model.classes_.tolist()==[0,1,2]
                    joblib.dump(model,folder/'model.joblib')
                    state.update(C=1.,max_iter=model.max_iter,n_iter=model.n_iter_.tolist(),iteration_extension=extended,
                        classes=model.classes_.tolist(),warnings=[str(v.message) for v in notices])
                core.write_json(folder/'model.json',state)
            else:
                state=json.loads((old/'model.json').read_text())
                for name in ('model.json','model.joblib'):
                    if (old/name).exists():shutil.copyfile(old/name,folder/name)
                shutil.copyfile(old/'calibration.json',folder/'source_calibration.json')
                assert state['fit_count']==len(y)
                for phase,ix in subsets.items():
                    previous=pd.read_parquet(old/f'{phase}_predictions.parquet');current=pop[phase].iloc[ix]
                    assert core.same_keys(current,previous)
                    value=metrics.base.normalized(probabilities(folder,state,features[phase][ix]))
                    np.testing.assert_allclose(value,previous[metrics.P_COLUMNS],atol=1e-13,rtol=1e-13)
                    reuse.append(dict(fold=fold,model=f'{method}_{label}',phase=phase,count=len(ix),
                        source_sha256=core.digest(old/f'{phase}_predictions.parquet'),maximum_probability_error=float(np.max(np.abs(value-previous[metrics.P_COLUMNS].to_numpy())))))
            core.write_json(folder/'fit.json',dict(fold=fold,method=method,league=label,updated=updated,source=str(old),
                fitting_seconds=time.perf_counter()-began,state_sha256=core.digest(folder/'model.json')))
            records.append(dict(fold=fold,method=method,league=label,path=str(folder.relative_to(run)),updated=updated))
        for league in protocol['competitions']:
            folder=destination/f'goal_mean_{league}';folder.mkdir(exist_ok=True)
            fitting=pop['fit'][pop['fit'].competition_id.eq(league)]
            mean=fitting[['goals_for','goals_against']].to_numpy(float).mean(0);source_goal=source/f'goal_mean_{league}'
            if league==38:
                core.write_json(folder/'model.json',dict(observed_mean=mean.tolist(),predictive_rates=np.maximum(mean,1e-6).tolist(),
                    floor_applied=(mean==0).tolist(),fit_count=len(fitting),fitting_identity=core.frame_hash(fitting)))
            else:
                state=json.loads((source_goal/'model.json').read_text())
                np.testing.assert_array_equal(mean,state['observed_mean']);assert state['fit_count']==len(fitting)
                shutil.copyfile(source_goal/'model.json',folder/'model.json')
                for phase,frame in pop.items():
                    current=frame[frame.competition_id.eq(league)]
                    previous=pd.read_parquet(source_goal/f'{phase}_predictions.parquet');assert core.same_keys(current,previous)
                    np.testing.assert_array_equal(current[['goals_for','goals_against']].to_numpy(),previous[['goals_for','goals_against']].to_numpy())
            core.write_json(folder/'fit.json',dict(updated=league==38,source=str(source_goal),state_sha256=core.digest(folder/'model.json')))
    assert len(records)==54 and sum(r['updated'] for r in records)==6
    core.write_json(run/'shared/checks/baseline_reuse.json',dict(outcome_references_reused=48,goal_references_reused=24,
        outcome_models_updated=6,goal_references_updated=2,feature_checks=feature_checks,prediction_checks=reuse,
        numerical_tolerance=dict(atol=1e-13,rtol=1e-13),no_outer_scores_computed=True))
    core.write_json(root/'PREPARED.json',dict(models=records,outcome_models=54,goal_references=26,updated_outcome_models=6,
        updated_goal_references=2,reused_outcome_models=48,reused_goal_references=24,seconds=time.perf_counter()-start))
    print(json.dumps(dict(status='baselines_prepared',updated=8,reused_outcome=48,reused_goal=24)),flush=True)

def evaluate(run):
    run=Path(run);selection.verify_gate(run);protocol,store=core.load(run,frozen=True)
    root=run/'02_baselines';prepared=json.loads((root/'PREPARED.json').read_text())
    if (root/'COMPLETE.json').exists():return
    for item in prepared['models']:
        folder=run/item['path'];state=json.loads((folder/'model.json').read_text());pop=core.populations(run,item['fold'])
        league=None if item['league']=='pooled' else int(item['league'])
        subsets={phase:np.flatnonzero(np.ones(len(f),bool) if league is None else f.competition_id.eq(league)) for phase,f in pop.items()}
        probabilities_by_phase={phase:probabilities(folder,state,np.load(root/item['fold']/f'{phase}_design.npy')[ix]) for phase,ix in subsets.items()}
        inner=pop['inner'].iloc[subsets['inner']].reset_index(drop=True)
        calibrators=calibration_by_league(inner,probabilities_by_phase['inner'])
        if not item['updated']:
            previous=json.loads((folder/'source_calibration.json').read_text())
            for key,value in calibrators.items():np.testing.assert_allclose(value['intercept'],previous[key]['intercept'],rtol=1e-12,atol=1e-12)
            calibrators={key:previous[key] for key in calibrators}
        core.write_json(folder/'calibration.json',calibrators);scores={}
        for phase,ix in subsets.items():
            frame=pop[phase].iloc[ix].reset_index(drop=True)
            if item['updated']:table=prediction_table(frame,probabilities_by_phase[phase],None,calibrators,store,core.config())
            else:
                old=core.PARENT/'03_baselines'/item['fold']/f'{item["method"]}_{item["league"]}'/f'{phase}_predictions.parquet'
                table=attach_schedule(pd.read_parquet(old),store.schedule)
                assert core.same_keys(frame,table)
                for columns in (metrics.P_COLUMNS,metrics.C_COLUMNS):np.testing.assert_array_equal(table[columns],pd.read_parquet(old,columns=columns))
            table['fold']=item['fold'];table['seed']=17;metrics.add_rps(table)
            table.to_parquet(folder/f'{phase}_predictions.parquet',index=False);scores[phase]=metrics.table_scores(table)
        core.write_json(folder/'result.json',dict(**item,metrics=scores,selection_sha256=core.digest(run/'SELECTION.json')))
    for fold in core.FOLDS:
        pop=core.populations(run,fold)
        for league in protocol['competitions']:
            folder=root/fold/f'goal_mean_{league}';state=json.loads((folder/'model.json').read_text())
            for phase,frame in pop.items():
                table=frame[frame.competition_id.eq(league)].copy();rates=np.tile(state['predictive_rates'],(len(table),1))
                table[['lambda_home','lambda_away']]=rates;table[['goals_home','goals_away']]=table[['goals_for','goals_against']].to_numpy()
                for key,value in metrics.base.count_rows(rates,table[['goals_for','goals_against']].to_numpy(float)).items():table[key]=value
                issue=store.issue_times(table,core.config());table['issuance_at']=issue.to_numpy();table['cutoff_at']=[core.config().history_cutoff(t) for t in issue]
                table['fold']=fold;table.to_parquet(folder/f'{phase}_predictions.parquet',index=False)
    core.write_json(root/'COMPLETE.json',dict(**prepared,selection_sha256=core.digest(run/'SELECTION.json')))
