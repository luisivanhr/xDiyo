"""Native full-population fitting, Bundesliga-only selection, and durable jobs."""
from dataclasses import asdict
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
from scipy.special import gammaln
from . import core
from .initialization import initial_path
from gat_suite import metrics
from gat_suite.fitting import setup, predict, optimizer_receipt
from compare_gat_temporal import MemoryProbe, timed, write_json, digest, validate_training_progress
from compare_gat_goals import digest_state
from temporal_team_gat import make_context
from xdiyo_analytics.graphs import TemporalGraphAdapter, TemporalGraphSerializer
from xdiyo_analytics.graphs.adapter import _Backend
from xdiyo_analytics.training import TrainingControl,EarlyStopping
from xdiyo_analytics.training.resume_loop import run_resumable_iterations
from xdiyo_analytics.training.residency import InputCachePolicy


class CrossBackend(_Backend):
    def __init__(self,run,protocol,store,job):
        self.run,self.job=Path(run),job
        cfg=core.config(job['configuration'])
        assert json.dumps(asdict(cfg),sort_keys=True)==json.dumps(protocol['configurations'][job['configuration']],sort_keys=True)
        state=json.loads((self.run/'shared/preprocessing'/f'{job["fold"]}.json').read_text())['state']
        r=protocol['resources']
        super().__init__(cfg,store,job['seed'],r['device'],InputCachePolicy(r['input_cache_bytes'],r['headroom_bytes']),
                         preprocessor_state=state)

    def initialize(self,context):
        super().initialize(context)
        self.model.load_state_dict(torch.load(initial_path(self.run,self.job),map_location=self.device,weights_only=True))
        self.initial_digest=digest_state(self.model.state_dict())
        self.local_fit=core.populations(self.run,self.job['fold'],'local')['fit']
        self.local_groups=self.train_groups if self.job['arm']=='local' else self._stream(make_context(self.local_fit))
        self.refresh_version()

    def step(self):
        self._update_started=time.perf_counter()
        return super().step()

    def checkpoint_metrics(self,monitor,values):
        values=super().checkpoint_metrics('train_loss',values)
        if self.job['arm']=='local':
            values['bundesliga_train_loss']=values['train_loss']
            values['bundesliga_train_goal_loss']=values['train_goal_loss']
        else:
            prob,rates,_=predict(self,self.local_groups,self.local_fit)
            ys=self.local_fit[['goals_for','goals_against']].to_numpy(float)
            values['bundesliga_train_loss']=metrics.outcome_metrics(prob,core.truth(self.local_fit))['log_loss']
            values['bundesliga_train_goal_loss']=float((rates-ys*np.log(rates)).sum(1).mean())
        if not all(np.isfinite(v) for v in values.values()):
            raise FloatingPointError('Nonfinite training metrics; no clipping')
        if str(self.device).startswith('cuda'):torch.cuda.synchronize(self.device)
        values['update_seconds']=time.perf_counter()-self._update_started
        return values


def context_for(run,job):
    population=core.populations(run,job['fold'],job['arm'])
    local=core.populations(run,job['fold'],'local')
    context=make_context(population['fit'])
    context.validation=make_context(local['inner'])
    return context,population


def calibration_by_league(frame,prob):
    result={}
    for league,indices in frame.groupby('competition_id',sort=True).indices.items():
        result[str(league)]=metrics.draw_calibration(prob[indices],core.truth(frame.iloc[indices]))
        result[str(league)]['population_sha256']=core.frame_hash(frame.iloc[indices])
        result[str(league)]['available']=True
    return result


def prediction_table(frame,prob,rates,calibrators,store,cfg,job=None):
    parts=[]
    for league,positions in frame.groupby('competition_id',sort=True).indices.items():
        selected=frame.iloc[positions]
        cal=calibrators.get(str(league),dict(intercept=0.,available=False,population='missing inner'))
        goals=selected[['goals_for','goals_against']].to_numpy(float)
        table=metrics.prediction_table(selected,prob[positions],None if rates is None else rates[positions],goals,cal)
        table['_position']=positions
        table['calibration_available']=cal.get('available',True)
        table['calibration_intercept']=cal['intercept']
        parts.append(table)
    output=pd.concat(parts,ignore_index=True).sort_values('_position').drop(columns='_position').reset_index(drop=True)
    issue=store.issue_times(frame,cfg)
    output['issuance_at']=issue.to_numpy()
    output['cutoff_at']=[cfg.history_cutoff(t) for t in issue]
    output['fixture_key']=['/'.join(map(str,k)) for k in core.keys(frame)]
    if job:
        for name in ('fold','seed','configuration','arm'):output[name]=job[name]
    return output


def per_league_metrics(table):
    return {str(league):metrics.table_metrics(group) for league,group in table.groupby('competition_id')}


def fit(run,job_id):
    run=Path(run);p,store=core.load(run,frozen=True)
    rows=[j for j in p['matrix'] if j['job_id']==job_id]
    if len(rows)!=1:raise ValueError('Job absent from fixed matrix')
    job=rows[0];folder=run/job['path'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():raise ValueError('Completed fit cannot be overwritten')
    attempts=sorted(folder.glob('attempt_*'));number=len(attempts)+1
    if number>3:raise RuntimeError('Engineering retry budget exhausted')
    attempt=folder/f'attempt_{number:02}';attempt.mkdir()
    identity=dict(job=job,protocol=digest(run/'PROTOCOL.json'),freeze=digest(run/'FREEZE.json'),
                  initial=digest(initial_path(run,job)),preprocessor=digest(run/'shared/preprocessing'/f'{job["fold"]}.json'))
    setup(p);previous=None
    if attempts:
        for name in ('completed_training','last_training'):
            last=attempts[-1]/f'{name}.pt';receipt=attempts[-1]/f'{name}.json'
            if last.exists() and receipt.exists():
                meta=json.loads(receipt.read_text())
                if meta['identity']!=identity or meta['sha256']!=digest(last):raise ValueError('Resume identity mismatch')
                previous=torch.load(last,map_location='cpu',weights_only=True);break
    write_json(attempt/'attempt.json',dict(identity=identity,resumed=previous is not None,pid=os.getpid()))
    context,pop=context_for(run,job)
    def observer(event):
        if not event.final:
            write_json(folder/'progress.json',dict(job=job_id,step=event.step,metrics=event.metrics,pid=os.getpid(),attempt=number))
    context.observer=observer
    def save(capture,*,complete):
        state=capture();name='completed_training' if complete else 'last_training'
        path=attempt/f'{name}.pt';temporary=path.with_suffix('.pt.tmp')
        torch.save(state,temporary);temporary.replace(path)
        write_json(attempt/f'{name}.json',dict(identity=identity,sha256=digest(path),complete=complete))
    control=TrainingControl(max_steps=200,seed=job['seed'],early_stopping=EarlyStopping(15,0.),monitor='validation_loss',restore_best=True)
    start=time.perf_counter()
    try:
        with MemoryProbe(p['resources']['rss_limit_gib']) as probe:
            preparation=[]
            def factory(seed):
                backend=CrossBackend(run,p,store,job)
                initialize=backend.initialize
                def initialized(ctx):
                    begin=time.perf_counter();initialize(ctx);preparation.append(time.perf_counter()-begin)
                backend.initialize=initialized
                return backend
            (backend,history,summary),fit_seconds=timed(lambda:run_resumable_iterations(factory,context,control,state=previous,save=save),'cuda:0')
            curve=history.pivot(index='step',columns='metric',values='value').reset_index()
            validate_training_progress(curve.to_dict('records'),summary,dict(max_steps=200,early_stopping=dict(patience=15,min_delta=0.)))
            history.to_csv(folder/'history_long.csv',index=False);curve.to_csv(folder/'training.csv',index=False)
            # All secondary inner data are diagnostic only, after BL selection.
            inner_groups=backend._stream(make_context(pop['inner']))
            inner,inner_seconds=timed(lambda:predict(backend,inner_groups,pop['inner']),'cuda:0')
            calibrators=calibration_by_league(pop['inner'],inner[0])
            for league in p['competitions']:
                calibrators.setdefault(str(league),dict(intercept=0.,available=False,count=0,population='missing inner',boundary_solution=False))
            write_json(folder/'calibration.json',calibrators)
            outer_context=make_context(pop['outer'],fitting=False)
            backend._boundary(backend._validate(outer_context))
            outer_groups=backend._stream(outer_context)
            outer,outer_seconds=timed(lambda:predict(backend,outer_groups,pop['outer']),'cuda:0')
            fitting,fitting_seconds=timed(lambda:predict(backend,backend.train_groups,pop['fit']),'cuda:0')
            evaluation={};league_scores={}
            for phase,values in [('fit',fitting),('inner',inner),('outer',outer)]:
                rates=values[1] if job['configuration']=='time_C' else None
                table=prediction_table(pop[phase],values[0],rates,calibrators,store,backend.config,job)
                table.to_parquet(folder/f'{phase}_predictions.parquet',index=False)
                evaluation[phase]=metrics.table_metrics(table)
                league_scores[phase]=per_league_metrics(table)
                if phase=='fit':
                    contributions=[]
                    for league,part in table.groupby('competition_id'):
                        contribution=dict(competition_id=int(league),count=len(part),share=len(part)/len(table),
                            outcome_contribution=float(part.raw_log_loss.sum()/len(table)))
                        if rates is not None:
                            constant=gammaln(part[['goals_home','goals_away']].to_numpy()+1).sum(1)
                            contribution['goal_contribution']=float((part.goal_joint_nll-constant).sum()/len(table))
                        contributions.append(contribution)
                    pd.DataFrame(contributions).to_csv(folder/'fitting_contributions.csv',index=False)
            best=curve.set_index('step').loc[summary['best_step']]
            np.testing.assert_allclose(evaluation['fit']['raw']['log_loss'],best.train_loss,rtol=1e-5,atol=1e-6)
            np.testing.assert_allclose(league_scores['inner']['35']['raw']['log_loss'],best.validation_loss,rtol=1e-5,atol=1e-6)
            canonical=torch.load(initial_path(run,job),map_location='cpu',weights_only=True)
            if job['configuration']!='time_C':
                for name,value in backend.model.state_dict().items():
                    if name.startswith('goal_head.'):
                        torch.testing.assert_close(value.cpu(),canonical[name],atol=0,rtol=0)
            adapter=TemporalGraphAdapter(backend.config,store,control=control,device=backend.device)
            adapter.backend_,adapter.training_history_,adapter.training_summary_=backend,history,summary
            serializer=TemporalGraphSerializer(store,device=backend.device)
            serializer.save(adapter,folder/'model');restored=serializer.load(folder/'model')
            # Replay exactly the same groups and retain the public availability guard.
            reloaded=predict(restored.backend_,restored.backend_._stream(outer_context),pop['outer'])
            np.testing.assert_array_equal(outer[0],reloaded[0]);np.testing.assert_array_equal(outer[1],reloaded[1])
            for method in (restored.predict,restored.predict_goals):
                try:method(make_context(core.populations(run,job['fold'],'local')['inner'],fitting=False))
                except ValueError:pass
                else:raise AssertionError('Public selection-period boundary weakened')
            result=dict(job=job,identity=identity,summary=summary,metrics=evaluation,by_league=league_scores,
                selected_initial_digest=backend.initial_digest,optimizer=optimizer_receipt(backend),
                fit_seconds=fit_seconds,preprocessing_seconds=sum(preparation),inner_seconds=inner_seconds,
                outer_seconds=outer_seconds,selected_fit_replay_seconds=fitting_seconds,total_seconds=time.perf_counter()-start,
                warm_update_seconds=float(curve.update_seconds.iloc[1:].mean()),
                memory=probe.result(),cuda_allocated_peak=torch.cuda.max_memory_allocated(),cuda_reserved_peak=torch.cuda.max_memory_reserved(),
                parameters=sum(v.numel() for v in backend.model.parameters()),goal_parameters=sum(v.numel() for v in backend.model.goal_head.parameters()),
                reload_exact=True,public_boundaries_verified=True,attempts=number,completed_at=pd.Timestamp.now(tz='UTC').isoformat())
            write_json(folder/'result.json',result)
        print(json.dumps(dict(job=job_id,status='complete',completed=summary['attempts'][0]['steps'],selected=summary['best_step'],seconds=result['total_seconds'])),flush=True)
    except BaseException:
        (attempt/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


def profile(run):
    run=Path(run);p,store=core.load(run);setup(p)
    job=dict(configuration='time_C',fold='fold_2',seed=17,arm='pooled')
    context,pop=context_for(run,job);backend=CrossBackend(run,p,store,job)
    variant='components' if p['resources'].get('complete_components') else 'full'
    print(json.dumps(dict(profile=variant,stage='initialize',pid=os.getpid())),flush=True)
    with MemoryProbe(p['resources']['rss_limit_gib']) as probe:
        _,preparation=timed(lambda:backend.initialize(context),'cuda:0')
        print(json.dumps(dict(profile=variant,stage='initialized',seconds=preparation)),flush=True)
        updates=[]
        for i in range(3):
            _,elapsed=timed(lambda:backend.checkpoint_metrics('validation_loss',backend.step()),'cuda:0')
            updates.append(elapsed)
            print(json.dumps(dict(profile=variant,update=i+1,seconds=elapsed)),flush=True)
        outer_groups=backend._stream(make_context(pop['outer'],fitting=False))
        _,outer=timed(lambda:predict(backend,outer_groups,pop['outer']),'cuda:0')
        result=dict(disposable=True,predictive_scores_inspected=False,variant=variant,fit_fixtures=len(pop['fit']),
            fitting_groups=len(backend.train_groups),preparation_seconds=preparation,update_seconds=updates,
            outer_replay_seconds=outer,memory=probe.result(),cuda_allocated_peak=torch.cuda.max_memory_allocated(),
            cuda_reserved_peak=torch.cuda.max_memory_reserved(),max_36x200_update_hours=float(np.mean(updates[1:])*36*200/3600))
    write_json(run/'shared/checks'/f'profile_{variant}.json',result)
    print(json.dumps(result),flush=True)


def freeze(run):
    run=Path(run);p,_=core.load(run)
    if (run/'FREEZE.json').exists():raise ValueError('Already frozen')
    for name in ('correctness.json','focused_tests.txt','regression.txt','canonical_mapping.json','secondary_bootstrap.json'):
        assert (run/'shared/checks'/name).exists(),name
    for name in ('focused_tests.txt','regression.txt'):
        text=(run/'shared/checks'/name).read_text()
        assert 'passed' in text and 'FAILED' not in text and 'ERROR' not in text,name
    assert list((run/'shared/checks').glob('profile_*.json'))
    from . import reporting
    p['primary_contrasts']=reporting.contrasts();p['secondary_contrasts']=reporting.secondary_contrasts()
    p['scoring']=dict(primary='raw outer Bundesliga CE, pooled time_C minus local time_C',
        class_order=['home','draw','away'],calibration=dict(method='inner-only league draw intercept',bounds=[-20,20],missing_inner_intercept=0.),
        combined_primary_weights=[306/611,305/611],seed_reduction='separately scored then averaged',
        secondary='fixed original within-league fold weights; macro equal leagues, micro original fixture counts')
    write_json(run/'PROTOCOL.json',p)
    # Report presentation may be repaired with a separate receipt; statistical
    # definitions above and every fitting dependency remain frozen.
    sources=[x for x in (core.ROOT/'examples/gat_cross_league').glob('*.py') if x.name!='reporting.py']+list((core.ROOT/'src/xdiyo_analytics').rglob('*.py'))
    sources+=list((core.ROOT/'examples/gat_suite').glob('*.py'))
    sources+=[core.ROOT/f'examples/{n}.py' for n in ('run_gat_cross_league','compare_gat_temporal','compare_gat_goals','temporal_team_gat')]
    sources.append(core.ROOT/'tests/analytics/test_gat_cross_league.py')
    write_json(run/'REPORTING_SOURCE.json',dict(path='examples/gat_cross_league/reporting.py',sha256=digest(Path(reporting.__file__))))
    write_json(run/'FREEZE.json',dict(protocol_sha256=digest(run/'PROTOCOL.json'),
        source_hashes={f.relative_to(core.ROOT).as_posix():digest(f) for f in sources},
        input_hashes={f.relative_to(run).as_posix():digest(f) for base in ('data','populations','preprocessing','initialization','bootstrap')
                      for f in (run/'shared'/base).rglob('*') if f.is_file()},frozen_at=pd.Timestamp.now(tz='UTC').isoformat()))
    import platform,psutil,sklearn
    write_json(run/'ENVIRONMENT.json',dict(python=sys.version,executable=sys.executable,torch=str(torch.__version__),cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,
        host_ram_gib=psutil.virtual_memory().total/2**30,deterministic=True,precision='native highest',cublas_workspace=':4096:8',
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=core.ROOT,text=True).strip()))
    core.status(run,stage='frozen')


def checks(run):
    from .checks import check
    check(Path(run))


def baselines(run):
    from .baselines import fit_baselines
    fit_baselines(Path(run))


def report(run):
    from .reporting import render
    render(Path(run))


def audit(run):
    from .reporting import audit_all
    audit_all(Path(run))


def run(run):
    from .orchestration import execute
    execute(Path(run))
