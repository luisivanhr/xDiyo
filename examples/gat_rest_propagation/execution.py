"""Fresh native fits; outer evaluation is a separate, selection-gated phase."""
from pathlib import Path
import json,os,time,traceback
import numpy as np
import pandas as pd
import torch
from . import core,metrics,selection
from .backend import AllLeagueBackend,context_for
from gat_suite.fitting import setup,predict,optimizer_receipt
from gat_cross_league.execution import calibration_by_league,prediction_table
from compare_gat_temporal import MemoryProbe,timed,validate_training_progress
from temporal_team_gat import make_context
from xdiyo_analytics.graphs import TemporalGraphAdapter,TemporalGraphSerializer
from xdiyo_analytics.training import TrainingControl,EarlyStopping
from xdiyo_analytics.training.resume_loop import run_resumable_iterations

def control(seed):return TrainingControl(max_steps=200,seed=seed,early_stopping=EarlyStopping(15,0.),monitor=core.MONITOR,restore_best=True)

def identity(run,job):
    run=Path(run)
    return dict(job=job,protocol=core.digest(run/'PROTOCOL.json'),freeze=core.digest(run/'FREEZE.json'),monitor=core.MONITOR,
        initial=core.digest(core.initial_path(run,job)),preprocessor=core.digest(run/'shared/preprocessing'/f'{job["fold"]}.json'),
        populations={phase:core.digest(run/'shared/populations'/f'{job["fold"]}_{phase}.parquet') for phase in ('fit','inner')})

def continuation(folder,wanted):
    for attempt in reversed(sorted(Path(folder).glob('attempt_*'))):
        for name in ('completed_training','last_training'):
            path=attempt/f'{name}.pt';receipt=attempt/f'{name}.json'
            if path.exists() and receipt.exists():
                meta=json.loads(receipt.read_text())
                if meta['identity']!=wanted or meta['sha256']!=core.digest(path):raise ValueError('Resume population, monitor, source, or initialization identity mismatch')
                return torch.load(path,map_location='cpu',weights_only=True)
    return None

def raw_table(frame,probability,rates,store,cfg,job):
    output=frame.copy().reset_index(drop=True);output['y']=core.truth(frame)
    output[metrics.P_COLUMNS]=metrics.base.normalized(probability)
    for key,value in metrics.base.outcome_rows(output[metrics.P_COLUMNS],output.y).items():output[f'raw_{key}']=value.to_numpy()
    metrics.add_rps(output)
    if rates is not None:
        y=frame[['goals_for','goals_against']].to_numpy(float)
        output[['lambda_home','lambda_away']]=rates;output[['goals_home','goals_away']]=y
        for key,value in metrics.base.count_rows(rates,y).items():output[key]=value
    issue=store.issue_times(frame,cfg);output['issuance_at']=issue.to_numpy();output['cutoff_at']=[cfg.history_cutoff(v) for v in issue]
    output['fixture_key']=['/'.join(map(str,key)) for key in core.keys(frame)]
    for name in ('fold','seed','configuration','arm'):output[name]=job[name]
    return output

def verify_progress(curve,summary):
    assert summary['monitor']==core.MONITOR
    np.testing.assert_array_equal(curve[core.MONITOR],curve.validation_loss)
    validate_training_progress(curve.to_dict('records'),summary,dict(max_steps=200,early_stopping=dict(patience=15,min_delta=0.)))

def fit(run,job_id):
    run=Path(run);protocol,store=core.load(run,frozen=True);setup(protocol)
    jobs=[j for j in protocol['matrix'] if j['job_id']==job_id]
    if len(jobs)!=1:raise ValueError('Job is absent from the fixed matrix')
    job=jobs[0];folder=run/job['path'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'fit.json').exists():raise ValueError('Completed fitting evidence cannot be overwritten')
    wanted=identity(run,job);previous=continuation(folder,wanted)
    number=len(list(folder.glob('attempt_*')))+1
    if number>3:raise RuntimeError('Engineering retry budget exhausted')
    attempt=folder/f'attempt_{number:02}';attempt.mkdir()
    core.write_json(attempt/'attempt.json',dict(identity=wanted,resumed=previous is not None,pid=os.getpid()))
    context,population=context_for(run,job)
    def observer(event):
        if not event.final:core.write_json(folder/'progress.json',dict(job=job_id,step=event.step,metrics=event.metrics,attempt=number,pid=os.getpid()))
    context.observer=observer
    def save(capture,*,complete):
        state=capture();name='completed_training' if complete else 'last_training'
        path=attempt/f'{name}.pt';temporary=path.with_suffix('.pt.tmp')
        torch.save(state,temporary);temporary.replace(path)
        core.write_json(attempt/f'{name}.json',dict(identity=wanted,sha256=core.digest(path),complete=complete))
    started=time.perf_counter()
    try:
        with MemoryProbe(protocol['resources']['rss_limit_gib']) as probe:
            preparation=[]
            def factory(seed):
                backend=AllLeagueBackend(run,protocol,store,job);initialize=backend.initialize
                def initialize_timed(ctx):
                    start=time.perf_counter();initialize(ctx);preparation.append(time.perf_counter()-start)
                backend.initialize=initialize_timed;return backend
            (backend,history,summary),seconds=timed(lambda:run_resumable_iterations(factory,context,control(job['seed']),state=previous,save=save),'cuda:0')
            curve=history.pivot(index='step',columns='metric',values='value').reset_index();verify_progress(curve,summary)
            timing=folder/'TRAINING_FINISHED.json'
            if not timing.exists():core.write_json(timing,dict(identity=wanted,fit_seconds=seconds,preparation_seconds=sum(preparation),
                cuda_allocated_peak=torch.cuda.max_memory_allocated(),cuda_reserved_peak=torch.cuda.max_memory_reserved(),memory=probe.result(),summary=summary))
            training_receipt=json.loads(timing.read_text());assert training_receipt['identity']==wanted
            history.to_csv(folder/'history_long.csv',index=False);curve.to_csv(folder/'training.csv',index=False)
            fitting,fit_replay=timed(lambda:predict(backend,backend.train_groups,population['fit']),'cuda:0')
            inner,inner_replay=timed(lambda:predict(backend,backend.validation_groups,population['inner']),'cuda:0')
            scored={};league={}
            for phase,values in [('fit',fitting),('inner',inner)]:
                table=raw_table(population[phase],values[0],values[1] if job['configuration']=='time_C' else None,store,backend.config,job)
                table.to_parquet(folder/f'{phase}_raw.parquet',index=False)
                scored[phase]=metrics.scores(table);league[phase]={str(k):metrics.scores(g) for k,g in table.groupby('competition_id')}
            best=curve.set_index('step').loc[summary['best_step']]
            np.testing.assert_allclose(scored['fit']['log_loss'],best.train_loss,rtol=1e-5,atol=1e-6)
            np.testing.assert_allclose(scored['inner']['log_loss'],best[core.MONITOR],rtol=1e-5,atol=1e-6)
            canonical=torch.load(core.initial_path(run,job),map_location='cpu',weights_only=True)
            if job['configuration']!='time_C':
                for name,value in backend.model.state_dict().items():
                    if name.startswith('goal_head.'):torch.testing.assert_close(value.cpu(),canonical[name],atol=0,rtol=0)
            adapter=TemporalGraphAdapter(backend.config,store,control=control(job['seed']),device=backend.device)
            adapter.backend_,adapter.training_history_,adapter.training_summary_=backend,history,summary
            serializer=TemporalGraphSerializer(store,device=backend.device);serializer.save(adapter,folder/'model')
            restored=serializer.load(folder/'model')
            reloaded=predict(restored.backend_,restored.backend_._stream(make_context(population['inner'])),population['inner'])
            for a,b in zip(inner[:2],reloaded[:2]):np.testing.assert_array_equal(a,b)
            for method in (restored.predict,restored.predict_goals):
                try:method(make_context(population['inner'].iloc[:1],fitting=False))
                except ValueError:pass
                else:raise AssertionError('Public selection-period availability boundary was weakened')
            result=dict(job=job,identity=wanted,summary=summary,selection_score=scored['inner']['log_loss'],raw_metrics=scored,by_league=league,
                initial_digest=backend.initial_digest,optimizer=optimizer_receipt(backend),training=training_receipt,
                fit_replay_seconds=fit_replay,inner_replay_seconds=inner_replay,total_fit_stage_seconds=time.perf_counter()-started,
                warm_update_seconds=float(curve.update_seconds.iloc[1:].mean()),parameters=sum(v.numel() for v in backend.model.parameters()),
                reload_exact=True,public_boundaries_verified=True,attempts=number,outer_scored=False,completed_at=pd.Timestamp.now(tz='UTC').isoformat())
            core.write_json(folder/'fit.json',result)
        print(json.dumps(dict(job=job_id,status='fitted_inner_only',completed=len(curve),selected=summary['best_step'])),flush=True)
    except BaseException:
        (attempt/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

def evaluate(run):
    run=Path(run);decision=selection.verify_gate(run);protocol,store=core.load(run,frozen=True);setup(protocol)
    for job in protocol['matrix']:
        folder=run/job['path']
        if (folder/'result.json').exists():
            assert json.loads((folder/'result.json').read_text())['selection_sha256']==core.digest(run/'SELECTION.json');continue
        fit_result=json.loads((folder/'fit.json').read_text());population=core.populations(run,job['fold'])
        inner=pd.read_parquet(folder/'inner_raw.parquet')
        calibrators=calibration_by_league(inner,inner[metrics.P_COLUMNS].to_numpy())
        assert set(calibrators)=={str(k) for k in protocol['competitions']}
        core.write_json(folder/'calibration.json',calibrators)
        adapter=TemporalGraphSerializer(store,device='cuda:0').load(folder/'model');backend=adapter.backend_
        context=make_context(population['outer'],fitting=False);backend._boundary(backend._validate(context))
        groups=backend._stream(context)
        outer,seconds=timed(lambda:predict(backend,groups,population['outer']),'cuda:0')
        evaluation={}
        for phase in ('fit','inner','outer'):
            if phase=='outer':probability,rates=outer[:2]
            else:
                raw=pd.read_parquet(folder/f'{phase}_raw.parquet');probability=raw[metrics.P_COLUMNS].to_numpy()
                rates=raw[['lambda_home','lambda_away']].to_numpy() if job['configuration']=='time_C' else None
            table=prediction_table(population[phase],probability,rates if job['configuration']=='time_C' else None,calibrators,store,backend.config,job)
            metrics.add_rps(table);table.to_parquet(folder/f'{phase}_predictions.parquet',index=False)
            evaluation[phase]=metrics.table_scores(table)
        core.write_json(folder/'result.json',dict(job=job,fit_sha256=core.digest(folder/'fit.json'),selection_sha256=core.digest(run/'SELECTION.json'),
            metrics=evaluation,outer_seconds=seconds,evaluated_at=pd.Timestamp.now(tz='UTC').isoformat()))
        print(json.dumps(dict(job=job['job_id'],status='evaluated_after_selection')),flush=True)
