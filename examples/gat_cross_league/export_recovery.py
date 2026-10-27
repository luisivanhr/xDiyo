"""Export an already completed native checkpoint; optimizer updates are forbidden."""
from pathlib import Path
import json,shutil,time
import numpy as np
import pandas as pd
import torch
from scipy.special import gammaln
from . import core
from .execution import CrossBackend,context_for,calibration_by_league,prediction_table,per_league_metrics
from .initialization import initial_path
from gat_suite import metrics
from gat_suite.fitting import setup,predict,optimizer_receipt
from compare_gat_temporal import MemoryProbe,timed,write_json,digest,validate_training_progress
from compare_gat_goals import digest_state
from temporal_team_gat import make_context
from xdiyo_analytics.graphs import TemporalGraphAdapter,TemporalGraphSerializer
from xdiyo_analytics.training import TrainingControl,EarlyStopping
from xdiyo_analytics.training.resume_loop import run_resumable_iterations


def recover(run,job_id):
    run=Path(run);p,store=core.load(run,frozen=True)
    job=next(j for j in p['matrix'] if j['job_id']==job_id)
    folder=run/job['path']
    assert not (folder/'result.json').exists()
    # Prefer the first completed trajectory, never a more favorable retry.
    candidates=sorted(folder.glob('attempt_*/completed_training.pt'))
    if not candidates:raise ValueError('No completed training checkpoint')
    checkpoint=candidates[0];receipt=json.loads(checkpoint.with_suffix('.json').read_text())
    identity=receipt['identity']
    assert identity['job']==job and identity['protocol']==digest(run/'PROTOCOL.json')
    assert identity['freeze']==digest(run/'FREEZE.json')
    assert receipt['sha256']==digest(checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert state['active'] is None and state['attempt']==1 and len(state['attempts'])==1
    audit=folder/'export_recovery';audit.mkdir(exist_ok=True)
    for name in ('fit_predictions.parquet','inner_predictions.parquet','outer_predictions.parquet','calibration.json','fitting_contributions.csv','training.csv','history_long.csv','progress.json'):
        path=folder/name
        if path.exists() and not (audit/name).exists():shutil.copyfile(path,audit/name)
    setup(p);context,pop=context_for(run,job)
    def forbidden(*args,**kwargs):raise AssertionError('Completed export recovery attempted an optimizer update or checkpoint write')
    context.observer=forbidden
    def factory(seed):
        backend=CrossBackend(run,p,store,job);backend.step=forbidden;return backend
    control=TrainingControl(max_steps=200,seed=job['seed'],early_stopping=EarlyStopping(15,0.),monitor='validation_loss',restore_best=True)
    start=time.perf_counter();number=len(list(folder.glob('attempt_*')))
    # The internal fit timer did not survive the export exception. Preserve a
    # clearly labelled wall-time estimate from the original durable receipts.
    fit_seconds=checkpoint.with_suffix('.json').stat().st_mtime-(checkpoint.parent/'attempt.json').stat().st_mtime
    with MemoryProbe(p['resources']['rss_limit_gib']) as probe:
        backend,history,summary=run_resumable_iterations(factory,context,control,state=state,save=forbidden)
        weights_before=digest_state(backend.model.state_dict())
        assert weights_before==digest_state(state['winner']['model'])
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
            fit_seconds=fit_seconds,preprocessing_seconds=None,inner_seconds=inner_seconds,
            outer_seconds=outer_seconds,selected_fit_replay_seconds=fitting_seconds,total_seconds=time.perf_counter()-start,
            warm_update_seconds=float(curve.update_seconds.iloc[1:].mean()),
            memory=dict(sampled_peak_rss_bytes=None,source='Original training peak was not persisted before export failure'),cuda_allocated_peak=None,cuda_reserved_peak=None,
            parameters=sum(v.numel() for v in backend.model.parameters()),goal_parameters=sum(v.numel() for v in backend.model.goal_head.parameters()),
            reload_exact=True,public_boundaries_verified=True,attempts=number,completed_at=pd.Timestamp.now(tz='UTC').isoformat())


        assert digest_state(backend.model.state_dict())==weights_before
        assert digest(checkpoint)==receipt['sha256']
        prior=audit/'fit_predictions.parquet'
        if prior.exists():
            old=pd.read_parquet(prior);new=pd.read_parquet(folder/'fit_predictions.parquet')
            assert core.keys(old).equals(core.keys(new))
            np.testing.assert_array_equal(old[metrics.P_COLUMNS],new[metrics.P_COLUMNS])
            np.testing.assert_array_equal(old[metrics.C_COLUMNS],new[metrics.C_COLUMNS])
        result['recovery_seconds']=time.perf_counter()-start
        result['total_seconds']=fit_seconds+result['recovery_seconds']
        result['recovery_memory']=probe.result()
        result['recovery_cuda_allocated_peak']=torch.cuda.max_memory_allocated()
        result['recovery_cuda_reserved_peak']=torch.cuda.max_memory_reserved()
        result['fit_seconds_source']='Completed training receipt mtime minus original attempt receipt mtime; internal timer unavailable'
        latest=max(path.stat().st_mtime for path in folder.glob('attempt_*/last_training.json'))
        result['engineering_active_elapsed_seconds']=max(0.,latest-checkpoint.with_suffix('.json').stat().st_mtime)
        result['original_training_peak_available']=False
        result['export_recovery']=dict(checkpoint=str(checkpoint.relative_to(run)),sha256=receipt['sha256'],optimizer_updates=0,
            selected_parameters_unchanged=True,preserved_fit_probabilities_exact=True,
            abandoned_retry='attempt_03 was inadvertently reinitialized after attempt_02 completed-state export failed; its 14-update partial trajectory is preserved and excluded')
        result['scoring_repair']=digest(run/'SCORING_REPAIR.json')
        write_json(folder/'result.json',result)
        write_json(audit/'receipt.json',result['export_recovery'])
    print(json.dumps(dict(job=job_id,status='recovered_exports',updates=0,selected=summary['best_step'],completed=summary['attempts'][0]['steps'])),flush=True)
