"""Native AdamW and resumable loop, frozen initial states, and audited diagnostics."""
from dataclasses import asdict
from pathlib import Path
from copy import deepcopy
import gc
import json
import os
import time
import traceback
import numpy as np
import pandas as pd
import torch

from . import core, metrics
from xdiyo_analytics.graphs import TemporalGraphAdapter, TemporalGraphSerializer
from xdiyo_analytics.graphs.adapter import _Backend, inference_mode, torch_threads
from xdiyo_analytics.graphs.execution import configured_execution, prepare_determinism
from xdiyo_analytics.graphs.goals import auxiliary_goal_loss, goal_labels
from xdiyo_analytics.training import TrainingControl, EarlyStopping
from xdiyo_analytics.training.resume_loop import run_resumable_iterations
from xdiyo_analytics.training.native_resume import capture_backend, rng_state, restore_rng, _cpu
from xdiyo_analytics.training.residency import InputCachePolicy
from temporal_team_gat import make_context
from compare_gat_temporal import MemoryProbe, timed, write_json, digest, validate_training_progress
from compare_gat_goals import digest_state


def setup(protocol):
    resources=protocol['resources']
    torch.set_num_threads(resources['threads'])
    torch.set_float32_matmul_precision('highest')
    torch.cuda.set_per_process_memory_fraction(resources['cuda_fraction'])
    prepare_determinism(core.config(),resources['device'])
    torch.cuda.reset_peak_memory_stats()


def initial_path(run,job):
    name=f"{job['arm']}.pt" if job['arm'] in core.ARMS[3:] else 'canonical.pt'
    return Path(run)/'shared/initialization'/job['fold']/job['mode']/f"seed_{job['seed']}"/name


class SuiteBackend(_Backend):
    def __init__(self,run,protocol,store,job):
        self.suite_run,self.job=Path(run),job
        cfg=core.config(job['mode'],job['arm'],tuple(protocol['competitions']))
        if 'configurations' in protocol:
            if json.dumps(asdict(cfg),sort_keys=True)!=json.dumps(protocol['configurations'][job['mode']][job['arm']],sort_keys=True):
                raise ValueError('Worker configuration disagrees with frozen specification.')
        state=json.loads((self.suite_run/'shared/preprocessing'/f"{job['fold']}.json").read_text())['state']
        r=protocol['resources']
        super().__init__(cfg,store,job['seed'],r['device'],InputCachePolicy(r['input_cache_bytes'],r['headroom_bytes']),preprocessor_state=state)

    def initialize(self,context):
        super().initialize(context)
        initial=torch.load(initial_path(self.suite_run,self.job),map_location=self.device,weights_only=True)
        self.model.load_state_dict(initial)
        self.initial_state_digest=digest_state(self.model.state_dict())
        self.refresh_version()

    def checkpoint_metrics(self,monitor,values):
        values=super().checkpoint_metrics('train_loss',values)
        if self.model.goal_head is not None and hasattr(self.model.goal_head,'raw_alpha') and self.model.goal_head.raw_alpha is not None:
            alpha=float(torch.nn.functional.softplus(self.model.goal_head.raw_alpha).detach())
            values.update(alpha=alpha,kappa=1/alpha)
        if not all(np.isfinite(v) for v in values.values()):
            raise FloatingPointError('Nonfinite objective or metric; fit invalid, no clamping.')
        if hasattr(self,'_update_started'):
            if str(self.device).startswith('cuda'):torch.cuda.synchronize(self.device)
            values['update_seconds']=time.perf_counter()-self._update_started
        return values

    def step(self):
        self._update_started=time.perf_counter()
        return super().step()


@configured_execution
def gradients(backend):
    model=backend.model
    names,params=zip(*[(n,p) for n,p in model.named_parameters() if not n.startswith(('head.','goal_head.'))])
    before=digest_state(capture_backend(backend)); rng=rng_state(backend.device); training=model.training
    accum=[[torch.zeros_like(p) for p in params] for _ in range(2)]
    try:
        model.eval()
        for i in range(len(backend.train_groups)):
            with backend._group_lease(backend.train_groups,i) as (group,_):
                positions,data,h,a,f=group; ix=torch.as_tensor(positions,device=backend.device)
                logits,details=model(data,h,a,f,return_goal_details=True)
                weights=backend.train_weights[ix]/backend.train_weights.sum()
                outcome=(torch.nn.functional.cross_entropy(logits,backend.train_targets[ix],reduction='none')*weights).sum()
                goal=(auxiliary_goal_loss(model,details,backend.train_goals[ix])*weights).sum()
                values=[torch.autograd.grad(outcome,params,retain_graph=True,allow_unused=True),
                        torch.autograd.grad(goal,params,allow_unused=True)]
                for row,items in zip(accum,values):
                    for total,grad in zip(row,items):
                        if grad is not None: total.add_(grad.detach())
        x,y=[torch.cat([v.flatten().double() for v in row]) for row in accum]
        a,b=float(x.norm()),float(y.norm()); beta=backend.config.goal_auxiliary.beta
        result=dict(outcome_norm=a,goal_norm=b,weighted_goal_norm=beta*b,weighted_gradient_ratio=beta*b/a if a else None,
                    cosine_similarity=float(torch.dot(x,y)/(a*b)) if a and b else None,
                    zero_norms=[name for name,value in [('outcome',a),('goal',b)] if value==0],
                    parameter_names=names,parameter_count=sum(p.numel() for p in params),population=len(backend.train_targets),
                    regularizer_included=False)
    finally:
        model.train(training)
        assert digest_state(rng_state(backend.device))==digest_state(rng), 'Diagnostics changed RNG state'
        assert digest_state(capture_backend(backend))==before, 'Diagnostics changed model/optimizer/gradients/mode'
    return result


@configured_execution
def predict(backend,groups,frame):
    n=len(frame); probabilities=np.empty((n,3)); rates=np.empty((n,2)); extras={}
    with inference_mode(backend.model):
        for i in range(len(groups)):
            with backend._group_lease(groups,i) as (group,_):
                positions,data,h,a,f=group
                logits,details=backend.model(data,h,a,f,return_goal_details=True)
                probabilities[positions]=logits.softmax(1).cpu().numpy()
                rates[positions]=details['rates'].cpu().numpy()
                for key in ('attack','defence','scoring','home_effect','log_rates'):
                    if key not in details: continue
                    value=details[key].cpu().numpy()
                    if key not in extras: extras[key]=np.empty((n,*value.shape[1:]))
                    extras[key][positions]=value
    return probabilities,rates,extras


def context_for(run,job):
    frames=core.frames(run,job['fold'])
    train=frames['recent' if job['arm']=='A_recent' else 'expanded']
    ctx=make_context(train); ctx.validation=make_context(frames['inner'])
    return ctx,dict(train=train,inner=frames['inner'],outer=frames['outer'])


def optimizer_receipt(backend):
    names={id(p):n for n,p in backend.model.named_parameters()}
    return [dict(lr=g['lr'],weight_decay=g['weight_decay'],parameters=[names[id(p)] for p in g['params']]) for g in backend.optimizer.param_groups]


def fit(run,job_id):
    run=Path(run); protocol,store=core.load(run,frozen=True)
    matches=[r for r in protocol['matrix'] if r['job_id']==job_id]
    if len(matches)!=1: raise ValueError('Fit request absent from frozen manifest.')
    job=matches[0]; folder=run/job['path']; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists(): raise ValueError('Completed fit cannot be overwritten.')
    attempts=sorted(folder.glob('attempt_*')); number=len(attempts)+1
    if number>3: raise RuntimeError('Engineering retry budget exhausted.')
    attempt=folder/f'attempt_{number:02d}'; attempt.mkdir()
    identity=dict(job=job,protocol=digest(run/'MASTER_PROTOCOL.json'),source=digest(run/'frozen_source.zip'),initial=digest(initial_path(run,job)))
    setup(protocol)
    previous=None
    if attempts:
        last=attempts[-1]/'last_training.pt'; receipt=attempts[-1]/'last_training.json'
        if last.exists() and receipt.exists():
            meta=json.loads(receipt.read_text())
            if meta['identity']==identity and meta['sha256']==digest(last):
                previous=torch.load(last,map_location='cpu',weights_only=True)
    write_json(attempt/'attempt.json',dict(identity=identity,resume_from=str(attempts[-1]) if previous is not None else None,
        started=pd.Timestamp.now(tz='UTC').isoformat(),pid=os.getpid()))
    context,populations=context_for(run,job)
    rows=[]
    def observer(event):
        if not event.final:
            write_json(folder/'progress.json',dict(job_id=job_id,step=event.step,metrics=event.metrics,pid=os.getpid(),attempt=number))
    context.observer=observer
    def save(capture,*,complete):
        state=capture()
        name='completed_training' if complete else 'last_training'
        temp=attempt/(name+'.pt.tmp'); torch.save(state,temp); temp.replace(attempt/(name+'.pt'))
        write_json(attempt/(name+'.json'),dict(identity=identity,sha256=digest(attempt/(name+'.pt')),complete=complete))
    control=TrainingControl(max_steps=200,seed=job['seed'],early_stopping=EarlyStopping(15,0.),monitor='validation_loss',restore_best=True)
    backend=SuiteBackend(run,protocol,store,job)
    start=time.perf_counter()
    try:
        with MemoryProbe(protocol['resources']['rss_limit_gib']) as probe:
            _,prep_seconds=timed(lambda:backend.initialize(context),'cuda:0')
            initial_grad,initial_gradient_seconds=timed(lambda:gradients(backend),'cuda:0')
            initial_digest=backend.initial_state_digest
            initial_predictions=predict(backend,backend.train_groups,populations['train'])
            initial_receipt=dict(state=initial_digest,outcome_sha256=digest_state(initial_predictions[0]),rates_sha256=digest_state(initial_predictions[1]),
                rate_means=initial_predictions[1].mean(0).tolist(),rate_min=initial_predictions[1].min(0).tolist(),rate_max=initial_predictions[1].max(0).tolist(),
                rate_quantile_levels=[0,.01,.05,.25,.5,.75,.95,.99,1],
                rate_quantiles=np.quantile(initial_predictions[1],[0,.01,.05,.25,.5,.75,.95,.99,1],axis=0).tolist())
            # The native resumable loop owns initialization; reuse this initialized
            # backend for its first initialize call only, with state still canonical.
            original_initialize=backend.initialize
            backend.initialize=lambda ctx: setattr(backend,'initialize',original_initialize)
            factory=lambda seed:backend
            (selected,history,summary),fit_seconds=timed(lambda:run_resumable_iterations(factory,context,control,state=previous,save=save),'cuda:0')
            backend=selected
            curve=history.pivot(index='step',columns='metric',values='value').reset_index()
            progress=curve.to_dict('records')
            validate_training_progress(progress,summary,dict(max_steps=200,early_stopping=dict(patience=15,min_delta=0.)))
            selected_grad,selected_gradient_seconds=timed(lambda:gradients(backend),'cuda:0')
            predictions={}
            replay_start=time.perf_counter()
            for split,groups in [('train',backend.train_groups),('inner',backend.validation_groups)]:
                predictions[split]=predict(backend,groups,populations[split])
            replay_seconds=time.perf_counter()-replay_start
            inner_goals=backend.validation_goals.cpu().numpy()
            inner_y=np.where(inner_goals[:,0]>inner_goals[:,1],0,np.where(inner_goals[:,0]==inner_goals[:,1],1,2))
            calibrator=metrics.draw_calibration(predictions['inner'][0],inner_y)
            calibrator.update(population_sha256=core.frame_hash(populations['inner']),trained_through=str(backend.trained_through))
            write_json(folder/'calibration.json',calibrator)
            if backend.config.goal_auxiliary.architecture=='attack_defence':
                extra=predictions['train'][2]
                offsets=np.array([extra['attack'].mean(),extra['defence'].mean()])
                backend.model.goal_head.centering_offsets.copy_(torch.tensor(offsets,device=backend.device))
                write_json(folder/'centering.json',dict(attack=float(offsets[0]),defence=float(offsets[1]),
                    population_sha256=core.frame_hash(populations['train']),team_appearances=2*len(populations['train'])))
                backend.refresh_version()
            outer=make_context(populations['outer'],fitting=False)
            backend._boundary(backend._validate(outer))
            outer_groups=backend._stream(outer)
            predictions['outer'],outer_seconds=timed(lambda:predict(backend,outer_groups,populations['outer']),'cuda:0')
            family=backend.config.goal_auxiliary.likelihood
            alpha=float(torch.nn.functional.softplus(backend.model.goal_head.raw_alpha).detach()) if family!='poisson' else 0.
            evaluated={}
            for split,(prob,rates,extra) in predictions.items():
                frame=populations[split]
                goals=goal_labels(store,frame,store.label_availability(frame,backend.config).max())
                table=metrics.prediction_table(frame,prob,rates,goals,calibrator,family=family,alpha=alpha)
                issue=store.issue_times(frame,backend.config)
                table['issuance_at']=issue.to_numpy(); table['cutoff_at']=[backend.config.history_cutoff(t) for t in issue]
                if 'attack' in extra:
                    ca,cd=backend.model.goal_head.centering_offsets.cpu().numpy()
                    for j,side in enumerate(('home','away')):
                        table[f'attack_{side}']=extra['attack'][:,j]-ca
                        table[f'defence_{side}']=extra['defence'][:,j]-cd
                    table['scoring_centered']=extra['scoring']+ca-cd; table['home_effect']=extra['home_effect']
                    neutral=frame.get('neutral_venue',pd.Series(np.nan,index=frame.index)).to_numpy()
                    centered=np.column_stack((table.scoring_centered+(neutral!=1)*table.home_effect+table.attack_home-table.defence_away,
                                              table.scoring_centered+table.attack_away-table.defence_home))
                    np.testing.assert_allclose(np.exp(centered),rates,rtol=2e-6,atol=1e-6)
                table.to_parquet(folder/f'{split}_predictions.parquet',index=False)
                evaluated[split]=metrics.table_metrics(table)
            for split,native in [('train','train'),('inner','validation')]:
                np.testing.assert_allclose(evaluated[split]['raw']['log_loss'],curve.set_index('step').loc[summary['best_step'],f'{native}_loss'],rtol=1e-5,atol=1e-6)
            adapter=TemporalGraphAdapter(backend.config,store,control=control,device=backend.device)
            adapter.backend_,adapter.training_history_,adapter.training_summary_=backend,history,summary
            serializer=TemporalGraphSerializer(store,device=backend.device); serializer.save(adapter,folder/'model')
            restored=serializer.load(folder/'model')
            restored_out=predict(restored.backend_,restored.backend_._stream(outer),populations['outer'])
            np.testing.assert_array_equal(predictions['outer'][0],restored_out[0]); np.testing.assert_array_equal(predictions['outer'][1],restored_out[1])
            for key,value in predictions['outer'][2].items(): np.testing.assert_array_equal(value,restored_out[2][key])
            for method in (restored.predict,restored.predict_goals):
                try: method(make_context(populations['inner'],fitting=False))
                except ValueError: pass
                else: raise AssertionError('Public selection-period boundary weakened.')
            history.to_csv(folder/'history_long.csv',index=False); curve.to_csv(folder/'training.csv',index=False)
            write_json(folder/'gradients.json',dict(initial=initial_grad,selected=selected_grad))
            write_json(folder/'initialization.json',initial_receipt)
            result=dict(job=job,identity=identity,summary=summary,metrics=evaluated,initial=initial_receipt,
                gradients=dict(initial=initial_grad,selected=selected_grad),optimizer_groups=optimizer_receipt(backend),
                preprocessing_seconds=prep_seconds,fit_seconds=fit_seconds,inner_replay_seconds=replay_seconds,
                warm_update_seconds=float(curve.update_seconds.iloc[1:].mean()),
                initial_gradient_seconds=initial_gradient_seconds,selected_gradient_seconds=selected_gradient_seconds,
                outer_prediction_seconds=outer_seconds,total_seconds=time.perf_counter()-start,
                memory=probe.result(),cuda_allocated_peak=torch.cuda.max_memory_allocated(),cuda_reserved_peak=torch.cuda.max_memory_reserved(),
                parameters=sum(p.numel() for p in backend.model.parameters()),goal_parameters=sum(p.numel() for p in backend.model.goal_head.parameters()),
                outcome_parameters=sum(p.numel() for p in backend.model.head.parameters()),shared_parameters=selected_grad['parameter_count'],
                alpha=alpha,reload_exact=True,public_boundaries_verified=True,
                attempts=number,completed_at=pd.Timestamp.now(tz='UTC').isoformat())
            write_json(folder/'result.json',result)
        print(json.dumps(dict(job=job_id,status='completed',steps=summary['attempts'][0]['steps'],selected=summary['best_step'],seconds=result['total_seconds'])),flush=True)
    except BaseException:
        (attempt/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise


def profile(run,mode):
    run=Path(run); p,store=core.load(run); setup(p)
    job=dict(fold='fold_2',mode=mode,arm='C_flexible',seed=17)
    ctx,pop=context_for(run,job); backend=SuiteBackend(run,p,store,job)
    print(json.dumps(dict(profile=mode,stage='initializing',pid=os.getpid())),flush=True)
    with MemoryProbe(p['resources']['rss_limit_gib']) as probe:
        _,prep=timed(lambda:backend.initialize(ctx),'cuda:0')
        print(json.dumps(dict(profile=mode,stage='initialized',seconds=prep)),flush=True)
        steps=[]
        for step in range(3):
            _,elapsed=timed(lambda:backend.checkpoint_metrics('validation_loss',backend.step()),'cuda:0');steps.append(elapsed)
            print(json.dumps(dict(profile=mode,step=step+1,seconds=elapsed)),flush=True)
        result=dict(disposable=True,outer_evaluated=False,mode=mode,expanded_fixtures=len(pop['train']),inner_fixtures=len(pop['inner']),
            preprocessing_seconds=prep,update_seconds=steps,warm_update_seconds=float(np.mean(steps[1:])),memory=probe.result(),
            cuda_allocated_peak=torch.cuda.max_memory_allocated(),cuda_reserved_peak=torch.cuda.max_memory_reserved())
    write_json(run/'preflight'/f'profile_{mode}.json',result);print(json.dumps(result),flush=True)


def input_map(backend):
    result={};common={}
    for i in range(len(backend.train_groups)):
        with backend._group_lease(backend.train_groups,i) as (group,_):
            positions,data,h,a,f=group
            normal={k:v for k,v in data.items() if not k.startswith(('goal_','temporal_'))}
            value=digest_state((normal,h,a,f))
            shared=dict(normal)
            if backend.config.temporal_mode!='snapshot': shared['x']=shared['x'][:,:-14]
            common_value=digest_state((shared,h,a,f))
            for key in backend.train_groups.plan.metadata.iloc[positions][list(core.MATCH_KEYS)].itertuples(index=False,name=None):
                result[tuple(key)]=value; common[tuple(key)]=common_value
    return result,common


def pairing(run):
    run=Path(run);p,store=core.load(run);setup(p);receipts={}
    for fold in ('fold_1','fold_2'):
        common_reference={}
        for mode in core.MODES:
            reference=None
            for arm in core.ARMS:
                job=dict(fold=fold,mode=mode,arm=arm,seed=17)
                ctx,pop=context_for(run,job);backend=SuiteBackend(run,p,store,job);backend.initialize(ctx)
                mapping,common=input_map(backend)
                if reference is None: reference=mapping
                else:
                    assert all(mapping[k]==v for k,v in reference.items())
                    reference=mapping
                if arm=='B_expanded':
                    if mode=='snapshot': common_reference=common
                    else: assert common==common_reference
                assert torch.unique(backend.train_weights).numel()==1
                receipts[f'{fold}/{mode}/{arm}']=dict(fixtures=len(mapping),encoded_sha256=digest_state(mapping),
                    common_initial_sha256=digest_state({k:v for k,v in backend.model.state_dict().items() if not k.startswith('goal_head.')}),
                    preprocessor_sha256=digest_state(backend.preprocessor.state()))
                del backend;gc.collect();torch.cuda.empty_cache()
            print(json.dumps(dict(pairing=fold,mode=mode,arms_verified=6)),flush=True)
    write_json(run/'preflight/pairing.json',receipts)
