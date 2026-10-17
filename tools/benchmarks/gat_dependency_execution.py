"""Disposable benchmark of frozen combined 74-target GAT execution.

One fresh process per arm/mode. No full fit, checkpoint selection or research
continuation. Canonical input hashes are pinned; only a new external output
directory is written. Historical validation predictions are labeled replays.
"""
from pathlib import Path
import argparse, datetime, gc, hashlib, json, os, platform, sys, subprocess, threading, time
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import asdict, replace
sys.dont_write_bytecode=True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
SOURCE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE/'src'))
COMMIT='0cb13cac57255f5e2f8e0eeb692e2b6f22344381'
# Essential loader copied verbatim from the author recipe; the reference module
# is never imported and therefore cannot inject its frozen source checkout.
HASHES = {
 'INPUT_MANIFEST.json':'89054f75169d44ba01fa1fd1668f2636c8129a05d9a0c33b8b7c115b83166f08',
 'baseline/model_history.parquet':'9e7e1d89a5dd57c1b71b14b469a417b9a98f5c735bb490ef195d309c7f3d6e51',
 'baseline/fixtures.parquet':'153c01006b2a2a863701d1a8ee018db4c7f9fb66783e9803ee7b2aeffd9c59af',
 'baseline/memberships_schedule_effective.parquet':'d3b84e4c7310c88a58cb0ecbf6c81da7797bcb6091c39216d760e35989991308',
 'edge_stats_totals.parquet':'892946be0f2a8457388566819d87a1560b63da66c0829a063da9eb7699d49c6e',
 'fixture_head.parquet':'cb26859190a4a5abca3d83853af5acc6122e495f85f0a6c1fcfcd39a9118d130',
 'baseline/selection/fit.parquet':'5a65fc9db664da776908297dbd68304603c544d857144d5dcfb00a69a8594661',
 'baseline/selection/validation.parquet':'0cb7d9f390d14e758231590c355d0cb331701e6cfea2b8bc87ba0b665f90df1b',
 'baseline/selection/purged_selection_only.parquet':'ff20204a97998b15d6555597603b52d1353fc984e934bad9a887ee3c2d857cd4',
 'long200_aux2_plain_selection.json':'e81e5b928c15d3c4ea6ca7cc0b508a66e8c0f42e8793da7a782ae68e27ac4c87',
 'long200_aux2_bayes11_selection.json':'05b873e71a7ca2260eab2981d004e602ac793cf9f9ef66a6dae58c9cf434bf70',
 'runtime_d1219063_long200_aux2/runner/data_adapter.py':'fd53d7f1be1bd9ce86ea585bf7e7472e286e5cffee249089a56f0155f846f482',
}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def recipe(arm):
    from xdiyo_analytics.graphs import TemporalGraphConfig,StatisticsForecast
    from xdiyo_analytics.training import TrainingControl
    old=json.loads((OLD/f'long200_aux2_{arm}_selection.json').read_text())
    cfg=dict(old['config']); sf=dict(cfg['statistics_forecast'])
    sf.update(route='combined',weight=1.0)
    cfg['statistics_forecast']=StatisticsForecast.from_dict(sf)
    config=TemporalGraphConfig(**cfg)
    controls=dict(old['control']); controls.update(max_steps=100,monitor='validation_loss')
    control=TrainingControl(**controls)
    assert config.boundary_memory is None and config.heatmaps is None
    assert config.forecast_policy=='by_round' and config.deterministic
    assert len(config.statistics)==72 and len(config.statistics_forecast.targets)==74
    assert all(t.quantiles==(.25,.5,.75) for t in config.statistics_forecast.targets)
    assert control.seed==17 and control.early_stopping is None and control.scheduler is None and control.restarts==0 and control.restore_best
    return config,control

def construct(arm,device='cuda:0'):
    import runpy,numpy as np,pandas as pd
    from xdiyo_analytics.graphs import TemporalGraphAdapter,TemporalEventStore
    from xdiyo_analytics.graphs.config import MATCH_KEYS
    from xdiyo_analytics.training.contracts import FitContext
    config,control=recipe(arm)
    helpers=runpy.run_path(str(OLD/'runtime_d1219063_long200_aux2/runner/data_adapter.py'))
    fixtures=pd.read_parquet(OLD/'baseline/fixtures.parquet')
    history=pd.read_parquet(OLD/'baseline/model_history.parquet')
    members=pd.read_parquet(OLD/'baseline/memberships_schedule_effective.parquet')
    history=helpers['attach_edge_stats'](history,pd.read_parquet(OLD/'edge_stats_totals.parquet'),list(config.statistics))
    inputs=json.loads((OLD/'INPUT_MANIFEST.json').read_text())
    store=TemporalEventStore(history,members,schedule=fixtures,provenance=dict(input_manifest_sha256=sha(OLD/'INPUT_MANIFEST.json'),source_commit=COMMIT,arm_family='totals',availability_policy=inputs['edge_availability_policy']))
    head=pd.read_parquet(OLD/'fixture_head.parquet') if config.fixture_features else None
    def context(name):
        frame=pd.read_parquet(OLD/f'baseline/selection/{name}.parquet').set_index('cohort_position')
        X=pd.DataFrame(index=frame.index); meta=frame.copy()
        if config.fixture_features:
            meta=helpers['attach_fixture_head'](meta,head,list(config.fixture_features))
            X=meta[list(config.fixture_features)].copy()
        return FitContext(X=X,y=pd.DataFrame({'result':np.sign(frame.goals_for-frame.goals_against).astype(int)},index=frame.index),metadata=meta,layout='match',match_columns=MATCH_KEYS,fold_id=0)
    ctx=context('fit'); ctx.validation=context('validation')
    assert len(ctx.X)==7542 and len(ctx.validation.X)==1756
    return TemporalGraphAdapter(config,store,control=control,device=device,inner_threads=1),ctx

def model_counts(backend):
    m=backend.model; p=backend.preprocessor
    return dict(parameters=sum(x.numel() for x in m.parameters()),trainable_parameters=sum(x.numel() for x in m.parameters() if x.requires_grad),edge_numeric_channels=len(p.edge_columns),edge_encoded_width=m.layers[0].e.in_features,fixture_features=len(p.fixture_columns),auxiliary_targets=len(m.statistics_config.targets),auxiliary_outputs=sum(m.statistics_head.sizes),outcome_head_input_width=m.head[0].in_features,statistics_head_input_width=m.statistics_head.head[0].in_features,league_categories=len(p.leagues),training_groups=len(backend.train_groups),validation_groups=len(backend.validation_groups))


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def verify():
    actual={name:sha(OLD/name) for name in HASHES}
    if actual!=HASHES:raise ValueError('Canonical input hashes differ')
    return actual

class Memory:
    def __enter__(self):
        import psutil
        self.process=psutil.Process();self.stop=threading.Event();self.peak={};self.lock=threading.Lock()
        def sample():
            m=self.process.memory_info()
            with self.lock:
                for k in ('rss','private','vms'):
                    if hasattr(m,k):self.peak[k]=max(self.peak.get(k,0),getattr(m,k))
        self.sample=sample;sample()
        def run():
            while not self.stop.wait(.002):sample()
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start();return self
    def __exit__(self,*exc):
        self.stop.set();self.thread.join();self.sample();self.os_info=self.process.memory_info()._asdict()
    def reset(self):
        with self.lock:self.peak={}
        self.sample()
    def snapshot(self):
        with self.lock:return dict(self.peak)

def cpu(value):
    import torch
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return type(value)(cpu(v) for v in value)
    return value

def replay(backend,capture=False):
    import torch
    from xdiyo_analytics.graphs.adapter import inference_mode
    from xdiyo_analytics.graphs.execution import deterministic_execution
    from xdiyo_analytics.graphs.packing import stream_batches
    groups=backend.validation_groups
    output=None
    if capture:
        n=len(groups.plan.metadata)
        output=dict(logits=torch.empty((n,3)),probabilities=torch.empty((n,3)),quantiles=torch.empty((n,74,3)))
    scalars=[]
    with deterministic_execution(backend.config,backend.device),inference_mode(backend.model):
        for group,_,_ in stream_batches(backend,groups):
            pos,data,home,away,fixture=group
            logits,_,forecasts=backend.model(data,home,away,fixture,return_statistics=True)
            probabilities=logits.softmax(-1)
            scalars.append(probabilities[:,0].sum().detach())
            if capture:
                output['logits'][pos]=logits.cpu();output['probabilities'][pos]=probabilities.cpu()
                output['quantiles'][pos]=torch.stack([f['quantiles'] for f in forecasts],1).cpu()
            del group,data,home,away,fixture,logits,forecasts,probabilities
    return sum(torch.stack(scalars).cpu().tolist()),output

def main():
    global OLD
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=('plain','bayes11'),required=True)
    p.add_argument('--mode',choices=('native','dependency','dependency_lean','dependency_packed'),required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--steps',type=int,default=1);p.add_argument('--state',action='store_true')
    p.add_argument('--cold-repeats',type=int,default=3)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--concurrent-gpu-active',action='store_true',help='Record caller GPU preflight finding; does not change execution')
    p.add_argument('--host-memory-bytes',type=int,default=128*1024**2)
    p.add_argument('--device-memory-bytes',type=int,default=256*1024**2)
    p.add_argument('--headroom-bytes',type=int,default=1024**3);p.add_argument('--max-graphs',type=int,default=8)
    args=p.parse_args();OLD=args.input_root.resolve();output=args.output.resolve()
    if args.repeats<1 or args.steps<1 or args.cold_repeats<1:p.error('repeats, cold-repeats and steps must be positive')
    if output.is_relative_to(OLD) or output.is_relative_to(SOURCE):p.error('output must be external to source and canonical inputs')
    output.mkdir(parents=True,exist_ok=False)
    started=utc();wall_started=time.perf_counter();inputs=verify()
    import torch
    from xdiyo_analytics.graphs.config import GraphExecution
    from xdiyo_analytics.graphs.adapter import _Backend,inference_mode
    from xdiyo_analytics.graphs.packing import stream_batches
    from xdiyo_analytics.graphs.profiling import GraphProfile
    from xdiyo_analytics.graphs.execution import deterministic_execution
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    def flags():
        return dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    caller_flags_before=flags()
    def sync():
        if args.device.startswith('cuda'):torch.cuda.synchronize(args.device)
    def timed(call):
        sync();start=time.perf_counter();value=call();sync();return value,time.perf_counter()-start
    records=[]
    with Memory() as memory:
        def initialize():
            adapter,ctx=construct(args.arm,args.device)
            adapter.control=replace(adapter.control,seed=args.seed)
            cfg=replace(adapter.config,execution=GraphExecution(mode=args.mode,
                host_memory_bytes=args.host_memory_bytes,device_memory_bytes=args.device_memory_bytes,
                headroom_bytes=args.headroom_bytes,max_graphs=args.max_graphs))
            backend=_Backend(cfg,adapter.event_store,adapter.control.seed,args.device)
            backend.initialize(ctx)
            return adapter,ctx,backend
        (adapter,ctx,backend),initialization_seconds=timed(initialize)
        print(json.dumps(dict(progress='initialized',seconds=initialization_seconds,utc=utc())),flush=True)
        with deterministic_execution(backend.config,backend.device):effective_execution_flags=flags()
        initial=cpu(backend.model.state_dict())
        initialization_host_peak=memory.snapshot()
        memory.reset()
        backend.preprocessor._edge_cache=None;backend.preprocessor._map_cache=None;gc.collect()
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(args.device)
        def preparation():
            counts=[];traces=[]
            for groups in (backend.train_groups,backend.validation_groups):
                total=0
                for group,indices,_ in stream_batches(backend,groups):total+=len(indices);del group
                counts.append(total);traces.append(deepcopy(getattr(backend,'_execution_trace',None)))
            return counts,traces
        cold_records=[]
        for cold_repeat in range(args.cold_repeats):
            # Logical source-cache cold only. OS file caches and fitted model,
            # scalers, labels and observations are deliberately not reset.
            backend.preprocessor._edge_cache=None
            backend.preprocessor._map_cache=None
            gc.collect()
            (cold_counts,cold_traces),cold_seconds=timed(preparation)
            cold_records.append(dict(repeat=cold_repeat,seconds=cold_seconds,
                counts=cold_counts,traces=cold_traces))
            print(json.dumps(dict(progress='cold_preparation',repeat=cold_repeat,seconds=cold_seconds,counts=cold_counts,utc=utc())),flush=True)
        cold_host_peak=memory.snapshot()
        cold_cuda=None if not args.device.startswith('cuda') else dict(allocated=torch.cuda.max_memory_allocated(args.device),reserved=torch.cuda.max_memory_reserved(args.device))
        fixed_capture_seconds=None
        initial_export_seconds=None
        if args.state:
            def capture_initial():
                _,fixed=replay(backend,True)
                components={}
                with deterministic_execution(backend.config,backend.device),inference_mode(backend.model):
                    for name,groups,targets,weights in (
                        ('train',backend.train_groups,backend.train_targets,backend.train_weights),
                        ('validation',backend.validation_groups,backend.validation_targets,backend.validation_weights)):
                        outcome=backend._loss(groups,targets,weights)
                        auxiliary=backend._statistics_loss
                        penalty=float(backend.config.prior_penalty*(backend.model.league.weight.square().sum()+backend.model.entry.weight.square().sum()))
                        components[name]=dict(outcome=outcome,statistics=auxiliary,prior_penalty=penalty,
                            objective=outcome+backend.config.statistics_forecast.weight*auxiliary+(penalty if name=='train' else 0.))
                return fixed,components
            (fixed,initial_components),fixed_capture_seconds=timed(capture_initial)
            export_started=time.perf_counter()
            torch.save(dict(model=initial,fixed_validation=fixed,components=initial_components),output/'initial-state.pt');del fixed
            initial_export_seconds=time.perf_counter()-export_started
            print(json.dumps(dict(progress='initial_state_captured',capture_seconds=fixed_capture_seconds,export_seconds=initial_export_seconds,utc=utc())),flush=True)
        warm_start=memory.process.memory_info()._asdict()
        memory.reset()
        if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(args.device)
        warm_started=time.perf_counter()
        for repeat in range(args.repeats):
            backend.model.load_state_dict(initial)
            backend.optimizer=torch.optim.AdamW(backend.model.parameters(),lr=backend.config.learning_rate,weight_decay=backend.config.weight_decay)
            backend.model.zero_grad(set_to_none=True)
            profile=GraphProfile(args.device) if args.profile else None
            steps=[]
            # Boundary-only loss timers split train and validation without adding
            # a CUDA synchronization per graph/kernel. Detailed profiling is opt-in.
            original_loss=backend._loss;loss_phases=[]
            def measured_loss(groups,targets,weights,backward=False):
                value,seconds=timed(lambda:original_loss(groups,targets,weights,backward))
                loss_phases.append(dict(population='train' if groups is backend.train_groups else 'validation',backward=backward,seconds=seconds,trace=deepcopy(getattr(backend,'_execution_trace',None))))
                return value
            backend._loss=measured_loss
            try:
                with profile if profile is not None else nullcontext():
                    for step in range(args.steps):
                        metrics,seconds=timed(backend.step)
                        steps.append(dict(step=step,seconds=seconds,components=metrics))
                    replay_metrics,replay_seconds=timed(lambda:backend.checkpoint_metrics('train_loss',metrics))
                    (checksum,_),prediction_seconds=timed(lambda:replay(backend))
            finally:backend._loss=original_loss
            row=dict(repeat=repeat,steps=steps,loss_phases=loss_phases,
                train_metric_replay_seconds=replay_seconds,train_metric_replay_components=replay_metrics,
                validation_prediction_replay_seconds=prediction_seconds,prediction_checksum=checksum,
                detailed_phases=None if profile is None else profile.timings,final_trace=deepcopy(getattr(backend,'_execution_trace',None)))
            if args.state:
                export_started=time.perf_counter()
                torch.save(dict(model=cpu(backend.model.state_dict()),optimizer=cpu(backend.optimizer.state_dict()),
                    gradients={name:cpu(t.grad) for name,t in backend.model.named_parameters() if t.grad is not None},
                    components=steps,replay_components=replay_metrics),output/f'final-state-{repeat}.pt')
                row['state_export_seconds']=time.perf_counter()-export_started
            else:row['state_export_seconds']=None
            records.append(row);print(json.dumps(row,allow_nan=False),flush=True)
        sync()
        warm_loop_elapsed_seconds=time.perf_counter()-warm_started
        cuda_memory=None if not args.device.startswith('cuda') else dict(peak_allocated=torch.cuda.max_memory_allocated(args.device),peak_reserved=torch.cuda.max_memory_reserved(args.device))
        counts=model_counts(backend)
        config=asdict(backend.config);control=asdict(adapter.control)
    verify()
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    except (OSError,subprocess.SubprocessError):commit=None
    receipt=dict(started_utc=started,finished_utc=utc(),arm=args.arm,mode=args.mode,device=args.device,
        executable=sys.executable,python=sys.version,platform=platform.platform(),torch=str(torch.__version__),
        cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
        gpu_name=None if not args.device.startswith('cuda') else torch.cuda.get_device_name(args.device),
        source=str(SOURCE),source_head=commit,source_sha256={str(f.relative_to(SOURCE)):sha(f) for f in sorted((SOURCE/'src/xdiyo_analytics/graphs').glob('*.py'))},
        script_sha256=sha(__file__),input_root=str(OLD),input_hashes=inputs,inputs_unchanged=True,
        config=config,control=control,counts=counts,initialization_seconds=initialization_seconds,
        cold_preparation_seconds=cold_seconds,cold_preparation_counts=cold_counts,cold_traces=cold_traces,
        cold_preparation_records=cold_records,cold_repeats=args.cold_repeats,seed=args.seed,
        initialization_host_peak=initialization_host_peak,cold_host_peak=cold_host_peak,
        cold_cuda_memory=cold_cuda,fixed_capture_seconds=fixed_capture_seconds,
        warm_start_memory=warm_start,sampled_peak_host=memory.peak,os_memory_info=memory.os_info,
        cuda_memory=cuda_memory,records=records,intrusive_profile=args.profile,
        caller_flags_before=caller_flags_before,caller_flags_after=flags(),effective_execution_flags=effective_execution_flags,
        initial_state_export_seconds=initial_export_seconds,warm_loop_elapsed_seconds=warm_loop_elapsed_seconds,
        warm_timed_workload_seconds=sum(sum(s['seconds'] for s in row['steps'])+row['train_metric_replay_seconds']+row['validation_prediction_replay_seconds'] for row in records),
        concurrent_gpu_active=args.concurrent_gpu_active,
        notes=['Fresh process per arm/mode; concurrent GPU users are not stopped.',
            'Native step includes validation. Train checkpoint metric replay and historical validation prediction replay measured separately.',
            'Prediction replay uses historical validation inputs internally and is not a public causal held-out prediction.',
            'Every repeat restores one identical initialized model snapshot and a fresh AdamW; steps within each repeat are sequential.',
            'Cold source cache reset before each cold repeat; this is logical source-cache cold, not OS/disk cold. Warm repeats retain the existing source cache.',
            'With --state, fixed initial train/validation loss capture prewarms statistics observations identically before all timed warm repeats; no first repeat is discarded.',
            'No full fit, checkpoint selection, test evaluation or canonical run changes.',
            'Host sampled every2ms; initialization/cold/warm intervals recorded separately. Warm interval includes optional final CPU state exports; OS high-water is process lifetime.',
            'Torch allocator peaks exclude driver/library/external allocations; estimates and measured receipts are not universal memory guarantees.'])
    receipt['end_to_end_seconds']=time.perf_counter()-wall_started
    receipt['end_to_end_scope']='Includes input verification, imports, initialization, cold preparation, optional state capture/exports, warm repeats and receipt construction; excludes output-directory creation and final receipt JSON write/print.'
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(receipt=str(output/'receipt.json'),counts=counts,cuda_memory=cuda_memory)),flush=True)

if __name__=='__main__':main()
