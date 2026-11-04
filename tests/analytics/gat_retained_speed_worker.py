"""Bounded exact speed gates; immutable archived inputs, new output only."""
from pathlib import Path
import argparse,json,sys,time,random,gc,os,threading
import psutil
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples')]
import numpy as np,pandas as pd,torch
from dataclasses import replace,asdict,fields
from contextlib import contextmanager
from gat_retained.replay import prepare
from xdiyo_analytics.graphs import RetainedGATExecution,TemporalGraphSerializer
from xdiyo_analytics.graphs.research_training import MONITOR,digest_state
from xdiyo_analytics.graphs.execution import deterministic_execution,prepare_determinism
from xdiyo_analytics.training.native_resume import capture_backend,rng_state,execution_profile,write_checkpoint,read_checkpoint
from xdiyo_analytics.training.resume_loop import run_resumable_iterations
from xdiyo_analytics.experiments.recovery import execution_key

p=argparse.ArgumentParser();p.add_argument('--research',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--mode',choices=('short','full','timing'),required=True);p.add_argument('--family',default='time_C/E1');p.add_argument('--flag',default='all');p.add_argument('--budget',type=int)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
S=a.research/'experiments/gat_speed_residuals_20261102_01/00_speedup/parity'
FLAGS=[f.name for f in fields(RetainedGATExecution) if f.name!='profile']
OFF=RetainedGATExecution(**dict.fromkeys(FLAGS,False))
changes=RetainedGATExecution() if a.flag=='all' else replace(OFF,**{a.flag:True})

class MemoryProbe:
    def __enter__(self):
        self.process=psutil.Process();self.baseline=self.process.memory_info().rss
        self.peak=self.baseline;self.stop=threading.Event()
        def sample():
            while not self.stop.wait(.02):self.peak=max(self.peak,self.process.memory_info().rss)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join()
    def result(self):
        info=self.process.memory_info();self.peak=max(self.peak,info.rss)
        return dict(baseline_rss_bytes=self.baseline,sampled_peak_rss_bytes=self.peak,
                    final_rss_bytes=info.rss,private_bytes=getattr(info,'private',None),sampling_seconds=.02)


def exact(x,y,path='root'):
    if isinstance(x,torch.Tensor):torch.testing.assert_close(x,y,rtol=0,atol=0,msg=path)
    elif isinstance(x,np.ndarray):
        assert isinstance(y,np.ndarray) and x.dtype==y.dtype,path
        np.testing.assert_array_equal(x,y,err_msg=path)
    elif isinstance(x,dict):
        assert x.keys()==y.keys(),(path,'keys',x.keys(),y.keys())
        for k in x:exact(x[k],y[k],path+'/'+str(k))
    elif isinstance(x,(tuple,list)):
        assert type(x)==type(y) and len(x)==len(y),path
        for i,(l,r) in enumerate(zip(x,y)):exact(l,r,path+'/'+str(i))
    else:assert x==y,(path,x,y)


def legacy_state(state):
    state=dict(state)
    semantics=dict(state['graph_semantics'])
    assert semantics.pop('statistic_availability_policy')=='per_stat_release_v1'
    assert semantics.pop('identity_schema') is None
    state['graph_semantics']=semantics
    assert state.pop('research_contract')['contract']=='retained_all13_raw_ce_v1'
    state.pop('research_execution');state.pop('research_state')
    return state


def reset():
    random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.cuda.manual_seed_all(17)


def construction(execution=changes):
    reset();spec,store,ctx,outer=prepare(a.research,family=a.family,fold='fold_2',seed=17,
                                      window=20 if a.family=='time_B/CD_DW' else None)
    del outer  # No new outer predictions or scores are produced by this gate.
    spec=replace(spec,research_execution=execution);adapter=spec.build(store);adapter.configure_device('cuda:0')
    resources=json.loads((a.research/'experiments/gat_corners_propagation_20261101_01/PROTOCOL.json').read_text())['resources']
    adapter.configure_input_cache(a.budget if a.budget is not None else resources['input_cache_bytes'],resources['headroom_bytes'])
    prepare_determinism(spec.config,adapter.device)
    return adapter,ctx


def close(b):
    if hasattr(b,'close'):b.close()
    else:
        for name in ('train_groups','validation_groups'):getattr(b,name)._resident.close()


def snapshot(b,raw,values,best,stale):
    return dict(backend=legacy_state(capture_backend(b)),rng=rng_state(b.device),raw=raw,
        validation={k:v for k,v in values.items() if k.startswith(('validation_','all13_','inner_raw_'))},
        probability=torch.from_numpy(b._inner_probability.copy()),rates=torch.from_numpy(b._inner_rates.copy()),best=best,stale=stale)


def input_audit(b):
    original=b._group_lease;seen=set();records={}
    @contextmanager
    def lease(groups,index,replay=None):
        with original(groups,index,replay) as (group,owned):
            key=('fit' if groups is b.train_groups else 'inner',index)
            if key not in seen:
                pos,*rest=group
                if isinstance(pos,torch.Tensor):pos=pos.detach().cpu().numpy()
                records['%s/%s'%key]=digest_state((np.asarray(pos),dict(rest[0]),*rest[1:]));seen.add(key)
            yield group,owned
    b._group_lease=lease;return records


def inference(b,phases=('fit','inner')):
    output={}
    with deterministic_execution(b.config,b.device),torch.no_grad():
        b.model.eval()
        for phase,groups in [('fit',b.train_groups),('inner',b.validation_groups)]:
            if phase not in phases:continue
            rows=[]
            for i in range(len(groups)):
                with b._group_lease(groups,i) as (group,_):
                    positions,data,h,away,f=group;logits,rates=b.model(data,h,away,f,return_goals=True)
                    rows.append((np.asarray(positions).copy(),logits.cpu().clone(),logits.softmax(1).cpu().clone(),rates.cpu().clone()))
            output[phase]=rows
    return output


def short():
    adapter,ctx=construction();b=adapter._make_backend(17);b.audit_borrow=True
    started=time.perf_counter();b.initialize(ctx);inputs=input_audit(b)
    reference=S/a.family.replace('/','_')/'unchanged_a';best=float('inf');stale=0
    try:
        for step in range(1,4):
            raw=b.step();values=b.checkpoint_metrics(MONITOR,raw);v=values[MONITOR]
            if v<best:best=v;stale=0
            else:stale+=1
            actual=snapshot(b,raw,values,best,stale)
            exact(torch.load(reference/f'{step:03}.pt',weights_only=True),actual)
            (a.output/'progress.json').write_text(json.dumps(dict(step=step,validation_ce=v)))
        exact(json.loads((reference/'inputs.json').read_text()),inputs)
        (a.output/'inputs.json').write_text(json.dumps(inputs,indent=2))
        return dict(passed=True,mode='short',family=a.family,flags=asdict(changes),updates=3,
                    input_groups=len(inputs),exact=True,seconds=time.perf_counter()-started,
                    cache=None if b.borrow is None else b.borrow.stats())
    finally:close(b)


def full():
    adapter,ctx=construction();reference=S/'full_C_E1_fold_2_seed_17/reference';holder={}
    profile=execution_profile(adapter)
    identity=execution_key(type(adapter),ctx,adapter.control,adapter.config,
        (adapter.event_store.version,adapter.event_store.history,adapter.event_store.team_seasons,adapter.event_store.schedule),profile)
    def factory(seed):
        b=adapter._make_backend(seed);holder['backend']=b;original=b.step
        def step():
            value=original();holder['raw']=value.copy();return value
        b.step=step;return b
    def save(capture,*,complete):
        state=capture()
        if not complete:
            active=state['active'];b=holder['backend'];step=active['step']
            actual=snapshot(b,holder['raw'],active['last_metrics'],active['best'],active['stale'])
            actual['selection']={k:active[k] for k in ('best','best_step','progress_best','stale','schedule_stale','last_value','reason','step','stopped')}
            actual['checkpoint']=active['checkpoint']
            exact(torch.load(reference/f'{step:03}.pt',weights_only=True),actual)
            torch.save(actual,a.output/f'{step:03}.pt')
            (a.output/'progress.json').write_text(json.dumps(dict(step=step,best_step=active['best_step'],stale=active['stale'],validation_ce=active['last_value'])))
            print('exact update',step,'best',active['best_step'],'stale',active['stale'],flush=True)
        name='complete' if complete else f'update_{state["active"]["step"]:03}'
        # Native writer requires a fresh caller-owned staging directory.
        from uuid import uuid4
        directory=a.output/'checkpoints'/f'{name}_{uuid4().hex[:12]}'
        directory.mkdir(parents=True,exist_ok=False)
        write_checkpoint(directory,state,identity,profile)
        temp=a.output/'latest.tmp';temp.write_text(json.dumps(dict(directory=str(directory))));temp.replace(a.output/'latest.json')
    latest=a.output/'latest.json';state=None
    if latest.exists():
        state=read_checkpoint(Path(json.loads(latest.read_text())['directory']),identity,profile)[0]
        if state['active'] is not None:holder['raw']=torch.load(a.output/f'{state["active"]["step"]:03}.pt',weights_only=True)['raw']
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    try:
        b,history,summary=run_resumable_iterations(factory,ctx,adapter.control,state=state,save=save)
        selected=dict(backend=legacy_state(capture_backend(b)),rng=rng_state(b.device),summary=summary)
        exact(torch.load(reference/'selected.pt',weights_only=True),selected)
        torch.save(selected,a.output/'selected.pt');history.to_csv(a.output/'history.csv',index=False)
        actual=inference(b);exact(torch.load(reference/'inference.pt',weights_only=False),actual)
        torch.save(actual,a.output/'inference.pt')
        assert summary['best_step']==88 and summary['attempts'][0]['steps']==103
        return dict(passed=True,mode='full',exact=True,summary=summary,flags=asdict(changes),
            seconds=time.perf_counter()-started,cuda_peak_allocated=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved=torch.cuda.max_memory_reserved(),new_outer_predictions=False,
            comparison='all 103 optimizer states, gradients, RNG, raw losses, probabilities/rates, stopping counters and restored inference',
            cache=b.borrow.stats())
    finally:
        if 'backend' in holder:close(holder['backend'])


def timing():
    # Fixed ABBA/BAAB order, four cold starts and eight warm updates per arm.
    rows=[];reference=S/'full_C_E1_fold_2_seed_17/reference'
    for trial,arm in enumerate(('reference','candidate','candidate','reference','candidate','reference','reference','candidate')):
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        adapter,ctx=construction(None if arm=='reference' else changes);b=adapter._make_backend(17);b.initialize(ctx)
        torch.cuda.synchronize();initialization=time.perf_counter()-started
        probe=MemoryProbe().__enter__()
        try:
            for update in range(1,4):
                torch.cuda.synchronize();start=time.perf_counter();raw=b.step();values=b.checkpoint_metrics(MONITOR,raw);torch.cuda.synchronize()
                elapsed=time.perf_counter()-start
                rows.append(dict(trial=trial,arm=arm,task='training',update=update,cold=update==1,seconds=elapsed,
                    initialization_seconds=initialization,cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved(),**probe.result()))
            expected=torch.load(S/'time_C_E1/unchanged_a/003.pt',weights_only=True)
            exact(expected['backend'],legacy_state(capture_backend(b)))
            b.restore(torch.load(reference/'selected.pt',weights_only=True)['backend']['model']);close(b)
            for cold in (True,False):
                torch.cuda.synchronize();start=time.perf_counter()
                # Ordered logits/probabilities/rates are actually copied to CPU.
                output=inference(b,phases=('inner',))
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                exact(torch.load(reference/'inference.pt',weights_only=False)['inner'],output['inner'])
                rows.append(dict(trial=trial,arm=arm,task='inference',update=0,cold=cold,seconds=elapsed,**probe.result()))
            pd.DataFrame(rows).to_csv(a.output/'timing.csv',index=False)
            print('timing trial',trial,arm,flush=True)
        finally:probe.__exit__();close(b);del b,adapter,ctx
    frame=pd.DataFrame(rows);summary=frame.groupby(['arm','task','cold']).seconds.agg(['count','median','mean']).reset_index()
    summary.to_csv(a.output/'timing_summary.csv',index=False)
    return dict(passed=True,mode='timing',order='ABBA BAAB',rows=len(rows),summary=summary.to_dict('records'),scope='C/E1 fold2 seed17; training and inner inference measured separately')

try:
    if a.mode in ('full','timing') and (a.family!='time_C/E1' or a.flag!='all'):
        raise ValueError('The full/timing gate is fixed to C/E1 fold2 seed17 with all accepted flags.')
    with MemoryProbe() as probe:
        result={'short':short,'full':full,'timing':timing}[a.mode]()
        result['memory']=probe.result()
    (a.output/'PASS.json').write_text(json.dumps(result,indent=2))
except BaseException as error:
    (a.output/'FAILED.json').write_text(json.dumps(dict(error=repr(error)),indent=2));raise
