"""Three complete native updates against the frozen COR_FA fitting procedure."""
from pathlib import Path
import argparse,json,sys,hashlib,random,gc
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--research',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--family',required=True);p.add_argument('--fold',default='fold_2');p.add_argument('--seed',type=int,default=17);p.add_argument('--updates',type=int,default=3);p.add_argument('--candidate',action='store_true');a=p.parse_args()
sys.path.insert(0,str(a.source/'src'))
C=a.research/'experiments/gat_corners_propagation_20261101_01';A=a.research/'experiments/gat_capacity_shots_for_20261031_01'
sys.path.insert(1,str(C/'shared/source_frozen/examples'))
import pandas as pd,numpy as np,torch
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
random.seed(17);np.random.seed(17);torch.manual_seed(17)
from xdiyo_analytics.graphs import PhaseRankLookup
from gat_cross_league.components import ComponentStore
from temporal_team_gat import make_context
from xdiyo_analytics.training import TrainingControl,EarlyStopping
from xdiyo_analytics.training.resume_loop import run_resumable_iterations
from xdiyo_analytics.training.native_resume import _cpu,rng_state
protocol=json.loads((C/'PROTOCOL.json').read_text())
from types import SimpleNamespace
from xdiyo_analytics.graphs import TemporalGraphConfig,GoalAuxiliary,PhaseRankFeatures
job=dict(recipe=a.family,fold=a.fold,seed=a.seed,condition='COR_FA',configuration=a.family+'/d32/E_FOR/COR_FA')
def frozen_config(run,job):
 raw=dict(protocol['configurations'][job['configuration']+'/'+job['fold']])
 raw['goal_auxiliary']=GoalAuxiliary(**raw['goal_auxiliary'])
 if raw['phase_rank_features'] is not None:raw['phase_rank_features']=PhaseRankFeatures(**raw['phase_rank_features'])
 return TemporalGraphConfig(**raw)
core=SimpleNamespace(MONITOR='all13_inner_raw_hda_ce_micro',job_config=frozen_config,
 config_key=lambda j:j['configuration']+'/'+j['fold'],
 preprocessing_path=lambda run,j:Path(run)/'shared/preprocessing'/j['fold']/j['recipe']/'COR_FA.json',
 initial_path=lambda run,j:Path(run)/'shared/initialization'/j['fold']/('seed_'+str(j['seed']))/j['recipe']/'COR_FA.pt')
store=ComponentStore(pd.read_parquet(C/'shared/data/history_corners.parquet'),pd.read_parquet(A/'shared/data/memberships.parquet'),schedule=pd.read_parquet(A/'shared/data/schedule.parquet'),provenance=json.loads((A/'PROTOCOL.json').read_text())['provenance'],phase_context=PhaseRankLookup.load(A/'shared/features'))
ctx=make_context(pd.read_parquet(C/'shared/populations'/(a.fold+'_fit.parquet')));ctx.validation=make_context(pd.read_parquet(C/'shared/populations'/(a.fold+'_inner.parquet')))
a.output.mkdir(parents=True,exist_ok=True)
records=[];offers=[];initial={};resources=protocol['resources']
if a.candidate:
 from xdiyo_analytics.graphs import GATResearchBNV1,GATResearchCE1V1,GATResearchBCDDWV1,RetainedGATTraining,TemporalGraphAdapter
 from xdiyo_analytics.training.residency import InputCachePolicy
 from dataclasses import asdict
 cfg={'time_B/N':lambda:GATResearchBNV1(),'time_C/E1':lambda:GATResearchCE1V1(store.phase_context.source_id),'time_B/CD_DW':lambda:GATResearchBCDDWV1(store.phase_context.source_id,10 if a.fold=='fold_1' else 20)}[a.family]()
 expanded=asdict(cfg);assert expanded.pop('statistic_availability_policy')=='per_stat_release_v1'
 assert json.loads(json.dumps(expanded))==protocol['configurations'][core.config_key(job)]
 pre=core.preprocessing_path(C,job);weights=core.initial_path(C,job)
 spec=RetainedGATTraining(competitions=tuple(protocol['competitions']),preprocessing_path=str(pre),preprocessing_sha256=hashlib.sha256(pre.read_bytes()).hexdigest(),initial_state_path=str(weights),initial_state_sha256=hashlib.sha256(weights.read_bytes()).hexdigest())
 adapter=TemporalGraphAdapter(cfg,store,research_training=spec,control=TrainingControl(max_steps=a.updates,monitor=core.MONITOR),device=resources['device'])
 adapter.configure_input_cache(resources['input_cache_bytes'],resources['headroom_bytes'])
 def base_factory(seed):return adapter._make_backend(seed)
else:
 # Compile the original frozen class and its exact numerical helpers. Explicit
 # file/config bindings avoid importing unrelated reporting orchestration whose
 # transitive helper was not included in the COR_FA source archive.
 import ast,time
 from dataclasses import asdict
 from hashlib import sha256
 from xdiyo_analytics.graphs.adapter import _Backend
 from xdiyo_analytics.graphs.execution import configured_execution
 from xdiyo_analytics.graphs.goals import poisson_goal_loss
 from xdiyo_analytics.training.residency import InputCachePolicy
 def frozen_definitions(relative,names,namespace):
  path=C/'shared/source_frozen'/relative;tree=ast.parse(path.read_text())
  selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
  assert {n.name for n in selected}==set(names)
  exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
  return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),definitions=names)
 helpers={'np':np};source_evidence=[]
 source_evidence.append(frozen_definitions('examples/gat_all_league/metrics.py',['summarize','rps'],helpers))
 metrics=SimpleNamespace(**{k:helpers[k] for k in ('summarize','rps')})
 namespace=dict(globals())
 source_evidence.append(frozen_definitions('examples/compare_gat_goals.py',['digest_state'],namespace))
 source_evidence.append(frozen_definitions('examples/gat_corners_propagation/backend.py',['AllLeagueBackend'],namespace))
 AllLeagueBackend=namespace['AllLeagueBackend']
 (a.output/'frozen_reference_bindings.json').write_text(json.dumps(source_evidence,indent=2))
 def base_factory(seed):return AllLeagueBackend(C,protocol,store,{**job,'seed':seed})

def tensors_hash(value):
 h=hashlib.sha256()
 def add(v):
  if isinstance(v,torch.Tensor):
   t=v.detach().cpu().contiguous();h.update(str((t.dtype,tuple(t.shape))).encode());h.update(t.numpy().tobytes())
  elif isinstance(v,np.ndarray):h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes())
  elif isinstance(v,dict):
   for k,x in v.items():h.update(str(k).encode());add(x)
  elif isinstance(v,(tuple,list)):
   for x in v:add(x)
  else:h.update(str(v).encode())
 add(value);return h.hexdigest()

def factory(seed):
 b=base_factory(seed);initialize=b.initialize;metrics=b.checkpoint_metrics
 def traced_initialize(context):
  initialize(context)
  initial.update(model=_cpu(b.model.state_dict()),optimizer=_cpu(b.optimizer.state_dict()),preprocessing=b.preprocessor.state(),digest=b.initial_digest,
    train_targets=_cpu(b.train_targets),validation_targets=_cpu(b.validation_targets),train_goals=_cpu(b.train_goals),validation_goals=_cpu(b.validation_goals))
  for name in ('train_groups','validation_groups'):
   groups=getattr(b,name);hashes=[]
   for i in range(len(groups)):
    with b._group_lease(groups,i) as (group,_):hashes.append(tensors_hash(group))
   initial[name]=hashes
 def traced_metrics(monitor,values):
  values=metrics(monitor,values);clean={k:v for k,v in values.items() if k!='update_seconds'}
  records.append(dict(metrics=clean,model=_cpu(b.model.state_dict()),optimizer=_cpu(b.optimizer.state_dict()),gradients=[None if v.grad is None else _cpu(v.grad) for v in b.model.parameters()],probability=b._inner_probability.copy(),rates=b._inner_rates.copy(),rng=rng_state(b.device),validation_calls=b._validation_forward_calls))
  print(a.family+': completed update '+str(len(records)),flush=True)
  return values
 b.initialize=traced_initialize;b.checkpoint_metrics=traced_metrics
 return b

def clean(value):
 if isinstance(value,dict):
  # Additional explicit migration and replay metadata has no frozen analogue.
  omit={'update_seconds','research_contract','research_execution','research_state'}
  if 'graph_semantics' in value:
   value=dict(value);v=dict(value['graph_semantics'])
   if a.candidate:assert v['statistic_availability_policy']=='per_stat_release_v1' and v['identity_schema'] is None
   v.pop('statistic_availability_policy',None);v.pop('identity_schema',None);value['graph_semantics']=v
  return {k:clean(v) for k,v in value.items() if k not in omit}
 if isinstance(value,list):return [clean(v) for v in value if not isinstance(v,dict) or v.get('metric')!='update_seconds']
 if isinstance(value,tuple):return tuple(clean(v) for v in value)
 return value

def save(capture,*,complete):offers.append(clean(capture()))
control=TrainingControl(max_steps=a.updates,early_stopping=EarlyStopping(15,0),scheduler=None,restarts=0,seed=a.seed,monitor=core.MONITOR,direction='minimize',restore_best=True)
b,history,summary=run_resumable_iterations(factory,ctx,control,save=save)
torch.save(dict(initial=initial,updates=records,offers=offers,selected=_cpu(b.model.state_dict()),summary=summary),a.output/'trajectory.pt')
(a.output/'receipt.json').write_text(json.dumps(dict(family=a.family,fold=a.fold,seed=a.seed,fit_count=len(ctx.X),validation_count=len(ctx.validation.X),updates=len(records),summary=summary,preprocessing='canonical frozen',initialization='canonical frozen',excluded_comparison_fields=['wall time','explicit migration metadata','new replay metadata']),indent=2))
for name in ('train_groups','validation_groups'):getattr(b,name)._resident.close()
