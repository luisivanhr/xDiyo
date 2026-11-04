"""Read-only schema and output checks against one original frozen runtime."""
from pathlib import Path
import argparse,json,sys,gc
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--jobs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--roundtrip',action='store_true');a=p.parse_args()
sys.path.insert(0,str(a.source/'src'));sys.path.insert(1,str(a.source/'examples'))
import pandas as pd,numpy as np,torch
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
from xdiyo_analytics.graphs import TemporalGraphSerializer, TemporalEventStore
from xdiyo_analytics.training.contracts import PredictionContext
from xdiyo_analytics.graphs.config import MATCH_KEYS
jobs=json.loads(a.jobs.read_text());a.output.mkdir(parents=True,exist_ok=True);receipts=[]
store=None
for job in jobs:
 directory=Path(job['model']);meta=json.loads((directory/'graph.json').read_text())
 if job['store_kind']=='embedded':
  model=TemporalGraphSerializer(device=a.device).load(directory);store=model.event_store
 else:
  if store is None:
   cls=TemporalEventStore
   if job['store_kind']=='component':
    # Keep the archived experiment's ordering/restriction, independent of candidate code.
    sys.path.insert(1,job['component_examples'])
    from gat_cross_league.components import ComponentStore
    cls=ComponentStore
   data=Path(job['data'])
   store=cls(pd.read_parquet(data/'history.parquet'),pd.read_parquet(data/'memberships.parquet'),schedule=pd.read_parquet(data/'schedule.parquet'),provenance=meta['event_store_provenance'])
  model=TemporalGraphSerializer(store,device=a.device).load(directory)
 backend=model.backend_;frame=pd.read_parquet(job['population']) if job.get('population') else store.schedule
 # Four fixed fixtures are schema probes, not a new model evaluation.
 frame=frame.iloc[:4].copy()
 context=PredictionContext(X=pd.DataFrame({'graph_input':np.zeros(len(frame))},index=frame.index),metadata=frame,layout='match',match_columns=MATCH_KEYS,fold_id=0)
 group_source=backend._stream(context);z=np.empty((len(frame),3),np.float64);rates=np.empty((len(frame),2),np.float64);has_goals=getattr(model.config,'goal_auxiliary',None) is not None
 seen=np.zeros(len(frame),int)
 with torch.no_grad():
  backend.model.eval()
  for i in range(len(group_source)):
   # Public materialization supports older PreparedGroups predating leases.
   pos,data,home,away,fixture=group_source[i]
   if has_goals:
    logits,lam=backend.model(data,home,away,fixture,return_goals=True);rates[pos]=lam.cpu().numpy()
   else:logits=backend.model(data,home,away,fixture)
   z[pos]=logits.cpu().numpy();seen[pos]+=1
 assert (seen==1).all()
 public_boundary='allowed'
 try:
  output=model.predict(context)['predict_proba'].to_numpy()
 except ValueError as error:
  if 'boundary' not in str(error):raise
  public_boundary='rejected_before_selection_boundary';output=np.empty((0,3))
 data=dict(logits=z,probabilities=output)
 if has_goals:data['rates']=rates
 np.savez(a.output/(job['id']+'.npz'),**data)
 record=dict(id=job['id'],model_version=backend.model_version,store_version=model.event_store.version,public_boundary=public_boundary,fixtures=frame[list(MATCH_KEYS)].astype(str).values.tolist())
 if a.roundtrip:
  target=a.output/'roundtrip'/job['id'];TemporalGraphSerializer().save(model,target)
  again=TemporalGraphSerializer(store,device=a.device).load(target)
  assert again.backend_.model_version==backend.model_version
  with torch.no_grad():
   q=again.backend_._stream(context)
   for i in range(len(q)):
    pos,d,h,w,f=q[i]
    if has_goals:
     zz,ll=again.backend_.model(d,h,w,f,return_goals=True);np.testing.assert_array_equal(ll.cpu().numpy(),rates[pos])
    else:zz=again.backend_.model(d,h,w,f)
    np.testing.assert_array_equal(zz.cpu().numpy(),z[pos])
  record['roundtrip']=True
 receipts.append(record);(a.output/'receipt.json').write_text(json.dumps(receipts,indent=2));print(job['id']+': schema/output probe complete',flush=True)
 del model,backend,group_source;gc.collect()
