"""Independent frozen/candidate inference; archived inputs are never modified."""
from pathlib import Path
import argparse, json, sys, hashlib, gc
parser=argparse.ArgumentParser()
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--research',type=Path,required=True)
parser.add_argument('--references',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--device',default='cuda:0')
parser.add_argument('--roundtrip',action='store_true')
a=parser.parse_args()
sys.path.insert(0,str(a.source/'src'))
frozen=a.research/'experiments/gat_corners_propagation_20261101_01/shared/source_frozen'
sys.path.insert(1,str(frozen/'examples'))
import pandas as pd
import numpy as np
import torch
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
from xdiyo_analytics.graphs import TemporalGraphSerializer, PhaseRankLookup
from xdiyo_analytics.training.contracts import PredictionContext
from xdiyo_analytics.graphs.config import MATCH_KEYS
from gat_cross_league.components import ComponentStore
C=a.research/'experiments/gat_corners_propagation_20261101_01'
A=a.research/'experiments/gat_capacity_shots_for_20261031_01'
store=ComponentStore(pd.read_parquet(C/'shared/data/history_corners.parquet'),
 pd.read_parquet(A/'shared/data/memberships.parquet'),schedule=pd.read_parquet(A/'shared/data/schedule.parquet'),
 provenance=json.loads((A/'PROTOCOL.json').read_text())['provenance'],phase_context=PhaseRankLookup.load(A/'shared/features'))
a.output.mkdir(parents=True,exist_ok=True)
receipt=[]
def fingerprint(value):
 h=hashlib.sha256()
 def add(x):
  if isinstance(x,torch.Tensor):
   t=x.detach().cpu().contiguous();h.update(str((t.dtype,tuple(t.shape))).encode());h.update(t.numpy().tobytes())
  elif isinstance(x,np.ndarray):
   h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
  elif isinstance(x,dict):
   for k,v in x.items(): h.update(str(k).encode());add(v)
  elif isinstance(x,(list,tuple)):
   for v in x:add(v)
  else:h.update(str(x).encode())
 add(value);return h.hexdigest()
for row in pd.read_csv(a.references).to_dict('records'):
 name=row['family'].replace('/','__')+'__'+row['fold']+'__'+str(row['seed'])
 model=TemporalGraphSerializer(store,device=a.device).load(row['model_directory'])
 population=pd.read_parquet(C/'shared/populations'/(row['fold']+'_outer.parquet'))
 context=PredictionContext(X=pd.DataFrame({'graph_input':np.zeros(len(population))},index=population.index),metadata=population.copy(),layout='match',match_columns=MATCH_KEYS,fold_id=0)
 backend=model.backend_
 backend._boundary(backend._validate(context))
 logits=np.empty((len(population),3),np.float32);rates=np.empty((len(population),2),np.float32)
 inputs=[];seen=np.zeros(len(population),int)
 groups=backend._stream(context)
 try:
  with torch.no_grad():
   backend.model.eval()
   for i in range(len(groups)):
    with groups.lease(i) as group:
     pos,data,home,away,fixture=group
     inputs.append(fingerprint(group));seen[pos]+=1
     z,lam=backend.model(data,home,away,fixture,return_goals=True)
     logits[pos]=z.cpu().numpy();rates[pos]=lam.cpu().numpy()
 finally:
  groups._resident.close()
 assert (seen==1).all()
 public=model.predict(context)['predict_proba'].to_numpy()
 public_rates=model.predict_goals(context).to_numpy()
 np.testing.assert_array_equal(rates,public_rates)
 np.savez(a.output/(name+'.npz'),logits=logits,rates=rates,probabilities=public)
 record=dict(name=name,count=len(population),model_version=backend.model_version,store_version=model.event_store.version,inputs=inputs)
 if a.roundtrip:
  folder=a.output/'roundtrip'/name
  TemporalGraphSerializer().save(model,folder)
  again=TemporalGraphSerializer(store,device=a.device).load(folder)
  assert again.backend_.model_version==backend.model_version
  np.testing.assert_array_equal(public,again.predict(context)['predict_proba'].to_numpy())
  np.testing.assert_array_equal(public_rates,again.predict_goals(context).to_numpy())
  record['roundtrip']=True
 receipt.append(record)
 (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2))
 print(name+': '+str(len(population))+' fixtures',flush=True)
 del model,backend,groups;gc.collect()
