"""Run a pinned package in its own process; emits exact tiny CPU trajectory."""
from pathlib import Path
import sys
source,out,kind=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
sys.path.insert(0,str(source/'src'))
import torch
import numpy as np
import pandas as pd
import json
from dataclasses import asdict
from test_temporal_graph import sample,config,context
from xdiyo_analytics.graphs import TemporalGraphAdapter,TemporalGraphSerializer
from xdiyo_analytics.graphs.adapter import _Backend
from xdiyo_analytics.training import TrainingControl
from xdiyo_analytics.experiments.recovery import pack

def cpu(v):
    if isinstance(v,np.ndarray):return torch.as_tensor(v.copy())
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return type(v)(cpu(x) for x in v)
    return v

torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
if kind.startswith('gat_'):
    from test_graph_integration_gates import extension_fixture
    a,train,query=extension_fixture(kind)
else:
    if kind=='cnn':
        from test_heatmap_cnn import graph_setup
        from xdiyo_analytics.neural import Reconstruction
        spec,store=graph_setup(reconstruction=Reconstruction(loss='kl'));cfg=spec.config
    else:store,cfg=sample(),config(deterministic=True)
    a=TemporalGraphAdapter(cfg,store);train=context(store,[1,2,3,4]);train.validation=context(store,[5,6]);query=context(store,[7,8],fit=False)
b=_Backend(a.config,a.event_store,19,'cpu');b.initialize(train)
trajectory=[cpu(dict(model=b.model.state_dict(),preprocessor=b.preprocessor.state(),groups=list(b.train_groups)))]
for _ in range(2):
    values=b.checkpoint_metrics('train_loss',b.step())
    trajectory.append(cpu(dict(loss=values,model=b.model.state_dict(),optimizer=b.optimizer.state_dict(),gradients=[p.grad for p in b.model.parameters()])))
out.mkdir(parents=True,exist_ok=True);torch.save(trajectory,out/'trajectory.pt')
a.backend_=b;a.control=TrainingControl(max_steps=2,seed=19);a.training_summary_={};a.training_history_=pd.DataFrame()
queries=pd.DataFrame({'team_id':[3,2,1], 'issuance_at':[pd.Timestamp('2020-02-02T12:00Z')]*3})
outputs=dict(predictions=a.predict(query),embeddings=a.encode_teams(queries),attention=a.attention_weights(queries),version=b.model_version)
(out/'outputs.json').write_text(json.dumps(pack(outputs),sort_keys=True,allow_nan=False),encoding='utf-8')
TemporalGraphSerializer(embed_event_store=True).save(a,out/'model')
