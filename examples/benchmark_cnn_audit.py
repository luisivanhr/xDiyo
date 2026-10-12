"""Bounded synthetic CPU timing for CNN audit repairs; no research data.

Run from the repository root with PYTHONPATH=src. Requires the graphs extra
and git history containing baseline 5eded399. Reports medians after warm-up;
absolute timings depend on hardware and do not establish GPU performance.
"""
import json, statistics, subprocess, time, types
import numpy as np
import pandas as pd
import torch
from xdiyo_analytics.neural import HeatmapCNNModel, HeatmapCNNConfig, Reconstruction
from xdiyo_analytics.neural.adapter import CNNBackend
from xdiyo_analytics.training.contracts import FitContext
from xdiyo_analytics.features.point_geometry import validate_exact_point_ids

torch.set_num_threads(1)
legacy = types.ModuleType('xdiyo_analytics.neural.audit_baseline')
legacy.__package__ = 'xdiyo_analytics.neural'
exec(subprocess.check_output(['git','show','5eded399:src/xdiyo_analytics/neural/adapter.py'], text=True, encoding='utf-8'), legacy.__dict__)
rng=np.random.default_rng(17)
N=512
names=[f'{side}::recent::heatmap_for_mass_10_y{y:02}_x{x:02}' for side in ('home','away') for y in range(10) for x in range(10)]
a=rng.random((N,2,100));a/=a.sum(-1,keepdims=True)
X=pd.DataFrame(a.reshape(N,200),columns=names)
y=pd.DataFrame({'target':rng.random(N)})
c=FitContext(X=X,y=y,metadata=pd.DataFrame(index=X.index),layout='match',match_columns=('event_id',),fold_id=0)
c.validation=c
spec=HeatmapCNNModel(task='regression',batch_size=128,encoder=HeatmapCNNConfig(reconstruction=Reconstruction()))
results={}
for label, cls, monitor in [('baseline_validation',legacy.CNNBackend,'validation_loss'),('repaired_validation',CNNBackend,'validation_loss'),('repaired_train_monitor',CNNBackend,'train_loss')]:
    b=cls(spec,17,'cpu');b.initialize(c)
    durations=[]
    for i in range(23):
        start=time.perf_counter();v=b.step()
        if hasattr(b,'checkpoint_metrics'):b.checkpoint_metrics(monitor,v)
        elapsed=time.perf_counter()-start
        if i>=3:durations.append(elapsed)
    results[label]={'median_seconds':statistics.median(durations),'min_seconds':min(durations)}
ids=pd.DataFrame({'event_id':np.arange(1,1000001,dtype=np.int64),'team_id':np.full(1000000,17,dtype=np.int64)})
t=[]
for _ in range(100):
    start=time.perf_counter();validate_exact_point_ids(ids,ids);t.append(time.perf_counter()-start)
results['one_million_typed_ids_validation_seconds']=statistics.median(t)

# GAT replay comparison uses the same prepared grids and chronological groups.
from xdiyo_analytics.graphs import TemporalGraphConfig, TemporalGraphModel, GraphHeatmaps
from xdiyo_analytics.graphs.adapter import _Backend
rows=[]
for event in range(1, 31):
    kick=pd.Timestamp('2020-01-01',tz='UTC')+pd.Timedelta(days=3*(event-1))
    for team, opponent, side in [(1,2,'home'),(2,1,'away')]:
        rows.append(dict(event_id=event,competition_id=10,season_id=900,round=event,
            team_id=team,opponent_id=opponent,side=side,kickoff_at=kick,status='finished',
            goals_for=team-1,goals_against=opponent-1))
h=pd.DataFrame(rows)
members=pd.DataFrame([dict(team_id=team,competition_id=10,season_id=900,
    entry_at=pd.Timestamp('2019-07-01',tz='UTC'),movement='retained') for team in (1,2)])
points=h[['event_id','team_id']].assign(kind='player',x=15.,y=25.,point_order=0)
cfg=TemporalGraphConfig('UTC',heatmaps=GraphHeatmaps())
store=TemporalGraphModel(config=cfg,membership_policy='explicit').event_store(h,members,points)
def graph_context(positions):
    meta=store.schedule.iloc[positions].copy()
    target=np.sign(meta.goals_for-meta.goals_against)
    return FitContext(X=pd.DataFrame(index=meta.index),y=pd.DataFrame({'result':target}),
        metadata=meta,layout='match',match_columns=('competition_id','season_id','event_id'),fold_id=0)
train=graph_context(list(range(1,21)))
train.validation=graph_context(list(range(22,28)))
old_graph=types.ModuleType('xdiyo_analytics.graphs.audit_baseline')
old_graph.__package__='xdiyo_analytics.graphs'
exec(subprocess.check_output(['git','show','5eded399:src/xdiyo_analytics/graphs/adapter.py'],
                             text=True,encoding='utf-8'),old_graph.__dict__)
for label,cls in [('baseline_gat_validation',old_graph._Backend),('repaired_gat_validation',_Backend)]:
    b=cls(cfg,store,17,'cpu');b.initialize(train)
    durations=[]
    for i in range(10):
        start=time.perf_counter();values=b.step()
        if hasattr(b,'checkpoint_metrics'):b.checkpoint_metrics('validation_loss',values)
        if i>=2:durations.append(time.perf_counter()-start)
    results[label]={'median_seconds':statistics.median(durations),'min_seconds':min(durations)}
results['scope']={'device':'cpu','threads':1,'standalone_rows':N,'grid_size':10,
                  'gat_train_issuances':20,'gat_validation_issuances':6,
                  'torch':torch.__version__,'pandas':pd.__version__}
print(json.dumps(results,indent=2))
