"""Structured routes, frozen preprocessing, exact continuation and legacy exports."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import sys
import numpy as np
import pytest
import torch
from test_temporal_graph import sample,config,context
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'examples'))
from gat_suite import core
from gat_suite.fitting import gradients,input_map
from compare_gat_goals import digest_state
from xdiyo_analytics.graphs import GoalAuxiliary,TemporalGraphAdapter,TemporalGraphSerializer
from xdiyo_analytics.graphs.adapter import _Backend
from xdiyo_analytics.graphs.goals import auxiliary_goal_loss
from xdiyo_analytics.graphs.execution import prepare_determinism
from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor
from xdiyo_analytics.training import TrainingControl,EarlyStopping
from xdiyo_analytics.training.resume_loop import run_resumable_iterations
from xdiyo_analytics.training.native_resume import _cpu,capture_backend,rng_state

@pytest.fixture(autouse=True)
def one_thread():
    n=torch.get_num_threads();torch.set_num_threads(1);yield;torch.set_num_threads(n)

@pytest.fixture(params=['cpu','cuda:0'])
def device(request):
    if request.param.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    prepare_determinism(config(deterministic=True),request.param)
    return request.param

def build(family,device,detach=False):
    store=sample();ctx=context(store,[2,3,4]);ctx.validation=context(store,[6,7])
    cfg=config(deterministic=True,goal_auxiliary=GoalAuxiliary(architecture='attack_defence',likelihood=family,competitions=(10,),detach_encoder=detach))
    b=_Backend(cfg,store,17,device);b.initialize(ctx)
    return b,ctx

@pytest.mark.parametrize('family',['poisson','independent_nb','shared_gamma'])
def test_routes_zero_initial_gradient_optimizer_exceptions_and_persistence(device,family,tmp_path):
    b,ctx=build(family,device)
    diag=gradients(b);assert diag['goal_norm']==0 and diag['cosine_similarity'] is None
    with torch.no_grad():
        b.model.goal_head.attack.weight.fill_(.13);b.model.goal_head.defence.weight.fill_(-.09)
    _,data,h,a,f=b.train_groups[0]
    logits,detail=b.model(data,h,a,f,return_goal_details=True)
    y=b.train_goals[:len(logits)]
    params=dict(b.model.named_parameters());names=list(params)
    g=torch.autograd.grad(auxiliary_goal_loss(b.model,detail,y).sum(),tuple(params.values()),allow_unused=True)
    assert any(v is not None and v.abs().sum()>0 for n,v in zip(names,g) if n.startswith('layers.'))
    assert all(v is None for n,v in zip(names,g) if n.startswith('head.'))
    assert any(v is not None and v.abs().sum()>0 for n,v in zip(names,g) if n.startswith('goal_head.'))
    logits,detail=b.model(data,h,a,f,return_goal_details=True)
    g=torch.autograd.grad(logits.square().sum(),tuple(params.values()),allow_unused=True)
    assert all(v is None for n,v in zip(names,g) if n.startswith('goal_head.'))
    b.model.goal_config=replace(b.model.goal_config,detach_encoder=True)
    logits,detail=b.model(data,h,a,f,return_goal_details=True)
    g=torch.autograd.grad(auxiliary_goal_loss(b.model,detail,y).sum(),tuple(params.values()),allow_unused=True)
    assert all(v is None for n,v in zip(names,g) if not n.startswith('goal_head.'))
    assert any(v is not None and v.abs().sum()>0 for n,v in zip(names,g) if n.startswith('goal_head.'))
    b.model.goal_config=b.config.goal_auxiliary
    groups=b.optimizer.param_groups
    assert groups[0]['weight_decay']==.001 and groups[0]['lr']==.001
    assert groups[1]['weight_decay']==0 and groups[1]['lr']==.001
    if family!='poisson': assert groups[2]['weight_decay']==0 and groups[2]['lr']==.05
    before=digest_state((capture_backend(b),rng_state(device)));gradients(b)
    assert digest_state((capture_backend(b),rng_state(device)))==before
    adapter=TemporalGraphAdapter(b.config,b.store,control=TrainingControl(max_steps=2),device=device)
    adapter.fit(ctx);pred=context(b.store,[8],fit=False)
    pp=adapter.predict(pred)['predict_proba'];gg=adapter.predict_goals(pred)
    serializer=TemporalGraphSerializer(b.store,device=device);serializer.save(adapter,tmp_path);loaded=serializer.load(tmp_path)
    np.testing.assert_array_equal(pp,loaded.predict(pred)['predict_proba']);np.testing.assert_array_equal(gg,loaded.predict_goals(pred))
    assert digest_state(adapter.backend_.model.state_dict())==digest_state(loaded.backend_.model.state_dict())
    with pytest.raises(ValueError,match='boundary'):loaded.predict_goals(context(b.store,[3],fit=False))


def test_preprocessor_pairing_initial_temporal_columns_and_cache_identity(device):
    store=sample();recent=context(store,[3,4]);expanded=context(store,[1,2,3,4])
    cfg=config(deterministic=True,goal_auxiliary=GoalAuxiliary(beta=0))
    ref=_Backend(cfg,store,17,device);ref.initialize(recent);state=ref.preprocessor.state()
    b=_Backend(cfg,store,17,device,preprocessor_state=state);b.initialize(expanded)
    short,_=input_map(ref);long,_=input_map(b)
    assert all(long[k]==v for k,v in short.items())
    snap=core.initialize_state(cfg,ref.preprocessor,17)
    timed=replace(cfg,temporal_mode='time_features');pp=GraphPreprocessor.from_state(timed,state)
    time_model=core.initialize_state(timed,pp,17,snap.state_dict()).to(device)
    ref.model=snap.to(device)
    temporal=_Backend(timed,store,17,device,preprocessor_state=state);temporal.initialize(recent);temporal.model=time_model
    g0=ref.train_groups[0];g1=temporal.train_groups[0]
    torch.testing.assert_close(ref.model(*g0[1:]),temporal.model(*g1[1:]),rtol=1e-5,atol=1e-6)
    temporal.model(*g1[1:]).square().sum().backward()
    assert any(layer.e.weight.grad[:,-14:].abs().sum()>0 for layer in temporal.model.layers)
    hist=core.initialize_state(replace(timed,temporal_mode='historical'),pp,17,snap.state_dict())
    assert digest_state(hist.state_dict())==digest_state(time_model.state_dict())


def test_resume_captures_last_optimizer_before_best_restore():
    store=sample();ctx=context(store,[2,3,4]);ctx.validation=context(store,[6,7])
    cfg=config(deterministic=True,goal_auxiliary=GoalAuxiliary())
    control=TrainingControl(max_steps=6,early_stopping=EarlyStopping(2,0),seed=17,monitor='validation_loss',restore_best=True)
    class Scripted(_Backend):
        def checkpoint_metrics(self,monitor,values):
            # AdamW's completed update counter survives continuation.
            step=int(next(iter(self.optimizer.state.values()))['step'])
            values=super().checkpoint_metrics('train_loss',values)
            values['validation_loss']={1:.5,2:.4,3:.7,4:.8,5:.9,6:1.}[step]
            return values
    factory=lambda seed:Scripted(cfg,store,seed,'cpu')
    uninterrupted=[]
    selected,_,summary=run_resumable_iterations(factory,ctx,control,save=lambda capture,complete:uninterrupted.append(_cpu(capture())))
    assert summary['best_step']==2 and summary['attempts'][0]['steps']==4
    active=uninterrupted[-2]['active'];assert active['step']==4 and active['stopped']
    assert digest_state(active['backend']['model'])!=digest_state(selected.model.state_dict())
    interrupted=[]
    def stop(capture,*,complete):
        state=_cpu(capture());interrupted.append(state)
        if state['active'] and state['active']['step']==3:raise InterruptedError('Disposable interruption')
    with pytest.raises(InterruptedError):run_resumable_iterations(factory,ctx,control,save=stop)
    resumed=[]
    restored,_,receipt=run_resumable_iterations(factory,ctx,control,state=interrupted[-1],save=lambda capture,complete:resumed.append(_cpu(capture())))
    assert receipt==summary
    assert digest_state(restored.model.state_dict())==digest_state(selected.model.state_dict())
    assert digest_state(resumed[-2]['active']['backend'])==digest_state(active['backend'])


def test_manifest_exact_matrix_and_legacy_pilot_loads():
    rows=core.matrix();assert len(rows)==108
    assert [sum(r['study']==n for r in rows) for n in (1,2,3,4)]==[18,36,18,36]
    parent=core.PARENT
    if not parent.exists():pytest.skip('Local pilot archive unavailable')
    from xdiyo_analytics.graphs import TemporalEventStore
    import pandas as pd
    p=json.loads((parent/'protocol.json').read_text())
    store=TemporalEventStore(pd.read_parquet(parent/'history.parquet'),pd.read_parquet(parent/'memberships.parquet'),
        schedule=pd.read_parquet(parent/'schedule.parquet'),provenance=p['provenance'])
    for path in sorted((parent/'fit').glob('*/model')):
        loaded=TemporalGraphSerializer(store).load(path)
        assert loaded.config.goal_auxiliary.architecture=='linear' and loaded.config.goal_auxiliary.likelihood=='poisson'
