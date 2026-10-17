"""Unsupported execution combinations retain the complete native objective."""
from dataclasses import replace
from copy import deepcopy
import numpy as np
import pytest
import torch
from test_temporal_graph import sample, config, context, history
from test_heatmap_cnn import graph_setup
from test_graph_preparation_memory import assert_exact
from test_graph_dependency import groups_for
from xdiyo_analytics.graphs import GraphExecution, BoundaryMemory, TemporalEventStore
from xdiyo_analytics.graphs.adapter import _Backend, torch_threads
from xdiyo_analytics.graphs.packed_execution import fallback_reason
from xdiyo_analytics.graphs.dependency import prepare_dependency_group
from xdiyo_analytics.neural import Reconstruction


@pytest.mark.parametrize('mode',['dependency','dependency_lean','dependency_packed'])
@pytest.mark.parametrize('kind',['cnn','boundary'])
def test_unsupported_execution_preserves_native_full_updates_and_prediction(mode,kind):
    with torch_threads(1):
        if kind=='cnn':
            spec,store=graph_setup(Reconstruction(loss='kl'),deterministic=True)
            native_cfg=spec.config
            reason='unsupported_cnn'
        else:
            store=sample()
            native_cfg=config(boundary_memory=BoundaryMemory('fixed_days',7,'2020-01-01T00:00:00+01:00'),deterministic=True)
            reason='unsupported_boundary_memory'
        accelerated_cfg=replace(native_cfg,execution=GraphExecution(mode=mode))
        native=_Backend(native_cfg,store,7,'cpu')
        accelerated=_Backend(accelerated_cfg,store,7,'cpu')
        fit=context(store,[1,2,3,4])
        native.initialize(fit); accelerated.initialize(fit)
        assert fallback_reason(accelerated)==reason
        for _ in range(2):
            a=native.step(); b=accelerated.step()
            assert a==b
            assert_exact(native.model.state_dict(),accelerated.model.state_dict())
            assert_exact(native.optimizer.state_dict(),accelerated.optimizer.state_dict())
            assert_exact([p.grad for p in native.model.parameters()],[p.grad for p in accelerated.model.parameters()])
            assert native.checkpoint_metrics('train_loss',a)==accelerated.checkpoint_metrics('train_loss',b)
        prediction=context(store,[7,8],fit=False)
        a=native.predict(prediction); b=accelerated.predict(prediction)
        import pandas as pd
        pd.testing.assert_frame_equal(a['predict_proba'],b['predict_proba'],check_exact=True)
        assert accelerated._execution_trace['fallback_reasons']=={reason:1}
        assert accelerated._execution_trace['whole_pass_native']


@pytest.mark.parametrize('layers',[1,2,3])
def test_float64_dependency_logits_and_gradients_match_native(layers):
    from xdiyo_analytics.graphs.network import TemporalNetwork
    h=history([(1,'2020-01-01',1,2,1,0,10,900,1),
               (2,'2020-01-02',2,3,1,0,10,900,2),
               (3,'2020-01-03',3,4,1,0,10,900,3),
               (4,'2020-01-04',8,9,1,0,10,900,4)])
    groups=groups_for(TemporalEventStore(h,sample().team_seasons),layers=layers)
    full=groups[0]
    cone=prepare_dependency_group(groups,0)
    pre=groups.preprocessor
    torch.manual_seed(431)
    model=TemporalNetwork(groups.plan.config,len(pre.leagues),full[1]['x'].shape[1],full[4].shape[1]).double()
    other=deepcopy(model)
    def run(net,group):
        _,data,home,away,fixture=group
        data={k:([None if x is None else x for x in v] if k=='_dependency_rows' else v.double() if v.is_floating_point() else v) for k,v in data.items()}
        logits=net(data,home,away,fixture.double())
        logits.square().sum().backward()
        return logits
    with torch_threads(1):
        a=run(model,full); b=run(other,cone)
    torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-13)
    for (name,p),(_,q) in zip(model.named_parameters(),other.named_parameters()):
        assert (p.grad is None)==(q.grad is None),name
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-11,atol=1e-12,msg=name)


@pytest.mark.parametrize('mode',['dependency','dependency_lean','dependency_packed'])
def test_nonquantile_statistics_falls_back_with_native_auxiliary_loss(mode):
    from test_graph_extensions import forecast
    store=sample()
    cfg=config(statistics_forecast=forecast('combined',family='point'),deterministic=True)
    native=_Backend(cfg,store,21,'cpu')
    other=_Backend(replace(cfg,execution=GraphExecution(mode=mode)),store,21,'cpu')
    fit=context(store,[1,2,3,4])
    with torch_threads(1):
        native.initialize(fit); other.initialize(fit)
        assert fallback_reason(other)=='unsupported_statistics_family'
        a=native.step(); b=other.step()
        assert a==b
        assert 'train_statistics_loss' in b
        assert_exact(native.model.state_dict(),other.model.state_dict())
        assert_exact(native.optimizer.state_dict(),other.optimizer.state_dict())
        assert_exact([p.grad for p in native.model.parameters()],[p.grad for p in other.model.parameters()])


@pytest.mark.parametrize('kind,monitor,direction',[
    ('cnn','validation_objective','minimize'),
    ('boundary','train_loss','maximize'),
])
def test_adapter_fallback_retains_native_monitor_choices(kind,monitor,direction):
    from xdiyo_analytics.graphs import TemporalGraphAdapter
    from xdiyo_analytics.training import TrainingControl
    if kind=='cnn':
        spec,store=graph_setup(Reconstruction(loss='kl'),deterministic=True)
        cfg=spec.config
    else:
        store=sample()
        cfg=config(boundary_memory=BoundaryMemory('fixed_days',7,'2020-01-01T00:00:00+01:00'),deterministic=True)
    fit=context(store,[1,2,3])
    fit.validation=context(store,[5,6])
    control=TrainingControl(max_steps=1,seed=5,monitor=monitor,direction=direction)
    a=TemporalGraphAdapter(cfg,store,control=control)
    b=TemporalGraphAdapter(replace(cfg,execution=GraphExecution(mode='dependency_packed')),store,control=control)
    a.fit(fit); b.fit(fit)
    assert_exact(a.backend_.model.state_dict(),b.backend_.model.state_dict())
    assert_exact(a.backend_.optimizer.state_dict(),b.backend_.optimizer.state_dict())
    assert a.control.monitor==b.control.monitor==monitor
    assert a.control.direction==b.control.direction==direction
    assert b.training_summary_['compatibility_fallback'].startswith('unsupported_')


@pytest.mark.parametrize('mode',['dependency','dependency_lean','dependency_packed'])
def test_empty_dependency_over_nonempty_history_preserves_adamw_zero_gradients(mode):
    # The only forecast teams have never appeared in eligible history. Native
    # graph operations still create zero gradients for edge-layer parameters,
    # so AdamW must apply weight decay rather than skip parameters with grad=None.
    h=history([(1,'2019-12-01',8,9,1,0,10,900,1),
               (2,'2020-01-01',90,91,1,0,10,900,2)])
    store=TemporalEventStore(h,sample().team_seasons)
    cfg=config(weight_decay=.1,deterministic=True)
    accelerated_cfg=replace(cfg,execution=GraphExecution(mode=mode,headroom_bytes=1))
    fit=context(store,[1])
    native=_Backend(cfg,store,19,'cpu')
    accelerated=_Backend(accelerated_cfg,store,19,'cpu')
    with torch_threads(1):
        native.initialize(fit); accelerated.initialize(fit)
        from xdiyo_analytics.graphs.dependency import select_dependencies
        selection=select_dependencies(accelerated.train_groups.plan,0)
        assert selection.native_edge_count>0
        assert len(selection.layer_rows[0])==0
        before=native.model.layers[0].q.weight.detach().clone()
        for _ in range(2):
            a=native.step(); b=accelerated.step()
            assert a==b
            assert native.model.layers[0].q.weight.grad is not None
            assert not bool(native.model.layers[0].q.weight.grad.any())
            assert_exact([p.grad for p in native.model.parameters()],[p.grad for p in accelerated.model.parameters()])
            assert_exact(native.model.state_dict(),accelerated.model.state_dict())
            assert_exact(native.optimizer.state_dict(),accelerated.optimizer.state_dict())
            assert accelerated._execution_trace['fallback_reasons']=={'empty_dependency':1}
        assert not torch.equal(before,native.model.layers[0].q.weight)



def test_direct_empty_cone_requires_fallback_for_optimizer_semantics():
    # Demonstrate why the admission fallback is required, independently of the
    # admission implementation: equal logits alone do not preserve AdamW decay.
    h=history([(1,'2019-12-01',8,9,1,0,10,900,1),
               (2,'2020-01-01',90,91,1,0,10,900,2)])
    store=TemporalEventStore(h,sample().team_seasons)
    backend=_Backend(config(weight_decay=.1),store,19,'cpu')
    with torch_threads(1):
        backend.initialize(context(store,[1]))
        full=backend.train_groups[0]
        cone=prepare_dependency_group(backend.train_groups,0)
        _,data,home,away,fixture=full
        native=backend.model(data,home,away,fixture)
        native.sum().backward()
        zero=backend.model.layers[0].q.weight.grad
        assert zero is not None and not bool(zero.any())
        backend.optimizer.zero_grad(set_to_none=True)
        _,data,home,away,fixture=cone
        reduced=backend.model(data,home,away,fixture)
        reduced.sum().backward()
        assert backend.model.layers[0].q.weight.grad is None
        torch.testing.assert_close(native,reduced,rtol=0,atol=0)
