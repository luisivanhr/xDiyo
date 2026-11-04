"""Exact retained acceleration and private cache ownership acceptance gates."""
from dataclasses import fields,replace,asdict
import gc,json,weakref
import numpy as np
import pytest
import torch
from test_gat_reconciliation import retained_fixture
from test_graph_preparation_memory import assert_exact
from xdiyo_analytics.graphs import RetainedGATExecution,TemporalGraphModel
from xdiyo_analytics.graphs.research_training import MONITOR
from xdiyo_analytics.graphs.research_execution import AcceleratedRetainedGATBackend
from xdiyo_analytics.graphs.research_cache import BorrowCache,tensors
from xdiyo_analytics.graphs.execution import prepare_determinism
from xdiyo_analytics.training.residency import InputCachePolicy,tensor_bytes
from xdiyo_analytics.training import TrainingControl
from xdiyo_analytics.training.native_resume import capture_backend,restore_backend,rng_state

FLAGS=tuple(f.name for f in fields(RetainedGATExecution) if f.name!='profile')
OFF=RetainedGATExecution(**dict.fromkeys(FLAGS,False))


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def build(family,device,execution=None,budget=64*1024**2):
    a,ctx,_=retained_fixture(family);a.research_execution=execution
    a.control=TrainingControl(max_steps=3,monitor=MONITOR);a.configure_device(device)
    a.configure_input_cache(budget,0);prepare_determinism(a.config,device)
    b=a._make_backend(17);b.initialize(ctx)
    if execution is not None:b.audit_borrow=True
    return a,b,ctx


def math_state(b):
    state=capture_backend(b);state.pop('research_execution')
    return state


def close(b):
    if hasattr(b,'close'):b.close()
    else:
        for name in ('train_groups','validation_groups'):getattr(b,name)._resident.close()


@pytest.mark.parametrize('device',['cpu','cuda:0'])
@pytest.mark.parametrize('family',['bn','ce1','cddw'])
@pytest.mark.parametrize('flag',[*FLAGS,'all','none'])
def test_each_flag_and_combined_exact_updates(family,device,flag):
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    changes=RetainedGATExecution() if flag=='all' else OFF if flag=='none' else replace(OFF,**{flag:True})
    _,a,_=build(family,device);_,b,_=build(family,device,changes)
    try:
        assert_exact(math_state(a),math_state(b))
        assert list(a.model.state_dict())==list(b.model.state_dict())
        for step in range(3):
            va=a.step();vb=b.step();assert_exact(va,vb)
            rng=rng_state(device);ma=a.checkpoint_metrics(MONITOR,va);mb=b.checkpoint_metrics(MONITOR,vb)
            assert_exact(rng,rng_state(device));assert_exact(math_state(a),math_state(b))
            assert_exact(a._inner_probability,b._inner_probability);assert_exact(a._inner_rates,b._inner_rates)
            for key in ma:
                if key.startswith(('validation_','all13_','inner_raw_')):assert ma[key]==mb[key]
            if changes.omit_diagnostic_replay:
                assert 'train_loss' not in mb and mb['pre_update_train_ce']==vb['train_loss']
            else:assert ma['train_loss']==mb['train_loss']
        if changes.borrow_inputs:
            assert b.train_groups._resident.pool is b.validation_groups._resident.pool is None
            assert b.borrow.resident<=b.input_cache_policy.max_bytes and b.borrow.transient==0
        if changes.cache_indices and not changes.borrow_inputs and device.startswith('cuda'):
            assert all(owner.pool is b.train_groups._resident.pool for owner in b._index_owners.values())
    finally:close(a);close(b)


def test_private_borrow_lifetimes_indices_mutation_and_oversize():
    _,b,_=build('ce1','cpu');groups=b.train_groups
    cache=BorrowCache('cpu',InputCachePolicy(10**8,0),audit=True)
    try:
        with cache.lease(groups,0,cache_indices=True) as (a,ix):
            expected=sum(t.numel()*t.element_size() for t in tensors((a,ix)))
            assert cache.resident==expected and cache.transient==0
            with cache.lease(groups,0,cache_indices=True) as (other,jx):
                assert a[1]['x'].data_ptr()==other[1]['x'].data_ptr() and ix.data_ptr()==jx.data_ptr()
                with pytest.raises(TypeError):a[1]['new']=1
                with pytest.raises(ValueError):a[0][0]=3
                cache.close_owner(groups)
                with pytest.raises(RuntimeError,match='closed'):
                    with cache.lease(groups,0):pass
                assert cache.resident>0
            assert cache.resident>0
        assert cache.resident==0
        audit=BorrowCache('cpu',InputCachePolicy(10**8,0),audit=True)
        with pytest.raises(RuntimeError,match='mutated'):
            with audit.lease(groups,0) as (value,ix):value[1]['x'].add_(1)
        assert audit.resident==audit.transient==0;audit.close()
        with groups.lease(0) as public:
            expected=public[1]['x'].clone();public[1]['x'].add_(55)
        with groups.lease(0) as public:torch.testing.assert_close(public[1]['x'],expected,atol=0,rtol=0)
        tiny=BorrowCache('cpu',InputCachePolicy(1,0),audit=True)
        with tiny.lease(groups,0) as (value,ix):
            assert tiny.resident==0 and tiny.transient==sum(t.numel()*t.element_size() for t in tensors((value,ix)))
            b.model(*value[1:]).square().sum().backward()
        assert tiny.transient==0;tiny.close()
    finally:cache.close();close(b)


def test_weak_owners_frozen_schema_and_failure_cleanup(monkeypatch):
    a,b,ctx=build('bn','cpu',RetainedGATExecution());groups=b.train_groups
    cache=b._borrow_cache()
    with cache.lease(groups,0) as (value,ix):saved=value[1]['x'].clone()
    groups.preprocessor.edge_scale.mean[:]=123
    with cache.lease(groups,0) as (value,ix):torch.testing.assert_close(value[1]['x'],saved,atol=0,rtol=0)
    ref=weakref.ref(groups);del b.train_groups;del groups;gc.collect();cache._trim(0)
    assert ref() is None and cache.resident==0
    close(b)
    seen=[];original=AcceleratedRetainedGATBackend.step
    def fail(self):
        seen.append(self)
        with self._group_lease(self.train_groups,0):raise RuntimeError('injected consumer failure')
    monkeypatch.setattr(AcceleratedRetainedGATBackend,'step',fail)
    with pytest.raises(RuntimeError,match='injected'):a.fit(ctx)
    assert len(seen)==1 and seen[0].borrow.closed and seen[0].borrow.resident==seen[0].borrow.transient==0


def test_opt_in_eligibility_catalogue_persistence_and_diagnostics(tmp_path):
    from xdiyo_analytics.ui.recipe import catalog_for_ui
    from xdiyo_analytics.graphs import TemporalGraphSerializer
    changes=RetainedGATExecution();catalog=catalog_for_ui()
    assert catalog.build(json.loads(json.dumps(catalog.encode(changes))))==changes
    a,ctx,query=retained_fixture('ce1');a.control=TrainingControl(max_steps=1,monitor=MONITOR)
    a.research_execution=changes
    with pytest.raises(ValueError,match='exact supported'):
        changes.validate_config(replace(a.config,embedding_dim=64))
    with pytest.raises(ValueError,match='explicit boolean'):RetainedGATExecution(cache_indices=1)
    a.fit(ctx)
    assert not a.training_summary_['research_execution_diagnostics']['post_update_training_replay']
    assert a.backend_.borrow.closed and a.backend_.borrow.resident==0
    serializer=TemporalGraphSerializer(event_store=a.event_store);serializer.save(a,tmp_path/'model')
    loaded=serializer.load(tmp_path/'model')
    assert loaded.research_execution is None and loaded.saved_research_execution_==asdict(changes)
    for key,value in a.predict(query).items():
        import pandas as pd
        pd.testing.assert_frame_equal(value,loaded.predict(query)[key],check_exact=True)
    serializer.save(loaded,tmp_path/'reexport')
    assert json.loads((tmp_path/'reexport/graph.json').read_text())['research_execution']==asdict(changes)
