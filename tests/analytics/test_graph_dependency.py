"""Causal cone preparation preserves native inputs, validation and residuals."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import torch
from test_temporal_graph import sample, config, context, history
from xdiyo_analytics.graphs import TemporalEventStore
from xdiyo_analytics.graphs.streaming import GroupPlan, PreparedGroups
from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor
from xdiyo_analytics.graphs.dependency import select_dependencies, prepare_dependency_group


def groups_for(store=None, layers=2, issuance='2020-03-01T12:00Z', cold=False):
    store = sample() if store is None else store
    ctx = context(store, [0])
    if cold:
        ctx.metadata['home_id'] = 999
    cfg = config(layers=layers)
    plan = GroupPlan(store,cfg,ctx,pd.Series(pd.Timestamp(issuance),index=ctx.metadata.index))
    numeric, leagues = plan.fitting_inputs()
    prep = GraphPreprocessor(cfg).fit(store.history,(),ctx.metadata,numeric,extra_leagues=leagues)
    return PreparedGroups(plan,prep,'cpu')


@pytest.mark.parametrize('layers',[1,2,3])
@pytest.mark.parametrize('cold',[False,True])
@pytest.mark.parametrize('issuance',['2019-01-01T12:00Z','2020-03-01T12:00:00.000000999Z'])
def test_lean_matches_snapshot_inputs(layers,cold,issuance):
    groups=groups_for(layers=layers,cold=cold,issuance=issuance)
    selection=select_dependencies(groups.plan,0)
    a=prepare_dependency_group(groups,0,selection,lean=False)
    groups.preprocessor._edge_cache=None
    b=prepare_dependency_group(groups,0,selection,lean=True)
    np.testing.assert_array_equal(a[0],b[0])
    for key in a[1]:
        if key=='_dependency_rows':
            for x,y in zip(a[1][key],b[1][key]):
                assert (x is None)==(y is None)
                if x is not None: assert torch.equal(x,y)
        else:
            assert torch.equal(a[1][key],b[1][key]),key
    for x,y in zip(a[2:],b[2:]): assert torch.equal(x,y)


def test_chain_residual_closure_and_order():
    h=history([(1,'2020-01-01',1,2,1,0,10,900,1),
               (2,'2020-01-02',2,3,1,0,10,900,2),
               (3,'2020-01-03',3,4,1,0,10,900,3),
               (4,'2020-01-04',8,9,1,0,10,900,4)])
    m=sample().team_seasons
    store=TemporalEventStore(h,m)
    groups=groups_for(store,layers=3)
    selection=select_dependencies(groups.plan,0)
    assert 8 not in selection.teams and 9 not in selection.teams
    for left,right in zip(selection.layer_rows,selection.layer_rows[1:]):
        assert set(right)<=set(left)
    for rows in selection.layer_rows:
        assert np.all(rows[1:]>rows[:-1])
    assert len(selection.layer_rows[0])<selection.native_edge_count


@pytest.mark.parametrize('budget',[0,128*1024**2])
def test_unused_invalid_eligible_rows_validate_on_cold_cache(budget):
    h=history([(1,'2020-01-01',1,2,1,0,10,900,1),
               (2,'2020-01-02',8,9,1,0,10,900,2)])
    store=TemporalEventStore(h,sample().team_seasons)
    groups=groups_for(store)
    groups.preprocessor._edge_cache_limit=budget
    # Trusted immutable store is deliberately corrupted only for this test.
    store._history['goals_for']=store._history.goals_for.astype(float)
    store._history.loc[store._history.team_id.eq(8),'goals_for']=np.inf
    with pytest.raises(ValueError,match='Infinite'):
        prepare_dependency_group(groups,0)


def test_source_cache_scaler_mutation_invalidates():
    groups=groups_for()
    prep=groups.preprocessor
    rows=np.arange(len(groups.plan.store._history))
    before=prep.source_edges(groups.plan.store,rows)
    prep.edge_scale.mean[0]+=1
    after=prep.source_edges(groups.plan.store,rows)
    assert not np.array_equal(before,after)


@pytest.mark.parametrize('offset',[2**63,2**70])
def test_exact_large_ids_and_release_cutoffs(offset):
    source=sample()
    h,m=source.history,source.team_seasons
    for frame,columns in ((h,('team_id','opponent_id')),(m,('team_id',))):
        for col in columns:
            frame[col]=pd.Series([int(v)+offset for v in frame[col]],dtype=object)
    h['result_available_at']=h.kickoff_at+pd.Timedelta(hours=2)
    h.loc[h.event_id.eq(2),'result_available_at']=pd.Timestamp('2020-03-01T00:00Z')
    h.loc[h.event_id.eq(3),'status']='scheduled'
    store=TemporalEventStore(h,m)
    groups=groups_for(store,issuance='2020-03-01T12:00:00.000000999Z')
    selection=select_dependencies(groups.plan,0)
    assert all(type(t) is int for t in selection.teams)
    native=store._eligible_rows(groups.plan.config.history_cutoff(selection.issuance))
    np.testing.assert_array_equal(selection.eligible_rows,native)
    a=prepare_dependency_group(groups,0,selection,lean=False)
    b=prepare_dependency_group(groups,0,selection,lean=True)
    for key in ('x','receiver','sender','log_weight','gate','league','entry'):
        assert torch.equal(a[1][key],b[1][key]),key
    assert torch.equal(a[4],b[4])


def test_future_invalid_rows_remain_unvalidated_until_release():
    h=history([(1,'2020-01-01',1,2,1,0,10,900,1),
               (2,'2020-01-02',8,9,1,0,10,900,2)])
    h['result_available_at']=h.kickoff_at+pd.Timedelta(hours=2)
    h.loc[h.event_id.eq(2),'result_available_at']=pd.Timestamp('2020-04-01T00:00Z')
    store=TemporalEventStore(h,sample().team_seasons)
    groups=groups_for(store)
    store._history['goals_for']=store._history.goals_for.astype(float)
    store._history.loc[store._history.team_id.eq(8),'goals_for']=np.inf
    prepare_dependency_group(groups,0)
    ctx=context(store,[0])
    later_plan=GroupPlan(store,groups.plan.config,ctx,pd.Series(pd.Timestamp('2020-04-02T12:00Z'),index=ctx.metadata.index))
    later=PreparedGroups(later_plan,groups.preprocessor,'cpu')
    with pytest.raises(ValueError,match='Infinite'):
        prepare_dependency_group(later,0)


def test_timestamp_overflow_matches_native_failure():
    h=history([(1,'1700-01-01',1,2,1,0,10,900,1)])
    store=TemporalEventStore(h,sample().team_seasons)
    groups=groups_for(store,issuance='1700-01-01T12:00Z')
    _,positions=groups.plan.descriptors[0]
    groups.plan.descriptors=((pd.Timestamp('2200-01-01T12:00Z'),positions),)
    selection=select_dependencies(groups.plan,0)
    with pytest.raises(OverflowError):
        prepare_dependency_group(groups,0,selection,lean=False)
    with pytest.raises(OverflowError):
        prepare_dependency_group(groups,0,selection,lean=True)


def test_source_cache_configuration_identity_changes():
    groups=groups_for()
    prep=groups.preprocessor
    rows=np.arange(len(groups.plan.store._history))
    before=prep.source_edges(groups.plan.store,rows)
    old=prep._edge_cache
    prep.config=replace(prep.config,half_life_days=prep.config.half_life_days+1)
    after=prep.source_edges(groups.plan.store,rows)
    assert prep._edge_cache is not old
    np.testing.assert_array_equal(before,after)


@pytest.mark.parametrize('layers',[1,2,3])
def test_sparse_chain_sender_support_and_fixture_parity(layers):
    h=history([(1,'2020-01-01',1,2,1,0,10,900,1),
               (2,'2020-01-02',2,3,1,0,10,900,2),
               (3,'2020-01-03',3,4,1,0,10,900,3),
               (4,'2020-01-04',8,9,1,0,10,900,4)])
    groups=groups_for(TemporalEventStore(h,sample().team_seasons),layers=layers)
    a=prepare_dependency_group(groups,0,lean=False)
    b=prepare_dependency_group(groups,0,lean=True)
    for key in ('x','receiver','sender','log_weight','gate','league','entry'):
        assert torch.equal(a[1][key],b[1][key]),key
    assert torch.equal(a[4],b[4])
