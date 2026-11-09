"""Exact-cutoff context routes, zero initialization, isolation and persistence."""
from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
import torch

from test_temporal_graph import sample, config, context
from xdiyo_analytics.graphs import TemporalEventStore, TemporalGraphAdapter, TemporalGraphSerializer, GoalAuxiliary
from xdiyo_analytics.graphs.phase_context import PhaseRankLookup, PhaseRankFeatures, VALUES, bounded
from xdiyo_analytics.graphs.adapter import _Backend
from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor
from xdiyo_analytics.graphs.goals import poisson_goal_loss
from xdiyo_analytics.graphs.execution import prepare_determinism
from xdiyo_analytics.training import TrainingControl
from xdiyo_analytics.training.native_resume import capture_backend, restore_backend
from examples.gat_phase_form_rank.initialization import mapped_initial


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(params=['cpu', 'cuda:0'])
def device(request):
    if request.param.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    prepare_determinism(config(deterministic=True), request.param)
    return request.param


def source():
    store = sample(); cfg = config()
    table = store.schedule[['competition_id','season_id','event_id','home_id','away_id','kickoff_at']].copy()
    table['cutoff_at'] = [cfg.history_cutoff(t) for t in store.issue_times(store.schedule, cfg)]
    for n, column in enumerate(VALUES):
        table[column] = ((np.arange(len(table))+n)%5)/4
    table.loc[0,list(VALUES)] = 1.  # Native opening gate is observed, not missing.
    table.loc[1,list(VALUES)] = np.nan
    lookup = PhaseRankLookup(table, provenance={'test':'fixed native inputs'})
    return TemporalEventStore(store.history, store.team_seasons, schedule=store.schedule, phase_context=lookup)


def backend(store, cfg, device='cpu'):
    ctx = context(store,[2,3,4]); ctx.validation = context(store,[6,7])
    obj = _Backend(cfg,store,17,device); obj.initialize(ctx)
    return obj


def spec(store, route):
    window = {'D1':1,'D5':5,'D10':10,'D20':20,'DE1':1,'CD_DW':10}.get(route)
    return PhaseRankFeatures(route,store.phase_context.source_id,window)


def test_lookup_bounded_gate_masks_identity_and_ownership(tmp_path):
    store=source(); lookup=store.phase_context
    table=lookup.frame; table.loc[0,'countdown']=.123
    assert lookup.frame.countdown.iloc[0] == 1.
    np.testing.assert_array_equal(bounded([[1.,np.nan,0.]]),[[1.,.5,0.,0.,1.,0.]])
    for invalid in (np.inf,-.1,1.1):
        with pytest.raises(ValueError): bounded([[invalid]])
    lookup.save(tmp_path); loaded=PhaseRankLookup.load(tmp_path)
    assert lookup.source_id == loaded.source_id
    rows=store.schedule.iloc[[3,1,0]].copy()
    rows['_phase_cutoff']=[config().history_cutoff(t) for t in store.issue_times(rows,config())]
    expected=lookup.fixture(rows,spec(store,'D1'))
    np.testing.assert_array_equal(expected[1],[.5,.5,1,1])
    np.testing.assert_array_equal(expected[2],[1,1,0,0])
    np.testing.assert_array_equal(expected,loaded.fixture(rows,spec(store,'D1')))
    rows['_phase_cutoff'] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError,match='exact saved'): lookup.fixture(rows,spec(store,'D1'))


def test_reciprocal_and_historical_context_invariance():
    store=source(); original=store.phase_context.edges(store.history)
    h=store.history
    for _,indices in h.groupby(['competition_id','season_id','event_id']).indices.items():
        np.testing.assert_array_equal(original[indices[0]],original[indices[1]][[1,0,3,2]])
    # Own-result edits may change ordinary goal inputs, never fixed pre-match context.
    h[['goals_for','goals_against']]=99
    np.testing.assert_array_equal(original,store.phase_context.edges(h))
    for index in ([8,3,2,0],[0,1],list(range(len(h)-1,-1,-1))):
        np.testing.assert_array_equal(original[index],store.phase_context.edges(h.iloc[index]))
    cfg=replace(config(),phase_rank_features=spec(store,'E1'))
    prep=backend(store,cfg).preprocessor
    earlier=store.snapshot(pd.Timestamp('2020-01-12',tz='UTC'),cfg)
    later=store.snapshot(pd.Timestamp('2020-02-01',tz='UTC'),cfg)
    a=prep.native_tensors(earlier,'cpu',store)['x'][:,-4:]
    b=prep.native_tensors(later,'cpu',store)['x'][:,-4:]
    torch.testing.assert_close(a,b[earlier.edges.index],atol=0,rtol=0)


@pytest.mark.parametrize('mode',['snapshot','time_features'])
@pytest.mark.parametrize('route',['CD','D1','D5','D10','D20','E1','DE1','CD_DW'])
def test_zero_columns_preserve_family_forward_and_new_columns_learn(device,mode,route):
    store=source()
    cfg=config(temporal_mode='time_features',deterministic=True,goal_auxiliary=GoalAuxiliary(beta=0))
    canonical=backend(store,cfg,device)
    base=backend(store,replace(cfg,temporal_mode=mode),device)
    treatment=backend(store,replace(cfg,temporal_mode=mode,phase_rank_features=spec(store,route)),device)
    for model in (base,treatment):
        model.model.load_state_dict(mapped_initial(canonical.model.state_dict(),canonical.preprocessor,model.preprocessor))
    group=base.train_groups[0]; changed=treatment.train_groups[0]
    _,data,home,away,fixture=group
    _,x,h,a,f=changed
    torch.testing.assert_close(data['x'],x['x'][:,:base.preprocessor.edge_dim],atol=0,rtol=0)
    torch.testing.assert_close(fixture,f[:,:base.preprocessor.fixture_dim],atol=0,rtol=0)
    b_logits,b_rates=base.model(data,home,away,fixture,return_goals=True)
    logits,rates=treatment.model(x,h,a,f,return_goals=True)
    torch.testing.assert_close(logits,b_logits,atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(rates,b_rates,atol=1e-7,rtol=1e-6)
    names=treatment.preprocessor.fixture_encoded_columns
    assert len(names)==len(set(names))
    assert treatment.preprocessor.edge_columns==base.preprocessor.edge_columns
    (logits.square().sum()+rates.sum()).backward()
    if spec(store,route).fixture_columns:
        for name in ('head.0.weight','goal_head.weight'):
            assert dict(treatment.model.named_parameters())[name].grad[:,4*cfg.embedding_dim+base.preprocessor.fixture_dim:].abs().sum()>0
    if spec(store,route).edge_columns:
        for layer in treatment.model.layers:
            assert layer.e.weight.grad[:,-4:].abs().sum()>0
            assert layer.u.weight.grad[:,-4:].abs().sum()>0


def test_bc_matching_gradient_routes_dormant_head_and_resume_rejection(device):
    store=source()
    b_cfg=config(temporal_mode='time_features',deterministic=True,goal_auxiliary=GoalAuxiliary(beta=0),phase_rank_features=spec(store,'DE1'))
    b=backend(store,b_cfg,device); c=backend(store,replace(b_cfg,goal_auxiliary=GoalAuxiliary(beta=.1)),device)
    for name,value in b.model.state_dict().items(): torch.testing.assert_close(value,c.model.state_dict()[name],atol=0,rtol=0)
    group=c.train_groups[0]; positions,data,home,away,fixture=group
    logits,rates=c.model(data,home,away,fixture,return_goals=True)
    b_values=b.model(*group[1:],return_goals=True)
    for expected,actual in zip(b_values,(logits,rates)): torch.testing.assert_close(expected,actual,atol=0,rtol=0)
    params=dict(c.model.named_parameters())
    goal=poisson_goal_loss(rates,c.train_goals[positions]).mean()
    gradients=dict(zip(params,torch.autograd.grad(goal,tuple(params.values()),allow_unused=True,retain_graph=True)))
    assert all(gradients[n] is None for n in params if n.startswith('head.'))
    assert gradients['goal_head.weight'].abs().sum()>0
    assert gradients['layers.0.u.weight'].abs().sum()>0
    outcome=torch.nn.functional.cross_entropy(logits,c.train_targets[positions])
    assert torch.autograd.grad(outcome,params['goal_head.weight'],allow_unused=True)[0] is None
    original={n:p.detach().clone() for n,p in b.model.named_parameters() if n.startswith('goal_head.')}
    b.step()
    for name,expected in original.items():
        p=dict(b.model.named_parameters())[name]
        torch.testing.assert_close(p,expected,atol=0,rtol=0)
        assert p.grad is None and p not in b.optimizer.state
    with pytest.raises(ValueError): restore_backend(c,capture_backend(b))


def test_phase_context_resume_preserves_next_update(device):
    store=source();cfg=config(deterministic=True,goal_auxiliary=GoalAuxiliary(beta=.1),phase_rank_features=spec(store,'DE1'))
    original=backend(store,cfg,device)
    original.step()
    saved=capture_backend(original)
    expected=original.step()
    resumed=backend(store,cfg,device)
    restore_backend(resumed,saved)
    actual=resumed.step()
    for key in expected: assert actual[key]==expected[key]
    for name,value in original.model.state_dict().items():
        torch.testing.assert_close(value,resumed.model.state_dict()[name],atol=0,rtol=0)
    a,b=original.optimizer.state_dict(),resumed.optimizer.state_dict()
    assert a['param_groups']==b['param_groups']
    for index,values in a['state'].items():
        for name,value in values.items():
            torch.testing.assert_close(value,b['state'][index][name],atol=0,rtol=0)


@pytest.mark.parametrize('daily', [False, True])
@pytest.mark.parametrize('reviewed', [False, True])
def test_context_persistence_and_cross_route_schema_rejection(tmp_path,device,daily,reviewed):
    store=source()
    if daily:
        from xdiyo_analytics.graphs.phase_context import DAILY_POLICY
        lookup=PhaseRankLookup(store.phase_context.frame,provenance={'test':'daily policy'},policy=DAILY_POLICY)
        store=TemporalEventStore(store.history,store.team_seasons,schedule=store.schedule,phase_context=lookup)
    if reviewed:
        from xdiyo_analytics.graphs.phase_context import REVIEWED_POLICY, REVIEWED_DAILY_POLICY
        lookup=PhaseRankLookup(store.phase_context.frame,provenance={'test':'reviewed points policy'},
                               policy=REVIEWED_DAILY_POLICY if daily else REVIEWED_POLICY)
        store=TemporalEventStore(store.history,store.team_seasons,schedule=store.schedule,phase_context=lookup)
    phase=replace(spec(store,'CD_DW'),policy=store.phase_context.policy)
    cfg=config(deterministic=True,goal_auxiliary=GoalAuxiliary(beta=.1),phase_rank_features=phase)
    adapter=TemporalGraphAdapter(cfg,store,control=TrainingControl(max_steps=1),device=device)
    adapter.fit(context(store,[2,3,4]))
    query=context(store,[7,8],fit=False)
    before=adapter.predict(query); goals=adapter.predict_goals(query)
    serializer=TemporalGraphSerializer(store,device=device)
    serializer.save(adapter,tmp_path)
    restored=serializer.load(tmp_path)
    after=restored.predict(query)
    for key in before: pd.testing.assert_frame_equal(before[key],after[key],check_exact=True)
    pd.testing.assert_frame_equal(goals,restored.predict_goals(query),check_exact=True)
    with pytest.raises(ValueError): restored.predict(context(store,[0],fit=False))
    state=adapter.backend_.preprocessor.state()
    for route in ('D10','D5'):
        with pytest.raises(ValueError): GraphPreprocessor.from_state(replace(cfg,phase_rank_features=spec(store,route)),state)
    metadata=json.loads((tmp_path/'graph.json').read_text());metadata['config']['phase_rank_features']['window']=5
    (tmp_path/'graph.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError): serializer.load(tmp_path)
