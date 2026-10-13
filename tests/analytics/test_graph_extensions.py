"""Opt-in memory/statistics: causal partitions, gradients, native persistence."""
from dataclasses import replace, asdict
import numpy as np
import pandas as pd
import pytest
import torch
from test_temporal_graph import sample, config, context
from xdiyo_analytics.graphs import BoundaryMemory, StatisticsForecast, StatisticTarget, TemporalGraphAdapter, TemporalGraphSerializer, TemporalEventStore
from xdiyo_analytics.graphs.adapter import _Backend, torch_threads
from xdiyo_analytics.graphs.memory import BoundaryReplay
from xdiyo_analytics.graphs.statistics import labels, StatisticsScale
from xdiyo_analytics.graphs.statistics_network import StatisticsHead
from xdiyo_analytics.training import TrainingControl


@pytest.fixture(autouse=True)
def threads():
    with torch_threads(1):
        yield


def forecast(route='auxiliary', family='point'):
    return StatisticsForecast(targets=(StatisticTarget('goals', 'home', family=family), StatisticTarget('goals', 'away', family=family)), route=route)


@pytest.mark.parametrize('route', ['auxiliary','combined','bottleneck'])
@pytest.mark.parametrize('memory', [None, BoundaryMemory('fixed_days', days=7, anchor='2020-01-01T00:00:00+01:00')])
def test_native_fit_save_inference_inspection(tmp_path, route, memory):
    s = sample(); cfg = config(statistics_forecast=forecast(route), boundary_memory=memory, deterministic=True)
    fit = context(s, [1,2,3]); fit.validation = context(s, [5,6])
    a = TemporalGraphAdapter(cfg, s, control=TrainingControl(max_steps=2)); a.fit(fit)
    assert {'train_statistics_loss','validation_statistics_loss'} <= set(a.training_history_.metric)
    pred = context(s, [7,8], fit=False)
    p = a.predict(pred); f = a.predict_statistics(pred)
    assert np.isfinite(f.to_numpy()).all() and (f.to_numpy() >= 0).all()
    ser = TemporalGraphSerializer(embed_event_store=True); ser.save(a,tmp_path)
    b = ser.load(tmp_path)
    pd.testing.assert_frame_equal(p['predict_proba'], b.predict(pred)['predict_proba'], check_exact=True)
    pd.testing.assert_frame_equal(f, b.predict_statistics(pred), check_exact=True)
    queries = pd.DataFrame({'team_id':[1,2,3], 'issuance_at':pd.Timestamp('2020-03-01',tz='UTC')})
    pd.testing.assert_frame_equal(a.encode_teams(queries), b.encode_teams(queries), check_exact=True)
    pd.testing.assert_frame_equal(a.attention_weights(queries), b.attention_weights(queries), check_exact=True)
    assert b.backend_.model_version == a.backend_.model_version


def test_routes_and_end_to_end_gradient():
    for route in ('auxiliary','combined','bottleneck'):
        b = _Backend(config(statistics_forecast=forecast(route)), sample(), 2, 'cpu'); b.initialize(context(b.store,[1,2,3]))
        _, data, home, away, fixture = b.train_groups[1]
        b.model(data,home,away,fixture).sum().backward()
        grads = [p.grad for p in b.model.statistics_head.parameters()]
        assert any(g is not None and g.abs().sum() > 0 for g in grads) == (route != 'auxiliary')
        expected = 2 if route == 'bottleneck' else 4*4+fixture.shape[1]+(2 if route == 'combined' else 0)
        assert b.model.head[0].in_features == expected
        # A constant statistics head must remove ALL direct input dependence.
        if route == 'bottleneck':
            with torch.no_grad():
                for p in b.model.statistics_head.parameters(): p.zero_()
            torch.testing.assert_close(b.model(data,home,away,fixture), b.model(data,away,home,fixture+100), rtol=0,atol=0)


def test_boundary_partition_detachment_no_accumulation_refresh():
    cfg = config(boundary_memory=BoundaryMemory('fixed_days',7,'2020-01-01T00:00:00+01:00'))
    b = _Backend(cfg,sample(),1,'cpu'); b.initialize(context(b.store,[1,2,3]))
    t = pd.Timestamp('2020-01-18T12:00Z'); r = BoundaryReplay(b,t)
    snap, data = r.snapshot(t,[1,2,3])
    assert r.start == pd.Timestamp('2020-01-14T23:00Z')
    old = set(b.store._eligible_rows(r.start)); new = set(snap.edges.index)
    assert not old & new and old | new == set(b.store._eligible_rows(snap.cutoff))
    assert all(not x.requires_grad for x in r.state.values())
    original = b.model.encode(data).detach().clone()
    _, again = r.snapshot(t,[1,2,3])
    torch.testing.assert_close(original,b.model.encode(again),rtol=0,atol=0)
    b.model.encode(again).sum().backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in b.model.layers.parameters())
    before = {k:v.clone() for k,v in r.state.items()}
    with torch.no_grad(): b.model.layers[0].output.weight.add_(.1)
    r2 = BoundaryReplay(b,t);r2.snapshot(t,[1,2,3])
    assert any(not torch.equal(before[k],r2.state[k]) for k in before)


def test_late_release_boundary_tie_and_future_mutation():
    s=sample();h=s.history
    h['result_available_at']=h.kickoff_at+pd.Timedelta(hours=3)
    h.loc[h.event_id==1,'result_available_at']=pd.Timestamp('2020-01-15T00:00Z')
    s=TemporalEventStore(h,s.team_seasons)
    cfg=config(boundary_memory=BoundaryMemory('fixed_days',7,'2020-01-01T00:00:00+01:00'))
    boundary=pd.Timestamp('2020-01-14T23:00Z')
    snap=s.snapshot(pd.Timestamp('2020-01-16T12:00Z'),cfg,_since=boundary)
    assert 1 in snap.edges.event_id.values
    assert 1 not in s.snapshot(boundary,cfg).edges.event_id.values
    b=_Backend(cfg,s,2,'cpu');b.initialize(context(s,[1,2,3]))
    t=pd.Timestamp('2020-01-18T12:00Z')
    first=BoundaryReplay(b,t).snapshot(t,[1,2])[1]
    h=s.history;h.loc[h.event_id==9,'goals_for']=42;h.loc[h.event_id==9,'goals_against']=42
    b.store=TemporalEventStore(h,s.team_seasons)
    second=BoundaryReplay(b,t).snapshot(t,[1,2])[1]
    for key in first: torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)


@pytest.mark.parametrize('device',['cpu','cuda:0'])
@pytest.mark.parametrize('family,proportion', [('point',False),('poisson',False),('negative_binomial',False),('inflated_beta',True)])
def test_distribution_support_masks_and_finite_gradients(family,proportion,device):
    import os
    if device.startswith('cuda') and (os.environ.get('XDIYO_RUN_GPU_AUDIT')!='1' or not torch.cuda.is_available()):
        pytest.skip('Explicit idle-GPU audit required.')
    cfg=StatisticsForecast((StatisticTarget('ballPossession' if proportion else 'goals',family=family),))
    head=StatisticsHead(cfg,3,4).to(device)
    x=torch.randn(4,3,requires_grad=True,device=device)
    pred,summary=head(x)
    y=torch.tensor([[0.],[1.],[.5 if proportion else 100.],[float('nan')]],device=device)
    w=torch.ones(4,device=device);den=torch.tensor([3.],device=device)
    loss=head.losses(pred,y,w,den);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all() and torch.isfinite(summary).all()
    assert (pred[0]['mean']>=0).all()
    if proportion: assert (pred[0]['mean']<=1).all()
    empty=head.losses(head(x)[0],y*float('nan'),w,torch.zeros(1,device=device));empty.backward()
    assert empty.item()==0


def test_training_scale_constant_missing_and_weight_normalization():
    cfg=forecast();y=np.array([[2,np.nan],[2,3],[2,3.]])
    scale=StatisticsScale(cfg).fit(y)
    assert scale.counts.tolist()==[3,2] and np.array_equal(scale.scale,[1,1])
    assert StatisticsScale.from_state(cfg,scale.state()).state()==scale.state()
    with pytest.raises(ValueError,match='No usable'): StatisticsScale(cfg).fit(y*np.nan)
    head=StatisticsHead(cfg,3,4);p,_=head(torch.ones(3,3));a=torch.tensor(y,dtype=torch.float32);w=torch.tensor([.5,1.,2.]);d=(torch.isfinite(a)*w[:,None]).sum(0)
    torch.testing.assert_close(head.losses(p,a,w,d),head.losses(p,a,w*100,d*100))


def test_targets_not_forward_inputs_and_half_goals_rejected():
    with pytest.raises(ValueError,match='ALL only'): StatisticTarget('goals',period='1ST')
    s=sample();b=_Backend(config(statistics_forecast=forecast()),s,1,'cpu');b.initialize(context(s,[1,2,3]))
    for group in b.train_groups: assert set(group[1]) == {'x','receiver','sender','log_weight','gate','league','entry'}
    # Future target changed, including reverse edge: earlier inputs identical.
    h=s.history;h.loc[h.event_id==9,['goals_for','goals_against']]=99
    s2=TemporalEventStore(h,s.team_seasons);b2=_Backend(b.config,s2,1,'cpu');b2.initialize(context(s2,[1,2,3]))
    for g1,g2 in zip(b.train_groups,b2.train_groups):
        for k in g1[1]:torch.testing.assert_close(g1[1][k],g2[1][k],rtol=0,atol=0)
    assert b.statistics_scale.state()==b2.statistics_scale.state()



from test_ui_workflow import ui_recipe


def test_native_ui_recipe_exports_and_reporters(ui_recipe,tmp_path):
    import json
    from test_graph_integration import graph_recipe
    from xdiyo_analytics.ui.recipe import catalog_for_ui, prepare_recipe, run_recipe, export_python, node
    recipe=graph_recipe(ui_recipe,tmp_path)
    settings=recipe['model']['params']['config']['params']
    settings.update(boundary_memory=node('graphs.BoundaryMemory',policy='fixed_days',days=4),
        statistics_forecast=node('graphs.StatisticsForecast',route='bottleneck',targets=[
            node('graphs.StatisticTarget',statistic='goals',side='home',family='poisson'),
            node('graphs.StatisticTarget',statistic='goals',side='away',family='negative_binomial'),
            node('graphs.StatisticTarget',statistic='cornerKicks',side='home',family='poisson')]))
    recipe['post_reporters']['statistics']=node('reporting.GraphStatisticsReporter',type='overall',partition='score')
    recipe['post_reporters']['loss']['params']['metrics']=['train_loss','train_statistics_loss']
    c=catalog_for_ui(); built=c.build(recipe['model'])
    assert c.build(json.loads(json.dumps(c.encode(built))))==built
    namespace={};exec(export_python(recipe).split('prepared =')[0],namespace)
    assert namespace['recipe']['model']==recipe['model']
    prepared=prepare_recipe(recipe);result=run_recipe(recipe,prepared=prepared)
    assert 'statistics' in result.training.folds[0].predictions
    from xdiyo_analytics.reporting import GraphStatisticsReporter, GATLossReporter, BetOutcomeReporter, BetPerformanceReporter
    from xdiyo_analytics.analysis import PostTrainingAnalysis
    from xdiyo_analytics.labels import Outcome, BetOption
    from xdiyo_analytics.evaluation import BetOffer, HighestExpectedProfit, Parlay
    reports=PostTrainingAnalysis({
        'statistics':GraphStatisticsReporter(type='overall',partition='score'),
        'loss':GATLossReporter(type='overall',metrics=('train_loss','train_statistics_loss')),
        'decisions':BetOutcomeReporter(type='overall',partition='score',history=prepared.outputs['history'],
            offers={'home':BetOffer(BetOption(Outcome(perspective='home'),selection='win'),odds=2.)},
            policy=HighestExpectedProfit(min_ev=-1),composition=Parlay(size=1)),
        'performance':BetPerformanceReporter(type='overall',partition='score',source='decisions')
    }).run(result.training)
    assert reports.studies[0].result.tables['scores'].n.sum()>0
    assert np.isfinite(reports.studies[0].result.tables['scores'].nll).all()
    assert reports.studies[-1].result.tables
    html=reports.to_html()
    assert 'Upcoming match-statistics' in html and 'Cumulative' in html


def test_season_movement_and_empty_segment_retains_carry():
    s=sample();m=s.team_seasons
    extra=m.iloc[[0]].copy();extra['entry_at']=pd.Timestamp('2020-01-08T00:00Z');extra['season_id']=901;extra['competition_id']=11;extra['movement']='promoted'
    m=pd.concat([m,extra],ignore_index=True)
    s=TemporalEventStore(s.history,m)
    cfg=config(boundary_memory=BoundaryMemory())
    b=_Backend(cfg,s,1,'cpu');b.initialize(context(s,[1,2,3]))
    t=pd.Timestamp('2020-03-01T12:00Z');r=BoundaryReplay(b,t);snap,data=r.snapshot(t,[1,2,3])
    assert r.start==pd.Timestamp('2020-01-07T23:00Z')
    assert snap.league[snap.teams.index(1)]==11
    # After all games, subsequent empty fixed segments preserve the state.
    b.config=replace(cfg,boundary_memory=BoundaryMemory('fixed_days',7,'2020-01-01T00:00:00+01:00'))
    r=BoundaryReplay(b,t);r.snapshot(pd.Timestamp('2020-02-08T12:00Z'),[1,2])
    before={k:v.clone() for k,v in r.state.items()}
    snap,data=r.snapshot(t,[1,2])
    assert len(snap.edges)==0
    for k in before: torch.testing.assert_close(before[k],r.state[k],atol=0,rtol=0)


def test_all_missing_auxiliary_group_and_checkpoint_metric_current_state():
    s=sample();h=s.history
    col=StatisticTarget().column;opp=col.replace('team::','opponent::')
    h[col]=np.where(h.event_id%2==0,3.,np.nan);h[opp]=h[col]
    s=TemporalEventStore(h,s.team_seasons)
    b=_Backend(config(statistics_forecast=StatisticsForecast()),s,4,'cpu');b.initialize(context(s,[1,2,3]))
    result=b.step();monitored=b.checkpoint_metrics('train_objective',result)
    expected=b._loss(b.train_groups,b.train_targets,b.train_weights)
    penalty=float((b.config.prior_penalty*(b.model.league.weight.square().sum()+b.model.entry.weight.square().sum())).detach())
    assert monitored['train_objective']==pytest.approx(expected+b._statistics_loss+penalty)
    assert all(np.isfinite(v) for v in monitored.values())


@pytest.mark.parametrize('kwargs', [dict(policy='sliding'),dict(days=0),dict(retention=1.1),dict(anchor='2020-01-01')])
def test_invalid_boundary_configuration(kwargs):
    with pytest.raises(ValueError): BoundaryMemory(**kwargs)


@pytest.mark.parametrize('kwargs', [dict(statistic='made_up'),dict(period='3RD'),dict(side='both'),dict(weight=0),dict(statistic='goals',family='inflated_beta'),dict(statistic='ballPossession',family='poisson')])
def test_invalid_statistic_configuration(kwargs):
    with pytest.raises(ValueError): StatisticTarget(**kwargs)


def test_half_targets_masks_transform_and_validation_is_not_fit():
    targets=(StatisticTarget('cornerKicks',period='1ST'),StatisticTarget('ballPossession',side='away',period='2ND',family='inflated_beta'))
    cfg=StatisticsForecast(targets)
    s=sample();h=s.history
    for t in targets:
        h[t.column]=h.event_id.astype(float) if not t.proportion else 50.
        h[t.column.replace('team::','opponent::')]=h[t.column]
    h.loc[h.event_id==2,targets[0].column]=np.nan
    h.loc[h.event_id==2,targets[0].column.replace('team::','opponent::')]=np.nan
    s=TemporalEventStore(h,s.team_seasons)
    fit=context(s,[1,2,3]);fit.validation=context(s,[5,6])
    b=_Backend(config(statistics_forecast=cfg),s,4,'cpu');b.initialize(fit)
    assert b.statistics_scale.counts.tolist()==[2,3]
    y=labels(s,fit.metadata,cfg,b.fitting_boundary)
    native=y.copy();native[:,1]*=100
    np.testing.assert_allclose(b.statistics_scale.inverse(b.statistics_scale.transform(y)),native,equal_nan=True)
    h2=s.history
    for col in [targets[0].column,targets[0].column.replace('team::','opponent::')]:h2.loc[h2.event_id>=6,col]=1000.
    s2=TemporalEventStore(h2,s.team_seasons)
    b2=_Backend(b.config,s2,4,'cpu');b2.initialize(fit)
    assert b.statistics_scale.state()==b2.statistics_scale.state()
    b.step();b2.step()
    for key,value in b.model.state_dict().items():torch.testing.assert_close(value,b2.model.state_dict()[key],rtol=0,atol=0)


def test_optional_cnn_reconstruction_memory_statistics_all_coexist(tmp_path):
    from test_heatmap_cnn import graph_setup
    from xdiyo_analytics.neural import Reconstruction
    spec,s=graph_setup(reconstruction=Reconstruction(loss='kl'))
    cfg=replace(spec.config,boundary_memory=BoundaryMemory('fixed_days',7,'2020-01-01T00:00:00+01:00'),statistics_forecast=forecast('combined'))
    a=TemporalGraphAdapter(cfg,s,control=TrainingControl(max_steps=1));a.fit(context(s,[1,2,3]))
    assert {'train_loss','train_statistics_loss','train_reconstruction_loss'} <= set(a.training_history_.metric)
    assert np.isfinite(a.training_history_.value).all()
    pred=context(s,[7,8],fit=False)
    p=a.predict(pred)
    ser=TemporalGraphSerializer(embed_event_store=True);ser.save(a,tmp_path)
    restored=ser.load(tmp_path)
    for key in p:pd.testing.assert_frame_equal(p[key],restored.predict(pred)[key],check_exact=True)


def test_deterministic_extensions_repeated_full_adamw_state():
    from test_graph_preparation_memory import assert_exact
    cfg=config(deterministic=True,boundary_memory=BoundaryMemory('fixed_days',7,'2020-01-01T00:00:00+01:00'),statistics_forecast=forecast('bottleneck','negative_binomial'))
    results=[]
    for _ in range(2):
        b=_Backend(cfg,sample(),7,'cpu');fit=context(b.store,[1,2,3]);fit.validation=context(b.store,[5,6]);b.initialize(fit)
        for step in range(2):b.checkpoint_metrics('train_objective',b.step())
        results.append((b.model.state_dict(),b.optimizer.state_dict(),[p.grad for p in b.model.parameters()]))
    assert_exact(results[0],results[1])


@pytest.mark.parametrize('cnn',[False,True])
def test_disabled_network_exact_against_base_revision(cnn):
    import types
    from pathlib import Path
    from copy import deepcopy
    from test_graph_preparation_memory import assert_exact
    from test_heatmap_cnn import graph_setup
    from xdiyo_analytics.neural import Reconstruction
    module=types.ModuleType('xdiyo_analytics.graphs._base_network');module.__package__='xdiyo_analytics.graphs'
    path=Path(__file__).parent/'reference/gat_network_de32880.txt'
    exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),module.__dict__)
    if cnn:
        spec,store=graph_setup(reconstruction=Reconstruction(loss='kl'));cfg=spec.config
    else:cfg,store=config(),sample()
    fit=context(store,[1,2,3]);fit.validation=context(store,[5,6])
    current=_Backend(cfg,store,11,'cpu');current.initialize(fit)
    base=_Backend(cfg,store,11,'cpu');base.initialize(fit)
    pre=base.preprocessor
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        base.model=module.TemporalNetwork(cfg,len(pre.leagues),2*len(pre.edge_columns)+len(pre.leagues)+2,2*len(pre.fixture_columns)+len(pre.leagues)+1)
    assert_exact(current.model.state_dict(),base.model.state_dict())
    base.optimizer=torch.optim.AdamW(base.model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    for _ in range(2):
        a=current.checkpoint_metrics('train_loss',current.step());b=base.checkpoint_metrics('train_loss',base.step())
        assert_exact(a,b);assert_exact(current.model.state_dict(),base.model.state_dict());assert_exact(current.optimizer.state_dict(),base.optimizer.state_dict())
        assert_exact([p.grad for p in current.model.parameters()],[p.grad for p in base.model.parameters()])


@pytest.mark.parametrize('route',['auxiliary','combined','bottleneck'])
def test_explicitly_scheduled_gpu_extensions(tmp_path,route):
    import os
    if os.environ.get('XDIYO_RUN_GPU_AUDIT')!='1':
        pytest.skip('Opt-in GPU audit: run only when research workers are idle.')
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from xdiyo_analytics.graphs.execution import prepare_determinism
    cfg=config(deterministic=True,boundary_memory=BoundaryMemory('fixed_days',7,'2020-01-01T00:00:00+01:00'),statistics_forecast=forecast(route,'negative_binomial'))
    prepare_determinism(cfg,'cuda:0')
    store=sample();fit=context(store,[1,2,3]);fit.validation=context(store,[5,6]);pred=context(store,[7,8],fit=False)
    a=TemporalGraphAdapter(cfg,store,device='cuda:0',control=TrainingControl(max_steps=2));a.fit(fit)
    p=a.predict(pred)
    from test_graph_preparation_memory import assert_exact
    repeated=TemporalGraphAdapter(cfg,store,device='cuda:0',control=TrainingControl(max_steps=2));repeated.fit(fit)
    pd.testing.assert_frame_equal(a.training_history_,repeated.training_history_,check_exact=True)
    assert_exact(a.backend_.model.state_dict(),repeated.backend_.model.state_dict())
    assert_exact(a.backend_.optimizer.state_dict(),repeated.backend_.optimizer.state_dict())
    assert_exact([v.grad for v in a.backend_.model.parameters()],[v.grad for v in repeated.backend_.model.parameters()])
    assert np.isfinite(a.training_history_.value).all()
    ser=TemporalGraphSerializer(embed_event_store=True,device='cuda:0');ser.save(a,tmp_path);b=ser.load(tmp_path)
    for key in p:pd.testing.assert_frame_equal(p[key],b.predict(pred)[key],check_exact=True)
    c=TemporalGraphSerializer(device='cpu').load(tmp_path)
    for key in p:np.testing.assert_allclose(p[key],c.predict(pred)[key],atol=1e-6,rtol=1e-5)



def test_first_boundary_numeric_carry_matches_causal_native_encoder():
    cfg=config(boundary_memory=BoundaryMemory('fixed_days',7,'2020-01-01T00:00:00+01:00'))
    b=_Backend(cfg,sample(),5,'cpu');b.initialize(context(b.store,[1,2,3]))
    boundary=pd.Timestamp('2020-01-07T23:00Z')
    snap=b.store.snapshot(boundary,cfg)
    with torch.no_grad():expected=b.model.encode(b.preprocessor.tensors(snap,'cpu'))
    replay=BoundaryReplay(b,boundary);replay.advance(boundary)
    for i,team in enumerate(snap.teams):torch.testing.assert_close(replay.state[team],expected[i],rtol=0,atol=0)


def test_point_normalization_extreme_relative_target_weights():
    cfg=StatisticsForecast((StatisticTarget('goals','home',weight=1e308),StatisticTarget('goals','away',weight=1e308)))
    head=StatisticsHead(cfg,3,4);forecasts,_=head(torch.ones(2,3))
    y=torch.tensor([[1.,2.],[2.,3.]])
    loss=head.losses(forecasts,y,torch.ones(2),torch.ones(2)*2)
    assert torch.isfinite(loss);loss.backward()
