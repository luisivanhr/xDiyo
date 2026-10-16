"""Ayre f224a48 repairs and native ordered-quantile regression contracts."""
from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
import torch
from test_temporal_graph import sample, config, context
from test_graph_extensions import threads
from xdiyo_analytics.graphs import BoundaryMemory, StatisticsForecast, StatisticTarget, TemporalGraphAdapter, TemporalGraphSerializer, TemporalEventStore
from xdiyo_analytics.graphs.adapter import _Backend
from xdiyo_analytics.graphs.memory import BoundaryReplay
from xdiyo_analytics.graphs.statistics_network import StatisticsHead
from xdiyo_analytics.graphs.statistics import StatisticsScale
from xdiyo_analytics.graphs.extensions import quantile_column
from xdiyo_analytics.training import TrainingControl
from xdiyo_analytics.reporting import GraphStatisticsReporter


def report(adapter, query, output=None):
    output = adapter.predict(query)['statistics'] if output is None else output.copy()
    meta = query.metadata.copy()
    meta.index = pd.MultiIndex.from_arrays([[0]*len(meta),meta.index],names=['fold_id','row_position'])
    output.index = meta.index
    return GraphStatisticsReporter(type='overall',partition='score').run(SimpleNamespace(pooling=None,predictions={'statistics':output},metadata=meta,models={0:adapter}))


def quantile_config(levels=(.25,.5,.75),route='combined'):
    return StatisticsForecast(tuple(StatisticTarget('goals',side,family='quantile',quantiles=levels) for side in ('home','away')),route=route)


@pytest.mark.parametrize('memory',[BoundaryMemory(),BoundaryMemory('fixed_days',7,'2020-01-01T00:00:00+01:00')])
def test_empty_history_prior_only_save_attach_first_evidence(tmp_path,memory):
    full=sample(); empty=TemporalEventStore(full.history.iloc[:0],full.team_seasons,schedule=full.schedule)
    cfg=config(boundary_memory=memory,deterministic=True)
    a=TemporalGraphAdapter(cfg,empty,control=TrainingControl(max_steps=1));a.fit(context(empty,[1,2,3]))
    query=context(empty,[7,8],fit=False);p=a.predict(query)
    assert np.isfinite(p['predict_proba']).all().all()
    r=BoundaryReplay(a.backend_,pd.Timestamp('2020-03-01T12:00Z'));snap,data=r.snapshot(pd.Timestamp('2020-03-01T12:00Z'),[1])
    assert not r.state and r.start is None and len(snap.edges)==0
    torch.testing.assert_close(a.backend_.model.encode(data),a.backend_.model.prior(data['league'],data['entry']),atol=0,rtol=0)
    queries=pd.DataFrame({'team_id':[1],'issuance_at':[pd.Timestamp('2020-03-01T12:00Z')]})
    assert len(a.encode_teams(queries))==1
    assert a.encode_teams(queries.iloc[:0]).empty
    assert a.predict(context(empty,[],fit=False))['predict_proba'].empty
    ser=TemporalGraphSerializer(embed_event_store=True);ser.save(a,tmp_path);loaded=ser.load(tmp_path)
    pd.testing.assert_frame_equal(p['predict_proba'],loaded.predict(query)['predict_proba'],check_exact=True)
    loaded.attach_event_store(full)
    snap=loaded.event_store.snapshot(pd.Timestamp('2020-01-03T12:00Z'),cfg)
    assert set(snap.edges.event_id)=={1}
    assert np.isfinite(loaded.predict(query)['predict_proba']).all().all()
    with pytest.raises(ValueError,match='No usable'):
        b=_Backend(replace(cfg,statistics_forecast=quantile_config()),empty,1,'cpu');b.initialize(context(empty,[1,2,3]))


def test_fixed_calendar_midnight_morning_evening_and_availability(tmp_path):
    s=sample();h=s.history;m=s.team_seasons
    extra=m.iloc[[0]].copy();extra['entry_at']=pd.Timestamp('2020-01-20T18:00Z');extra['team_id']=99;extra['season_id']=901
    m=pd.concat([m,extra],ignore_index=True)
    h['result_available_at']=h.kickoff_at+pd.Timedelta(hours=3)
    h.loc[h.event_id==6,'result_available_at']=pd.Timestamp('2020-01-20T16:00Z')
    s=TemporalEventStore(h,m);cfg=config(boundary_memory=BoundaryMemory())
    a=TemporalGraphAdapter(cfg,s,control=TrainingControl(max_steps=1));a.fit(context(s,[1,2,3]))
    morning,evening=pd.Timestamp('2020-01-20T10:00Z'),pd.Timestamp('2020-01-20T21:00Z')
    r=BoundaryReplay(a.backend_,evening);snap1,d1=r.snapshot(morning,[1,2]);snap2,d2=r.snapshot(evening,[1,2])
    assert r.start==pd.Timestamp('2020-01-19T23:00Z')
    assert 6 not in s.snapshot(morning,cfg).edges.event_id.values
    assert 6 not in s.snapshot(evening,cfg).edges.event_id.values
    for key in d1:torch.testing.assert_close(d1[key],d2[key],rtol=0,atol=0)
    assert 6 in s.snapshot(pd.Timestamp('2020-01-21T12:00Z'),cfg).edges.event_id.values
    q=pd.DataFrame({'team_id':[1,1],'issuance_at':[morning,evening]})
    z=a.encode_teams(q);ser=TemporalGraphSerializer(embed_event_store=True);ser.save(a,tmp_path)
    pd.testing.assert_frame_equal(z,ser.load(tmp_path).encode_teams(q),check_exact=True)


@pytest.mark.parametrize('value,component',[(0.,0),(100.,1),(43.,2)])
def test_inflated_beta_logs_preserve_extreme_native_scores(tmp_path,value,component):
    t=StatisticTarget('ballPossession',family='inflated_beta');s=sample();h=s.history
    h[t.column]=value;h[t.column.replace('team::','opponent::')]=value
    h.loc[h.event_id==9,t.column]=np.nan;h.loc[h.event_id==9,t.column.replace('team::','opponent::')]=np.nan
    s=TemporalEventStore(h,s.team_seasons);cfg=config(statistics_forecast=StatisticsForecast((t,)))
    a=TemporalGraphAdapter(cfg,s,control=TrainingControl(max_steps=1));a.fit(context(s,[1,2,3]))
    head=a.backend_.model.statistics_head
    with torch.no_grad():
        for p in head.parameters():p.zero_()
        head.head[-1].bias[component]=-110.
    a.backend_.refresh_version()
    f,_=head(torch.zeros(1,head.head[0].in_features))
    native=head.losses(f,torch.tensor([[value/100]]),torch.ones(1),torch.ones(1)).item()
    q=context(s,[7,8],fit=False);out=a.predict(q)['statistics'];r=report(a,q,out)
    assert out[t.name+'::p_'+('zero','one','interior')[component]].eq(0).all()
    assert r.tables['scores'].missing.iloc[0]==1
    assert r.tables['scores'].nll.iloc[0]==pytest.approx(native,rel=1e-6,abs=1e-5)
    assert np.isfinite(r.tables['forecasts'][['lower_10','upper_90']]).all().all()
    ser=TemporalGraphSerializer(embed_event_store=True);ser.save(a,tmp_path);loaded=ser.load(tmp_path)
    pd.testing.assert_frame_equal(out,loaded.predict(q)['statistics'],check_exact=True)
    pd.testing.assert_frame_equal(r.tables['scores'],report(loaded,q).tables['scores'],check_exact=True)


@pytest.mark.parametrize('levels',[(),(0.,), (1.,),(.5,.5),(float('nan'),),(float('inf'),),(-.1,),None])
def test_invalid_quantiles(levels):
    with pytest.raises(ValueError,match='Quantiles'):StatisticTarget(family='quantile',quantiles=levels)


def test_pinball_hand_values_gradients_masks_and_normalization():
    t=StatisticTarget('ballPossession',family='quantile',quantiles=(.8,.2));cfg=StatisticsForecast((t,))
    assert t.quantiles==(.2,.8)
    head=StatisticsHead(cfg,2,3);head.scale.fill_(2.)
    q=torch.tensor([[.1,.9],[.2,.8],[.5,.5],[float('nan'),float('nan')]],requires_grad=True)
    y=torch.tensor([[.5],[.5],[.5],[float('nan')]])
    w=torch.tensor([1.,2.,3.,0.]);den=torch.tensor([6.])
    loss=head.losses([dict(quantiles=q)],y,w,den)
    # Each row averages the two pinballs, scaled by 2, then weights / 6.
    assert loss.item()==pytest.approx((.04+2*.03)/6)
    loss.backward()
    np.testing.assert_allclose(q.grad[:2].numpy(),[[-.2/24,.2/24],[-.2/12,.2/12]],atol=1e-8)
    assert torch.isfinite(q.grad).all() and (q.grad[3]==0).all()
    empty=head.losses([dict(quantiles=q)],y*float('nan'),w,torch.zeros(1));assert empty.item()==0
    # Equal forecasts and symmetric extra levels retain the effective weight.
    for levels in [( .5,),(.25,.5,.75)]:
        h=StatisticsHead(StatisticsForecast((replace(t,quantiles=levels),)),2,3)
        f=[dict(quantiles=torch.full((1,len(levels)),.2))]
        assert h.losses(f,torch.tensor([[.8]]),torch.ones(1),torch.ones(1)).item()==pytest.approx(.3)


@pytest.mark.parametrize('levels',[(.1,),(.25,.5,.75),(.9,.1,.3,.7,.5)])
@pytest.mark.parametrize('route',['auxiliary','combined','bottleneck'])
def test_quantile_native_shape_routes_save_report_and_schema(tmp_path,levels,route,monkeypatch):
    cfg=config(statistics_forecast=quantile_config(levels,route),boundary_memory=BoundaryMemory('fixed_days',7,'2020-01-01T00:00:00+01:00'))
    s=sample();a=TemporalGraphAdapter(cfg,s,control=TrainingControl(max_steps=1));fit=context(s,[1,2,3]);fit.validation=context(s,[5,6]);a.fit(fit)
    head=a.backend_.model.statistics_head;k=len(levels)
    assert head.head[-1].out_features==2*k and head.summary_dim==2*k
    original=head.head[0].in_features
    assert a.backend_.model.head[0].in_features==({'auxiliary':original,'combined':original+2*k,'bottleneck':2*k}[route])
    pred=context(s,[7,8],fit=False);out=a.predict(pred)['statistics'];ordered=sorted(levels)
    assert list(out)==[quantile_column(t,q) for t in cfg.statistics_forecast.targets for q in ordered]
    assert np.isfinite(out).all().all()
    for t in cfg.statistics_forecast.targets:assert (np.diff(out[[quantile_column(t,q) for q in ordered]],axis=1)>=0).all()
    assert not any('::mean' in c for c in out)
    r=report(a,pred,out)
    assert len(r.tables['quantile_scores'])==2*k
    assert 'median' in r.tables['forecasts'] if .5 in levels else 'median' not in r.tables['forecasts']
    if k==3:assert 'central 50%' in r.tables['scores'].interval_label.iloc[0]
    # No label lookup on inference, including loaded models.
    import xdiyo_analytics.graphs.statistics as stat
    monkeypatch.setattr(stat,'labels',lambda *args:(_ for _ in ()).throw(AssertionError('inference used labels')))
    ser=TemporalGraphSerializer(embed_event_store=True);ser.save(a,tmp_path);loaded=ser.load(tmp_path)
    pd.testing.assert_frame_equal(out,loaded.predict(pred)['statistics'],check_exact=True)
    for t in cfg.statistics_forecast.targets:assert t.quantiles==tuple(ordered)
    metadata=json.loads((tmp_path/'graph.json').read_text());metadata['config']['statistics_forecast']['targets'][0]['quantiles']=[.15]
    (tmp_path/'graph.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='target order/configuration'):ser.load(tmp_path)


@pytest.mark.parametrize('proportion',[False,True])
def test_quantile_ties_extremes_and_outcome_gradient(proportion):
    for route in ('auxiliary','combined','bottleneck'):
        t=StatisticTarget('ballPossession' if proportion else 'goals',family='quantile')
        cfg=config(statistics_forecast=StatisticsForecast((t,),route=route))
        from xdiyo_analytics.graphs.network import TemporalNetwork
        model=TemporalNetwork(cfg,1,5,2)
        h=model.statistics_head
        with torch.no_grad():
            for p in h.parameters():p.zero_()
        f,summary=h(torch.ones(3,h.head[0].in_features))
        assert torch.equal(f[0]['quantiles'][:,0],f[0]['quantiles'][:,1])
        for bias in (-1e20,1e20):
            with torch.no_grad():h.head[-1].bias.fill_(bias)
            f,summary=h(torch.ones(1,h.head[0].in_features))
            loss=h.losses(f,torch.zeros(1,1),torch.ones(1),torch.ones(1));loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(summary).all()
            assert (f[0]['quantiles']>=0).all()
            if proportion:assert (f[0]['quantiles']<=1).all()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in h.parameters())


def test_quantile_route_gradient_and_bottleneck_no_bypass():
    for route in ('auxiliary','combined','bottleneck'):
        b=_Backend(config(statistics_forecast=quantile_config(route=route)),sample(),9,'cpu');b.initialize(context(b.store,[1,2,3]))
        _,data,home,away,fixture=b.train_groups[1]
        b.model(data,home,away,fixture).sum().backward()
        assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in b.model.statistics_head.parameters())==(route!='auxiliary')
        assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in b.model.layers.parameters())
        if route=='bottleneck':
            with torch.no_grad():
                for p in b.model.statistics_head.parameters():p.zero_()
            torch.testing.assert_close(b.model(data,home,away,fixture),b.model(data,away,home,fixture+100),rtol=0,atol=0)
        elif route=='auxiliary':
            before=b.model(data,home,away,fixture).detach()
            with torch.no_grad():
                for p in b.model.statistics_head.parameters():p.add_(100)
            torch.testing.assert_close(before,b.model(data,home,away,fixture),rtol=0,atol=0)


@pytest.mark.parametrize('family',['point','poisson','negative_binomial','inflated_beta','quantile'])
@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_cached_loss_exact_full_update_against_uncached_reference(family,device,monkeypatch):
    import os, types
    from pathlib import Path
    from test_graph_preparation_memory import assert_exact
    from xdiyo_analytics.graphs.execution import prepare_determinism
    import xdiyo_analytics.graphs.supervision as supervision
    if device.startswith('cuda'):
        if os.environ.get('XDIYO_RUN_GPU_AUDIT')!='1' or not torch.cuda.is_available():pytest.skip('Explicit idle GPU audit required.')
        prepare_determinism(config(deterministic=True),device)
    module=types.ModuleType('xdiyo_analytics.graphs._reference_statistics');module.__package__='xdiyo_analytics.graphs'
    path=Path(__file__).parent/'reference/gat_supervision_f224a48.txt'
    exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),module.__dict__)
    optimized=supervision.statistics_loss
    s=sample();h=s.history
    t=StatisticTarget('ballPossession' if family=='inflated_beta' else 'cornerKicks',family=family)
    h[t.column]=np.where(h.event_id==3,np.nan,50. if t.proportion else 3.)
    h[t.column.replace('team::','opponent::')]=h[t.column]
    s=TemporalEventStore(h,s.team_seasons)
    cfg=config(statistics_forecast=StatisticsForecast((t,replace(t,side='away')),route='combined'),deterministic=True)
    fit=context(s,[1,2,3]);fit.validation=context(s,[5,6]);fit.sample_weight=np.array([.25,1.,2.])
    current=_Backend(cfg,s,17,device);current.initialize(fit)
    original=_Backend(cfg,s,17,device);original.initialize(fit)
    if family!='quantile':
        network=types.ModuleType('xdiyo_analytics.graphs._reference_statistics_network');network.__package__='xdiyo_analytics.graphs'
        path=Path(__file__).parent/'reference/gat_statistics_network_f224a48.txt'
        exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),network.__dict__)
        prior=original.model.statistics_head
        old=network.StatisticsHead(cfg.statistics_forecast,prior.head[0].in_features,cfg.hidden_width).to(device)
        old.load_state_dict(prior.state_dict());original.model.statistics_head=old
        original.optimizer=torch.optim.AdamW(original.model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    for _ in range(2):
        monkeypatch.setattr(supervision,'statistics_loss',optimized)
        a=current.checkpoint_metrics('train_objective',current.step())
        monkeypatch.setattr(supervision,'statistics_loss',module.statistics_loss)
        b=original.checkpoint_metrics('train_objective',original.step())
        assert_exact(a,b);assert_exact(current.model.state_dict(),original.model.state_dict())
        assert_exact(current.optimizer.state_dict(),original.optimizer.state_dict())
        assert_exact([p.grad for p in current.model.parameters()],[p.grad for p in original.model.parameters()])
    assert len(current._statistics_observations)==2


@pytest.mark.parametrize('levels',[(.1,),(.25,.5,.75),(.1,.3,.6,.8)])
def test_quantile_recipe_exports(levels):
    from xdiyo_analytics.ui.recipe import catalog_for_ui,node,export_python
    c=catalog_for_ui()
    value=node('graphs.StatisticsForecast',targets=[node('graphs.StatisticTarget',statistic='cornerKicks',side='away',period='1ST',family='quantile',quantiles=list(reversed(levels)))])
    built=c.build(value);assert built.targets[0].quantiles==levels
    assert c.build(json.loads(json.dumps(c.encode(built))))==built


@pytest.mark.parametrize('route,levels',[('auxiliary',(.1,)),('combined',(.25,.5,.75)),('bottleneck',(.1,.3,.7,.9))])
def test_quantile_idle_gpu_save_and_cross_device(tmp_path,route,levels):
    import os
    if os.environ.get('XDIYO_RUN_GPU_AUDIT')!='1' or not torch.cuda.is_available():pytest.skip('Explicit idle GPU audit required.')
    s=sample();cfg=config(statistics_forecast=quantile_config(levels,route),boundary_memory=BoundaryMemory(),deterministic=True)
    a=TemporalGraphAdapter(cfg,s,control=TrainingControl(max_steps=2,seed=19));a.configure_device('cuda:0');a.fit(context(s,[1,2,3]))
    q=context(s,[7,8],fit=False);before=a.predict(q)
    ser=TemporalGraphSerializer(embed_event_store=True,device='cuda:0');ser.save(a,tmp_path)
    for key,value in before.items():pd.testing.assert_frame_equal(value,ser.load(tmp_path).predict(q)[key],check_exact=True)
    cpu=TemporalGraphSerializer(device='cpu').load(tmp_path).predict(q)
    for key in ('predict_proba','statistics'):np.testing.assert_allclose(before[key],cpu[key],atol=1e-6,rtol=1e-5)


def test_quantile_future_mutation_and_discrete_reporting():
    s=sample();cfg=config(statistics_forecast=quantile_config((.1,.9),'bottleneck'))
    a=TemporalGraphAdapter(cfg,s,control=TrainingControl(max_steps=1));a.fit(context(s,[1,2,3]))
    query=context(s,[7],fit=False);before=a.predict(query)
    h=s.history;h.loc[h.event_id==8,['goals_for','goals_against']]=99
    # Controlled immutable-input perturbation: no refitting or teacher forcing.
    changed=TemporalEventStore(h,s.team_seasons);a.backend_.store=changed
    for key,value in before.items():pd.testing.assert_frame_equal(value,a.predict(query)[key],check_exact=True)
    a.backend_.store=s
    out=before['statistics'].copy();out[:]=0.
    r=report(a,query,out)
    assert set(r.tables['quantile_scores'].empirical_coverage)<= {0.,1.}
    assert r.tables['quantile_scores'].pinball.ge(0).all()
    assert 'median' not in r.tables['forecasts']


from test_ui_workflow import ui_recipe


def test_quantile_native_python_recipe_and_pipeline(ui_recipe,tmp_path):
    from test_graph_integration import graph_recipe
    from xdiyo_analytics.ui.recipe import node,prepare_recipe,run_recipe,export_python
    recipe=graph_recipe(ui_recipe,tmp_path)
    recipe['model']['params']['config']['params']['statistics_forecast']=node('graphs.StatisticsForecast',route='bottleneck',targets=[node('graphs.StatisticTarget',statistic='goals',side='home',family='quantile',quantiles=[.9,.1]),node('graphs.StatisticTarget',statistic='cornerKicks',side='away',family='quantile',quantiles=[.25,.5,.75])])
    recipe['post_reporters']['statistics']=node('reporting.GraphStatisticsReporter',type='overall',partition='score')
    namespace={};exec(export_python(recipe).split('prepared =')[0],namespace)
    assert namespace['recipe']['model']==recipe['model']
    prepared=prepare_recipe(recipe);result=run_recipe(recipe,prepared=prepared)
    frame=result.training.folds[0].predictions['statistics']
    assert frame.shape[1]==5 and np.isfinite(frame).all().all()


def test_twenty_four_outputs_and_training_only_scale():
    targets=tuple(StatisticTarget(stat,side,period='1ST',family='quantile') for stat in ('cornerKicks','totalShotsOnGoal','shotsOnGoal','accuratePasses') for side in ('home','away'))
    head=StatisticsHead(StatisticsForecast(targets),5,8)
    f,s=head(torch.randn(4,5));assert torch.stack([v['quantiles'] for v in f],dim=1).shape==(4,8,3)
    assert s.shape==(4,24) and head.head[-1].out_features==24
    # Quantile targets share the already-native train-only scale and missing mask.
    cfg=quantile_config();scale=StatisticsScale(cfg).fit(np.array([[0.,np.nan],[2.,4.],[3.,4.]]))
    assert scale.counts.tolist()==[3,2] and scale.scale[1]==1
    raw=np.array([[0.,1.],[2.,4.]])
    np.testing.assert_allclose(scale.inverse(scale.transform(raw)),raw,rtol=1e-15,atol=1e-15)


def test_cache_eliminates_repeated_missing_mask_nonzero():
    from xdiyo_analytics.graphs.supervision import prepare_observations
    cfg=quantile_config();head=StatisticsHead(cfg,3,4)
    y=torch.tensor([[1.,float('nan')],[0.,3.],[float('nan'),2.]])
    weights=torch.ones(3)
    groups=SimpleNamespace(plan=SimpleNamespace(descriptors=((None,np.arange(3)),)))
    denominator,obs=prepare_observations(groups,y,weights)
    f,_=head(torch.ones(3,3))
    def count(prepared):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            head.losses(f,y,weights,denominator,prepared=prepared)
        return sum(e.count for e in profile.key_averages() if e.key=='aten::nonzero')
    assert count(None)>0
    assert count(obs[0])==0


def test_legacy_artifact_without_quantile_fields_keeps_identity(tmp_path):
    cfg=config(statistics_forecast=StatisticsForecast((StatisticTarget('goals'),)))
    s=sample();a=TemporalGraphAdapter(cfg,s,control=TrainingControl(max_steps=1));a.fit(context(s,[1,2,3]))
    ser=TemporalGraphSerializer(embed_event_store=True);ser.save(a,tmp_path)
    p=tmp_path/'graph.json';meta=json.loads(p.read_text())
    for t in meta['config']['statistics_forecast']['targets']:del t['quantiles']
    assert all('quantiles' not in t for t in meta['statistics_scale']['targets'])
    p.write_text(json.dumps(meta))
    loaded=ser.load(tmp_path)
    assert loaded.backend_.model_version==a.backend_.model_version
    q=context(s,[7,8],fit=False)
    for key,value in a.predict(q).items():pd.testing.assert_frame_equal(value,loaded.predict(q)[key],check_exact=True)
