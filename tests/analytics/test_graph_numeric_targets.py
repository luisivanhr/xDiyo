"""Generic own-team labels, transformed pinball, causal fitting and native export."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import numpy as np
import pandas as pd
import pytest
import torch
from test_temporal_graph import sample, config, context
from test_graph_extensions import threads
from test_graph_statistics_repair import report
from xdiyo_analytics.graphs import (
    NumericStatisticTarget as Target, StatisticsForecast, StatisticTarget,
    TemporalEventStore, TemporalGraphAdapter, TemporalGraphSerializer)
from xdiyo_analytics.graphs.statistics import labels, StatisticsScale, transform_target
from xdiyo_analytics.graphs.statistics_network import StatisticsHead
from xdiyo_analytics.training import TrainingControl, EarlyStopping

DOC = Path(__file__).resolve().parents[2]/'docs/analytics/full_statistics_quantiles'


def full_config():
    from xdiyo_analytics.ui.recipe import catalog_for_ui
    return catalog_for_ui().build(json.loads((DOC/'full_totals_auxiliary.json').read_text())['forecast'])


def populated(cfg):
    s=sample();h=s.history
    for i,t in enumerate(cfg.targets):
        if t.column in h:
            continue
        value = np.arange(len(h), dtype=float)
        if t.support == 'signed_continuous': value=(value-9)/10
        elif t.support == 'nonnegative_continuous': value=value/10
        elif t.support == 'bounded_continuous': value=value/2+30.25
        h[t.column]=value
    return TemporalEventStore(h,s.team_seasons)


def test_full_74_exact_side_order_and_222_outputs():
    cfg=full_config()
    assert len(cfg.targets)==74
    assert [t.side for t in cfg.targets]==['home']*37+['away']*37
    s=populated(cfg);meta=context(s,[1,2,3]).metadata
    y=labels(s,meta,cfg,pd.Timestamp('2021-01-01',tz='UTC'))
    for i,t in enumerate(cfg.targets):
        expected=s.history[s.history.side.eq(t.side)].set_index('event_id').loc[meta.event_id,t.column]
        np.testing.assert_array_equal(y[:,i],expected)
    head=StatisticsHead(cfg,4,8);f,z=head(torch.zeros(3,4))
    assert z.shape==(3,222) and sum(v['quantiles'].shape[1] for v in f)==222
    assert cfg.route=='auxiliary'
    assert {t.support for t in cfg.targets}=={'count','nonnegative_continuous','signed_continuous','bounded_continuous'}
    # Misleading provider names retain their verified numerator semantics.
    by={t.semantic_name:t for t in cfg.targets}
    assert by['ALL::aerialDuelsPercentage::value'].support=='count'
    assert by['ALL::wonTacklePercent::value'].support=='count'
    assert by['ALL::duelWonPercent::value'].support=='bounded_continuous'


def test_signed_pinball_hand_weights_missing_and_target_normalization():
    t=Target('cov','team::cov','coordinate squared','signed_continuous')
    cfg=StatisticsForecast((t,replace(t,semantic_name='abs',source_column='team::abs',support='nonnegative_continuous',weight=2)))
    head=StatisticsHead(cfg,2,4);head.scale.copy_(torch.tensor([2.,1.]))
    q1=torch.tensor([[-2.,0.,2.],[0.,1.,2.],[float('nan')]*3],requires_grad=True)
    q2=torch.tensor([[0.,1.,2.],[1.,2.,3.],[float('nan')]*3],requires_grad=True)
    y=torch.tensor([[-1.,0.],[2.,float('nan')],[float('nan'),float('nan')]])
    w=torch.tensor([1.,3.,0.]);den=torch.tensor([4.,1.])
    actual=head.losses([dict(quantiles=q1),dict(quantiles=q2)],y,w,den)
    expected=0
    for i,q in enumerate([q1,q2]):
        keep=torch.isfinite(y[:,i])&(w>0)
        residual=(y[keep,i,None]-q[keep])/head.scale[i]
        per_row=torch.maximum(torch.tensor(t.quantiles)*residual,(torch.tensor(t.quantiles)-1)*residual).mean(1)
        expected+=(i+1)/3*(per_row*w[keep]).sum()/den[i]
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    actual.backward()
    assert torch.isfinite(q1.grad).all() and torch.isfinite(q2.grad).all()
    assert (q1.grad[-1]==0).all() and (q2.grad[-1]==0).all()


@pytest.mark.parametrize('transform',['identity','signed_log1p'])
@pytest.mark.parametrize('scaling',['standard','robust','none'])
def test_training_only_scale_inverse_and_constant(transform,scaling):
    t=Target('cov','team::cov','coordinate squared','signed_continuous',transform=transform,scaling=scaling)
    cfg=StatisticsForecast((t,))
    values=np.array([[-3.],[0.],[3.],[1e10]])
    a=StatisticsScale(cfg).fit(values, [1,1,1,0])
    b=StatisticsScale(cfg).fit(values[:3])
    np.testing.assert_array_equal(a.mean,b.mean);np.testing.assert_array_equal(a.scale,b.scale)
    np.testing.assert_allclose(a.inverse(a.transform(values)),values,rtol=1e-14,atol=1e-14)
    assert a.counts.tolist()==[3]
    constant=StatisticsScale(cfg).fit(np.zeros((3,1)));assert constant.scale[0]==1
    z=torch.tensor([0.],requires_grad=True);transform_target(t,z).sum().backward();assert z.grad.item()==1
    with pytest.raises(ValueError,match='No usable'):StatisticsScale(cfg).fit(values,[0,0,0,0])


@pytest.mark.parametrize('change',[
    {'support':'bad'}, {'support':'signed_continuous','transform':'log1p'},
    {'support':'bounded_continuous'}, {'source_column':'opponent::x'},
    {'source_column':'team::1ST::x::value'}, {'source_column':'team::ALL::x::total'},
    {'weight':0}, {'quantiles':(.5,.5)}, {'family':'point'}, {'source_column':'goals_for','period':'1ST'}])
def test_schema_rejects_unsafe_combinations(change):
    args=dict(semantic_name='x',source_column='team::x',units='count',support='count')
    with pytest.raises(ValueError):Target(**(args|change))


def test_duplicate_aliases_and_wrong_numeric_support():
    t=Target('a','team::a','count','count')
    with pytest.raises(ValueError,match='Duplicate source'):
        StatisticsForecast((t,replace(t,semantic_name='alias')))
    cfg=StatisticsForecast((t,));s=populated(cfg)
    for value in (-1.,.5,np.inf,'bad'):
        h=s.history;h[t.column]=value
        bad=TemporalEventStore(h,s.team_seasons)
        with pytest.raises(ValueError):labels(bad,context(bad,[1]).metadata,cfg,pd.Timestamp('2021-01-01',tz='UTC'))
    with pytest.raises(ValueError,match='missing native column'):
        labels(sample(),context(sample(),[1]).metadata,cfg,pd.Timestamp('2021-01-01',tz='UTC'))


def test_release_masks_and_two_gathers_not_74(monkeypatch):
    cfg=full_config();s=populated(cfg);h=s.history
    h['result_available_at']=h.kickoff_at+pd.Timedelta(hours=3)
    h.loc[h.event_id.eq(3),'result_available_at']=pd.Timestamp('2021-01-01',tz='UTC')
    s=TemporalEventStore(h,s.team_seasons);meta=context(s,[1,2,3]).metadata
    original=pd.Series.reindex;calls=[]
    def spy(self,*args,**kwargs):
        calls.append(1);return original(self,*args,**kwargs)
    monkeypatch.setattr(pd.Series,'reindex',spy)
    y=labels(s,meta,cfg,pd.Timestamp('2020-02-01',tz='UTC'))
    assert len(calls)==2 and np.isnan(y[1]).all()
    assert np.isfinite(y[[0,2]]).all()


def test_full74_training_save_report_schema_and_auxiliary_isolation(tmp_path,monkeypatch):
    cfg=full_config();s=populated(cfg)
    a=TemporalGraphAdapter(config(statistics_forecast=cfg,deterministic=True),s,
        control=TrainingControl(max_steps=2,monitor='validation_loss',direction='minimize'))
    fit=context(s,[1,2,3]);fit.validation=context(s,[5,6]);a.fit(fit)
    q=context(s,[7,8],fit=False);pred=a.predict(q)
    assert pred['statistics'].shape==(2,222)
    r=report(a,q);assert len(r.tables['quantile_scores'])==222
    state=a.backend_.statistics_scale.state()
    assert len(state['target_manifest'])==74
    assert state['target_manifest'][3]['units'].startswith('squared')
    ser=TemporalGraphSerializer(embed_event_store=True);ser.save(a,tmp_path);loaded=ser.load(tmp_path)
    for key in ('predict_proba','statistics'):
        pd.testing.assert_frame_equal(pred[key],loaded.predict(q)[key],check_exact=True)
    with pytest.raises(ValueError,match='order/configuration'):
        StatisticsScale.from_state(replace(cfg,targets=cfg.targets[::-1]),state)
    tampered=copy.deepcopy(state);tampered['transforms'][0]='identity'
    with pytest.raises(ValueError,match='manifest'):
        StatisticsScale.from_state(cfg,tampered)
    with torch.no_grad():a.backend_.model.statistics_head.head[-1].bias.add_(1)
    a.backend_.refresh_version()
    after=a.predict(q)
    pd.testing.assert_frame_equal(pred['predict_proba'],after['predict_proba'],check_exact=True)
    assert not pred['statistics'].equals(after['statistics'])
    # Isolate auxiliary gradients from outcome loss.
    b=a.backend_;b.model.zero_grad(set_to_none=True)
    positions,data,home,away,fixture=b.train_groups[0]
    _,_,forecasts=b.model(data,home,away,fixture,return_statistics=True)
    ix=torch.as_tensor(positions)
    loss=b.model.statistics_head.losses(forecasts,b.train_statistics[ix],b.train_weights[ix],
        (torch.isfinite(b.train_statistics)*b.train_weights[:,None]).sum(0))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in b.model.statistics_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in b.model.named_parameters() if not n.startswith(('head.','statistics_head.')))
    assert all(p.grad is None for p in b.model.head.parameters())
    # Future labels and maps are not inference inputs; no refit.
    h=s.history
    for t in cfg.targets:
        if t.column != 'goals_for':h.loc[h.event_id.ge(8),t.column]=np.nan
    b.store=TemporalEventStore(h,s.team_seasons)
    for key,value in after.items():pd.testing.assert_frame_equal(value,a.predict(q)[key],check_exact=True)


def test_validation_auxiliary_unavailable_is_not_zero():
    t=Target('x','team::x','count','count',transform='log1p');cfg=StatisticsForecast((t,))
    s=populated(cfg);h=s.history;h.loc[h.event_id.ge(6),t.column]=np.nan
    s=TemporalEventStore(h,s.team_seasons)
    from xdiyo_analytics.graphs.adapter import _Backend
    b=_Backend(config(statistics_forecast=cfg),s,1,'cpu')
    fit=context(s,[1,2,3]);fit.validation=context(s,[5,6]);b.initialize(fit)
    values=b.step()
    assert np.isfinite(values['validation_loss'])
    assert np.isnan(values['validation_statistics_loss']) and np.isnan(values['validation_objective'])


from test_ui_workflow import ui_recipe


def test_generic_builder_json_python_and_pipeline(ui_recipe,tmp_path):
    from test_graph_integration import graph_recipe
    from xdiyo_analytics.ui.recipe import catalog_for_ui,node,prepare_recipe,run_recipe,export_python
    c=catalog_for_ui();cfg=full_config()
    assert c.build(json.loads(json.dumps(c.encode(cfg))))==cfg
    recipe=graph_recipe(ui_recipe,tmp_path)
    recipe['model']['params']['config']['params']['statistics_forecast']=node('graphs.StatisticsForecast',targets=[
        node('graphs.NumericStatisticTarget',semantic_name='goals',source_column='goals_for',units='count',support='count',transform='log1p'),
        node('graphs.NumericStatisticTarget',semantic_name='corners',source_column='team::ALL::Match overview::cornerKicks::value',units='count',support='count',side='away',transform='log1p')])
    ns={};exec(export_python(recipe).split('prepared =')[0],ns);assert ns['recipe']['model']==recipe['model']
    prepared=prepare_recipe(recipe);result=run_recipe(recipe,prepared=prepared)
    assert result.training.folds[0].predictions['statistics'].shape[1]==6

def test_ce_controls_checkpoint_and_patience_when_auxiliary_improves(monkeypatch):
    from xdiyo_analytics.graphs.adapter import _Backend
    original=_Backend.step
    def controlled(self):
        values=original(self)
        self.test_steps=getattr(self,'test_steps',0)+1
        i=self.test_steps-1
        values.update(validation_loss=[1.,.8,.9,1.][i],
                      validation_statistics_loss=[100.,50.,2.,0.][i],
                      validation_objective=[101.,50.8,2.9,1.][i])
        return values
    monkeypatch.setattr(_Backend,'step',controlled)
    cfg=StatisticsForecast((Target('x','team::x','count','count'),))
    s=populated(cfg)
    a=TemporalGraphAdapter(config(statistics_forecast=cfg),s,
        control=TrainingControl(max_steps=4,monitor='validation_loss',direction='minimize',early_stopping=EarlyStopping(patience=1)))
    fit=context(s,[1,2,3]);fit.validation=context(s,[5,6]);a.fit(fit)
    assert a.training_summary_['best_step']==2
    assert a.training_summary_['attempts'][0]['steps']==3
    assert a.training_summary_['attempts'][0]['termination_reason']=='early_stopping'


def test_two_arm_recipe_configuration():
    import runpy
    example=runpy.run_path(str(DOC/'recipe.py'))
    columns=tuple(f'bayes_{i}' for i in range(11))
    arms=example['full_totals_arms'](config(),columns)
    assert arms['totals'].fixture_features==()
    assert arms['totals_bayesian'].fixture_features==columns
    for arm in arms.values():
        assert arm.boundary_memory is None and arm.heatmaps is None
        assert len(arm.statistics_forecast.targets)==74
        assert arm.statistics_forecast.route=='auxiliary'
    control=example['selection_control'](max_steps=2)
    assert control.monitor=='validation_loss' and control.direction=='minimize'


def test_validation_changes_cannot_fit_scaler_and_zero_weight_coverage():
    from xdiyo_analytics.graphs.adapter import _Backend
    t=Target('x','team::x','count','count',transform='log1p')
    cfg=StatisticsForecast((t,));s=populated(cfg)
    fit=context(s,[1,2,3]);fit.validation=context(s,[5,6])
    fit.sample_weight=pd.Series([1.,0.,1.],index=fit.X.index)
    a=_Backend(config(statistics_forecast=cfg),s,1,'cpu');a.initialize(fit)
    h=s.history;h.loc[h.event_id.ge(6),t.column]=99999
    changed=TemporalEventStore(h,s.team_seasons)
    b=_Backend(config(statistics_forecast=cfg),changed,1,'cpu');b.initialize(fit)
    assert a.statistics_scale.state()==b.statistics_scale.state()
    assert a.statistics_scale.counts.tolist()==[2]


def test_all_missing_quantile_reporting_unavailable():
    from xdiyo_analytics.reporting.graph_quantiles import diagnose_quantiles
    from xdiyo_analytics.graphs.extensions import quantile_column
    t=Target('x','team::x','signed units','signed_continuous')
    q=pd.DataFrame({quantile_column(t,p):[-1.,0.] for p in t.quantiles})
    _,score,_,_=diagnose_quantiles(t,q,np.array([np.nan,np.nan]),0,q.index,1.)
    assert score['n']==0 and np.isnan(score['training_space_pinball'])
