"""Independent count CDF oracles and native NB/Poisson pipeline integration."""
from copy import deepcopy
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom, poisson

from xdiyo_analytics.evaluation import Metric, evaluate_metrics
from xdiyo_analytics.evaluation.count_scores import DEFAULT_OU_LINES
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.reporting import PerformanceReporter
from xdiyo_analytics.selection import Candidate, ModelSelection, MetricSelection
from test_count_ou_brier import training_sample
from test_ui_workflow import ui_recipe


def inputs(means, dispersions=None, truth=None):
    index=pd.Index(range(len(means)),name='row_position')
    y=pd.DataFrame({'target':truth if truth is not None else [25]*len(means)},index=index)
    p=pd.DataFrame({('target','mean'):means},index=index)
    if dispersions is not None:p[('target','dispersion')]=dispersions
    p.columns=pd.MultiIndex.from_tuples(p.columns,names=['target','parameter'])
    return y,{'count_distribution':p}


def request(distribution,**kw):
    return Metric('count_ou_brier',parameters={'distribution':distribution,**kw},key='ou_brier')


def oracle(y,means,dispersions,lines=DEFAULT_OU_LINES):
    losses=[]
    for value,mu,alpha in zip(y,means,dispersions):
        rv=poisson(mu) if alpha==0 else nbinom(1/alpha,1/(1+alpha*mu))
        losses.append(np.mean([(rv.cdf(np.floor(line))-(value<line))**2 for line in lines]))
    return np.array(losses)


@pytest.mark.parametrize('distribution',['negative_binomial','poisson'])
@pytest.mark.parametrize('lines',[list(DEFAULT_OU_LINES),[.5,7.5,30.5]])
def test_analytic_scores_and_metadata(distribution,lines):
    means=[0.,1.,7.,10.,40.,100.]
    alpha=[0.,.2,1.,10.,.001,50.] if distribution=='negative_binomial' else [0.]*6
    truth=[0,25,26,3,50,200]
    y,p=inputs(means,alpha,truth)
    row=evaluate_metrics(y,p,[request(distribution,lines=lines)]).iloc[0]
    assert row.value==pytest.approx(oracle(truth,means,alpha,lines).mean(),abs=1e-14)
    assert (row.n,row.n_total,row.n_missing,row.status)==(6,6,0,'ok')
    assert row.output=='count_distribution' and row.direction=='minimize'
    assert row.calculation=='count_ou_brier' and row.target=='target'
    assert json.loads(row.parameters)=={'distribution':distribution,'lines':lines}
    from xdiyo_analytics.evaluation.probabilities import negative_binomial_bet_probabilities
    from xdiyo_analytics.labels import MatchTotal,BetOption
    native=[]
    for line in lines:
        for side in ['under','over']:
            events=negative_binomial_bet_probabilities(p['count_distribution'][('target','mean')],
                p['count_distribution'][('target','dispersion')],BetOption(MatchTotal(None),side,line=line))
            actual=y.target.lt(line) if side=='under' else y.target.gt(line)
            native.append((events.p_win-actual)**2)
    assert row.value==pytest.approx(np.mean(native))


def test_poisson_mean_only_equals_zero_dispersion_nb_and_explicit_output():
    y,p=inputs([0.,7.,30.],truth=[26,2,25])
    score=evaluate_metrics(y,p,[request('poisson')]).iloc[0]
    p['count_distribution'][('target','dispersion')]=0.
    nb=evaluate_metrics(y,p,[request('negative_binomial')]).iloc[0]
    assert score.value==nb.value and score.sample_hash==nb.sample_hash
    req=Metric('count_ou_brier',output='retained_distribution',parameters={'distribution':'poisson'})
    custom=evaluate_metrics(y,{'retained_distribution':p['count_distribution']},[req]).iloc[0]
    assert custom.value==score.value and custom.output=='retained_distribution'


@pytest.mark.parametrize('distribution',['negative_binomial','poisson'])
@pytest.mark.parametrize('bad',[np.nan,np.inf,-1.,None])
@pytest.mark.parametrize('parameter',['mean','dispersion'])
def test_bad_parameters_never_dropped(distribution,bad,parameter):
    y,p=inputs([8.,9.],[0.,0.])
    p['count_distribution'].loc[0,('target',parameter)]=bad
    with pytest.raises(ValueError,match='complete, finite and nonnegative'):
        evaluate_metrics(y,p,[request(distribution)])


@pytest.mark.parametrize('distribution',['negative_binomial','poisson'])
@pytest.mark.parametrize('columns', [[],['dispersion'],['mean','junk'],['mean','mean'],['mean','dispersion','extra'],['mean',10]])
def test_parameter_shapes_are_strict(distribution,columns):
    y,p=inputs([9.],[0.])
    p['count_distribution']=pd.DataFrame([[0.]*len(columns)],index=y.index,
        columns=pd.MultiIndex.from_product([['target'],columns]))
    with pytest.raises((ValueError,KeyError)):evaluate_metrics(y,p,[request(distribution)])


def test_no_implicit_predict_fallback_or_missing_nb_parameter():
    y,p=inputs([9.])
    with pytest.raises(ValueError,match='parameter columns'):evaluate_metrics(y,p,[request('negative_binomial')])
    with pytest.raises(KeyError,match='count_distribution'):
        evaluate_metrics(y,{'predict':y},[request('poisson')])
    with pytest.raises(ValueError,match='zero'):
        evaluate_metrics(*inputs([9.],[.1]),[request('poisson')])
    with pytest.raises(ValueError,match='distribution'):
        evaluate_metrics(y,p,[request('other')])
    p['count_distribution'].index=[3]
    with pytest.raises(ValueError,match='index/order'):evaluate_metrics(y,p,[request('poisson')])
    p['count_distribution'].index=y.index
    p['count_distribution'].columns=['mean']
    with pytest.raises(ValueError,match='MultiIndex'):evaluate_metrics(y,p,[request('poisson')])


@pytest.mark.parametrize('distribution',['negative_binomial','poisson'])
def test_empty_and_invalid_targets_lines(distribution):
    y,p=inputs([],[])
    row=evaluate_metrics(y,p,[request(distribution)]).iloc[0]
    assert (row.n,row.n_total,row.n_missing,row.status)==(0,0,0,'no_valid_observations')
    for truth in [np.nan,np.inf,-1,1.5,True]:
        with pytest.raises(ValueError,match='Targets'):
            evaluate_metrics(*inputs([8.],[0.],[truth]),[request(distribution)])
    for lines in [[],[7.],[7.25],[7.5,7.5],[True]]:
        with pytest.raises(ValueError,match='lines'):
            evaluate_metrics(*inputs([8.],[0.]),[request(distribution,lines=lines)])


def converted_training(distribution,repeat=False):
    training=training_sample(repeat=repeat)
    for i,fold in enumerate(training.folds):
        _,p=inputs([4. if i==0 else 12.]*len(fold.y_true),
                   [(.5 if i==0 else 2.) if distribution=='negative_binomial' else 0.]*len(fold.y_true))
        p['count_distribution'].index=fold.y_true.index
        fold.predictions=p
    return training


def report(training,distribution,pooling=None):
    return PostTrainingAnalysis({'selection_metrics':PerformanceReporter(type='overall',partition='score',
        pooling=pooling,metrics=[request(distribution)])}).run(training).studies[0].result.tables['metrics'].iloc[0]


@pytest.mark.parametrize('distribution',['negative_binomial','poisson'])
def test_fold_weighting_repeats_and_missing_source(distribution):
    training=converted_training(distribution)
    before=deepcopy(training.folds[1].predictions)
    losses=oracle([10,5,25],[4.,12.,12.],[.5,2.,2.] if distribution=='negative_binomial' else [0.]*3)
    row=report(training,distribution)
    assert row.value==pytest.approx(losses.mean())
    assert row.value!=pytest.approx((losses[0]+losses[1:].mean())/2)
    assert (row.n,row.n_missing)==(3,0)
    from test_count_ou_brier import report as categorical_report
    assert row.sample_hash==categorical_report(training_sample()).studies[0].result.tables['metrics'].iloc[0].sample_hash
    pd.testing.assert_frame_equal(before['count_distribution'],training.folds[1].predictions['count_distribution'])
    repeated=converted_training(distribution,repeat=True)
    losses=oracle([10,10,25],[4.,12.,12.],[.5,2.,2.] if distribution=='negative_binomial' else [0.]*3)
    for mode,idx in [('occurrences',[0,1,2]),('first',[0,2]),('last',[1,2])]:
        row=report(repeated,distribution,mode)
        assert row.value==pytest.approx(losses[idx].mean()) and row.n==len(idx)
    with pytest.raises(ValueError,match='mean pooling'):report(training,distribution,'mean')
    training.folds[1].predictions['count_distribution'].loc[:,('target','mean')]=np.nan
    with pytest.raises(ValueError,match='complete, finite'):report(training,distribution)
    if distribution=='negative_binomial':
        training=converted_training(distribution)
        training.folds[1].predictions['count_distribution']=training.folds[1].predictions['count_distribution'].drop(columns=[('target','dispersion')])
        with pytest.raises(ValueError,match='parameter columns'):report(training,distribution)


def test_mixed_poisson_source_shapes_are_equivalent_without_mutation():
    training=converted_training('poisson')
    expected=report(training,'poisson')
    training.folds[0].predictions['count_distribution']=training.folds[0].predictions['count_distribution'].drop(columns=[('target','dispersion')])
    original=training.folds[0].predictions['count_distribution'].copy()
    actual=report(training,'poisson')
    assert actual.value==expected.value and actual.sample_hash==expected.sample_hash
    pd.testing.assert_frame_equal(original,training.folds[0].predictions['count_distribution'])


@dataclass
class NativeCountFactory:
    distribution:str
    alpha:float
    def __call__(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import PoissonRegressor
        from xdiyo_analytics.training import NegativeBinomialRegressor,EstimatorAdapter
        model=(PoissonRegressor(alpha=self.alpha,max_iter=500) if self.distribution=='poisson' else
               NegativeBinomialRegressor(dispersion=.5,learn_dispersion=True,penalty='l2',
                   alpha=self.alpha,prediction='mode',max_iter=500))
        return EstimatorAdapter(make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),model))


@pytest.mark.parametrize('distribution',['negative_binomial','poisson'])
def test_native_pipeline_selection_and_retained_replay(distribution,tmp_path,monkeypatch):
    from model_selection_samples import sample,plan,development
    from xdiyo_analytics.experiments import ExperimentStore
    data=sample()
    data.y=data.y.round()
    candidates=[Candidate(str(alpha),NativeCountFactory(distribution,alpha),config={'alpha':alpha}) for alpha in [.1,1.]]
    store=ExperimentStore(tmp_path,distribution)
    search=ModelSelection(candidates,metrics=[request(distribution)],decision=MetricSelection('ou_brier'))
    first=search.run(data,plan(data),development_positions=development(data),experiment=store,resume=True)
    values=[]
    for trial in first.trials:
        losses=[]
        for fold in trial.training.folds:
            p=fold.predictions['count_distribution'].loc[fold.score_positions,'target']
            losses.extend(oracle(fold.y_true.loc[fold.score_positions,'target'],p['mean'],p['dispersion']))
            if distribution=='poisson':assert p.dispersion.eq(0).all()
            else: assert not fold.predictions['predict'].target.equals(fold.predictions['count_distribution'][('target','mean')])
        values.append(np.mean(losses))
    assert first.winner.candidate.name==candidates[int(np.argmin(values))].name
    import xdiyo_analytics.selection.core as core
    def forbid(*a,**k):raise AssertionError('No candidate refitting on retained replay')
    monkeypatch.setattr(core,'_fit_candidate',forbid)
    search.metrics=[request(distribution,lines=[7.5,11.5])]
    replay=search.run(data,plan(data),development_positions=development(data),experiment=store,resume=True)
    assert [t.saved_run_id for t in replay.trials]==[t.saved_run_id for t in first.trials]


@pytest.mark.parametrize('distribution',['negative_binomial','poisson'])
def test_ui_catalog_recipe_roundtrip(distribution):
    from xdiyo_analytics.ui import catalog_for_ui
    from xdiyo_analytics.ui.inventory import inventory
    catalog=catalog_for_ui()
    metric=Metric('count_ou_brier',target='total_corners',output='count_distribution',key='ou_brier',
        parameters={'distribution':distribution,'lines':list(DEFAULT_OU_LINES)})
    node=catalog.encode(metric)
    assert catalog.build(json.loads(json.dumps(node)),{})==metric
    item=next(m for m in inventory()['metrics'] if m['name']=='count_ou_brier')
    field=next(f for f in item['fields'] if f['name']=='distribution')
    assert field['default']=='categorical' and distribution in field['choices']
    assert item['output_by_distribution'][distribution]=='count_distribution'


@pytest.mark.parametrize('distribution',['negative_binomial','poisson'])
def test_full_native_ui_recipe_search(ui_recipe,distribution):
    from xdiyo_analytics.ui import run_recipe,catalog_for_ui
    from xdiyo_analytics.ui.recipe import node
    ui_recipe['model']=(node('training.NegativeBinomialRegressor',dispersion=.5,penalty='l2',max_iter=500)
        if distribution=='negative_binomial' else node('sklearn.linear_model.PoissonRegressor',max_iter=500))
    ui_recipe['preprocessors']=[node('sklearn.impute.SimpleImputer',strategy='median',keep_empty_features=True),
        node('sklearn.preprocessing.StandardScaler')]
    ui_recipe['target_transformer']=None
    ui_recipe['prediction_methods']=['predict']
    metric=catalog_for_ui().encode(request(distribution))
    ui_recipe['search']=dict(nested=False,grid={'alpha':[.1,1.]},split=node('splits.MatchKFold',n_splits=2),
        options={'metrics':[metric],'pooling':'occurrences',
            'decision':node('selection.MetricSelection',metric='ou_brier',direction='minimize')})
    ui_recipe['post_reporters']={'score':node('reporting.PerformanceReporter',type='overall',partition='score',metrics=[metric])}
    result=run_recipe(json.loads(json.dumps(ui_recipe)))
    assert len(result.selection.trials)==2
    assert all(t.record['status']=='complete' for t in result.selection.trials)
    assert all('count_distribution' in fold.predictions for fold in result.training.folds)
    row=result.post_report.studies[0].result.tables['metrics'].iloc[0]
    assert np.isfinite(row.value) and row.n==row.n_total and row.n_missing==0


def test_actual_ui_metric_editor_switches_output_and_preserves_custom_output():
    import shutil,subprocess
    from pathlib import Path
    executable=shutil.which('node')
    if executable is None:pytest.skip('Node required for metric editor callback check')
    root=Path(__file__).resolve().parents[2]
    script=r"""
    const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
    const source=fs.readFileSync(process.argv[1],'utf8');
    const metrics=JSON.parse(fs.readFileSync(process.argv[2],'utf8')).metrics;
    const fn=source.slice(source.indexOf('function metricsEditor('),source.indexOf('export function valueEditor('));
    let config,callback;
    const context={metrics,el:()=>({append(){},replaceChildren(){}}),valueEditor:()=>({}),
      fieldsEditor:(fields,p,cb)=>{config=p;callback=cb;return {};},update:(setter,v)=>setter(v)};
    vm.createContext(context);vm.runInContext(fn,context);
    let value=['count_ou_brier'];context.metricsEditor(value,v=>value=v);
    assert.equal(value[0],'count_ou_brier'); // viewing does not change defaults
    config.distribution='negative_binomial';callback();
    assert.equal(value[0].params.output,'count_distribution');
    config.distribution='poisson';config.lines=[5.5,9.5];callback();
    assert.equal(value[0].params.output,'count_distribution');
    assert.deepEqual(JSON.parse(JSON.stringify(value[0].params.parameters.lines)),[5.5,9.5]);
    config.distribution='categorical';callback();
    assert.equal(value[0].params.output,'predict_proba');
    value[0].params.output='custom_retained';context.metricsEditor(value,v=>value=v);
    config.distribution='poisson';callback();assert.equal(value[0].params.output,'custom_retained');
    """
    subprocess.run([executable,'-e',script,str(root/'src/xdiyo_analytics/ui/static/forms.js'),
        str(root/'src/xdiyo_analytics/ui/inventory.json')],check=True,capture_output=True,text=True)
