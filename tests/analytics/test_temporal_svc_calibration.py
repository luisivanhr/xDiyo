"""Independent sigmoid oracle and strict temporal margin population contracts."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import least_squares
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from split_samples import dataset, rows_for, cases
from test_ui_workflow import ui_recipe
from xdiyo_analytics.splits import Fold, SplitPlan
from xdiyo_analytics.training import (EstimatorAdapter, ProbabilityCalibrator, TrainingRunner,
    save_model, load_model, refit_model, PredictionContext)
from xdiyo_analytics.training.calibration_split import fold_boundary
from xdiyo_analytics.ui import run_recipe, prepare_recipe
from xdiyo_analytics.ui.recipe import node, export_python, export_notebook, catalog_for_ui


def sample(layout='match', shuffle=False):
    data = dataset(60, layout=layout, specs=[dict(day=i//3, round=i//6+1) for i in range(60)])
    case = data.metadata.case.to_numpy()
    data.X = pd.DataFrame({'signal': np.sin(case)*2 + (case % 2), 'missing': np.where(case % 7 == 0, np.nan, case % 5),
                           'empty': np.where(case < 36, np.nan, 0.2)})
    data.y = pd.DataFrame({'label': np.where(case % 2 == 0, 'draw', 'not draw')})
    data.metadata['available'] = data.metadata.kickoff_at + pd.Timedelta(hours=3)
    if shuffle:
        order = np.random.default_rng(12).permutation(len(data.X))
        for name in ('X','y','metadata'):
            setattr(data, name, getattr(data,name).iloc[order].reset_index(drop=True))
    return data


def policy(**kwargs):
    return ProbabilityCalibrator(**(dict(method='sigmoid', response_method='decision_function', fraction=.25,
        availability_delay='3h', prediction_lead='1h', prediction_group_by=('competition_id','season_id','stage','round')) | kwargs))


def plan(data):
    test = rows_for(data, range(48,60))
    return SplitPlan([Fold(rows_for(data, range(48)), test, test)],len(data.X), np.arange(len(data.X)))


def factory():
    return EstimatorAdapter(Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scale', StandardScaler()), ('svc', SVC(probability=False, gamma='scale', C=.7))]), ('predict','decision_function'))


def run(data, cal=None):
    return TrainingRunner(factory, calibration=cal or policy()).run(data, plan(data))


@pytest.mark.parametrize('layout', ['match','team_match'])
def test_order_identity_batches_and_no_internal_probabilities(layout, monkeypatch):
    # A call to libsvm's probability path is always a failure, including predictions.
    monkeypatch.setattr(SVC, '_predict_proba', lambda *a, **k: pytest.fail('Internal probability CV/output used'), raising=False)
    from sklearn.svm import _libsvm
    original_fit = _libsvm.fit
    calls = []
    def guarded_fit(*args, **kwargs):
        calls.append(kwargs['probability'])
        assert not kwargs['probability']
        return original_fit(*args, **kwargs)
    monkeypatch.setattr(_libsvm, 'fit', guarded_fit)
    a, b = sample(layout), sample(layout, True)
    left, right = run(a).folds[0], run(b).folds[0]
    for name in ('fit_positions','calibration_positions','test_positions'):
        assert cases(a,getattr(left,name)) == cases(b,getattr(right,name))
    assert cases(a,left.fit_positions) == set(range(36))
    assert cases(a,left.calibration_positions) == set(range(36,48))
    pipe = left.model.estimator.estimator
    assert pipe[-1].probability is False
    assert pipe[1].n_samples_seen_ == len(left.fit_positions)
    np.testing.assert_array_equal(pipe[0].statistics_, [np.median(a.X.signal.iloc[left.fit_positions]),
                                                       np.nanmedian(a.X.missing.iloc[left.fit_positions]),0.])
    np.testing.assert_allclose(pipe[1].mean_, pipe[0].transform(a.X.iloc[left.fit_positions]).mean(axis=0))
    np.testing.assert_array_equal(pipe[-1].dual_coef_, right.model.estimator.estimator[-1].dual_coef_)
    assert left.model.calibrator.models_ == right.model.calibrator.models_
    assert len(calls) == 2
    assert set(left.predictions) == {'predict','decision_function','predict_proba'}
    audit = left.training_summary['calibration']['temporal_split']
    assert pd.Timestamp(audit['latest_base_label_available_at']) <= pd.Timestamp(audit['first_calibration_cutoff'])
    assert pd.Timestamp(audit['latest_calibration_label_available_at']) <= pd.Timestamp(audit['fit_at'])
    assert audit['availability']['kind'] == 'kickoff_plus_delay_proxy'


@pytest.mark.parametrize('change', ['calibration_X','calibration_y','test_X','test_y'])
def test_future_mutation_cannot_change_base_fit(change):
    original = sample()
    initial = run(original).folds[0]
    changed = deepcopy(original)
    rows = initial.calibration_positions if change.startswith('calibration') else initial.test_positions
    if change.endswith('_X'):
        changed.X.iloc[rows] += 0.2 if change == 'calibration_X' else 100
    else:
        changed.y.iloc[rows,0] = np.where(changed.y.iloc[rows,0] == 'draw','not draw','draw')
    final = run(changed).folds[0]
    for i, attr in ((0,'statistics_'), (1,'mean_'), (1,'scale_'), (2,'support_vectors_'), (2,'dual_coef_'), (2,'intercept_')):
        np.testing.assert_array_equal(getattr(initial.model.estimator.estimator[i],attr), getattr(final.model.estimator.estimator[i],attr))
    assert initial.model.estimator.estimator[-1]._gamma == final.model.estimator.estimator[-1]._gamma
    if change.startswith('test'):
        assert initial.model.calibrator.models_ == final.model.calibrator.models_
    if change == 'test_y':
        pd.testing.assert_frame_equal(initial.predictions['predict_proba'],final.predictions['predict_proba'])
    if change == 'calibration_y':
        assert initial.model.calibrator.models_ != final.model.calibrator.models_


def margin_frame(s, classes=('yes','no')):
    frame = pd.DataFrame({'outcome': s})
    frame.attrs['response_schema'] = {'outcome': dict(classes=list(classes),positive_class=classes[1])}
    return frame


@pytest.mark.parametrize('classes', [('yes','no'), ('not draw','draw'), (7,2)])
def test_smoothed_sigmoid_matches_independent_score_equation_oracle(classes):
    s = np.linspace(-4,4,80)
    binary = ((np.arange(80)*17 % 31) / 31 < expit(.6*s-.2)).astype(float)
    y = pd.DataFrame({'outcome': np.where(binary==1, classes[1],classes[0])})
    raw = margin_frame(s,classes)
    cal = policy().fit(raw,y)
    n1,n0 = binary.sum(),len(binary)-binary.sum()
    targets = np.where(binary==1,(n1+1)/(n1+2),1/(n0+2))
    design = np.column_stack([s,np.ones(len(s))])
    ref = least_squares(lambda theta: design.T@(expit(design@theta)-targets),[0.,0.],gtol=1e-13,xtol=1e-13,ftol=1e-13)
    diag = cal.diagnostics_['outcome']
    np.testing.assert_allclose([diag['a'],diag['b']],ref.x,atol=1e-6)
    q = cal.transform(raw)
    np.testing.assert_allclose(q.iloc[:,1],expit(design@ref.x),atol=1e-7)
    assert list(q.columns.get_level_values(1)) == list(classes)
    expected_loss = np.mean(np.logaddexp(0,design@ref.x)-targets*(design@ref.x))
    assert diag['loss'] == pytest.approx(expected_loss,abs=1e-12)
    extreme = cal.transform(margin_frame([-1e308,1e308],classes))
    assert np.isfinite(extreme).all().all()
    np.testing.assert_array_equal(extreme.sum(axis=1),[1.,1.])


@pytest.mark.parametrize('failure', ['constant','infinite','unknown','one_class','too_small','schema','optimizer'])
def test_margin_failures_are_explicit(failure,monkeypatch):
    raw = margin_frame(np.arange(12,dtype=float))
    y = pd.DataFrame({'outcome':['yes','no']*6})
    if failure=='constant': raw.iloc[:,0]=1.
    if failure=='infinite': raw.iloc[0,0]=np.inf
    if failure=='unknown': y.iloc[0,0]='other'
    if failure=='one_class': y.iloc[:,0]='yes'
    if failure=='too_small': raw,y=raw.iloc[:4],y.iloc[:4]
    if failure=='schema': raw.attrs['response_schema']['outcome']['positive_class']='yes'
    if failure=='optimizer':
        monkeypatch.setattr('xdiyo_analytics.training.margins.minimize',lambda *a,**k:SimpleNamespace(success=False,message='deliberate failure'))
    with pytest.raises((ValueError,RuntimeError)):
        policy().fit(raw,y)


@pytest.mark.parametrize('missing', [False,True])
def test_delayed_labels_purge_whole_prediction_groups(missing):
    data = sample('team_match')
    data.metadata.loc[data.metadata.case==34,'available'] = pd.NaT if missing else pd.Timestamp('2024-02-01',tz='UTC')
    data.metadata.loc[data.metadata.case==46,'available'] = pd.Timestamp('2024-02-01',tz='UTC')
    result = run(data,policy(availability_delay=None,availability_column='available',min_calibration_rows=4)).folds[0]
    assert not cases(data,result.fit_positions) & set(range(30,36))
    assert not cases(data,result.calibration_positions) & set(range(42,48))
    removed = result.training_summary['calibration']['temporal_split']['purged']
    assert {data.metadata.iloc[r['row_position']]['case'] for r in removed} == set(range(30,36))|set(range(42,48))
    assert all(r['identity']['event_id'] > 2**63 for r in removed)


def test_prediction_cutoff_not_first_calibration_kickoff():
    data = sample()
    # Available 30 minutes before first calibration kickoff, but after its 1h prediction cutoff.
    data.metadata.loc[data.metadata.case==34,'available'] = pd.Timestamp('2024-01-12T23:30Z')
    result = run(data,policy(availability_delay=None,availability_column='available')).folds[0]
    assert cases(data,result.fit_positions) == set(range(30))


def test_boundary_groups_are_purged_without_tail_expansion():
    data = sample()
    result = run(data,policy(fraction=.2,min_calibration_rows=4)).folds[0] # four batches, still whole groups
    assert cases(data,result.calibration_positions) == set(range(36,48))
    result = run(data,policy(fraction=.15,min_calibration_rows=4)).folds[0] # three batches cross round 7
    assert cases(data,result.calibration_positions) == set(range(42,48))
    assert cases(data,result.fit_positions) == set(range(36))


def test_unusable_availability_and_invalid_config_fail_before_fit(monkeypatch):
    data = sample()
    monkeypatch.setattr(SVC,'fit',lambda *a,**k:pytest.fail('Fit must not run'))
    for cal in [policy(availability_delay=None),policy(availability_delay='30d'),policy(min_calibration_rows=99)]:
        with pytest.raises(ValueError): run(data,cal)
    with pytest.raises(ValueError,match='sigmoid'): ProbabilityCalibrator(response_method='decision_function')
    with pytest.raises(ValueError,match='probability=False'):
        TrainingRunner(lambda:EstimatorAdapter(SVC(probability=True),('decision_function',)),calibration=policy()).run(data,plan(data))


def test_persistence_refit_and_original_prediction_toggle(tmp_path):
    data = sample()
    result = run(data,policy(update_predict=False))
    fold = result.folds[0]
    assert 'predict_proba_raw' not in fold.predictions
    save_model(result,tmp_path/'margin-model')
    loaded = load_model(tmp_path/'margin-model')
    for method,frame in loaded.predict(data,positions=fold.test_positions).items():
        pd.testing.assert_frame_equal(frame,fold.predictions[method],check_exact=True)
    with pytest.raises(ValueError,match='issue_at'):
        refit_model(data,factory,train_positions=plan(data).folds[0].train,calibration=policy())
    refit = refit_model(data,factory,train_positions=plan(data).folds[0].train,
                        calibration=policy(issue_at='2024-01-16T23:00Z'))
    assert cases(data,refit.fit_positions)==set(range(36))
    assert refit.model.calibrator.response_schema_ == fold.model.calibrator.response_schema_
    native = fold.model.estimator.estimator.predict(data.X.iloc[fold.test_positions])
    np.testing.assert_array_equal(fold.predictions['predict'].iloc[:,0],native)


def test_recipe_roundtrip_reuse_and_configuration_identity(ui_recipe, monkeypatch):
    ui_recipe['model']=node('sklearn.svm.SVC',probability=False)
    ui_recipe['labels']['corners']=node('labels.Above',source=ui_recipe['labels']['corners'],threshold=7)
    ui_recipe['candidate']['calibration']=node('training.ProbabilityCalibrator',method='sigmoid',response_method='decision_function',
        fraction=.5,availability_delay='3h',prediction_lead='1h',min_calibration_rows=2,min_calibration_per_class=1)
    recipe=json.loads(json.dumps(ui_recipe))
    scope={}
    exec(export_python(recipe).split('result = run_recipe')[0],scope)
    assert scope['recipe']==recipe
    for cell in export_notebook(recipe)['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'margin-recipe','exec')
    result=run_recipe(recipe)
    assert result.post_report.studies
    assert 'predict_proba_raw' not in result.training.folds[0].predictions
    reused=run_recipe(recipe)
    assert reused.reused
    for method,frame in reused.training.folds[0].predictions.items():
        pd.testing.assert_frame_equal(frame,result.training.folds[0].predictions[method],check_exact=True)
    recipe['post_reporters'] = {
        'calibration': node('reporting.CalibrationReporter',type='overall',partition='score'),
        'bets': node('reporting.BetOutcomeReporter',type='overall',partition='score',offers={
            'no':node('evaluation.BetOffer',option=node('labels.BetOption',source=recipe['labels']['corners'],selection='no'),odds=2.)},
            policy=node('evaluation.HighestExpectedProfit',min_ev=-1.))}
    with monkeypatch.context() as patch:
        patch.setattr(SVC,'fit',lambda *a,**k:pytest.fail('Reporting must not refit'))
        reports=run_recipe(recipe)
        assert reports.reused
        bets=next(study for study in reports.post_report.studies if study.name.endswith('bets'))
        np.testing.assert_allclose(bets.result.tables['bets'].probability,
                                   result.training.folds[0].predictions['predict_proba'][('corners',0)])
    recipe['candidate']['calibration']['params']['prediction_lead']='2h'
    changed=run_recipe(recipe)
    assert not changed.reused and changed.record['run_id']!=result.record['run_id']


def test_old_serialized_calibrator_resolves_probability_mode():
    cal=ProbabilityCalibrator()
    del cal.__dict__['response_method']
    p=pd.DataFrame([[.2,.8],[.8,.2]],columns=pd.MultiIndex.from_product([['x'],[0,1]]))
    assert cal.response_method=='predict_proba'
    assert np.isfinite(cal.fit(p,pd.DataFrame({'x':[1,0]})).transform(p)).all().all()


def test_selection_weights_and_preprocessing_see_only_earlier_fit(monkeypatch):
    from xdiyo_analytics.analysis import PreTrainingAnalysis
    from xdiyo_analytics.reporting import TopKCorrelationSelector, ClassWeightReporter
    from xdiyo_analytics.weighting import ClassWeightPolicy
    data = sample()
    # Numeric binary labels for correlation selection; response identities remain 2/7.
    data.y[data.y.columns[0]] = np.where(data.metadata.case % 2 == 0,2,7)
    data.y = data.y.astype(int)
    runner = TrainingRunner(factory,calibration=policy())
    scopes=[]
    original=TopKCorrelationSelector.select
    def spy(self,context):
        scopes.append(set(context.metadata.case))
        return original(self,context)
    monkeypatch.setattr(TopKCorrelationSelector,'select',spy)
    weights=[]
    original_weights=ClassWeightPolicy.compute
    def weight_spy(self,context):
        weights.append(set(context.metadata.case))
        return original_weights(self,context)
    monkeypatch.setattr(ClassWeightPolicy,'compute',weight_spy)
    report=PreTrainingAnalysis({'pick':TopKCorrelationSelector(type='per_fold',partition='train',k=1),
        'weights':ClassWeightReporter(type='per_fold')}).run(data,split_plan=runner.selection_plan(data,plan(data)))
    result=runner.run(data,plan(data),analysis_report=report,features_from='pick',weights_from='weights')
    assert scopes == [set(range(36))]
    assert weights and all(scope == set(range(36)) for scope in weights)
    assert result.folds[0].model.estimator.estimator[1].n_samples_seen_ == 36


def test_nested_search_keeps_original_purge_and_calibration_positions():
    from xdiyo_analytics.selection import Candidate, ModelSelection
    data=sample(shuffle=True)
    data.metadata.loc[data.metadata.case==16,'available']=pd.Timestamp('2024-02-01',tz='UTC')
    def inner(local):
        test=rows_for(local,range(36,48))
        return SplitPlan([Fold(rows_for(local,range(36)),test,test)],len(local.X),np.arange(len(local.X)))
    candidate=Candidate('margin',factory,calibration=policy(availability_delay=None,availability_column='available',min_calibration_rows=4))
    selected=ModelSelection([candidate],metrics='accuracy').run_nested(data,plan(data),inner)
    trial=selected.selections[0].trials[0].training.folds[0]
    assert cases(data,trial.calibration_positions)==set(range(30,36))
    assert set(trial.model.calibration_positions_)==set(trial.calibration_positions)
    for audit in (trial.training_summary['calibration']['temporal_split'],trial.model.calibrator.split_audit_):
        assert audit['purged']
        for record in audit['purged']:
            assert data.metadata.iloc[record['row_position']].event_id == record['identity']['event_id']


@pytest.mark.parametrize('change', [dict(availability_delay='4h'),dict(prediction_lead='2h'),
    dict(fraction=.3),dict(min_calibration_rows=11),dict(min_calibration_per_class=3),
    dict(prediction_group_by=()),dict(issue_at='2024-02-01'),dict(cutoff_column='predict_at')])
def test_new_policy_fields_are_part_of_recovery_fingerprint(change):
    from xdiyo_analytics.experiments.recovery import signature
    from xdiyo_analytics.experiments.store import configuration_hash
    assert configuration_hash(signature(policy())) != configuration_hash(signature(policy(**change)))


@pytest.mark.parametrize('failure', ['one_fit_class','missing_kickoff','missing_identity','broken_match','multiclass'])
def test_invalid_populations_rejected_before_svc_fit(failure,monkeypatch):
    data=sample('team_match')
    splits=plan(data)
    if failure=='one_fit_class': data.y.loc[data.metadata.case < 36,'label']='draw'
    if failure=='multiclass': data.y.loc[data.metadata.case==1,'label']='third'
    if failure=='missing_kickoff': data.metadata.loc[0,'kickoff_at']=pd.NaT
    if failure=='missing_identity': data.metadata.loc[0,'event_id']=pd.NA
    if failure=='broken_match': splits.folds[0].train=splits.folds[0].train[1:]
    monkeypatch.setattr(SVC,'fit',lambda *a,**k:pytest.fail('Fit must not run'))
    with pytest.raises(ValueError): TrainingRunner(factory,calibration=policy()).run(data,splits)
