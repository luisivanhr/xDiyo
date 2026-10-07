from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from xdiyo_analytics.composition import *
from xdiyo_analytics.training import TrainingRunner
from xdiyo_analytics.evaluation import *
from test_stake_bankroll import batch
from test_model_composition import setup
from test_bet_tickets import ledger


def test_examples_are_executable_and_restore_without_changing_real_data():
    import runpy
    from pathlib import Path
    example=runpy.run_path(str(Path(__file__).resolve().parents[2]/'examples/composition_and_staking.py'))
    results=example['models']()
    assert len(results)==8
    assert example['tickets']()[-1].summary['ending_wealth']==150
    assert example['learned_allocation']().amounts.between(0,5).all()


def test_decision_missing_models_are_explicit_abstentions():
    offers=batch().assign(economic_key=['one','two'],quote_id=['a','b'],quote_at='2024-12-31')
    values=offers.assign(issued_at='2024-12-31',artifact_vintage='2024-12-02',trained_through='2024-12-01')
    context=DecisionContext('2025-01-01',FrozenTable(offers))
    assert DecisionLayer(('a','b'),missing='reject').decide({'a':values},context).selected.empty
    result=DecisionLayer(('a','b'),missing='reject',gate='or').decide({'a':values},context)
    assert len(result.selected)==2
    assert result.audit.loc[result.audit.model.eq('b'),'reason'].eq('missing_model').all()
    # A later vintage cannot retroactively certify an earlier issued prediction.
    with pytest.raises(ValueError,match='issue time'):
        DecisionLayer(('a',)).decide({'a':values.assign(artifact_vintage='2025-01-01')},context)


def test_leg_settlement_waits_for_late_missing_results_and_is_atomic():
    membership=pd.DataFrame({'ticket_id':['one','one','two'],'leg_id':[1,2,3]})
    outcomes=pd.DataFrame({'available_at':['2025-01-01','2025-01-05'],'return_multiplier':[0.,2.]},index=[1,2])
    result=settlements_from_legs(membership,outcomes)
    assert result.loc['one','available_at']==pd.Timestamp('2025-01-05',tz='UTC')
    assert result.loc['one','return_multiplier']==0
    assert pd.isna(result.loc['two','available_at'])
    book=BankrollLedger(100,'u')
    with pytest.raises(ValueError,match='exposure'):
        book.place(FrozenTable(batch()),pd.Series([10.,10.],index=['a','b']),'2025-01-01',exposure_columns=('missing',))
    assert not book.events and not book.open_tickets and not book._seen


def test_probability_sources_require_time_and_explicit_rate_assumptions():
    context=StakeContext(100,100,'u','2025-01-01')
    with pytest.raises(ValueError,match='issue timestamps'):
        ModelProbabilitySource().probabilities(FrozenTable(batch().drop(columns='probability_issued_at')),context)
    with pytest.raises(ValueError,match='unavailable'):
        ModelProbabilitySource().probabilities(FrozenTable(batch().assign(probability_issued_at='2025-01-02')),context)
    h=pd.DataFrame({'selection_id':['chosen'],'selected':[True],'won':[1],'available_at':['2024-01-01']})
    source=HistoricalRateSource('chosen','2024-01-02',learned_selection=False).fit(h)
    with pytest.raises(ValueError,match='exchangeable'):
        source.probabilities(FrozenTable(batch()),context)
    native=HistoricalRateSource('chosen','2024-01-02',learned_selection=False,exchangeable_within_strata=True,history=h)
    assert native.probabilities(FrozenTable(batch()),context).tolist()==pytest.approx([2/3,2/3])


def test_fit_budget_and_invalid_weights_precede_fit(monkeypatch):
    data,plan=setup()
    monkeypatch.setattr(Ridge,'fit',lambda *a,**k:pytest.fail('must reject before fitting'))
    schema=OutputSchema('target')
    with pytest.raises(ValueError,match='Weights'):
        TrainingRunner(ModelEnsemble({'a':Ridge(),'b':Ridge()},schema,Mean((0,0))).build).run(data,plan)
    model=ResidualModel(ModelNode(Estimator(Ridge()),{'predict':schema}),Estimator(Ridge()),
                        TrainingPlan(mode='algorithmic_residual',max_fits=1),TargetSpec('residual'))
    with pytest.raises(ValueError,match='budget'):
        TrainingRunner(lambda:model).run(data,plan)


def test_later_training_labels_do_not_change_earlier_oof_predictions():
    data,plan=setup()
    schema=OutputSchema('target')
    spec=ModelStack({'base':ModelNode(Estimator(Ridge()),{'predict':schema})},Estimator(Ridge()),OutputFeatures(('base',)),schema,
        TrainingPlan(mode='chronological',min_train_groups=4,availability_delay='3h',features_as_of_issue=True))
    first=TrainingRunner(spec.build).run(data,plan).folds[0].model
    changed=deepcopy(data)
    changed.y.iloc[8:12]=1e8
    second=TrainingRunner(spec.build).run(changed,plan).folds[0].model
    pd.testing.assert_frame_equal(first.oof_[('base','predict')].loc[4:7],second.oof_[('base','predict')].loc[4:7])


def test_singles_preserve_nominal_series_and_zero_funded_status():
    from xdiyo_analytics.evaluation.ticket_allocation import allocate_singles
    offers=ledger(('win','loss')).assign(stake=[2.,5.],issued_at=pd.Timestamp('2024-12-31',tz='UTC'))
    context=StakeContext(100,100,'u','2025-01-01')
    tickets,_,metrics=allocate_singles(offers,FixedStake(0,'u'),context)
    assert tickets.nominal_stake.tolist()==[2,5]
    assert tickets.profit.eq(0).all() and tickets.accounting_status.eq('unfunded').all()
    funded,_,_=allocate_singles(offers,FractionalKelly('u'),context,payoff='binary')
    assert funded.stake.gt(0).all()


def test_utility_labels_fail_future_or_ambiguous_outcomes():
    data=pd.DataFrame({'available_at':['2024-01-02']*3,'net_return_per_unit':[-1,0,2]})
    assert UtilityTarget().build(data,cutoff='2024-01-03').utility.tolist()==[0,0,1]
    assert UtilityTarget('allocation',scale=.02,max_fraction=.03).build(data,cutoff='2024-01-03').utility.tolist()==[0,0,.03]
    with pytest.raises(ValueError,match='unavailable'):
        UtilityTarget().build(data,cutoff='2024-01-01')


def test_composition_report_uses_retained_models_and_outputs():
    from xdiyo_analytics.reporting import CompositionReporter
    data,plan=setup()
    result=TrainingRunner(ModelEnsemble({'a':Ridge(),'b':Ridge()},OutputSchema('target'),Mean()).build).run(data,plan)
    fold=result.folds[0]
    context=SimpleNamespace(models={0:fold.model},predictions=fold.predictions)
    report=CompositionReporter(type='overall',partition='score').run(context)
    assert report.tables['composition_summary'].fits.tolist()==[2]
    assert 'child/a/predict' in report.tables


def test_child_weighting_counts_only_each_inner_population():
    from xdiyo_analytics.weighting import ClassWeightPolicy
    from sklearn.linear_model import LogisticRegression
    from xdiyo_analytics.composition.training import fit_node
    from xdiyo_analytics.training.contracts import FitContext
    data,_=setup(True)
    context=FitContext(data.X.iloc[:6],data.metadata.iloc[:6],'match',data.match_columns,0,y=data.y.iloc[:6])
    counts=[]
    class ObservedWeights(ClassWeightPolicy):
        def compute(self,ctx):
            counts.append(ctx.y.index.tolist())
            return super().compute(ctx)
    node=ModelNode(Estimator(LogisticRegression()),{'predict':OutputSchema('target',kind='labels',classes=('a','b'))},weighting=ObservedWeights())
    fit_node(node,context)
    assert counts==[list(range(6))]


def test_calibrated_chronological_stack_restores_every_stage(tmp_path,monkeypatch):
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from xdiyo_analytics.training import ProbabilityCalibrator,save_model,load_model
    data,plan=setup(True)
    schema=OutputSchema('target',kind='probability',classes=('a','b'),units='probability')
    calibration=ProbabilityCalibrator(method='sigmoid',response_method='decision_function',fraction=.4,
        availability_delay='3h',min_calibration_rows=2,min_calibration_per_class=1)
    child=ModelNode(Estimator(SVC(probability=False),preprocessors=(StandardScaler(),),prediction_methods=('predict','decision_function')),
        {'predict_proba':schema},calibration=calibration)
    stack=ModelStack({'svc':child},Estimator(LogisticRegression(),preprocessors=(StandardScaler(),),prediction_methods=('predict_proba',)),
        OutputFeatures(('p_a','p_b')),schema,TrainingPlan(mode='chronological',min_train_groups=8,availability_delay='3h',features_as_of_issue=True),output='predict_proba')
    result=TrainingRunner(stack.build).run(data,plan)
    path=save_model(result,tmp_path/'calibrated-stack')
    for cls in (SVC,LogisticRegression,StandardScaler):
        monkeypatch.setattr(cls,'fit',lambda *a,**k:pytest.fail('restoration cannot refit'))
    restored=load_model(path).predict(data,positions=np.arange(12,16))['predict_proba']
    original=result.folds[0].predictions['predict_proba']
    pd.testing.assert_frame_equal(restored,original)
    offers=batch().assign(economic_key=['one','two'],quote_id=['a','b'],quote_at='2024-12-31')
    context=DecisionContext('2025-01-01',FrozenTable(offers))
    def decide(p):
        values=offers.assign(probability=p.iloc[:2,1].to_numpy(),issued_at='2024-12-31',artifact_vintage='2024-12-02',trained_through='2024-12-01')
        return DecisionLayer(('svc',)).decide({'svc':values},context)
    pd.testing.assert_frame_equal(decide(original).audit,decide(restored).audit)


def test_composition_parallel_outer_folds_match_serial():
    from xdiyo_analytics.training import ExecutionPolicy
    from xdiyo_analytics.splits import SplitPlan
    data,plan=setup()
    plan=SplitPlan([deepcopy(plan.folds[0]),deepcopy(plan.folds[0])],plan.n_rows,plan.row_order)
    spec=ModelEnsemble({'a':Estimator(Ridge()),'b':Estimator(Ridge(alpha=2))},OutputSchema('target'),Mean())
    serial=TrainingRunner(spec.build).run(data,plan)
    parallel=TrainingRunner(spec.build,execution=ExecutionPolicy(2)).run(data,plan)
    for first,second in zip(serial.folds,parallel.folds):
        for key in first.predictions:
            pd.testing.assert_frame_equal(first.predictions[key],second.predictions[key])
