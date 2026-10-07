from copy import deepcopy
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from xdiyo_analytics.composition import *
from xdiyo_analytics.training import EstimatorAdapter, TrainingRunner, save_model, load_model
from test_model_composition import setup


@pytest.mark.parametrize('mode',['algorithmic_residual','honest_error_meta'])
@pytest.mark.parametrize('classifier',[False,True])
def test_named_residual_schedules_and_restoration(mode,classifier,tmp_path,monkeypatch):
    data,plan=setup(classifier)
    output='predict_proba' if classifier else 'predict'
    schema=OutputSchema('target',kind='probability',classes=('a','b'),units='probability') if classifier else OutputSchema('target')
    base=ModelNode(EstimatorAdapter(LogisticRegression() if classifier else Ridge(alpha=100),(output,)),{output:schema})
    policy=TrainingPlan(mode=mode,min_train_groups=4,availability_delay='3h',features_as_of_issue=True)
    target=TargetSpec('residual' if mode=='algorithmic_residual' else 'oof_error','log_loss' if classifier else 'squared_error','logit' if classifier else 'identity')
    def factory():return ResidualModel(base,EstimatorAdapter(Ridge()),policy,target,learning_rate=.3)
    result=TrainingRunner(factory).run(data,plan)
    model=result.folds[0].model
    assert model.training_summary_['correction_rows']==(12 if mode=='algorithmic_residual' else 8)
    if classifier:
        p=result.folds[0].predictions['predict_proba']
        assert np.isfinite(p).all().all() and p.ge(0).all().all()
        np.testing.assert_allclose(p.sum(axis=1),1)
    changed=deepcopy(data)
    changed.y.iloc[12:]= 'b' if classifier else -10000
    again=TrainingRunner(factory).run(changed,plan)
    pd.testing.assert_frame_equal(result.folds[0].predictions[output],again.folds[0].predictions[output])
    path=save_model(result,tmp_path/'residual')
    monkeypatch.setattr(Ridge,'fit',lambda *a,**k:pytest.fail('no refit'))
    actual=load_model(path).predict(data,positions=np.arange(12,16))
    pd.testing.assert_frame_equal(actual[output],result.folds[0].predictions[output])


def test_learned_allocation_explicit_training_labels_and_cutoff():
    from xdiyo_analytics.training.contracts import FitContext
    from xdiyo_analytics.evaluation.stake_policy import LearnedAllocation,StakeContext,allocate_batch
    from xdiyo_analytics.evaluation.decision_layer import FrozenTable
    from test_stake_bankroll import batch
    index=pd.Index(range(4),name='row_position')
    X=pd.DataFrame({'odds':[2.,3.,4.,5.]},index=index)
    y=pd.DataFrame({'allocation':[.1,.15,.2,.25]},index=index)
    metadata=pd.DataFrame(dict(event_id=range(4),selection_id=['rule']*4,prediction_origin=['chronological_oof']*4,
                              issued_at=['2024-01-01']*4,trained_through=['2023-12-01']*4,available_at=['2024-01-02']*4),index=index)
    context=FitContext(X,metadata,'match',('event_id',),0,{'fit_at':'2024-02-01'},y=y)
    model=LearnedTargetAdapter(EstimatorAdapter(LinearRegression()),TargetSpec('allocation'),'available_at','rule')
    model.fit(context)
    policy=LearnedAllocation(model,('odds',),'u')
    result=allocate_batch(FrozenTable(batch()),policy,StakeContext(100,100,'u','2025-01-01'))
    np.testing.assert_allclose(result.amounts,[10,20],atol=.011)
    with pytest.raises(ValueError,match='strictly before'):
        policy.allocate(FrozenTable(batch()),StakeContext(100,100,'u','2024-01-01'))
    with pytest.raises(ValueError,match='unavailable'):
        model.fit(replace(context,metadata=metadata.assign(available_at='2025-01-01')))
