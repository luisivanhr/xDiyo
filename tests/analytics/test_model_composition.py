from copy import deepcopy
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xdiyo_analytics.composition import *
from xdiyo_analytics.training import EstimatorAdapter, TrainingRunner, save_model, load_model
from xdiyo_analytics.splits import Fold, SplitPlan
from xdiyo_analytics.ui.recipe import ModelFactory, catalog_for_ui, node, default_recipe
from model_selection_samples import sample


def setup(classifier=False):
    data = sample()
    if classifier:
        data.y['target'] = (data.metadata.case%2).map({0:'a',1:'b'})
    fold = Fold(np.arange(12),np.arange(12,16),np.arange(12,16),{'fit_at':data.metadata.kickoff_at.iloc[12]})
    return data,SplitPlan([fold],len(data.X),np.arange(len(data.X)))


def test_fixed_ensemble_matches_independent_models_and_restore(tmp_path,monkeypatch):
    data,plan = setup()
    schema = OutputSchema('target')
    source = EstimatorAdapter(make_pipeline(StandardScaler(),Ridge(alpha=2)))
    ensemble = ModelEnsemble({'a':source,'b':source},schema,Mean((1,3)))
    result = TrainingRunner(ensemble.build).run(data,plan)
    independent = TrainingRunner(lambda:deepcopy(source)).run(data,plan)
    pd.testing.assert_frame_equal(result.folds[0].predictions['predict'], independent.folds[0].predictions['predict'])
    assert not hasattr(source.estimator,'n_features_in_')
    assert result.folds[0].model.models_['a'] is not result.folds[0].model.models_['b']
    path = save_model(result,tmp_path/'graph')
    monkeypatch.setattr(Ridge,'fit',lambda *a,**k:pytest.fail('restore refit'))
    actual = load_model(path).predict(data,positions=np.arange(12,16))
    for key,frame in actual.items():
        pd.testing.assert_frame_equal(frame,result.folds[0].predictions[key])


@pytest.mark.parametrize('classifier',[False,True])
def test_chronological_stack_cutoffs_and_future_mutation(classifier):
    data,plan = setup(classifier)
    schema = OutputSchema('target',kind='probability',classes=('a','b'),units='probability') if classifier else OutputSchema('target')
    output = 'predict_proba' if classifier else 'predict'
    estimator = LogisticRegression() if classifier else Ridge()
    base = ModelNode(EstimatorAdapter(make_pipeline(StandardScaler(),estimator),(output,)),{output:schema})
    stack = ModelStack({'base':base},EstimatorAdapter(deepcopy(estimator),(output,)),
                      OutputFeatures(('p_a','p_b') if classifier else ('base',)),schema,
                      TrainingPlan(mode='chronological',n_splits=3,min_train_groups=4,availability_delay='3h',features_as_of_issue=True),output)
    result = TrainingRunner(stack.build).run(data,plan)
    fitted = result.folds[0].model
    assert fitted.training_summary_['oof_rows'] == 8
    for audit in fitted.audit_:
        assert set(audit['train_keys']).isdisjoint(audit['test_keys'])
        assert data.metadata.kickoff_at.iloc[audit['train_keys']].max()+pd.Timedelta('3h') <= pd.Timestamp(audit['fit_at'])
    changed = deepcopy(data)
    changed.y.iloc[12:] = 'b' if classifier else 9999.
    again = TrainingRunner(stack.build).run(changed,plan)
    pd.testing.assert_frame_equal(result.folds[0].predictions[output],again.folds[0].predictions[output])
    for key in fitted.oof_:
        pd.testing.assert_frame_equal(fitted.oof_[key],again.folds[0].model.oof_[key])


def test_catalog_native_composite_recipe():
    catalog = catalog_for_ui()
    recipe = default_recipe()
    recipe['preprocessors'] = []
    recipe['model'] = node('composition.ModelEnsemble',models={name:node('training.EstimatorAdapter',estimator=node('sklearn.linear_model.Ridge',alpha=a)) for name,a in [('a',1),('b',2)]},
                          schema=node('composition.OutputSchema',target='target'),reducer=node('composition.Mean'))
    factory = ModelFactory(recipe,catalog)
    data,plan = setup()
    result = TrainingRunner(factory).run(data,plan)
    assert set(result.folds[0].predictions) == {'predict','a/predict','b/predict'}
    recipe['preprocessors'] = [node('sklearn.preprocessing.StandardScaler')]
    with pytest.raises(ValueError,match='child preprocessing'):
        ModelFactory(recipe,catalog)()
