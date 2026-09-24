import numpy as np
import pandas as pd
import pytest
from training_samples import sample,plan
from xdiyo_analytics.training import EstimatorAdapter,TrainingRunner
from xdiyo_analytics.splits import Fold,SplitPlan


@pytest.mark.parametrize('layout',['match','team_match'])
@pytest.mark.parametrize('targets',[['first'],['second','first']])
def test_pipeline_preprocessing_and_ridge_match_independent_normal_equations(layout,targets):
    pytest.importorskip('sklearn')
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    data=sample(layout=layout); folds=plan(data)
    data.X.loc[data.metadata.case.isin([1,8]),'signal']=np.nan
    data.X.loc[data.metadata.case==10,'curve']=10000.
    alpha=1.25
    def factory():return EstimatorAdapter(make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=alpha)))
    results=TrainingRunner(factory,feature_columns=['signal','curve'],target_columns=targets).run(data,folds)
    pipelines=[]
    for result in results.folds:
        train=data.X.iloc[result.train_positions][['signal','curve']].to_numpy()
        test=data.X.iloc[result.test_positions][['signal','curve']].to_numpy()
        medians=np.nanmedian(train,axis=0)
        train=np.where(np.isnan(train),medians,train); test=np.where(np.isnan(test),medians,test)
        means=train.mean(axis=0); scale=train.std(axis=0)
        z=(train-means)/scale; z_test=(test-means)/scale
        y=data.y.iloc[result.train_positions][targets].to_numpy()
        coefficients=np.linalg.solve(z.T@z+alpha*np.eye(2),z.T@(y-y.mean(axis=0)))
        expected=z_test@coefficients+y.mean(axis=0)
        np.testing.assert_allclose(result.predictions['predict'],expected,rtol=1e-11,atol=1e-11)
        pipeline=result.model.estimator; pipelines.append(pipeline)
        np.testing.assert_allclose(pipeline[0].statistics_,medians)
        np.testing.assert_allclose(pipeline[1].mean_,means)
        assert pipeline[1].n_samples_seen_==len(train)
    assert pipelines[0] is not pipelines[1]
    assert all(pipelines[0][i] is not pipelines[1][i] for i in range(3))


@pytest.mark.parametrize('targets',[['second'],['second','first']])
def test_estimator_receives_series_or_dataframe_in_selected_order(targets):
    class Spy:
        def fit(self,X,y):self.y=y.copy(); return self
        def predict(self,X):
            values=np.zeros((len(X),len(targets)))
            return values[:,0] if len(targets)==1 else values
    data=sample()
    result=TrainingRunner(lambda:EstimatorAdapter(Spy()),target_columns=targets).run(data,plan(data),fold_ids=[0]).folds[0]
    fitted=result.model.estimator.y
    if len(targets)==1:assert isinstance(fitted,pd.Series) and fitted.name=='second'
    else:assert isinstance(fitted,pd.DataFrame) and list(fitted)==targets
    assert list(result.predictions['predict'])==targets


def test_classification_probabilities_use_fitted_labels_and_preserve_absent_class_missing():
    pytest.importorskip('sklearn')
    from sklearn.dummy import DummyClassifier
    data=sample(); data.y=pd.DataFrame({'category':['a','a','b','b','b','c','c','a','b','c','a','c']})
    folds=plan(data)
    result=TrainingRunner(lambda:EstimatorAdapter(DummyClassifier(strategy='prior'),('predict','predict_proba'))).run(data,folds)
    for fold in result.folds:
        expected=data.y.iloc[fold.train_positions]['category'].value_counts(normalize=True).sort_index()
        actual=fold.predictions['predict_proba']
        assert actual.columns.names==['target','class']
        assert actual.columns.tolist()==[('category',name) for name in expected.index]
        np.testing.assert_allclose(actual,np.tile(expected.to_numpy(),(len(actual),1)))
    combined=result.prediction_frame('predict_proba')
    assert combined.loc[0,('category','c')].isna().all()
    assert combined.loc[1,('category','c')].notna().all()
    assert result.predictions_by_fold('predict_proba',scored_only=True)[1].empty


def test_real_multioutput_classifier_probability_list_has_target_class_identity():
    pytest.importorskip('sklearn')
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.dummy import DummyClassifier
    data=sample()
    data.y=pd.DataFrame({'binary':np.arange(len(data.X))%2,'three':np.arange(len(data.X))%3})
    result=TrainingRunner(lambda:EstimatorAdapter(MultiOutputClassifier(DummyClassifier(strategy='prior')),('predict_proba',)),target_columns=['three','binary']).run(data,plan(data))
    for fold in result.folds:
        assert list(fold.predictions)==['predict_proba']
        probabilities=fold.predictions['predict_proba']
        for target in ('three','binary'):
            expected=data.y.iloc[fold.train_positions][target].value_counts(normalize=True).sort_index()
            assert list(probabilities[target])==list(expected.index)
            np.testing.assert_allclose(probabilities[target],np.tile(expected.to_numpy(),(len(probabilities),1)))


@pytest.mark.parametrize('kind',['reversed_index','wrong_columns','wrong_shape','wrong_proba_index','wrong_proba_columns','no_classes','wrong_class_shape','multi_probability_array'])
def test_estimator_output_identity_and_shape_guards(kind):
    class BadEstimator:
        def fit(self,X,y):
            self.classes_=[np.array([0,1]),np.array([0,1])] if kind=='multi_probability_array' else np.array([0,1])
            if kind=='no_classes':del self.classes_
        def predict(self,X):
            if kind=='reversed_index':return pd.Series(np.zeros(len(X)),index=X.index[::-1])
            if kind=='wrong_columns':return pd.DataFrame({'other':np.zeros(len(X))},index=X.index)
            return np.zeros((len(X),2))
        def predict_proba(self,X):
            if kind=='wrong_proba_index':return pd.DataFrame(np.ones((len(X),2))/2,index=X.index[::-1],columns=[0,1])
            if kind=='wrong_proba_columns':return pd.DataFrame(np.ones((len(X),2))/2,index=X.index,columns=[1,0])
            if kind=='wrong_class_shape':return np.ones((len(X),3))/3
            return np.ones((len(X),2))/2
    is_proba=kind not in ('reversed_index','wrong_columns','wrong_shape')
    targets=['first','second'] if kind=='multi_probability_array' else 'first'
    data=sample()
    with pytest.raises(ValueError):
        TrainingRunner(lambda:EstimatorAdapter(BadEstimator(),('predict_proba',) if is_proba else ('predict',)),target_columns=targets).run(data,plan(data),fold_ids=[0])


@pytest.mark.parametrize('methods',[(),('predict','predict'),('decision_function',),'predict'])
def test_invalid_method_selection(methods):
    class Estimator:
        def fit(self,X,y):raise AssertionError('Should validate before fit')
        def predict(self,X):return X
    data=sample()
    with pytest.raises(ValueError,match='prediction_methods'):
        TrainingRunner(lambda:EstimatorAdapter(Estimator(),methods)).run(data,plan(data),fold_ids=[0])


def test_missing_estimator_method_is_explicit_before_fit():
    data=sample()
    with pytest.raises(TypeError,match='does not expose'):
        TrainingRunner(lambda:EstimatorAdapter(object())).run(data,plan(data),fold_ids=[0])
