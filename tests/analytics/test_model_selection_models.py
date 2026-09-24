from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge,Lasso,ElasticNet,LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from model_selection_samples import sample,plan,development,outer
from split_samples import rows_for
from xdiyo_analytics.analysis import PreTrainingAnalysis
from xdiyo_analytics.reporting import FeatureSelector,FeatureSelection,StudyResult
from xdiyo_analytics.selection import Candidate,ModelSelection,MetricSelection
from xdiyo_analytics.training import EstimatorAdapter
from xdiyo_analytics.evaluation import Metric

@pytest.mark.parametrize('family',['ridge','lasso','elastic'])
@pytest.mark.parametrize('layout',['match','team_match'])
def test_real_linear_fits_use_inner_only_scaling_and_satisfy_independent_solutions(family,layout):
    data=sample(layout=layout,shuffle=True);splits=plan(data);made=[]
    def make():
        estimator=Ridge(alpha=.4) if family=='ridge' else Lasso(alpha=.05,tol=1e-11,max_iter=20000) if family=='lasso' else ElasticNet(alpha=.1,l1_ratio=.4,tol=1e-11,max_iter=20000)
        pipeline=Pipeline([('scale',StandardScaler()),('model',estimator)]);made.append(pipeline)
        return EstimatorAdapter(pipeline)
    search=ModelSelection([Candidate(family,make,feature_columns=['signal','wave'])],metrics='mse')
    result=search.run(data,splits,development_positions=development(data))
    assert len(made)==2 and made[0].named_steps['scale'] is not made[1].named_steps['scale']
    for saved,fold in zip(result.winner.training.folds,splits.folds):
        pipeline=saved.model.estimator;scale=pipeline.named_steps['scale'];model=pipeline.named_steps['model']
        x=data.X[['signal','wave']].iloc[fold.train].to_numpy();y=data.y.target.iloc[fold.train].to_numpy()
        np.testing.assert_allclose(scale.mean_,x.mean(axis=0))
        np.testing.assert_allclose(scale.var_,x.var(axis=0))
        z=(x-x.mean(axis=0))/x.std(axis=0);residual=z@model.coef_+model.intercept_-y
        assert residual.mean()==pytest.approx(0.,abs=1e-10)
        if family=='ridge':
            coefficients=np.linalg.solve(z.T@z+.4*np.eye(2),z.T@(y-y.mean()))
            np.testing.assert_allclose(model.coef_,coefficients,rtol=1e-10,atol=1e-10)
        else:
            alpha,l1=(.05,1.) if family=='lasso' else (.1,.4)
            gradient=z.T@residual/len(z)+alpha*(1-l1)*model.coef_
            active=np.abs(model.coef_)>1e-8
            np.testing.assert_allclose(gradient[active],-alpha*l1*np.sign(model.coef_[active]),atol=1e-8)
            assert (np.abs(gradient[~active])<=alpha*l1+1e-8).all()
        test=(data.X[['signal','wave']].iloc[fold.test].to_numpy()-scale.mean_)/scale.scale_
        np.testing.assert_allclose(saved.predictions['predict'].target,test@model.coef_+model.intercept_)

def test_selected_pipeline_relearns_feature_set_and_scaler_for_outer_fit():
    data=sample();seen=[]
    class Selector(FeatureSelector):
        def select(self,context):
            name='wave' if len(context.X)>=10 else 'signal'
            seen.append((len(context.X),set(context.metadata.case),name))
            return StudyResult('selection',selection=FeatureSelection((name,),pd.DataFrame({'feature':[name]})))
    made=[]
    def make():
        pipeline=Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=.1))]);made.append(pipeline);return EstimatorAdapter(pipeline)
    request=Candidate('relearn',make,pre_analysis=PreTrainingAnalysis({'pick':Selector(type='per_fold',partition='train')}),features_from='pick')
    result=ModelSelection([request],metrics='mse').run(data,plan(data),development_positions=development(data))
    final=result.evaluate(data,outer(data),validation=None,control=None).folds[0]
    assert seen[-1]==(12,set(range(12)),'wave') and len(seen)==3
    assert all(f.feature_columns==('signal',) for f in result.winner.training.folds)
    assert final.feature_columns==('wave',) and final.model.estimator is made[-1]
    assert len(made)==3 and len({id(p.named_steps['scale']) for p in made})==3
    np.testing.assert_allclose(made[-1].named_steps['scale'].mean_,data.X[['wave']].iloc[development(data)].mean())

def test_classifier_probability_evidence_uses_declared_classes_and_scored_rows():
    data=sample();data.y['target']=(data.metadata.case%2).astype(int)
    candidates=[Candidate(f'C={c}',lambda c=c:EstimatorAdapter(Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=c,random_state=7))]),('predict','predict_proba')),config={'C':c}) for c in (.1,10.)]
    result=ModelSelection(candidates,metrics=[Metric('log_loss',output='predict_proba')],decision=MetricSelection('log_loss')).run(data,plan(data),development_positions=development(data))
    expected={}
    for trial in result.trials:
        losses=[]
        for fold in trial.training.folds:
            probabilities=fold.predictions['predict_proba'].loc[fold.score_positions]
            assert probabilities.columns.tolist()==[('target',0),('target',1)]
            for row in fold.score_positions:
                truth=int(data.y.target.iloc[row]);losses.append(-np.log(probabilities.loc[row,('target',truth)]))
        expected[trial.candidate.name]=np.mean(losses)
        actual=next(m['value'] for m in trial.record['metrics'] if m['type']=='overall' and m['metric']=='log_loss')
        assert actual==pytest.approx(expected[trial.candidate.name])
    assert result.winner.candidate.name==min(expected,key=expected.get)
