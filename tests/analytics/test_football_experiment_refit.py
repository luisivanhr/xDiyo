from dataclasses import dataclass
from functools import partial
import numpy as np
import pandas as pd
import pytest
from football_experiment_samples import prepared,post,ridge
from split_samples import rows_for
from xdiyo_analytics.analysis import PreTrainingAnalysis
from xdiyo_analytics.experiments import FootballExperiment,RefitPolicy
from xdiyo_analytics.reporting import FeatureSelector,FeatureSelection,StudyResult
from xdiyo_analytics.selection import Candidate
from xdiyo_analytics.training import EstimatorAdapter
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

@dataclass(kw_only=True)
class PopulationSelector(FeatureSelector):
    def select(self,context):
        # Evaluation trains on 12 cases; deployment refit uses a different set.
        name='wave' if context.metadata.case.nunique()>12 else 'signal'
        table=pd.DataFrame({'feature':[name]})
        return StudyResult('picked',selection=FeatureSelection((name,),table))

class ContextRidge(EstimatorAdapter):
    def __init__(self):super().__init__(make_pipeline(StandardScaler(),Ridge(alpha=.1)))
    def fit(self,context):self.context=context;return super().fit(context)
    def fit_controlled(self,context,control):
        assert control is None
        self.context=context
        return super().fit(context)

@pytest.mark.parametrize('layout',['match','team_match'])
def test_refit_relearns_selector_and_scaler_on_its_own_fitting_rows_and_restores_metadata(tmp_path,layout):
    preparation=prepared(layout=layout,holdout=True);data=preparation.dataset
    candidate=Candidate('Ridge selection',ContextRidge,
        pre_analysis=PreTrainingAnalysis({'pick':PopulationSelector(type='per_fold',partition='train')}),features_from='pick')
    train=rows_for(data,range(16));valid=rows_for(data,[15])
    experiment=FootballExperiment('Separate refit',output_dir=tmp_path)
    result=experiment.run(preparation,model=candidate,post_analysis=post(),
        refit_policy=RefitPolicy(train,validation=valid))
    evaluation=result.training.folds[0];refit=result.refit
    assert evaluation.feature_columns==('signal',) and refit.feature_columns==('wave',)
    assert refit.model is not evaluation.model
    assert set(refit.model.context.metadata.case)==set(range(15))
    assert set(refit.model.context.validation.metadata.case)=={15}
    np.testing.assert_array_equal(refit.fit_positions,train[~np.isin(train,valid)])
    np.testing.assert_allclose(refit.model.estimator.named_steps['standardscaler'].mean_,
        data.X[['wave']].iloc[refit.fit_positions].mean())
    loaded=experiment.load(result.record['run_id'])
    assert loaded.refit.model is None and loaded.refit.feature_columns==('wave',)
    np.testing.assert_array_equal(loaded.refit.validation_positions,valid)
    with pytest.raises(RuntimeError,match='no live fitted model'):loaded.refit.predict(data)

def test_refit_controls_default_independently_of_candidate_evaluation_validation():
    preparation=prepared(holdout=True);data=preparation.dataset
    candidate=Candidate('Ridge',ContextRidge,validation=rows_for(data,[11]))
    fitted=RefitPolicy(rows_for(data,range(12))).run(data,candidate)
    assert fitted.validation_positions.size==0 and fitted.fit_positions.size==12
    assert fitted.model.context.validation is None

def test_nested_refit_requires_explicit_candidate_and_accepts_one():
    preparation=prepared(holdout=True);data=preparation.dataset;rows=rows_for(data,range(12))
    with pytest.raises(ValueError,match='explicit Candidate'):RefitPolicy(rows).run(data,None)
    result=RefitPolicy(rows,candidate=ridge()).run(data,None)
    assert result.model is not None and result.fit_positions.tolist()==rows.tolist()
